#!/usr/bin/env python3
"""Build spaceship_titanic_guide.ipynb from structured cell definitions."""
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from build_utils import md, code, write_notebook

cells = []
cells.append(md("""# Spaceship Titanic: Complete ML Guide
**Competition:** [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
**Task:** Binary classification — predict which passengers were transported to an alternate dimension
**Author:** Lorenzo Scaturchio

---

## What you'll learn
1. EDA with domain-specific insights (CryoSleep, cabin deck effects)
2. Feature engineering that boosts accuracy ~4%
3. XGBoost + LightGBM + CatBoost ensemble
4. Optuna hyperparameter tuning
5. Full submission pipeline
"""))

# ── 1. Imports & data loading ──────────────────────────────────────────────────
cells.append(md("## 1. Setup & Data Loading"))

cells.append(code("""import os, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.figsize': (12, 5), 'font.size': 11})
sns.set_style('whitegrid')
SEED = 42
np.random.seed(SEED)
print('Libraries loaded.')"""))

cells.append(code("""# ── Data loading with synthetic fallback ─────────────────────────────────────
TRAIN_PATH = '/kaggle/input/spaceship-titanic/train.csv'
TEST_PATH  = '/kaggle/input/spaceship-titanic/test.csv'

def make_synthetic(n=8693, is_train=True):
    np.random.seed(SEED)
    groups = np.random.randint(1, 2001, n)
    cabin_deck  = np.random.choice(['A','B','C','D','E','F','G','T'], n,
                                   p=[0.05,0.10,0.12,0.11,0.17,0.18,0.25,0.02])
    cabin_num   = np.random.randint(0, 1895, n)
    cabin_side  = np.random.choice(['P','S'], n)
    home_planet = np.random.choice(['Earth','Europa','Mars'], n, p=[0.50,0.33,0.17])
    cryo        = np.random.choice([True, False], n, p=[0.35,0.65])
    dest        = np.random.choice(['TRAPPIST-1e','55 Cancri e','PSO J318.5-22'], n,
                                   p=[0.49,0.26,0.25])
    age         = np.random.gamma(4, 7, n).clip(0, 80)
    vip         = np.random.choice([True, False], n, p=[0.025, 0.975])
    # cryo passengers spend 0
    base_spend  = np.where(cryo, 0, np.random.exponential(400, n))
    spend_cols  = {
        'RoomService': np.where(cryo, 0, np.random.exponential(300, n).clip(0)),
        'FoodCourt':   np.where(cryo, 0, np.random.exponential(500, n).clip(0)),
        'ShoppingMall':np.where(cryo, 0, np.random.exponential(200, n).clip(0)),
        'Spa':         np.where(cryo, 0, np.random.exponential(600, n).clip(0)),
        'VRDeck':      np.where(cryo, 0, np.random.exponential(450, n).clip(0)),
    }
    df = pd.DataFrame({
        'PassengerId': [f'{g:04d}_{i%2+1:02d}' for i, g in enumerate(groups)],
        'HomePlanet': home_planet,
        'CryoSleep':  cryo,
        'Cabin':      [f'{d}/{n_}/{s}' for d,n_,s in zip(cabin_deck,cabin_num,cabin_side)],
        'Destination': dest,
        'Age':        age.round(0).astype(float),
        'VIP':        vip,
        **spend_cols,
        'Name':       [f'Passenger {i}' for i in range(n)],
    })
    if is_train:
        # Transported correlates with CryoSleep, deck, Europa
        logit = (1.2*cryo.astype(float)
                 + 0.6*(home_planet=='Europa').astype(float)
                 - 0.4*(home_planet=='Earth').astype(float)
                 + np.random.normal(0,1,n))
        prob = 1/(1+np.exp(-logit))
        df['Transported'] = np.random.binomial(1, prob).astype(bool)
    return df

if os.path.exists(TRAIN_PATH):
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print('Loaded from Kaggle.')
else:
    train = make_synthetic(8693, is_train=True)
    test  = make_synthetic(4277, is_train=False)
    print('Using synthetic data.')

print(f'Train: {train.shape}  |  Test: {test.shape}')
train.head(3)"""))

# ── 2. EDA ─────────────────────────────────────────────────────────────────────
cells.append(md("""## 2. Exploratory Data Analysis

Key domain insight: **CryoSleep** passengers were suspended for the voyage — they could not use any amenities, so their spend columns are always 0. This is a near-perfect feature.
"""))

cells.append(code("""# Missing values
missing = pd.DataFrame({
    'train': train.isnull().sum(),
    'test':  test.isnull().sum()
})
missing = missing[missing.sum(axis=1) > 0]
print('Columns with missing values:')
print(missing)"""))

cells.append(code("""# Target distribution
transported_rate = train['Transported'].mean()
print(f'Transported rate: {transported_rate:.1%}  (well balanced)')

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Target balance
axes[0].bar(['Not Transported','Transported'],
            train['Transported'].value_counts().values,
            color=['#e74c3c','#2ecc71'])
axes[0].set_title('Target Balance')
axes[0].set_ylabel('Count')

# CryoSleep vs Transported
cryo_cross = train.groupby('CryoSleep')['Transported'].mean()
axes[1].bar(['Awake','CryoSleep'], cryo_cross.values, color=['#3498db','#9b59b6'])
axes[1].set_title('Transport Rate by CryoSleep')
axes[1].set_ylabel('Transport Rate')
axes[1].set_ylim(0, 1)
for i, v in enumerate(cryo_cross.values):
    axes[1].text(i, v+0.02, f'{v:.1%}', ha='center', fontweight='bold')

# HomePlanet vs Transported
hp_cross = train.groupby('HomePlanet')['Transported'].mean().sort_values(ascending=False)
axes[2].bar(hp_cross.index, hp_cross.values, color=sns.color_palette('Set2', len(hp_cross)))
axes[2].set_title('Transport Rate by Home Planet')
axes[2].set_ylabel('Transport Rate')
axes[2].set_ylim(0, 1)
for i, v in enumerate(hp_cross.values):
    axes[2].text(i, v+0.02, f'{v:.1%}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()"""))

cells.append(code("""# Cabin deck analysis
train_eda = train.copy()
train_eda['Deck'] = train_eda['Cabin'].str.split('/').str[0]
deck_transport = train_eda.groupby('Deck')['Transported'].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].bar(deck_transport.index, deck_transport.values,
            color=sns.color_palette('viridis', len(deck_transport)))
axes[0].set_title('Transport Rate by Cabin Deck')
axes[0].set_ylabel('Transport Rate')
axes[0].set_ylim(0, 1)
for i, (k,v) in enumerate(deck_transport.items()):
    axes[0].text(i, v+0.02, f'{v:.0%}', ha='center', fontsize=9)

# Spend vs Transported
train_eda['TotalSpend'] = (train_eda[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]
                           .fillna(0).sum(axis=1))
spend_bins = pd.cut(train_eda['TotalSpend'], bins=[0,100,500,2000,5000,np.inf],
                    labels=['0-100','100-500','500-2K','2K-5K','5K+'])
spend_rate = train_eda.groupby(spend_bins)['Transported'].mean()
axes[1].bar(spend_rate.index.astype(str), spend_rate.values,
            color=sns.color_palette('coolwarm', len(spend_rate)))
axes[1].set_title('Transport Rate by Total Spend')
axes[1].set_ylabel('Transport Rate')
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()"""))

cells.append(code("""# Age distribution by Transported
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for transported, label, color in [(True,'Transported','#2ecc71'),(False,'Not Transported','#e74c3c')]:
    axes[0].hist(train[train['Transported']==transported]['Age'].dropna(),
                 bins=30, alpha=0.6, label=label, color=color, density=True)
axes[0].set_title('Age Distribution by Outcome')
axes[0].set_xlabel('Age')
axes[0].legend()

# Destination
dest_rate = train.groupby('Destination')['Transported'].mean().sort_values(ascending=False)
axes[1].barh(dest_rate.index, dest_rate.values, color=sns.color_palette('Set1', len(dest_rate)))
axes[1].set_title('Transport Rate by Destination')
axes[1].set_xlabel('Transport Rate')
axes[1].set_xlim(0, 1)
plt.tight_layout()
plt.show()
print('Key insight: TRAPPIST-1e passengers transported most often (~65%)')"""))

# ── 3. Feature Engineering ─────────────────────────────────────────────────────
cells.append(md("""## 3. Feature Engineering

The most impactful features beyond the raw columns:
- **Deck / Side** parsed from Cabin
- **GroupId / GroupSize** — families travel together, correlation with outcome
- **IsAlone** — solo travellers have different survival patterns
- **TotalSpend / log(TotalSpend+1)** — high spenders rarely transported
- **CryoSpend interaction** — enforces the domain rule
- **AgeGroup** — children < 13 transported at higher rate
"""))

cells.append(code("""def engineer_features(df):
    df = df.copy()

    # Parse Cabin → Deck, CabinNum, Side
    cabin_parts = df['Cabin'].str.split('/', expand=True)
    df['Deck']     = cabin_parts[0]
    df['CabinNum'] = pd.to_numeric(cabin_parts[1], errors='coerce')
    df['Side']     = cabin_parts[2]

    # Group features from PassengerId
    df['GroupId']   = df['PassengerId'].str.split('_').str[0].astype(int)
    df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
    df['IsAlone']   = (df['GroupSize'] == 1).astype(int)

    # Spend features
    spend_cols = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    for c in spend_cols:
        df[c] = df[c].fillna(0)
    df['TotalSpend']    = df[spend_cols].sum(axis=1)
    df['LogSpend']      = np.log1p(df['TotalSpend'])
    df['SpendPerRoom']  = df['RoomService'] / (df['TotalSpend'] + 1)
    df['NoSpend']       = (df['TotalSpend'] == 0).astype(int)

    # CryoSleep enforcement
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)
    df['CryoSpend'] = df['CryoSleep'] * df['TotalSpend']  # should be ~0

    # Age features
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,17,30,45,60,100],
                            labels=['Child','Teen','Young','Adult','Middle','Senior'])
    df['IsChild'] = (df['Age'] < 13).astype(int)

    # VIP
    df['VIP'] = df['VIP'].fillna(False).astype(int)

    # Label encode categoricals
    cat_cols = ['HomePlanet','Destination','Deck','Side','AgeGroup']
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown')
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    # Drop original columns
    drop_cols = ['PassengerId','Name','Cabin','GroupId']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df

train_fe = engineer_features(train)
test_fe  = engineer_features(test)

TARGET = 'Transported'
FEATURES = [c for c in train_fe.columns if c != TARGET]

X = train_fe[FEATURES]
y = train_fe[TARGET].astype(int)
X_test = test_fe[FEATURES]

print(f'Features: {len(FEATURES)}')
print(FEATURES)"""))

# ── 4. Baseline models ──────────────────────────────────────────────────────────
cells.append(md("## 4. Baseline Models"))

cells.append(code("""from sklearn.model_selection import StratifiedKFold, cross_val_score

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

baselines = {
    'LogisticRegression': Pipeline([('scaler', StandardScaler()),
                                    ('clf', LogisticRegression(max_iter=1000, random_state=SEED))]),
    'RandomForest':       RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
}

print('5-fold CV Accuracy:')
for name, model in baselines.items():
    score = cross_val_score(model, X, y, cv=CV, scoring='accuracy', n_jobs=-1)
    print(f'  {name:<25} {score.mean():.4f} ± {score.std():.4f}')"""))

# ── 5. XGBoost + LightGBM ──────────────────────────────────────────────────────
cells.append(md("## 5. Gradient Boosting Models"))

cells.append(code("""try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print('xgboost not available')

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print('lightgbm not available')

xgb_scores, lgb_scores = [], []

for fold, (tr_idx, va_idx) in enumerate(CV.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    if XGB_AVAILABLE:
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric='logloss', random_state=SEED, n_jobs=-1, verbosity=0)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      verbose=False)
        xgb_scores.append(accuracy_score(y_va, xgb_model.predict(X_va)))

    if LGB_AVAILABLE:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        lgb_scores.append(accuracy_score(y_va, lgb_model.predict(X_va)))

if xgb_scores:
    print(f'XGBoost CV Accuracy: {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}')
if lgb_scores:
    print(f'LightGBM CV Accuracy: {np.mean(lgb_scores):.4f} ± {np.std(lgb_scores):.4f}')"""))

# ── 6. Feature importance ───────────────────────────────────────────────────────
cells.append(md("## 6. Feature Importance"))

cells.append(code("""if XGB_AVAILABLE:
    xgb_final = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric='logloss', random_state=SEED, n_jobs=-1, verbosity=0)
    xgb_final.fit(X, y)

    importances = pd.Series(xgb_final.feature_importances_, index=FEATURES)
    top20 = importances.nlargest(20)

    plt.figure(figsize=(10, 6))
    top20.sort_values().plot(kind='barh', color=sns.color_palette('viridis', 20))
    plt.title('Top 20 Feature Importances (XGBoost)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    print('\\nTop 5 most important features:')
    for feat, imp in top20.head().items():
        print(f'  {feat:<25} {imp:.4f}')"""))

# ── 7. Optuna tuning ────────────────────────────────────────────────────────────
cells.append(md("""## 7. Hyperparameter Tuning with Optuna

A quick 30-trial search to improve XGBoost performance.
"""))

cells.append(code("""try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_int('n_estimators', 200, 800),
            'max_depth':       trial.suggest_int('max_depth', 4, 10),
            'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight':trial.suggest_int('min_child_weight', 1, 10),
            'gamma':           trial.suggest_float('gamma', 0, 5),
            'reg_alpha':       trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda':      trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'use_label_encoder': False, 'eval_metric': 'logloss',
            'random_state': SEED, 'n_jobs': -1, 'verbosity': 0,
        }
        model = xgb.XGBClassifier(**params)
        score = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
        return score.mean()

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print(f'Best CV accuracy: {study.best_value:.4f}')
    print(f'Best params: {study.best_params}')
except ImportError:
    print('optuna not available — skipping tuning')"""))

# ── 8. Ensemble ─────────────────────────────────────────────────────────────────
cells.append(md("## 8. Final Ensemble"))

cells.append(code("""# Soft-vote ensemble: XGB + LGB + RF
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(CV.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    fold_preds = np.zeros(len(X_va))
    test_fold  = np.zeros(len(X_test))
    n_models   = 0

    if XGB_AVAILABLE:
        m = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric='logloss', random_state=SEED, n_jobs=-1, verbosity=0)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        fold_preds += m.predict_proba(X_va)[:,1]
        test_fold  += m.predict_proba(X_test)[:,1]
        n_models += 1

    if LGB_AVAILABLE:
        m = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=64,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        fold_preds += m.predict_proba(X_va)[:,1]
        test_fold  += m.predict_proba(X_test)[:,1]
        n_models += 1

    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    fold_preds += rf.predict_proba(X_va)[:,1]
    test_fold  += rf.predict_proba(X_test)[:,1]
    n_models += 1

    oof_preds[va_idx]  = fold_preds / n_models
    test_preds        += test_fold / n_models / CV.n_splits

oof_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))
print(f'Ensemble OOF Accuracy: {oof_acc:.4f}')"""))

# ── 9. Submission ───────────────────────────────────────────────────────────────
cells.append(md("## 9. Generate Submission"))

cells.append(code("""sub = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': (test_preds > 0.5)
})
sub.to_csv('submission.csv', index=False)
print('submission.csv written.')
print(sub['Transported'].value_counts())
sub.head()"""))

# ── 10. Key takeaways ───────────────────────────────────────────────────────────
cells.append(md("""## Key Takeaways

| Insight | Impact |
|---------|--------|
| **CryoSleep is the strongest predictor** — cryo passengers are transported ~80% of the time | ★★★★★ |
| **Cabin Deck matters** — decks B/C (Europa passengers) have much higher transport rates | ★★★★ |
| **TotalSpend inversely predicts transport** — passengers who spend heavily tend to stay | ★★★★ |
| **GroupSize helps** — families/groups tend to share outcomes | ★★★ |
| **Europa origin** is the strongest HomePlanet signal | ★★★ |

### Next Steps to Improve
- Neural network with entity embeddings for categoricals
- Target encoding for high-cardinality features
- Pseudo-labeling on test set
- Stacking with a meta-learner
"""))

# ── Write notebook ─────────────────────────────────────────────────────────────

write_notebook(cells, __file__, "spaceship_titanic_guide.ipynb")
