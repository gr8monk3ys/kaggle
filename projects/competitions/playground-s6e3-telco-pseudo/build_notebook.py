#!/usr/bin/env python3
"""Build the playground_s6e3_telco_pseudo.ipynb notebook."""
import os as _os
import sys as _sys


def _find_repo_root(start_dir):
    current = _os.path.abspath(start_dir)
    while True:
        if _os.path.exists(_os.path.join(current, "manage.sh")) and _os.path.isdir(
            _os.path.join(current, "kaggle_portfolio")
        ):
            return current
        parent = _os.path.dirname(current)
        if parent == current:
            return _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        current = parent


_sys.path.insert(0, _find_repo_root(_os.path.dirname(_os.path.abspath(__file__))))
from kaggle_portfolio.shared.build_utils import code, md, write_notebook

cells = []

cells.append(
    md(
        "# Playground Series S6E3: Telco Pseudo-Label XGBoost\n"
        "\n"
        "**Competition:** [Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3)  \n"
        "**Goal:** push a stronger telco churn baseline by combining original IBM churn data, aggressive tabular feature engineering, and conservative pseudo-labeling.  \n"
        "**Author:** Lorenzo Scaturchio\n"
        "\n"
        "---\n"
        "\n"
        "## Plan\n"
        "\n"
        "1. Load the competition train/test plus the original IBM telco churn dataset.\n"
        "2. Build the same high-signal telco feature family that already validated locally.\n"
        "3. Train an XGBoost model with target-encoding style statistics.\n"
        "4. Pseudo-label only the most confident test rows.\n"
        "5. Refit once on the augmented matrix and write `submission.csv`.\n"
        "\n"
        "This notebook is tuned for **competition score movement**, not tutorial completeness."
    )
)

cells.append(md("## 1. Setup"))

cells.append(
    code(
        "import json\n"
        "import os\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import StratifiedKFold, train_test_split\n"
        "from sklearn.metrics import roc_auc_score\n"
        "from sklearn.preprocessing import TargetEncoder\n"
        "import xgboost as xgb\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "pd.set_option('display.max_columns', 200)\n"
        "RANDOM_STATE = 42\n"
        "print('Environment ready.')"
    )
)

cells.append(md("## 2. Data Loading"))

cells.append(
    code(
        "def find_input_file(filename: str) -> Path | None:\n"
        "    candidates = [\n"
        "        Path('/kaggle/input/playground-series-s6e3') / filename,\n"
        "        Path('/kaggle/input/competitions/playground-series-s6e3') / filename,\n"
        "    ]\n"
        "    for candidate in candidates:\n"
        "        if candidate.exists():\n"
        "            return candidate\n"
        "    input_root = Path('/kaggle/input')\n"
        "    if input_root.exists():\n"
        "        matches = sorted(input_root.rglob(filename))\n"
        "        if matches:\n"
        "            return matches[0]\n"
        "    return None\n"
        "\n"
        "def find_original_telco_file() -> Path | None:\n"
        "    candidates = [\n"
        "        Path('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'),\n"
        "        Path('/kaggle/input/wa-fnusec-telcocustomerchurn/WA_Fn-UseC_-Telco-Customer-Churn.csv'),\n"
        "    ]\n"
        "    for candidate in candidates:\n"
        "        if candidate.exists():\n"
        "            return candidate\n"
        "    input_root = Path('/kaggle/input')\n"
        "    if input_root.exists():\n"
        "        matches = sorted(input_root.rglob('WA_Fn-UseC_-Telco-Customer-Churn.csv'))\n"
        "        if matches:\n"
        "            return matches[0]\n"
        "    return None\n"
        "\n"
        "train_path = find_input_file('train.csv')\n"
        "test_path = find_input_file('test.csv')\n"
        "orig_path = find_original_telco_file()\n"
        "\n"
        "if train_path is None or test_path is None or orig_path is None:\n"
        "    raise FileNotFoundError('Required competition or original telco files were not found under /kaggle/input.')\n"
        "\n"
        "train = pd.read_csv(train_path)\n"
        "test = pd.read_csv(test_path)\n"
        "orig = pd.read_csv(orig_path)\n"
        "\n"
        "print(f'train: {train.shape}')\n"
        "print(f'test : {test.shape}')\n"
        "print(f'orig : {orig.shape}')\n"
        "print(f'competition train path: {train_path}')\n"
        "print(f'original telco path  : {orig_path}')"
    )
)

cells.append(md("## 3. Feature Engineering"))

cells.append(
    code(
        """def concat_feature_block(df: pd.DataFrame, updates: dict[str, object]) -> pd.DataFrame:
    if not updates:
        return df
    return pd.concat([df, pd.DataFrame(updates, index=df.index)], axis=1).copy()

def prepare_target(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(series).astype(int)

def rank_against_reference(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = np.sort(reference.to_numpy(dtype=float))
    ranked = np.searchsorted(ref, values.to_numpy(dtype=float), side='right') / max(len(ref), 1)
    return pd.Series(ranked, index=values.index, dtype=float)

def advanced_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    target = 'Churn'
    train = train_df.copy()
    test = test_df.copy()
    orig = orig_df.copy()
    train[target] = prepare_target(train[target])
    orig[target] = prepare_target(orig[target])

    base_cat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    ]
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    ]
    pair_cols = [
        ('Contract', 'InternetService'),
        ('Contract', 'PaymentMethod'),
        ('InternetService', 'PaymentMethod'),
        ('PaperlessBilling', 'PaymentMethod'),
    ]

    for df in (train, test, orig):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
        for col in base_cat_cols:
            df[col] = df[col].fillna('Missing').astype(str)

        df['charges_deviation'] = df['TotalCharges'] - (df['MonthlyCharges'] * df['tenure'])
        df['abs_charges_dev'] = np.abs(df['charges_deviation'])
        df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1.0)
        df['total_to_monthly_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1.0)
        df['avg_monthly_charges'] = df['TotalCharges'] / np.maximum(df['tenure'], 1)
        df['tenure_x_monthly'] = df['tenure'] * df['MonthlyCharges']
        df['tenure_x_total'] = df['tenure'] * df['TotalCharges']
        yes_count = sum(df[col].eq('Yes') for col in service_cols)
        no_count = sum(df[col].eq('No') for col in service_cols)
        other_count = len(service_cols) - yes_count - no_count
        df['service_yes_count'] = yes_count.astype(int)
        df['service_no_count'] = no_count.astype(int)
        df['service_other_count'] = other_count.astype(int)
        df['service_count'] = (yes_count + (0.5 * other_count)).astype(float)
        df['has_internet'] = (~df['InternetService'].eq('No')).astype(int)
        df['has_phone'] = df['PhoneService'].eq('Yes').astype(int)

    bin_specs = {
        'tenure_bin': [-1, 0, 6, 12, 24, 48, 72, 200],
        'MonthlyCharges_bin': [0, 20, 40, 60, 80, 100, 200],
        'TotalCharges_bin': [-1, 0, 250, 1000, 2500, 5000, 10000],
    }
    for name, bins in bin_specs.items():
        source = name.replace('_bin', '')
        train[name] = pd.cut(train[source], bins=bins, labels=False, include_lowest=True).fillna(-1).astype(int).astype(str)
        test[name] = pd.cut(test[source], bins=bins, labels=False, include_lowest=True).fillna(-1).astype(int).astype(str)
        orig[name] = pd.cut(orig[source], bins=bins, labels=False, include_lowest=True).fillna(-1).astype(int).astype(str)

    indicator_triplets = [
        ('ISYES_', 'Yes'),
        ('ISNO_', 'No'),
    ]
    for prefix, matcher in indicator_triplets:
        train = concat_feature_block(train, {f'{prefix}{col}': train[col].eq(matcher).astype(int) for col in service_cols})
        test = concat_feature_block(test, {f'{prefix}{col}': test[col].eq(matcher).astype(int) for col in service_cols})
        orig = concat_feature_block(orig, {f'{prefix}{col}': orig[col].eq(matcher).astype(int) for col in service_cols})
    train = concat_feature_block(train, {f'ISOTHER_{col}': (~train[col].isin(['Yes', 'No'])).astype(int) for col in service_cols})
    test = concat_feature_block(test, {f'ISOTHER_{col}': (~test[col].isin(['Yes', 'No'])).astype(int) for col in service_cols})
    orig = concat_feature_block(orig, {f'ISOTHER_{col}': (~orig[col].isin(['Yes', 'No'])).astype(int) for col in service_cols})

    pair_feature_names = []
    for left, right in pair_cols:
        feature_name = f'BG_{left}_{right}'
        pair_feature_names.append(feature_name)
        train[feature_name] = train[left].astype(str) + '__' + train[right].astype(str)
        test[feature_name] = test[left].astype(str) + '__' + test[right].astype(str)
        orig[feature_name] = orig[left].astype(str) + '__' + orig[right].astype(str)

    enriched_cat_cols = base_cat_cols + ['tenure_bin', 'MonthlyCharges_bin', 'TotalCharges_bin'] + pair_feature_names
    orig_target_mean = float(orig[target].mean())
    combined_for_freq = pd.concat([train[enriched_cat_cols], orig[enriched_cat_cols]], axis=0, ignore_index=True)
    for col in enriched_cat_cols:
        freq = combined_for_freq[col].value_counts(dropna=False, normalize=True)
        train[f'FREQ_{col}'] = train[col].map(freq).fillna(0).astype(float)
        test[f'FREQ_{col}'] = test[col].map(freq).fillna(0).astype(float)
        orig[f'FREQ_{col}'] = orig[col].map(freq).fillna(0).astype(float)

        mapping = orig.groupby(col, observed=False)[target].mean()
        train[f'ORIG_proba_{col}'] = train[col].map(mapping).fillna(orig_target_mean).astype(float)
        test[f'ORIG_proba_{col}'] = test[col].map(mapping).fillna(orig_target_mean).astype(float)
        orig[f'ORIG_proba_{col}'] = orig[col].map(mapping).fillna(orig_target_mean).astype(float)

    churn_tc = train.loc[train[target] == 1, 'TotalCharges']
    non_tc = train.loc[train[target] == 0, 'TotalCharges']
    mc_mean_by_is = train.groupby('InternetService', observed=False)['MonthlyCharges'].mean().to_dict()
    is_rank_lookup = train.assign(_rank=train.groupby('InternetService', observed=False)['TotalCharges'].rank(pct=True)).groupby('InternetService', observed=False)['_rank'].mean().to_dict()
    contract_rank_lookup = train.assign(_rank=train.groupby('Contract', observed=False)['TotalCharges'].rank(pct=True)).groupby('Contract', observed=False)['_rank'].mean().to_dict()

    for df in (train, test, orig):
        df['pctrank_orig_TC'] = rank_against_reference(df['TotalCharges'], orig['TotalCharges'])
        df['pctrank_churner_TC'] = rank_against_reference(df['TotalCharges'], churn_tc)
        df['pctrank_nonchurner_TC'] = rank_against_reference(df['TotalCharges'], non_tc)
        df['zscore_churn_gap_TC'] = df['pctrank_churner_TC'] - df['pctrank_nonchurner_TC']
        df['zscore_nonchurner_TC'] = (df['TotalCharges'] - float(non_tc.mean())) / (float(non_tc.std()) + 1e-6)
        df['pctrank_churn_gap_TC'] = df['pctrank_churner_TC'] - df['pctrank_nonchurner_TC']
        df['resid_IS_MC'] = df['MonthlyCharges'] - df['InternetService'].map(mc_mean_by_is).fillna(float(train['MonthlyCharges'].mean()))
        df['cond_pctrank_IS_TC'] = df['InternetService'].map(is_rank_lookup).fillna(0.5).astype(float)
        df['cond_pctrank_C_TC'] = df['Contract'].map(contract_rank_lookup).fillna(0.5).astype(float)

    for df in (train, test, orig):
        for col in enriched_cat_cols:
            df[col] = df[col].astype(str).astype('category')

    indicator_cols = [c for c in train.columns if c.startswith(('ISYES_', 'ISNO_', 'ISOTHER_'))]
    freq_cols = [f'FREQ_{col}' for col in enriched_cat_cols]
    proba_cols = [f'ORIG_proba_{col}' for col in enriched_cat_cols]
    num_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'charges_deviation', 'abs_charges_dev',
        'monthly_to_total_ratio', 'total_to_monthly_ratio', 'avg_monthly_charges',
        'tenure_x_monthly', 'tenure_x_total', 'service_yes_count', 'service_no_count',
        'service_other_count', 'service_count', 'has_internet', 'has_phone',
        'pctrank_orig_TC', 'pctrank_churner_TC', 'pctrank_nonchurner_TC',
        'zscore_churn_gap_TC', 'zscore_nonchurner_TC', 'pctrank_churn_gap_TC',
        'resid_IS_MC', 'cond_pctrank_IS_TC', 'cond_pctrank_C_TC',
    ] + indicator_cols + freq_cols + proba_cols
    feature_cols = num_cols + enriched_cat_cols
    te_cols = enriched_cat_cols.copy()
    drop_raw_cols = te_cols.copy()
    return train, test, feature_cols, te_cols, drop_raw_cols

train_frame, test_frame, feature_cols, te_cols, drop_raw_cols = advanced_feature_frames(train, test, orig)
print({'feature_count': len(feature_cols), 'te_cols': len(te_cols), 'train_rows': len(train_frame), 'test_rows': len(test_frame)})
train_frame[feature_cols].head()"""
    )
)

cells.append(md("## 4. Pseudo-Label XGBoost"))

cells.append(
    code(
        """def pseudo_label_mask(predictions: np.ndarray, lower_quantile: float = 0.08, upper_quantile: float = 0.92, absolute_confidence: float = 0.92) -> np.ndarray:
    lower_threshold = min(float(np.quantile(predictions, lower_quantile)), 1.0 - absolute_confidence)
    upper_threshold = max(float(np.quantile(predictions, upper_quantile)), absolute_confidence)
    return (predictions <= lower_threshold) | (predictions >= upper_threshold)

def pseudo_label_weights(predictions: np.ndarray) -> np.ndarray:
    confidence = np.abs(predictions - 0.5) * 2.0
    return np.clip(0.15 + (0.25 * confidence), 0.2, 0.4)

def fit_encoded_xgb(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    x_test: pd.DataFrame,
    te_cols_local: list[str],
    drop_cols_local: list[str],
    sample_weight: np.ndarray | None = None,
) -> tuple[xgb.XGBClassifier, np.ndarray, np.ndarray]:
    stats = ['std', 'min', 'max']
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    te_stat_cols = [f'TE1_{col}_{stat}' for col in te_cols_local for stat in stats]
    x_train = concat_feature_block(x_train, {name: np.nan for name in te_stat_cols})

    for inner_train_idx, inner_valid_idx in inner_cv.split(x_train, y_train):
        x_inner_train = x_train.loc[inner_train_idx, feature_cols + ['Churn']].copy()
        x_inner_valid = x_train.loc[inner_valid_idx, feature_cols].copy()
        for col in te_cols_local:
            grouped = x_inner_train.groupby(col, observed=False)['Churn'].agg(stats)
            grouped.columns = [f'TE1_{col}_{stat}' for stat in stats]
            x_inner_valid = x_inner_valid.merge(grouped, on=col, how='left')
            for name in grouped.columns:
                x_train.loc[inner_valid_idx, name] = x_inner_valid[name].to_numpy(dtype='float32')

    for col in te_cols_local:
        grouped = x_train.groupby(col, observed=False)['Churn'].agg(stats)
        grouped.columns = [f'TE1_{col}_{stat}' for stat in stats]
        x_valid = x_valid.merge(grouped.astype('float32'), on=col, how='left')
        x_test = x_test.merge(grouped.astype('float32'), on=col, how='left')
        for name in grouped.columns:
            x_train[name] = x_train[name].fillna(0).astype('float32')
            x_valid[name] = x_valid[name].fillna(0).astype('float32')
            x_test[name] = x_test[name].fillna(0).astype('float32')

    mean_encoder = TargetEncoder(cv=3, shuffle=True, smooth='auto', target_type='binary', random_state=RANDOM_STATE)
    mean_cols = [f'TE_{col}' for col in te_cols_local]
    x_train = pd.concat([x_train, pd.DataFrame(mean_encoder.fit_transform(x_train[te_cols_local], y_train), columns=mean_cols, index=x_train.index)], axis=1).copy()
    x_valid = pd.concat([x_valid, pd.DataFrame(mean_encoder.transform(x_valid[te_cols_local]), columns=mean_cols, index=x_valid.index)], axis=1).copy()
    x_test = pd.concat([x_test, pd.DataFrame(mean_encoder.transform(x_test[te_cols_local]), columns=mean_cols, index=x_test.index)], axis=1).copy()

    for df in (x_train, x_valid, x_test):
        for col in te_cols_local:
            df[col] = df[col].astype(str).astype('category')
        df.drop(columns=drop_cols_local, inplace=True)
    x_train = x_train.drop(columns=['Churn'])

    params = {
        'n_estimators': 2400,
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'gamma': 0.05,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'enable_categorical': True,
        'tree_method': 'hist',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0,
        'early_stopping_rounds': 80,
    }
    model = xgb.XGBClassifier(**params)
    fit_kwargs = {'eval_set': [(x_valid, y_valid)], 'verbose': False}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    model.fit(x_train, y_train, **fit_kwargs)
    valid_pred = model.predict_proba(x_valid)[:, 1]
    test_pred = model.predict_proba(x_test)[:, 1]
    return model, valid_pred, test_pred

y = train_frame['Churn'].to_numpy()
train_idx, valid_idx = train_test_split(np.arange(len(train_frame)), test_size=0.2, random_state=RANDOM_STATE, stratify=y)
x_train = train_frame.iloc[train_idx][feature_cols + ['Churn']].reset_index(drop=True).copy()
y_train = y[train_idx]
x_valid = train_frame.iloc[valid_idx][feature_cols].reset_index(drop=True).copy()
y_valid = y[valid_idx]
x_test = test_frame[feature_cols].reset_index(drop=True).copy()

base_model, base_valid_pred, base_test_pred = fit_encoded_xgb(
    x_train.copy(),
    y_train,
    x_valid.copy(),
    y_valid,
    x_test.copy(),
    te_cols,
    drop_raw_cols,
)
base_auc = roc_auc_score(y_valid, base_valid_pred)
pseudo_mask = pseudo_label_mask(base_test_pred)
pseudo_count = int(pseudo_mask.sum())

if pseudo_count >= max(2000, len(base_test_pred) // 50):
    pseudo_x = x_test.loc[pseudo_mask].copy()
    pseudo_y = (base_test_pred[pseudo_mask] >= 0.5).astype(int)
    pseudo_w = pseudo_label_weights(base_test_pred[pseudo_mask])
    augmented_x = pd.concat([x_train, pseudo_x], axis=0, ignore_index=True).copy()
    augmented_y = np.concatenate([y_train, pseudo_y])
    augmented_x['Churn'] = augmented_y
    sample_weight = np.concatenate([np.ones(len(y_train), dtype=float), pseudo_w])
    pseudo_model, pseudo_valid_pred, _ = fit_encoded_xgb(
        augmented_x.copy(),
        augmented_y,
        x_valid.copy(),
        y_valid,
        x_test.copy(),
        te_cols,
        drop_raw_cols,
        sample_weight=sample_weight,
    )
    pseudo_auc = roc_auc_score(y_valid, pseudo_valid_pred)
else:
    pseudo_model = base_model
    pseudo_auc = base_auc

print({'base_holdout_auc': round(float(base_auc), 5), 'pseudo_holdout_auc': round(float(pseudo_auc), 5), 'pseudo_rows': pseudo_count})"""
    )
)

cells.append(md("## 5. Final Fit And Submission"))

cells.append(
    code(
        """def final_submission_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, te_cols_local: list[str], drop_cols_local: list[str]) -> np.ndarray:
    train_model = train_df[feature_cols + ['Churn']].reset_index(drop=True).copy()
    test_model = test_df[feature_cols].reset_index(drop=True).copy()
    y_full = train_model['Churn'].to_numpy()

    base_train = train_model.iloc[train_idx].reset_index(drop=True).copy()
    base_valid = train_model.iloc[valid_idx][feature_cols].reset_index(drop=True).copy()
    base_valid_y = y[valid_idx]

    _, _, base_test_pred = fit_encoded_xgb(
        base_train.copy(),
        y[train_idx],
        base_valid.copy(),
        base_valid_y,
        test_model.copy(),
        te_cols_local,
        drop_cols_local,
    )

    mask = pseudo_label_mask(base_test_pred)
    if mask.sum() >= max(2000, len(base_test_pred) // 50):
        pseudo_x = test_model.loc[mask].copy()
        pseudo_y = (base_test_pred[mask] >= 0.5).astype(int)
        pseudo_w = pseudo_label_weights(base_test_pred[mask])
        augmented_train = pd.concat([base_train, pseudo_x], axis=0, ignore_index=True).copy()
        augmented_target = np.concatenate([y[train_idx], pseudo_y])
        augmented_train['Churn'] = augmented_target
        sample_weight = np.concatenate([np.ones(len(base_train), dtype=float), pseudo_w])
        _, _, final_test_pred = fit_encoded_xgb(
            augmented_train.copy(),
            augmented_target,
            base_valid.copy(),
            base_valid_y,
            test_model.copy(),
            te_cols_local,
            drop_cols_local,
            sample_weight=sample_weight,
        )
        return final_test_pred
    return base_test_pred

final_pred = final_submission_predictions(train_frame.copy(), test_frame.copy(), te_cols, drop_raw_cols)
submission = pd.DataFrame({'id': test['id'], 'Churn': final_pred})
submission.to_csv('submission.csv', index=False)
submission.head()"""
    )
)

cells.append(
    code(
        "summary = {\n"
        "    'submission_rows': int(len(submission)),\n"
        "    'submission_min': float(submission['Churn'].min()),\n"
        "    'submission_max': float(submission['Churn'].max()),\n"
        "    'submission_mean': float(submission['Churn'].mean()),\n"
        "}\n"
        "print(json.dumps(summary, indent=2))\n"
        "print('submission.csv written to the working directory.')"
    )
)

write_notebook(cells, __file__, "playground_s6e3_telco_pseudo.ipynb")
