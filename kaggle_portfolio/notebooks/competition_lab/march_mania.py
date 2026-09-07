"""March Machine Learning Mania: tournament prediction.

Split out of local_competition_lab; the BENCHMARKS interface is unchanged.
"""

from __future__ import annotations

#!/usr/bin/env python3


from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from kaggle_portfolio.shared.errors import CommandError

from kaggle_portfolio.notebooks.competition_lab.runner import (  # noqa: F401
    LabResult,
    _benchmark_dir,
    _concat_feature_block,
    _ensure_data,
    _print_benchmarks,
    _safe_slug,
    _save_summary,
    _submission_dir,
    _submit,
)

RANDOM_STATE = 42
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
BLUE = "\033[0;34m"
RED = "\033[0;31m"
RESET = "\033[0m"


def _march_seed_number(value: Any) -> float:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return np.nan
    return float(digits[:2])


def _march_team_game_rows(results: pd.DataFrame) -> pd.DataFrame:
    winners = pd.DataFrame(
        {
            "Season": results["Season"],
            "DayNum": results["DayNum"],
            "TeamID": results["WTeamID"],
            "OppTeamID": results["LTeamID"],
            "Score": results["WScore"],
            "OppScore": results["LScore"],
            "Win": 1,
        }
    )
    losers = pd.DataFrame(
        {
            "Season": results["Season"],
            "DayNum": results["DayNum"],
            "TeamID": results["LTeamID"],
            "OppTeamID": results["WTeamID"],
            "Score": results["LScore"],
            "OppScore": results["WScore"],
            "Win": 0,
        }
    )
    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["Margin"] = team_games["Score"] - team_games["OppScore"]
    return team_games.sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)


def _march_team_game_rows_detailed(results: pd.DataFrame) -> pd.DataFrame:
    winners = pd.DataFrame(
        {
            "Season": results["Season"],
            "DayNum": results["DayNum"],
            "TeamID": results["WTeamID"],
            "OppTeamID": results["LTeamID"],
            "Score": results["WScore"],
            "OppScore": results["LScore"],
            "Win": 1,
            "Loc": results["WLoc"].fillna("N"),
            "FGM": results["WFGM"],
            "FGA": results["WFGA"],
            "FGM3": results["WFGM3"],
            "FGA3": results["WFGA3"],
            "FTM": results["WFTM"],
            "FTA": results["WFTA"],
            "OR": results["WOR"],
            "DR": results["WDR"],
            "Ast": results["WAst"],
            "TO": results["WTO"],
            "Stl": results["WStl"],
            "Blk": results["WBlk"],
            "PF": results["WPF"],
            "OppFGM": results["LFGM"],
            "OppFGA": results["LFGA"],
            "OppFGM3": results["LFGM3"],
            "OppFGA3": results["LFGA3"],
            "OppFTM": results["LFTM"],
            "OppFTA": results["LFTA"],
            "OppOR": results["LOR"],
            "OppDR": results["LDR"],
            "OppAst": results["LAst"],
            "OppTO": results["LTO"],
            "OppStl": results["LStl"],
            "OppBlk": results["LBlk"],
            "OppPF": results["LPF"],
        }
    )
    loser_loc = results["WLoc"].map({"H": "A", "A": "H"}).fillna("N")
    losers = pd.DataFrame(
        {
            "Season": results["Season"],
            "DayNum": results["DayNum"],
            "TeamID": results["LTeamID"],
            "OppTeamID": results["WTeamID"],
            "Score": results["LScore"],
            "OppScore": results["WScore"],
            "Win": 0,
            "Loc": loser_loc,
            "FGM": results["LFGM"],
            "FGA": results["LFGA"],
            "FGM3": results["LFGM3"],
            "FGA3": results["LFGA3"],
            "FTM": results["LFTM"],
            "FTA": results["LFTA"],
            "OR": results["LOR"],
            "DR": results["LDR"],
            "Ast": results["LAst"],
            "TO": results["LTO"],
            "Stl": results["LStl"],
            "Blk": results["LBlk"],
            "PF": results["LPF"],
            "OppFGM": results["WFGM"],
            "OppFGA": results["WFGA"],
            "OppFGM3": results["WFGM3"],
            "OppFGA3": results["WFGA3"],
            "OppFTM": results["WFTM"],
            "OppFTA": results["WFTA"],
            "OppOR": results["WOR"],
            "OppDR": results["WDR"],
            "OppAst": results["WAst"],
            "OppTO": results["WTO"],
            "OppStl": results["WStl"],
            "OppBlk": results["WBlk"],
            "OppPF": results["WPF"],
        }
    )
    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["Margin"] = team_games["Score"] - team_games["OppScore"]
    team_games["Possessions"] = (
        team_games["FGA"]
        - team_games["OR"]
        + team_games["TO"]
        + (0.475 * team_games["FTA"])
    ).clip(lower=1.0)
    team_games["OppPossessions"] = (
        team_games["OppFGA"]
        - team_games["OppOR"]
        + team_games["OppTO"]
        + (0.475 * team_games["OppFTA"])
    ).clip(lower=1.0)
    team_games["Pace"] = (
        (team_games["Possessions"] + team_games["OppPossessions"]) / 2.0
    ).clip(lower=1.0)
    team_games["OffEff"] = 100.0 * team_games["Score"] / team_games["Pace"]
    team_games["DefEff"] = 100.0 * team_games["OppScore"] / team_games["Pace"]
    team_games["NetEff"] = team_games["OffEff"] - team_games["DefEff"]
    team_games["eFG"] = (team_games["FGM"] + (0.5 * team_games["FGM3"])) / team_games[
        "FGA"
    ].clip(lower=1.0)
    team_games["OppEfg"] = (
        team_games["OppFGM"] + (0.5 * team_games["OppFGM3"])
    ) / team_games["OppFGA"].clip(lower=1.0)
    team_games["TOVRate"] = team_games["TO"] / team_games["Possessions"]
    team_games["OppTOVRate"] = team_games["OppTO"] / team_games["OppPossessions"]
    team_games["ORBRate"] = team_games["OR"] / (
        team_games["OR"] + team_games["OppDR"]
    ).clip(lower=1.0)
    team_games["OppORBRate"] = team_games["OppOR"] / (
        team_games["OppOR"] + team_games["DR"]
    ).clip(lower=1.0)
    team_games["FTRate"] = team_games["FTA"] / team_games["FGA"].clip(lower=1.0)
    team_games["OppFTRate"] = team_games["OppFTA"] / team_games["OppFGA"].clip(
        lower=1.0
    )
    team_games["AstRate"] = team_games["Ast"] / team_games["FGM"].clip(lower=1.0)
    team_games["OppAstRate"] = team_games["OppAst"] / team_games["OppFGM"].clip(
        lower=1.0
    )
    team_games["IsHome"] = team_games["Loc"].eq("H").astype(int)
    team_games["IsAway"] = team_games["Loc"].eq("A").astype(int)
    team_games["IsNeutral"] = team_games["Loc"].eq("N").astype(int)
    return team_games.sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)


def _march_elo_features(results: pd.DataFrame) -> pd.DataFrame:
    ratings_rows: list[dict[str, float]] = []
    for season, season_games in results.sort_values(["Season", "DayNum"]).groupby(
        "Season", sort=True
    ):
        season_ratings: dict[int, float] = {}
        for game in season_games.itertuples(index=False):
            winner_rating = season_ratings.get(int(game.WTeamID), 1500.0)
            loser_rating = season_ratings.get(int(game.LTeamID), 1500.0)
            expected_winner = 1.0 / (
                1.0 + 10 ** ((loser_rating - winner_rating) / 400.0)
            )
            margin = max(int(game.WScore) - int(game.LScore), 1)
            k_factor = 20.0 * min(2.5, 1.0 + (margin - 1) / 25.0)
            season_ratings[int(game.WTeamID)] = winner_rating + k_factor * (
                1.0 - expected_winner
            )
            season_ratings[int(game.LTeamID)] = loser_rating + k_factor * (
                0.0 - (1.0 - expected_winner)
            )

        for team_id, rating in season_ratings.items():
            ratings_rows.append({"Season": season, "TeamID": team_id, "elo": rating})
    return pd.DataFrame(ratings_rows)


def _march_massey_features(massey: pd.DataFrame) -> pd.DataFrame:
    if massey.empty:
        return pd.DataFrame(columns=["Season", "TeamID"])

    working = massey[
        ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]
    ].copy()
    working["season_latest_day"] = working.groupby("Season")["RankingDayNum"].transform(
        "max"
    )

    latest = working.loc[working["RankingDayNum"] == working["season_latest_day"]]
    latest_features = (
        latest.groupby(["Season", "TeamID"])
        .agg(
            massey_latest_mean=("OrdinalRank", "mean"),
            massey_latest_median=("OrdinalRank", "median"),
            massey_latest_best=("OrdinalRank", "min"),
            massey_latest_worst=("OrdinalRank", "max"),
            massey_latest_std=("OrdinalRank", "std"),
            massey_latest_count=("OrdinalRank", "size"),
        )
        .reset_index()
    )

    recent = working.loc[working["RankingDayNum"] >= (working["season_latest_day"] - 7)]
    recent_features = (
        recent.groupby(["Season", "TeamID"])
        .agg(
            massey_recent_mean=("OrdinalRank", "mean"),
            massey_recent_best=("OrdinalRank", "min"),
            massey_recent_std=("OrdinalRank", "std"),
        )
        .reset_index()
    )

    previous = working.loc[
        (working["RankingDayNum"] >= (working["season_latest_day"] - 14))
        & (working["RankingDayNum"] < (working["season_latest_day"] - 7))
    ]
    previous_features = (
        previous.groupby(["Season", "TeamID"])
        .agg(
            massey_prev_mean=("OrdinalRank", "mean"),
        )
        .reset_index()
    )

    features = latest_features.merge(
        recent_features, on=["Season", "TeamID"], how="left"
    )
    features = features.merge(previous_features, on=["Season", "TeamID"], how="left")
    features["massey_trend"] = (
        features["massey_prev_mean"] - features["massey_recent_mean"]
    )
    return features


def _march_training_weights(seasons: pd.Series) -> np.ndarray:
    season_values = seasons.astype(float).to_numpy()
    if len(season_values) == 0:
        return np.array([], dtype=float)
    min_season = float(np.min(season_values))
    max_season = float(np.max(season_values))
    if max_season <= min_season:
        return np.ones(len(season_values), dtype=float)
    scaled = (season_values - min_season) / (max_season - min_season)
    return 1.0 + (1.5 * scaled)


def _march_fit_model(
    model: Any, x_train: pd.DataFrame, y_train: pd.Series, sample_weight: np.ndarray
) -> Any:
    if isinstance(model, Pipeline):
        estimator_name = model.steps[-1][0]
        try:
            model.fit(
                x_train, y_train, **{f"{estimator_name}__sample_weight": sample_weight}
            )
            return model
        except TypeError:
            pass
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
        return model
    except TypeError:
        model.fit(x_train, y_train)
        return model


def _march_team_features(
    results: pd.DataFrame, seeds: pd.DataFrame, massey: pd.DataFrame | None = None
) -> pd.DataFrame:
    team_games = _march_team_game_rows_detailed(results)
    recent = (
        team_games.groupby(["Season", "TeamID"], group_keys=False)
        .tail(10)
        .groupby(["Season", "TeamID"])
        .agg(
            recent_win_pct=("Win", "mean"),
            recent_margin=("Margin", "mean"),
            recent_net_eff=("NetEff", "mean"),
            recent_off_eff=("OffEff", "mean"),
            recent_def_eff=("DefEff", "mean"),
            recent_efg=("eFG", "mean"),
            recent_tov_rate=("TOVRate", "mean"),
            recent_orb_rate=("ORBRate", "mean"),
        )
        .reset_index()
    )
    season_features = (
        team_games.groupby(["Season", "TeamID"])
        .agg(
            games=("Win", "size"),
            win_pct=("Win", "mean"),
            avg_score=("Score", "mean"),
            avg_allowed=("OppScore", "mean"),
            avg_margin=("Margin", "mean"),
            avg_pace=("Pace", "mean"),
            off_eff=("OffEff", "mean"),
            def_eff=("DefEff", "mean"),
            net_eff=("NetEff", "mean"),
            efg=("eFG", "mean"),
            opp_efg=("OppEfg", "mean"),
            tov_rate=("TOVRate", "mean"),
            opp_tov_rate=("OppTOVRate", "mean"),
            orb_rate=("ORBRate", "mean"),
            opp_orb_rate=("OppORBRate", "mean"),
            ft_rate=("FTRate", "mean"),
            opp_ft_rate=("OppFTRate", "mean"),
            ast_rate=("AstRate", "mean"),
            opp_ast_rate=("OppAstRate", "mean"),
            home_share=("IsHome", "mean"),
            away_share=("IsAway", "mean"),
            neutral_share=("IsNeutral", "mean"),
        )
        .reset_index()
    )
    neutral = (
        team_games.loc[team_games["IsNeutral"] == 1]
        .groupby(["Season", "TeamID"])
        .agg(
            neutral_win_pct=("Win", "mean"),
            neutral_margin=("Margin", "mean"),
        )
        .reset_index()
    )
    close = (
        team_games.loc[team_games["Margin"].abs() <= 5]
        .groupby(["Season", "TeamID"])
        .agg(
            close_games=("Win", "size"),
            close_win_pct=("Win", "mean"),
        )
        .reset_index()
    )
    elo = _march_elo_features(results)
    features = season_features.merge(recent, on=["Season", "TeamID"], how="left")
    features = features.merge(neutral, on=["Season", "TeamID"], how="left")
    features = features.merge(close, on=["Season", "TeamID"], how="left")
    features = features.merge(elo, on=["Season", "TeamID"], how="left")

    seed_features = seeds.copy()
    seed_features["seed"] = seed_features["Seed"].map(_march_seed_number)
    seed_features = seed_features[["Season", "TeamID", "seed"]]
    features = features.merge(seed_features, on=["Season", "TeamID"], how="left")
    features["seed_missing"] = features["seed"].isna().astype(int)
    features["seed"] = features["seed"].fillna(20.0)
    features["elo"] = features["elo"].fillna(1500.0)
    features["recent_win_pct"] = features["recent_win_pct"].fillna(features["win_pct"])
    features["recent_margin"] = features["recent_margin"].fillna(features["avg_margin"])
    features["recent_net_eff"] = features["recent_net_eff"].fillna(features["net_eff"])
    features["recent_off_eff"] = features["recent_off_eff"].fillna(features["off_eff"])
    features["recent_def_eff"] = features["recent_def_eff"].fillna(features["def_eff"])
    features["recent_efg"] = features["recent_efg"].fillna(features["efg"])
    features["recent_tov_rate"] = features["recent_tov_rate"].fillna(
        features["tov_rate"]
    )
    features["recent_orb_rate"] = features["recent_orb_rate"].fillna(
        features["orb_rate"]
    )
    features["neutral_win_pct"] = features["neutral_win_pct"].fillna(
        features["win_pct"]
    )
    features["neutral_margin"] = features["neutral_margin"].fillna(
        features["avg_margin"]
    )
    features["close_games"] = features["close_games"].fillna(0.0)
    features["close_win_pct"] = features["close_win_pct"].fillna(features["win_pct"])

    opp_base = features[
        ["Season", "TeamID", "win_pct", "net_eff", "off_eff", "def_eff", "elo"]
    ].rename(
        columns={
            "TeamID": "OppTeamID",
            "win_pct": "sos_win_pct",
            "net_eff": "sos_net_eff",
            "off_eff": "sos_off_eff",
            "def_eff": "sos_def_eff",
            "elo": "sos_elo",
        }
    )
    schedule_strength = (
        team_games[["Season", "TeamID", "OppTeamID"]]
        .merge(opp_base, on=["Season", "OppTeamID"], how="left")
        .groupby(["Season", "TeamID"])
        .agg(
            sos_win_pct=("sos_win_pct", "mean"),
            sos_net_eff=("sos_net_eff", "mean"),
            sos_off_eff=("sos_off_eff", "mean"),
            sos_def_eff=("sos_def_eff", "mean"),
            sos_elo=("sos_elo", "mean"),
        )
        .reset_index()
    )
    features = features.merge(schedule_strength, on=["Season", "TeamID"], how="left")
    features["net_eff_vs_schedule"] = features["net_eff"] - features["sos_net_eff"]
    features["elo_vs_schedule"] = features["elo"] - features["sos_elo"]

    if massey is not None and not massey.empty:
        features = features.merge(
            _march_massey_features(massey), on=["Season", "TeamID"], how="left"
        )

    numeric_cols = [col for col in features.columns if col not in {"Season", "TeamID"}]
    for col in numeric_cols:
        season_medians = features.groupby("Season")[col].transform("median")
        features[col] = features[col].fillna(season_medians)
        if features[col].isna().any():
            fallback = features[col].median()
            features[col] = features[col].fillna(
                0.0 if pd.isna(fallback) else float(fallback)
            )

    features["has_massey"] = (
        features.get("massey_latest_count", pd.Series(0.0, index=features.index))
        .gt(0)
        .astype(int)
    )
    return features


def _march_submission_pairs(sample: pd.DataFrame) -> pd.DataFrame:
    parsed = sample["ID"].astype(str).str.split("_", expand=True)
    if parsed.shape[1] != 3:
        raise ValueError("Unexpected March Mania submission ID format.")
    return pd.DataFrame(
        {
            "ID": sample["ID"].astype(str),
            "Season": parsed[0].astype(int),
            "Team1": parsed[1].astype(int),
            "Team2": parsed[2].astype(int),
        }
    )


def _march_matchups(
    games: pd.DataFrame,
    features: pd.DataFrame,
    *,
    include_target: bool,
) -> pd.DataFrame:
    feature_map = features.set_index(["Season", "TeamID"]).to_dict("index")
    base_cols = [col for col in features.columns if col not in {"Season", "TeamID"}]
    rows: list[dict[str, Any]] = []

    for game in games.itertuples(index=False):
        season = int(game.Season)
        if hasattr(game, "WTeamID") and hasattr(game, "LTeamID"):
            team_a = int(game.WTeamID)
            team_b = int(game.LTeamID)
        else:
            team_a = int(game.Team1)
            team_b = int(game.Team2)
        team1, team2 = sorted((team_a, team_b))
        feat_1 = feature_map.get((season, team1))
        feat_2 = feature_map.get((season, team2))
        if feat_1 is None or feat_2 is None:
            continue

        row: dict[str, Any] = {"Season": season, "Team1": team1, "Team2": team2}
        row["is_women"] = 1 if team1 >= 3000 else 0
        if include_target:
            row["target"] = 1 if team1 == team_a else 0
        for col in base_cols:
            value_1 = float(feat_1[col])
            value_2 = float(feat_2[col])
            row[f"{col}_1"] = value_1
            row[f"{col}_2"] = value_2
            row[f"{col}_diff"] = value_1 - value_2
        for col in ("seed", "elo", "net_eff", "recent_net_eff", "massey_latest_mean"):
            diff_key = f"{col}_diff"
            if diff_key in row:
                row[f"{col}_abs_diff"] = abs(row[diff_key])
        if "elo_diff" in row:
            row["elo_win_prob_1"] = 1.0 / (1.0 + (10.0 ** (-row["elo_diff"] / 400.0)))
        if "seed_diff" in row:
            row["seed_win_prob_1"] = 1.0 / (1.0 + np.exp(row["seed_diff"] / 1.5))
        if "net_eff_diff" in row:
            row["net_eff_win_prob_1"] = 1.0 / (1.0 + np.exp(-row["net_eff_diff"] / 5.0))
        if "recent_net_eff_diff" in row:
            row["recent_net_eff_win_prob_1"] = 1.0 / (
                1.0 + np.exp(-row["recent_net_eff_diff"] / 5.0)
            )
        if "massey_latest_mean_diff" in row:
            row["massey_win_prob_1"] = 1.0 / (
                1.0 + np.exp(row["massey_latest_mean_diff"] / 7.5)
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _march_build_models() -> dict[str, Any]:
    return {
        "lr": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=4000, C=0.8)),
            ]
        ),
        "hgb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.035,
                        max_depth=4,
                        max_iter=450,
                        min_samples_leaf=20,
                        l2_regularization=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def benchmark_march_mania(
    data_dir: Path, _folds: int, write_submission: bool
) -> LabResult:
    regular_season = pd.concat(
        [
            pd.read_csv(data_dir / "MRegularSeasonDetailedResults.csv"),
            pd.read_csv(data_dir / "WRegularSeasonDetailedResults.csv"),
        ],
        ignore_index=True,
    )
    tournament = pd.concat(
        [
            pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv"),
            pd.read_csv(data_dir / "WNCAATourneyCompactResults.csv"),
        ],
        ignore_index=True,
    )
    seeds = pd.concat(
        [
            pd.read_csv(data_dir / "MNCAATourneySeeds.csv"),
            pd.read_csv(data_dir / "WNCAATourneySeeds.csv"),
        ],
        ignore_index=True,
    )
    massey = pd.read_csv(data_dir / "MMasseyOrdinals.csv")

    features = _march_team_features(regular_season, seeds, massey)
    train_df = _march_matchups(tournament, features, include_target=True)
    if train_df.empty:
        raise CommandError(
            "Failed to build March Mania training rows from downloaded competition files."
        )

    feature_cols = [
        col for col in train_df.columns if col not in {"target", "Team1", "Team2"}
    ]
    holdout_seasons = sorted(
        season for season in train_df["Season"].unique() if season >= 2021
    )
    if not holdout_seasons:
        holdout_seasons = sorted(train_df["Season"].unique())[-5:]

    model_defs = _march_build_models()
    season_predictions: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        name: {} for name in model_defs
    }
    for season in holdout_seasons:
        train_mask = train_df["Season"] < season
        valid_mask = train_df["Season"] == season
        if int(train_mask.sum()) == 0 or int(valid_mask.sum()) == 0:
            continue
        x_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, "target"].astype(int)
        x_valid = train_df.loc[valid_mask, feature_cols]
        y_valid = train_df.loc[valid_mask, "target"].astype(int)
        sample_weight = _march_training_weights(train_df.loc[train_mask, "Season"])
        for name, model in _march_build_models().items():
            model = _march_fit_model(model, x_train, y_train, sample_weight)
            probs = model.predict_proba(x_valid)[:, 1]
            season_predictions[name][season] = (y_valid.to_numpy(), probs)

    benchmarks: list[dict[str, Any]] = []
    for name, rows in season_predictions.items():
        scores = [
            (season, brier_score_loss(y_true, probs))
            for season, (y_true, probs) in rows.items()
        ]
        if scores:
            benchmarks.append(
                {
                    "model": name,
                    "score": round(float(np.mean([score for _, score in scores])), 5),
                }
            )

    model_names = list(model_defs)
    for combo_size in range(2, len(model_names) + 1):
        for combo in combinations(model_names, combo_size):
            common_seasons = sorted(
                set.intersection(*(set(season_predictions[name]) for name in combo))
            )
            if not common_seasons:
                continue
            ensemble_scores = []
            for season in common_seasons:
                y_true = season_predictions[combo[0]][season][0]
                blended = np.mean(
                    [season_predictions[name][season][1] for name in combo], axis=0
                )
                ensemble_scores.append((season, brier_score_loss(y_true, blended)))
            benchmarks.append(
                {
                    "model": f"{'_'.join(combo)}_ensemble",
                    "score": round(
                        float(np.mean([score for _, score in ensemble_scores])), 5
                    ),
                    "members": list(combo),
                }
            )

    if not benchmarks:
        raise CommandError("March Mania benchmark did not produce any holdout scores.")

    best = min(benchmarks, key=lambda row: row["score"])
    submission_path = None
    if write_submission:
        sample_path = (
            data_dir / "SampleSubmissionStage2.csv"
            if (data_dir / "SampleSubmissionStage2.csv").exists()
            else data_dir / "SampleSubmissionStage1.csv"
        )
        sample = pd.read_csv(sample_path)
        submission_pairs = _march_submission_pairs(sample)
        submission_features = _march_matchups(
            submission_pairs, features, include_target=False
        )
        if submission_features.empty:
            raise CommandError("Failed to build March Mania submission rows.")

        train_x = train_df[feature_cols]
        train_y = train_df["target"].astype(int)
        submit_x = submission_features[feature_cols]
        sample_weight = _march_training_weights(train_df["Season"])
        models = _march_build_models()
        fitted: dict[str, Any] = {}
        for name, model in models.items():
            fitted[name] = _march_fit_model(model, train_x, train_y, sample_weight)

        if best.get("members"):
            preds = np.mean(
                [
                    fitted[name].predict_proba(submit_x)[:, 1]
                    for name in best["members"]
                ],
                axis=0,
            )
        else:
            preds = fitted[best["model"]].predict_proba(submit_x)[:, 1]

        submission_path = _submission_dir("march-machine-learning-mania-2026") / (
            f"submission_{_safe_slug(best['model'])}_{int(best['score'] * 100000)}.csv"
        )
        pd.DataFrame({"ID": submission_pairs["ID"], "Pred": preds}).to_csv(
            submission_path, index=False
        )

    return LabResult(
        competition="march-machine-learning-mania-2026",
        metric_name="brier",
        best_model=best["model"],
        best_score=float(best["score"]),
        benchmark_rows=benchmarks,
        submission_path=submission_path,
    )
