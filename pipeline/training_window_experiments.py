from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from pipeline.market_features import FOOTBALL_DATA_DIR, add_market_features
    from pipeline.model_features import LEAGUES, MARKET_AWARE_FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS
    from pipeline.stage3_train import (
        CALIBRATION_METHODS,
        DEFAULT_CALIBRATION_METHOD,
        DEFAULT_XGB_PARAMS,
        compute_class_sample_weights,
        evaluate_predictions,
        load_feature_data,
        normalize_probabilities,
        season_start_year,
        select_features_and_target,
        split_model_calibration_data,
        train_classifier,
        calibrate_classifier,
    )
    from pipeline.stage4_odds_gen import OUTPUT_COLUMNS, probabilities_to_odds
    from pipeline.stage5_compare import compute_value_bets, load_football_data_odds, match_model_to_bookmaker_odds
except ModuleNotFoundError:
    from market_features import FOOTBALL_DATA_DIR, add_market_features
    from model_features import LEAGUES, MARKET_AWARE_FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS
    from stage3_train import (
        CALIBRATION_METHODS,
        DEFAULT_CALIBRATION_METHOD,
        DEFAULT_XGB_PARAMS,
        compute_class_sample_weights,
        evaluate_predictions,
        load_feature_data,
        normalize_probabilities,
        season_start_year,
        select_features_and_target,
        split_model_calibration_data,
        train_classifier,
        calibrate_classifier,
    )
    from stage4_odds_gen import OUTPUT_COLUMNS, probabilities_to_odds
    from stage5_compare import compute_value_bets, load_football_data_odds, match_model_to_bookmaker_odds

FEATURES_DIR = Path("data/features")
OUTPUT_PATH = Path("data/model_artifacts/training_window_experiments.json")
WALK_FORWARD_OUTPUT_PATH = Path("data/model_artifacts/walk_forward_training_window.json")
EXPANDING_WALK_FORWARD_OUTPUT_PATH = Path("data/model_artifacts/expanding_walk_forward_training_window.json")
LEAGUE_SUBSET_OUTPUT_PATH = Path("data/model_artifacts/league_subset_walk_forward.json")
TEST_SEASONS = ("2023-24", "2024-25", "2025-26")
EXPERIMENTS = (
    ("long_history_to_2022", "2010-11", "2022-23"),
    ("recent_2018_to_2022", "2018-19", "2022-23"),
    ("post_covid_2021_to_2022", "2021-22", "2022-23"),
)
WALK_FORWARD_TEST_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
DEFAULT_RECENT_WINDOW_SEASONS = 5
DEFAULT_EXPANDING_START_SEASON = "2010-11"
LEAGUE_SUBSET_EXPERIMENTS = (
    ("all_five", ("ENG", "SPA", "FRA", "GER", "ITA")),
    ("eng_only", ("ENG",)),
    ("spa_only", ("SPA",)),
    ("fra_only", ("FRA",)),
    ("ger_only", ("GER",)),
    ("ita_only", ("ITA",)),
    ("eng_spa", ("ENG", "SPA")),
    ("eng_spa_ger", ("ENG", "SPA", "GER")),
    ("exclude_fra", ("ENG", "SPA", "GER", "ITA")),
    ("exclude_ita", ("ENG", "SPA", "FRA", "GER")),
    ("exclude_fra_ita", ("ENG", "SPA", "GER")),
)


@dataclass(frozen=True)
class ExperimentSummary:
    output_path: Path
    experiments: int

    def line(self) -> str:
        return f"training_window_experiments={self.output_path}, experiments={self.experiments}"


def _round(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def season_range(df: pd.DataFrame, start_season: str, end_season: str) -> pd.Series:
    starts = df["Season"].map(season_start_year)
    return (starts >= season_start_year(start_season)) & (starts <= season_start_year(end_season))


def filter_test_seasons(df: pd.DataFrame, seasons: Iterable[str]) -> pd.DataFrame:
    selected = tuple(seasons)
    test = df[df["Season"].isin(selected)].copy().reset_index(drop=True)
    if test.empty:
        raise ValueError(f"No test rows found for seasons: {selected}")
    return test


def build_odds_frame(scored_df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    matrix = normalize_probabilities(proba)
    odds = probabilities_to_odds(matrix)
    output = scored_df[["RBallID", "HomeTeam", "AwayTeam", "Date", "Season", "Result"]].copy()
    output["P_Home"] = matrix[:, 0]
    output["P_Draw"] = matrix[:, 1]
    output["P_Away"] = matrix[:, 2]
    output["ModelOdds_Home"] = odds[:, 0]
    output["ModelOdds_Draw"] = odds[:, 1]
    output["ModelOdds_Away"] = odds[:, 2]
    return output[OUTPUT_COLUMNS]


def summarize_group(value_bets: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    if value_bets.empty:
        return []
    df = value_bets.copy()
    df["Won"] = df["Outcome"] == df["Result"]
    df["FlatStakeProfit"] = np.where(df["Won"], pd.to_numeric(df["BestBookOdds"], errors="raise") - 1.0, -1.0)
    table = df.groupby(columns, sort=True).agg(
        bets=("Won", "size"),
        wins=("Won", "sum"),
        hit_rate=("Won", "mean"),
        profit=("FlatStakeProfit", "sum"),
        roi=("FlatStakeProfit", "mean"),
        avg_book_odds=("BestBookOdds", "mean"),
    ).reset_index()
    rows: list[dict[str, object]] = []
    for row in table.itertuples(index=False):
        item = {column: getattr(row, column) for column in columns}
        item.update(
            {
                "bets": int(row.bets),
                "wins": int(row.wins),
                "hit_rate": _round(row.hit_rate),
                "profit": _round(row.profit, 2),
                "roi": _round(row.roi),
                "avg_book_odds": _round(row.avg_book_odds),
            }
        )
        rows.append(item)
    return rows


def value_bet_records(value_bets: pd.DataFrame) -> list[dict[str, object]]:
    if value_bets.empty:
        return []
    selected = value_bets.sort_values(["Date", "RBallID", "Outcome"], kind="mergesort").copy()
    columns = [
        "RBallID",
        "HomeTeam",
        "AwayTeam",
        "Date",
        "Season",
        "League",
        "Result",
        "Outcome",
        "ModelOdds",
        "BestBookOdds",
        "Edge",
        "BestBookmaker",
    ]
    available = [column for column in columns if column in selected.columns]
    selected = selected[available]
    if "Date" in selected.columns:
        selected["Date"] = pd.to_datetime(selected["Date"]).dt.strftime("%Y-%m-%d")
    return selected.replace({np.nan: None}).to_dict(orient="records")


def summarize_value_bets(value_bets: pd.DataFrame) -> dict[str, object]:
    if value_bets.empty:
        overall = {"bets": 0, "wins": 0, "hit_rate": 0.0, "profit": 0.0, "roi": 0.0}
    else:
        won = value_bets["Outcome"] == value_bets["Result"]
        profit = np.where(won, pd.to_numeric(value_bets["BestBookOdds"], errors="raise") - 1.0, -1.0)
        overall = {
            "bets": int(len(value_bets)),
            "wins": int(won.sum()),
            "hit_rate": _round(won.mean()),
            "profit": _round(float(profit.sum()), 2),
            "roi": _round(float(profit.mean())),
        }
    return {
        "overall": overall,
        "by_season": summarize_group(value_bets, ["Season"]),
        "by_league": summarize_group(value_bets, ["League"]),
        "by_league_season": summarize_group(value_bets, ["League", "Season"]),
    }


def run_experiment(
    name: str,
    train_start: str,
    train_end: str,
    data: pd.DataFrame,
    bookmaker_odds: pd.DataFrame,
    test_seasons: Iterable[str] = TEST_SEASONS,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
    include_value_bet_records: bool = False,
) -> dict[str, object]:
    train_df = data[season_range(data, train_start, train_end)].copy().reset_index(drop=True)
    test_df = filter_test_seasons(data, test_seasons)
    if train_df.empty:
        raise ValueError(f"No train rows found for {name}: {train_start} to {train_end}")
    model_train_df, calibration_df = split_model_calibration_data(train_df)
    X_train, y_train = select_features_and_target(model_train_df, feature_columns=MARKET_AWARE_FEATURE_COLUMNS)
    X_calibration, y_calibration = select_features_and_target(calibration_df, feature_columns=MARKET_AWARE_FEATURE_COLUMNS)
    X_test, y_test = select_features_and_target(test_df, feature_columns=MARKET_AWARE_FEATURE_COLUMNS)

    model = train_classifier(
        X_train,
        y_train,
        params=dict(DEFAULT_XGB_PARAMS),
        sample_weight=compute_class_sample_weights(y_train),
    )
    calibrated = calibrate_classifier(model, X_calibration, y_calibration, method=calibration_method)
    proba = normalize_probabilities(calibrated.predict_proba(X_test))
    metrics = evaluate_predictions(y_test, proba)
    model_odds = build_odds_frame(test_df, proba)
    matched = match_model_to_bookmaker_odds(model_odds, bookmaker_odds)
    value_bets = compute_value_bets(matched)

    result = {
        "name": name,
        "train_start": train_start,
        "train_end": train_end,
        "test_seasons": list(test_seasons),
        "train_rows": int(len(train_df)),
        "model_train_rows": int(len(model_train_df)),
        "calibration_rows": int(len(calibration_df)),
        "test_rows": int(len(test_df)),
        "matched_rows": int(len(matched)),
        "metrics": {key: _round(value) for key, value in metrics.items()},
        "value_bets": summarize_value_bets(value_bets),
    }
    if include_value_bet_records:
        result["value_bet_records"] = value_bet_records(value_bets)
    return result


def build_experiments(
    experiments: Iterable[tuple[str, str, str]] = EXPERIMENTS,
    test_seasons: Iterable[str] = TEST_SEASONS,
    features_dir: Path = FEATURES_DIR,
    football_data_dir: Path = FOOTBALL_DATA_DIR,
    leagues: Iterable[str] = LEAGUES,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
    include_value_bet_records: bool = False,
) -> dict[str, object]:
    if calibration_method not in CALIBRATION_METHODS:
        raise ValueError(f"Unsupported calibration method: {calibration_method}")
    raw = load_feature_data(leagues=leagues, features_dir=features_dir)
    data, market_summary = add_market_features(raw, football_data_dir=football_data_dir)
    bookmaker_odds = load_football_data_odds(football_data_dir)
    rows = [
        run_experiment(
            name=name,
            train_start=train_start,
            train_end=train_end,
            data=data,
            bookmaker_odds=bookmaker_odds,
            test_seasons=test_seasons,
            calibration_method=calibration_method,
            include_value_bet_records=include_value_bet_records,
        )
        for name, train_start, train_end in experiments
    ]
    return {
        "model": "market_aware_xgboost_split_policy_experiment",
        "test_seasons": list(test_seasons),
        "market_feature_input_rows": market_summary.input_rows,
        "market_feature_output_rows": market_summary.output_rows,
        "experiments": rows,
    }


def season_label_from_start_year(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


def recent_train_window_for_test(test_season: str, window_seasons: int = DEFAULT_RECENT_WINDOW_SEASONS) -> tuple[str, str]:
    if window_seasons < 1:
        raise ValueError("window_seasons must be at least 1")
    test_start = season_start_year(test_season)
    train_end = test_start - 1
    train_start = train_end - window_seasons + 1
    return season_label_from_start_year(train_start), season_label_from_start_year(train_end)


def aggregate_fold_results(folds: list[dict[str, object]]) -> dict[str, object]:
    total_bets = sum(int(fold["value_bets"]["overall"]["bets"]) for fold in folds)
    total_wins = sum(int(fold["value_bets"]["overall"]["wins"]) for fold in folds)
    total_profit = sum(float(fold["value_bets"]["overall"]["profit"]) for fold in folds)
    fold_rois = [float(fold["value_bets"]["overall"]["roi"]) for fold in folds]
    return {
        "folds": int(len(folds)),
        "bets": int(total_bets),
        "wins": int(total_wins),
        "hit_rate": _round(total_wins / total_bets) if total_bets else 0.0,
        "profit": _round(total_profit, 2),
        "roi": _round(total_profit / total_bets) if total_bets else 0.0,
        "positive_roi_folds": int(sum(roi > 0.0 for roi in fold_rois)),
        "negative_roi_folds": int(sum(roi < 0.0 for roi in fold_rois)),
        "avg_fold_roi": _round(float(np.mean(fold_rois))) if fold_rois else 0.0,
        "median_fold_roi": _round(float(np.median(fold_rois))) if fold_rois else 0.0,
    }


def build_walk_forward_experiment(
    test_seasons: Iterable[str] = WALK_FORWARD_TEST_SEASONS,
    window_seasons: int = DEFAULT_RECENT_WINDOW_SEASONS,
    features_dir: Path = FEATURES_DIR,
    football_data_dir: Path = FOOTBALL_DATA_DIR,
    leagues: Iterable[str] = LEAGUES,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
    include_value_bet_records: bool = False,
) -> dict[str, object]:
    if calibration_method not in CALIBRATION_METHODS:
        raise ValueError(f"Unsupported calibration method: {calibration_method}")
    raw = load_feature_data(leagues=leagues, features_dir=features_dir)
    data, market_summary = add_market_features(raw, football_data_dir=football_data_dir)
    bookmaker_odds = load_football_data_odds(football_data_dir)
    folds = []
    for test_season in test_seasons:
        train_start, train_end = recent_train_window_for_test(test_season, window_seasons=window_seasons)
        folds.append(
            run_experiment(
                name=f"recent_{train_start}_to_{train_end}_test_{test_season}",
                train_start=train_start,
                train_end=train_end,
                data=data,
                bookmaker_odds=bookmaker_odds,
                test_seasons=(test_season,),
                calibration_method=calibration_method,
                include_value_bet_records=include_value_bet_records,
            )
        )
    return {
        "model": "market_aware_xgboost_walk_forward_recent_window",
        "window_seasons": int(window_seasons),
        "test_seasons": list(test_seasons),
        "market_feature_input_rows": market_summary.input_rows,
        "market_feature_output_rows": market_summary.output_rows,
        "aggregate": aggregate_fold_results(folds),
        "folds": folds,
    }


def expanding_train_window_for_test(
    test_season: str,
    start_season: str = DEFAULT_EXPANDING_START_SEASON,
) -> tuple[str, str]:
    test_start = season_start_year(test_season)
    train_end = test_start - 1
    if season_start_year(start_season) > train_end:
        raise ValueError("start_season must be before the test season")
    return start_season, season_label_from_start_year(train_end)


def build_expanding_walk_forward_experiment(
    test_seasons: Iterable[str] = WALK_FORWARD_TEST_SEASONS,
    start_season: str = DEFAULT_EXPANDING_START_SEASON,
    features_dir: Path = FEATURES_DIR,
    football_data_dir: Path = FOOTBALL_DATA_DIR,
    leagues: Iterable[str] = LEAGUES,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
    include_value_bet_records: bool = False,
) -> dict[str, object]:
    if calibration_method not in CALIBRATION_METHODS:
        raise ValueError(f"Unsupported calibration method: {calibration_method}")
    raw = load_feature_data(leagues=leagues, features_dir=features_dir)
    data, market_summary = add_market_features(raw, football_data_dir=football_data_dir)
    bookmaker_odds = load_football_data_odds(football_data_dir)
    folds = []
    for test_season in test_seasons:
        train_start, train_end = expanding_train_window_for_test(test_season, start_season=start_season)
        folds.append(
            run_experiment(
                name=f"expanding_{train_start}_to_{train_end}_test_{test_season}",
                train_start=train_start,
                train_end=train_end,
                data=data,
                bookmaker_odds=bookmaker_odds,
                test_seasons=(test_season,),
                calibration_method=calibration_method,
                include_value_bet_records=include_value_bet_records,
            )
        )
    return {
        "model": "market_aware_xgboost_expanding_walk_forward",
        "start_season": start_season,
        "test_seasons": list(test_seasons),
        "market_feature_input_rows": market_summary.input_rows,
        "market_feature_output_rows": market_summary.output_rows,
        "aggregate": aggregate_fold_results(folds),
        "folds": folds,
    }


def validate_league_subset(leagues: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(leagues)
    if not selected:
        raise ValueError("At least one league is required")
    unknown = sorted(set(selected) - set(LEAGUES))
    if unknown:
        raise ValueError(f"Unknown leagues: {unknown}")
    return selected


def build_league_subset_walk_forward_experiment(
    subset_specs: Iterable[tuple[str, tuple[str, ...]]] = LEAGUE_SUBSET_EXPERIMENTS,
    test_seasons: Iterable[str] = WALK_FORWARD_TEST_SEASONS,
    start_season: str = DEFAULT_EXPANDING_START_SEASON,
    features_dir: Path = FEATURES_DIR,
    football_data_dir: Path = FOOTBALL_DATA_DIR,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
) -> dict[str, object]:
    subsets = []
    for name, leagues in subset_specs:
        selected_leagues = validate_league_subset(leagues)
        payload = build_expanding_walk_forward_experiment(
            test_seasons=test_seasons,
            start_season=start_season,
            features_dir=features_dir,
            football_data_dir=football_data_dir,
            leagues=selected_leagues,
            calibration_method=calibration_method,
        )
        subsets.append(
            {
                "name": name,
                "leagues": list(selected_leagues),
                "aggregate": payload["aggregate"],
                "folds": payload["folds"],
            }
        )
    subsets.sort(key=lambda item: (float(item["aggregate"].get("roi", 0.0)), float(item["aggregate"].get("profit", 0.0))), reverse=True)
    return {
        "model": "market_aware_xgboost_league_subset_expanding_walk_forward",
        "start_season": start_season,
        "test_seasons": list(test_seasons),
        "subsets": subsets,
    }


def run_league_subset_pipeline(output_path: Path = LEAGUE_SUBSET_OUTPUT_PATH) -> ExperimentSummary:
    payload = build_league_subset_walk_forward_experiment()
    write_json(output_path, payload)
    return ExperimentSummary(output_path=output_path, experiments=len(payload["subsets"]))


def run_expanding_walk_forward_pipeline(output_path: Path = EXPANDING_WALK_FORWARD_OUTPUT_PATH) -> ExperimentSummary:
    payload = build_expanding_walk_forward_experiment(include_value_bet_records=True)
    write_json(output_path, payload)
    return ExperimentSummary(output_path=output_path, experiments=len(payload["folds"]))


def run_walk_forward_pipeline(output_path: Path = WALK_FORWARD_OUTPUT_PATH) -> ExperimentSummary:
    payload = build_walk_forward_experiment()
    write_json(output_path, payload)
    return ExperimentSummary(output_path=output_path, experiments=len(payload["folds"]))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_pipeline(output_path: Path = OUTPUT_PATH) -> ExperimentSummary:
    payload = build_experiments()
    write_json(output_path, payload)
    return ExperimentSummary(output_path=output_path, experiments=len(payload["experiments"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare XGBoost training windows on forward seasons.")
    parser.add_argument("--mode", choices=("split", "walk-forward", "expanding-walk-forward", "league-subsets"), default="split")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "walk-forward":
        summary = run_walk_forward_pipeline(output_path=args.output_path or WALK_FORWARD_OUTPUT_PATH)
    elif args.mode == "expanding-walk-forward":
        summary = run_expanding_walk_forward_pipeline(output_path=args.output_path or EXPANDING_WALK_FORWARD_OUTPUT_PATH)
    elif args.mode == "league-subsets":
        summary = run_league_subset_pipeline(output_path=args.output_path or LEAGUE_SUBSET_OUTPUT_PATH)
    else:
        summary = run_pipeline(output_path=args.output_path or OUTPUT_PATH)
    print(summary.line())
