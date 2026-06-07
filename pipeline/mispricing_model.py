from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

try:
    from pipeline.market_features import FOOTBALL_DATA_DIR, add_market_features
    from pipeline.model_features import LEAGUE_FEATURE_COLUMNS, LEAGUES, add_league_indicator_features
    from pipeline.stage3_train import FEATURES_DIR, season_start_year
except ModuleNotFoundError:
    from market_features import FOOTBALL_DATA_DIR, add_market_features
    from model_features import LEAGUE_FEATURE_COLUMNS, LEAGUES, add_league_indicator_features
    from stage3_train import FEATURES_DIR, season_start_year

OUTPUT_PATH = Path("data/output/mispricing_value_bets.parquet")
DASHBOARD_JSON_PATH = Path("dashboard/data/mispricing_value_bets.json")
ARTIFACTS_DIR = Path("data/model_artifacts/mispricing_model")
VALIDATION_SEASON = "2018-19"
HOLDOUT_START_SEASON = "2019-20"
FORWARD_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23")
OUTCOMES = ("H", "A")
MIN_BOOKMAKER_ODDS = 1.20
MAX_BOOKMAKER_ODDS = 8.00
MIN_PREDICTED_EV = 0.0
EV_THRESHOLD_CANDIDATES = tuple(round(value / 100.0, 2) for value in range(-5, 21))
MIN_VALIDATION_BETS = 200
MIN_VALIDATION_BETS_PER_OUTCOME = 50
ROLLING_VALIDATION_START_SEASON = "2015-16"
MIN_ROLLING_VALIDATION_FOLDS = 3

SIGNED_DIFF_COLUMNS = {
    "EloDiff": "SignalEloDiff",
    "HomeGoals_Last5": "RawHomeGoalsLast5",
    "AwayGoals_Last5": "RawAwayGoalsLast5",
    "AbsGoalsLast5Diff": "SignalAbsGoalsLast5Diff",
    "GoalsAgainstLast5Diff": "SignalDefenseDiff",
    "CornerForLast5Diff": "SignalCornerForDiff",
    "CornerAgainstLast5Diff": "SignalCornerAgainstDiff",
    "ShotsOnTargetForLast5Diff": "SignalShotsOnTargetForDiff",
    "ShotsOnTargetAgainstLast5Diff": "SignalShotsOnTargetAgainstDiff",
    "FoulsForLast5Diff": "SignalFoulsForDiff",
    "OffsidesForLast5Diff": "SignalOffsidesForDiff",
    "AbsPointsLast5Diff": "SignalAbsPointsLast5Diff",
    "AbsDrawRateLast5Diff": "SignalAbsDrawRateLast5Diff",
    "VenuePointsLast5Diff": "SignalVenuePointsDiff",
    "RestDaysDiff": "SignalRestDaysDiff",
    "CongestionDiff": "SignalCongestionDiff",
}

CANDIDATE_FEATURE_COLUMNS = (
    "OutcomeIsHome",
    "BestBookOdds",
    "MarketProbOutcome",
    "MarketFairOddsOutcome",
    "MarketFavoriteProb",
    "MarketBookmakerMargin",
    "SignedMarketHomeAwayProbDiff",
    "SignalEloDiff",
    "AbsEloDiff",
    "SignalGoalsForDiff",
    "SignalDefenseDiff",
    "SignalCornerForDiff",
    "SignalCornerAgainstDiff",
    "SignalShotsOnTargetForDiff",
    "SignalShotsOnTargetAgainstDiff",
    "SignalFoulsForDiff",
    "SignalOffsidesForDiff",
    "SignalPointsDiff",
    "SignalDrawRateDiff",
    "SignalVenuePointsDiff",
    "SignalRestDaysDiff",
    "SignalCongestionDiff",
    *LEAGUE_FEATURE_COLUMNS,
)
OUTPUT_COLUMNS = [
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
    "ValueBet",
    "BestBookmaker",
    "PredictedWinProbability",
    "PredictedExpectedProfit",
    "MarketProbOutcome",
]


@dataclass(frozen=True)
class MispricingRunSummary:
    train_rows: int
    test_rows: int
    selected_bets: int
    metrics_path: Path
    output_path: Path

    def line(self) -> str:
        return (
            f"train_rows={self.train_rows}, test_rows={self.test_rows}, selected_bets={self.selected_bets}, "
            f"metrics={self.metrics_path}, output={self.output_path}"
        )


def _round(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def load_feature_data(features_dir: Path = FEATURES_DIR, leagues: Iterable[str] = LEAGUES) -> pd.DataFrame:
    frames = []
    for league in leagues:
        path = features_dir / f"{league}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 2 feature file: {path}")
        frame = pd.read_parquet(path).copy()
        frame["League"] = league
        frames.append(frame)
    if not frames:
        raise ValueError("At least one league is required")
    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = add_league_indicator_features(combined, leagues=leagues)
    return combined.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True)


def _signed(row: pd.Series, column: str, outcome: str, reverse_home_advantage: bool = False) -> float:
    value = float(row[column])
    sign = 1.0 if outcome == "H" else -1.0
    if reverse_home_advantage:
        sign *= -1.0
    return sign * value


def build_candidate_bets(features: pd.DataFrame) -> pd.DataFrame:
    required = [
        "RBallID",
        "Date",
        "Season",
        "League",
        "HomeTeam",
        "AwayTeam",
        "Result",
        "MarketProb_H",
        "MarketProb_A",
        "MarketHomeAwayProbDiff",
        "MarketFavoriteProb",
        "MarketBookmakerMargin",
        "MarketBestOdds_H",
        "MarketBestOdds_A",
        "MarketBestBookmaker_H",
        "MarketBestBookmaker_A",
        "EloDiff",
        "AbsEloDiff",
        "HomeGoals_Last5",
        "AwayGoals_Last5",
        "GoalsAgainstLast5Diff",
        "CornerForLast5Diff",
        "CornerAgainstLast5Diff",
        "ShotsOnTargetForLast5Diff",
        "ShotsOnTargetAgainstLast5Diff",
        "FoulsForLast5Diff",
        "OffsidesForLast5Diff",
        "HomePoints_Last5",
        "AwayPoints_Last5",
        "HomeDrawRate_Last5",
        "AwayDrawRate_Last5",
        "VenuePointsLast5Diff",
        "RestDaysDiff",
        "CongestionDiff",
        *LEAGUE_FEATURE_COLUMNS,
    ]
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(f"Missing mispricing candidate columns: {missing}")

    rows: list[dict[str, object]] = []
    for row in features.itertuples(index=False):
        item = pd.Series(row._asdict())
        for outcome in OUTCOMES:
            odds = item[f"MarketBestOdds_{outcome}"]
            if pd.isna(odds):
                continue
            odds = float(odds)
            if odds < MIN_BOOKMAKER_ODDS or odds > MAX_BOOKMAKER_ODDS:
                continue
            won = item["Result"] == outcome
            profit = odds - 1.0 if won else -1.0
            candidate = {
                "RBallID": item["RBallID"],
                "HomeTeam": item["HomeTeam"],
                "AwayTeam": item["AwayTeam"],
                "Date": item["Date"],
                "Season": item["Season"],
                "League": item["League"],
                "Result": item["Result"],
                "Outcome": outcome,
                "BestBookOdds": odds,
                "BestBookmaker": item[f"MarketBestBookmaker_{outcome}"],
                "Won": bool(won),
                "FlatStakeProfit": float(profit),
                "OutcomeIsHome": 1.0 if outcome == "H" else 0.0,
                "MarketProbOutcome": float(item[f"MarketProb_{outcome}"]),
                "MarketFairOddsOutcome": 1.0 / float(item[f"MarketProb_{outcome}"]),
                "MarketFavoriteProb": float(item["MarketFavoriteProb"]),
                "MarketBookmakerMargin": float(item["MarketBookmakerMargin"]),
                "SignedMarketHomeAwayProbDiff": _signed(item, "MarketHomeAwayProbDiff", outcome),
                "SignalEloDiff": _signed(item, "EloDiff", outcome),
                "AbsEloDiff": float(item["AbsEloDiff"]),
                "SignalGoalsForDiff": float(item["HomeGoals_Last5"] - item["AwayGoals_Last5"]) * (1.0 if outcome == "H" else -1.0),
                "SignalDefenseDiff": _signed(item, "GoalsAgainstLast5Diff", outcome, reverse_home_advantage=True),
                "SignalCornerForDiff": _signed(item, "CornerForLast5Diff", outcome),
                "SignalCornerAgainstDiff": _signed(item, "CornerAgainstLast5Diff", outcome, reverse_home_advantage=True),
                "SignalShotsOnTargetForDiff": _signed(item, "ShotsOnTargetForLast5Diff", outcome),
                "SignalShotsOnTargetAgainstDiff": _signed(item, "ShotsOnTargetAgainstLast5Diff", outcome, reverse_home_advantage=True),
                "SignalFoulsForDiff": _signed(item, "FoulsForLast5Diff", outcome),
                "SignalOffsidesForDiff": _signed(item, "OffsidesForLast5Diff", outcome),
                "SignalPointsDiff": float(item["HomePoints_Last5"] - item["AwayPoints_Last5"]) * (1.0 if outcome == "H" else -1.0),
                "SignalDrawRateDiff": float(item["HomeDrawRate_Last5"] - item["AwayDrawRate_Last5"]) * (1.0 if outcome == "H" else -1.0),
                "SignalVenuePointsDiff": _signed(item, "VenuePointsLast5Diff", outcome),
                "SignalRestDaysDiff": _signed(item, "RestDaysDiff", outcome),
                "SignalCongestionDiff": _signed(item, "CongestionDiff", outcome, reverse_home_advantage=True),
            }
            for column in LEAGUE_FEATURE_COLUMNS:
                candidate[column] = float(item[column])
            rows.append(candidate)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return pd.DataFrame(columns=[*OUTPUT_COLUMNS, *CANDIDATE_FEATURE_COLUMNS, "Won", "FlatStakeProfit"])
    candidates["_OutcomeOrder"] = candidates["Outcome"].map({"H": 0, "A": 1})
    return candidates.sort_values(["Date", "RBallID", "_OutcomeOrder"], kind="mergesort").drop(columns=["_OutcomeOrder"]).reset_index(drop=True)


def split_train_forward(candidates: pd.DataFrame, holdout_start_season: str = HOLDOUT_START_SEASON) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = candidates["Season"].map(season_start_year)
    holdout_start = season_start_year(holdout_start_season)
    train = candidates[starts < holdout_start].copy().reset_index(drop=True)
    test = candidates[starts >= holdout_start].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError("No pre-holdout candidate rows available for mispricing training")
    if test.empty:
        raise ValueError("No forward candidate rows available for mispricing evaluation")
    if train["Won"].nunique() < 2:
        raise ValueError("Mispricing training target needs both win/loss classes")
    return train, test


def split_train_validation_forward(
    candidates: pd.DataFrame,
    validation_season: str = VALIDATION_SEASON,
    holdout_start_season: str = HOLDOUT_START_SEASON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    starts = candidates["Season"].map(season_start_year)
    validation_start = season_start_year(validation_season)
    holdout_start = season_start_year(holdout_start_season)
    train = candidates[starts < validation_start].copy().reset_index(drop=True)
    validation = candidates[candidates["Season"] == validation_season].copy().reset_index(drop=True)
    forward = candidates[starts >= holdout_start].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError("No rows available before validation season for mispricing training")
    if validation.empty:
        raise ValueError(f"No validation rows found for season: {validation_season}")
    if forward.empty:
        raise ValueError("No forward candidate rows available for mispricing evaluation")
    if train["Won"].nunique() < 2:
        raise ValueError("Mispricing training target needs both win/loss classes")
    return train, validation, forward


def select_features(df: pd.DataFrame, feature_columns: Iterable[str] = CANDIDATE_FEATURE_COLUMNS) -> pd.DataFrame:
    columns = list(feature_columns)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing mispricing feature columns: {missing}")
    return df[columns].apply(pd.to_numeric, errors="raise").astype("float64")


def train_mispricing_classifier(train: pd.DataFrame) -> HistGradientBoostingClassifier:
    X_train = select_features(train)
    y_train = train["Won"].astype("int64")
    sample_weight = pd.to_numeric(train["BestBookOdds"], errors="raise").clip(lower=1.0).to_numpy(dtype=float)
    sample_weight = sample_weight / sample_weight.mean()
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def predict_candidates(model: HistGradientBoostingClassifier, candidates: pd.DataFrame) -> pd.DataFrame:
    scored = candidates.copy()
    proba = model.predict_proba(select_features(scored))[:, 1]
    scored["PredictedWinProbability"] = proba
    scored["PredictedExpectedProfit"] = (scored["PredictedWinProbability"] * pd.to_numeric(scored["BestBookOdds"], errors="raise")) - 1.0
    scored["ModelOdds"] = 1.0 / scored["PredictedWinProbability"].clip(lower=1e-6)
    scored["Edge"] = (scored["BestBookOdds"] / scored["ModelOdds"]) - 1.0
    return scored


def select_mispriced_bets(scored: pd.DataFrame, min_predicted_ev: float = MIN_PREDICTED_EV) -> pd.DataFrame:
    selected = scored[scored["PredictedExpectedProfit"] > min_predicted_ev].copy()
    selected["ValueBet"] = True
    if selected.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    selected = selected.sort_values(["Date", "RBallID", "Outcome"], kind="mergesort").reset_index(drop=True)
    return selected[OUTPUT_COLUMNS]


def pre_holdout_validation_seasons(
    candidates: pd.DataFrame,
    validation_start_season: str = ROLLING_VALIDATION_START_SEASON,
    holdout_start_season: str = HOLDOUT_START_SEASON,
) -> list[str]:
    start_year = season_start_year(validation_start_season)
    holdout_year = season_start_year(holdout_start_season)
    seasons = sorted(candidates["Season"].dropna().unique(), key=season_start_year)
    return [season for season in seasons if start_year <= season_start_year(season) < holdout_year]


def rolling_validation_thresholds(
    candidates: pd.DataFrame,
    validation_seasons: Iterable[str] | None = None,
    threshold_candidates: Iterable[float] = EV_THRESHOLD_CANDIDATES,
    min_validation_bets_per_outcome: int = MIN_VALIDATION_BETS_PER_OUTCOME,
    min_validation_folds: int = MIN_ROLLING_VALIDATION_FOLDS,
) -> tuple[dict[str, float], dict[str, object]]:
    seasons = list(validation_seasons) if validation_seasons is not None else pre_holdout_validation_seasons(candidates)
    if not seasons:
        raise ValueError("At least one rolling validation season is required")
    rows: list[dict[str, object]] = []
    starts = candidates["Season"].map(season_start_year)
    for season in seasons:
        validation_year = season_start_year(season)
        train = candidates[starts < validation_year].copy().reset_index(drop=True)
        validation = candidates[candidates["Season"] == season].copy().reset_index(drop=True)
        if train.empty or validation.empty or train["Won"].nunique() < 2:
            continue
        model = train_mispricing_classifier(train)
        scored_validation = predict_candidates(model, validation)
        for outcome in OUTCOMES:
            outcome_scored = scored_validation[scored_validation["Outcome"] == outcome].copy().reset_index(drop=True)
            for threshold in threshold_candidates:
                selected = select_mispriced_bets(outcome_scored, min_predicted_ev=float(threshold))
                summary = summarize_bets(selected)
                rows.append(
                    {
                        "validation_season": season,
                        "outcome": outcome,
                        "threshold": float(threshold),
                        **summary,
                    }
                )
    if not rows:
        raise ValueError("Rolling validation produced no threshold candidates")
    raw = pd.DataFrame(rows)
    thresholds: dict[str, float] = {}
    aggregates: list[dict[str, object]] = []
    for outcome in OUTCOMES:
        outcome_rows = raw[(raw["outcome"] == outcome) & (raw["bets"] >= min_validation_bets_per_outcome)].copy()
        if outcome_rows.empty:
            continue
        grouped = outcome_rows.groupby("threshold", sort=True).agg(
            folds=("validation_season", "nunique"),
            avg_roi=("roi", "mean"),
            median_roi=("roi", "median"),
            min_roi=("roi", "min"),
            total_profit=("profit", "sum"),
            total_bets=("bets", "sum"),
        ).reset_index()
        grouped = grouped[grouped["folds"] >= min_validation_folds].copy()
        if grouped.empty:
            continue
        for row in grouped.itertuples(index=False):
            aggregates.append(
                {
                    "outcome": outcome,
                    "threshold": _round(row.threshold),
                    "folds": int(row.folds),
                    "avg_roi": _round(row.avg_roi),
                    "median_roi": _round(row.median_roi),
                    "min_roi": _round(row.min_roi),
                    "total_profit": _round(row.total_profit, 2),
                    "total_bets": int(row.total_bets),
                }
            )
        viable = grouped[grouped["avg_roi"] > 0.0].copy()
        if viable.empty:
            continue
        best = max(
            viable.itertuples(index=False),
            key=lambda row: (float(row.avg_roi), float(row.total_profit), int(row.total_bets), float(row.threshold)),
        )
        thresholds[outcome] = float(best.threshold)
    diagnostics = {
        "validation_seasons": seasons,
        "min_validation_bets_per_outcome": int(min_validation_bets_per_outcome),
        "min_validation_folds": int(min_validation_folds),
        "selected_thresholds": thresholds,
        "aggregates": aggregates,
        "raw": rows,
    }
    return thresholds, diagnostics


def select_ev_threshold(
    scored_validation: pd.DataFrame,
    threshold_candidates: Iterable[float] = EV_THRESHOLD_CANDIDATES,
    min_validation_bets: int = MIN_VALIDATION_BETS,
) -> tuple[float, list[dict[str, object]]]:
    candidates = [float(value) for value in threshold_candidates]
    if not candidates:
        raise ValueError("At least one EV threshold candidate is required")
    rows: list[dict[str, object]] = []
    for threshold in candidates:
        selected = select_mispriced_bets(scored_validation, min_predicted_ev=threshold)
        summary = summarize_bets(selected)
        rows.append({"threshold": threshold, **summary})
    eligible = [row for row in rows if int(row["bets"]) >= min_validation_bets]
    if not eligible:
        eligible = rows
    best = max(eligible, key=lambda row: (float(row["roi"]), float(row["profit"]), int(row["bets"]), float(row["threshold"])))
    return float(best["threshold"]), rows


def select_ev_threshold_by_outcome(
    scored_validation: pd.DataFrame,
    outcomes: Iterable[str] = OUTCOMES,
    threshold_candidates: Iterable[float] = EV_THRESHOLD_CANDIDATES,
    min_validation_bets: int = MIN_VALIDATION_BETS_PER_OUTCOME,
) -> tuple[dict[str, float], dict[str, list[dict[str, object]]]]:
    thresholds: dict[str, float] = {}
    diagnostics: dict[str, list[dict[str, object]]] = {}
    for outcome in outcomes:
        frame = scored_validation[scored_validation["Outcome"] == outcome].copy().reset_index(drop=True)
        if frame.empty:
            continue
        threshold, rows = select_ev_threshold(
            frame,
            threshold_candidates=threshold_candidates,
            min_validation_bets=min_validation_bets,
        )
        thresholds[outcome] = threshold
        diagnostics[outcome] = rows
    if not thresholds:
        raise ValueError("No outcomes available for EV threshold selection")
    return thresholds, diagnostics


def select_mispriced_bets_by_outcome(scored: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    selected_parts = []
    for outcome, threshold in thresholds.items():
        selected = select_mispriced_bets(scored[scored["Outcome"] == outcome], min_predicted_ev=threshold)
        if not selected.empty:
            selected_parts.append(selected)
    if not selected_parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined = pd.concat(selected_parts, ignore_index=True)
    combined["_OutcomeOrder"] = combined["Outcome"].map({"H": 0, "A": 1})
    return combined.sort_values(["Date", "RBallID", "_OutcomeOrder"], kind="mergesort").drop(columns=["_OutcomeOrder"]).reset_index(drop=True)


def summarize_bets(value_bets: pd.DataFrame) -> dict[str, object]:
    if value_bets.empty:
        return {"bets": 0, "wins": 0, "hit_rate": 0.0, "profit": 0.0, "roi": 0.0}
    won = value_bets["Outcome"] == value_bets["Result"]
    profit = np.where(won, pd.to_numeric(value_bets["BestBookOdds"], errors="raise") - 1.0, -1.0)
    return {
        "bets": int(len(value_bets)),
        "wins": int(won.sum()),
        "hit_rate": _round(won.mean()),
        "profit": _round(float(profit.sum()), 2),
        "roi": _round(float(profit.mean())),
    }


def summarize_group(value_bets: pd.DataFrame, group_column: str) -> list[dict[str, object]]:
    if value_bets.empty:
        return []
    rows = []
    for value, group in value_bets.groupby(group_column, sort=True):
        summary = summarize_bets(group)
        summary[group_column] = value
        rows.append(summary)
    return rows


def classification_metrics(scored: pd.DataFrame) -> dict[str, object]:
    y_true = scored["Won"].astype("int64")
    proba = scored["PredictedWinProbability"].to_numpy(dtype=float)
    predicted = proba >= 0.5
    metrics = {
        "log_loss": _round(log_loss(y_true, np.column_stack([1.0 - proba, proba]), labels=[0, 1])),
        "accuracy": _round(accuracy_score(y_true, predicted)),
        "candidate_rows": int(len(scored)),
    }
    if y_true.nunique() == 2:
        metrics["roc_auc"] = _round(roc_auc_score(y_true, proba))
    else:
        metrics["roc_auc"] = None
    return metrics


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_outputs(scored: pd.DataFrame, value_bets: pd.DataFrame, artifacts_dir: Path, output_path: Path, dashboard_json_path: Path, metrics: dict[str, object]) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_json_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(artifacts_dir / "candidate_predictions.parquet", index=False)
    value_bets.to_parquet(output_path, index=False)
    value_bets.to_json(dashboard_json_path, orient="records", date_format="iso")
    write_json(artifacts_dir / "metrics.json", metrics)


def run_pipeline(
    features_dir: Path = FEATURES_DIR,
    football_data_dir: Path = FOOTBALL_DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    output_path: Path = OUTPUT_PATH,
    dashboard_json_path: Path = DASHBOARD_JSON_PATH,
    leagues: Iterable[str] = LEAGUES,
    forward_seasons: Iterable[str] = FORWARD_SEASONS,
    min_predicted_ev: float | None = None,
    validation_season: str = VALIDATION_SEASON,
    min_validation_bets: int = MIN_VALIDATION_BETS,
    min_validation_bets_per_outcome: int = MIN_VALIDATION_BETS_PER_OUTCOME,
    use_outcome_thresholds: bool = True,
    use_rolling_validation: bool = True,
) -> MispricingRunSummary:
    features = load_feature_data(features_dir=features_dir, leagues=leagues)
    market_features, market_summary = add_market_features(features, football_data_dir=football_data_dir)
    candidates = build_candidate_bets(market_features)
    threshold_train, validation, test = split_train_validation_forward(candidates, validation_season=validation_season)
    forward = test[test["Season"].isin(tuple(forward_seasons))].copy().reset_index(drop=True)
    if forward.empty:
        raise ValueError("No candidate rows remain after forward-season filtering")

    threshold_model = train_mispricing_classifier(threshold_train)
    validation_scored = predict_candidates(threshold_model, validation)
    selected_threshold, threshold_results = select_ev_threshold(validation_scored, min_validation_bets=min_validation_bets)
    single_season_outcome_thresholds, outcome_threshold_results = select_ev_threshold_by_outcome(
        validation_scored,
        min_validation_bets=min_validation_bets_per_outcome,
    )
    rolling_outcome_thresholds, rolling_threshold_diagnostics = rolling_validation_thresholds(
        candidates,
        min_validation_bets_per_outcome=min_validation_bets_per_outcome,
    )
    outcome_thresholds = rolling_outcome_thresholds if use_rolling_validation else single_season_outcome_thresholds
    if min_predicted_ev is not None:
        selected_threshold = float(min_predicted_ev)
        outcome_thresholds = {outcome: selected_threshold for outcome in OUTCOMES}
    elif not use_outcome_thresholds:
        outcome_thresholds = {outcome: selected_threshold for outcome in OUTCOMES}
    if not outcome_thresholds:
        outcome_thresholds = {outcome: selected_threshold for outcome in OUTCOMES}
    train = pd.concat([threshold_train, validation], ignore_index=True)
    model = train_mispricing_classifier(train)
    scored = predict_candidates(model, forward)
    value_bets = select_mispriced_bets_by_outcome(scored, outcome_thresholds)
    metrics = {
        "model": "mispricing_hist_gradient_boosting",
        "train_rows": int(len(train)),
        "threshold_train_rows": int(len(threshold_train)),
        "validation_rows": int(len(validation)),
        "validation_season": validation_season,
        "test_rows": int(len(forward)),
        "market_feature_input_rows": market_summary.input_rows,
        "market_feature_output_rows": market_summary.output_rows,
        "market_feature_dropped_rows": market_summary.dropped_rows,
        "forward_seasons": list(forward_seasons),
        "min_predicted_ev": float(selected_threshold),
        "use_outcome_thresholds": use_outcome_thresholds and min_predicted_ev is None,
        "use_rolling_validation": use_rolling_validation and min_predicted_ev is None and use_outcome_thresholds,
        "outcome_thresholds": outcome_thresholds,
        "threshold_selection": threshold_results,
        "outcome_threshold_selection": outcome_threshold_results,
        "rolling_threshold_selection": rolling_threshold_diagnostics,
        "validation_strategy_at_selected_threshold": summarize_bets(select_mispriced_bets(validation_scored, min_predicted_ev=selected_threshold)),
        "validation_strategy_at_outcome_thresholds": summarize_bets(select_mispriced_bets_by_outcome(validation_scored, single_season_outcome_thresholds)),
        "validation_strategy_at_selected_outcome_thresholds": summarize_bets(select_mispriced_bets_by_outcome(validation_scored, outcome_thresholds)),
        "classification": classification_metrics(scored),
        "strategy_overall": summarize_bets(value_bets),
        "strategy_by_outcome": summarize_group(value_bets, "Outcome"),
        "strategy_by_league": summarize_group(value_bets, "League"),
        "strategy_by_season": summarize_group(value_bets, "Season"),
        "feature_columns": list(CANDIDATE_FEATURE_COLUMNS),
    }
    write_outputs(scored, value_bets, artifacts_dir, output_path, dashboard_json_path, metrics)
    return MispricingRunSummary(
        train_rows=len(train),
        test_rows=len(forward),
        selected_bets=len(value_bets),
        metrics_path=artifacts_dir / "metrics.json",
        output_path=output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a second-stage H/A market mispricing model.")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--football-data-dir", type=Path, default=FOOTBALL_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dashboard-json-path", type=Path, default=DASHBOARD_JSON_PATH)
    parser.add_argument("--min-predicted-ev", type=float, default=None, help="Manual EV cutoff. Omit to select on validation season.")
    parser.add_argument("--validation-season", default=VALIDATION_SEASON)
    parser.add_argument("--min-validation-bets", type=int, default=MIN_VALIDATION_BETS)
    parser.add_argument("--min-validation-bets-per-outcome", type=int, default=MIN_VALIDATION_BETS_PER_OUTCOME)
    parser.add_argument("--global-threshold", action="store_true", help="Use one validation-selected EV threshold for both H/A.")
    parser.add_argument("--single-validation-season", action="store_true", help="Use only the configured validation season for H/A thresholds instead of rolling validation.")
    parser.add_argument("--seasons", nargs="*", default=list(FORWARD_SEASONS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seasons = tuple(args.seasons) if args.seasons else FORWARD_SEASONS
    summary = run_pipeline(
        features_dir=args.features_dir,
        football_data_dir=args.football_data_dir,
        artifacts_dir=args.artifacts_dir,
        output_path=args.output_path,
        dashboard_json_path=args.dashboard_json_path,
        forward_seasons=seasons,
        min_predicted_ev=args.min_predicted_ev,
        validation_season=args.validation_season,
        min_validation_bets=args.min_validation_bets,
        min_validation_bets_per_outcome=args.min_validation_bets_per_outcome,
        use_outcome_thresholds=not args.global_threshold,
        use_rolling_validation=not args.single_validation_season,
    )
    print(summary.line())
