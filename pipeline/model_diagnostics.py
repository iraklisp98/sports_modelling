from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HOLDOUT_PREDICTIONS_PATH = Path("data/model_artifacts/stage3/holdout_predictions.parquet")
VALUE_BETS_PATH = Path("data/output/value_bets.parquet")
OUTPUT_PATH = Path("data/model_artifacts/stage3/model_diagnostics.json")
HOLDOUT_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23")
OUTCOME_PROBABILITY_COLUMNS = {"H": "P_Home", "D": "P_Draw", "A": "P_Away"}
PROBABILITY_BINS = tuple(round(value / 10, 1) for value in range(11))


@dataclass(frozen=True)
class DiagnosticsSummary:
    output_path: Path
    calibration_rows: int
    value_bet_rows: int
    worst_calibration_bucket: dict[str, object] | None

    def line(self) -> str:
        worst = self.worst_calibration_bucket or {}
        worst_text = (
            f"{worst.get('outcome')} {worst.get('bucket')} "
            f"abs_error={worst.get('abs_calibration_error')}"
            if worst
            else "none"
        )
        return (
            f"diagnostics={self.output_path}, calibration_rows={self.calibration_rows}, "
            f"value_bet_rows={self.value_bet_rows}, worst_calibration_bucket={worst_text}"
        )


def _round(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _bucket_label(left: float, right: float) -> str:
    return f"{left:.1f}-{right:.1f}"


def probability_bucket(probability: float, bins: Iterable[float] = PROBABILITY_BINS) -> str:
    values = list(bins)
    if len(values) < 2:
        raise ValueError("At least two probability bucket edges are required")
    if probability < values[0] or probability > values[-1]:
        raise ValueError(f"Probability out of bucket range: {probability}")
    for left, right in zip(values[:-1], values[1:]):
        if probability < right or np.isclose(probability, right):
            return _bucket_label(left, right)
    return _bucket_label(values[-2], values[-1])


def validate_holdout_predictions(df: pd.DataFrame) -> None:
    required = ["RBallID", "Result", *OUTCOME_PROBABILITY_COLUMNS.values()]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing holdout prediction columns: {missing}")
    unknown_results = sorted(set(df["Result"].dropna()) - set(OUTCOME_PROBABILITY_COLUMNS))
    if unknown_results:
        raise ValueError(f"Unknown holdout results: {unknown_results}")
    probabilities = df[list(OUTCOME_PROBABILITY_COLUMNS.values())].apply(pd.to_numeric, errors="raise")
    if probabilities.isna().any().any():
        raise ValueError("Holdout probabilities must not contain null values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any().any():
        raise ValueError("Holdout probabilities must be between 0 and 1")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.001):
        raise ValueError("Holdout probability rows must sum to 1.0 ± 0.001")


def validate_value_bets(df: pd.DataFrame) -> None:
    required = ["RBallID", "Season", "Outcome", "Result", "ModelOdds", "BestBookOdds", "Edge"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing value bet columns: {missing}")
    unknown_outcomes = sorted(set(df["Outcome"].dropna()) - set(OUTCOME_PROBABILITY_COLUMNS))
    if unknown_outcomes:
        raise ValueError(f"Unknown value bet outcomes: {unknown_outcomes}")
    unknown_results = sorted(set(df["Result"].dropna()) - set(OUTCOME_PROBABILITY_COLUMNS))
    if unknown_results:
        raise ValueError(f"Unknown value bet results: {unknown_results}")


def build_prediction_rows(holdout: pd.DataFrame, bins: Iterable[float] = PROBABILITY_BINS) -> pd.DataFrame:
    validate_holdout_predictions(holdout)
    rows: list[dict[str, object]] = []
    for outcome, probability_column in OUTCOME_PROBABILITY_COLUMNS.items():
        probabilities = pd.to_numeric(holdout[probability_column], errors="raise")
        actual = (holdout["Result"] == outcome).astype(int)
        for probability, hit in zip(probabilities, actual):
            rows.append(
                {
                    "outcome": outcome,
                    "probability": float(probability),
                    "actual": int(hit),
                    "bucket": probability_bucket(float(probability), bins=bins),
                }
            )
    return pd.DataFrame(rows)


def build_calibration_table(holdout: pd.DataFrame, bins: Iterable[float] = PROBABILITY_BINS) -> list[dict[str, object]]:
    rows = build_prediction_rows(holdout, bins=bins)
    if rows.empty:
        return []
    grouped = rows.groupby(["outcome", "bucket"], sort=True)
    table = grouped.agg(
        count=("actual", "size"),
        avg_predicted_probability=("probability", "mean"),
        empirical_rate=("actual", "mean"),
    ).reset_index()
    table["calibration_error"] = table["empirical_rate"] - table["avg_predicted_probability"]
    table["abs_calibration_error"] = table["calibration_error"].abs()
    return [
        {
            "outcome": row.outcome,
            "bucket": row.bucket,
            "count": int(row.count),
            "avg_predicted_probability": _round(row.avg_predicted_probability),
            "empirical_rate": _round(row.empirical_rate),
            "calibration_error": _round(row.calibration_error),
            "abs_calibration_error": _round(row.abs_calibration_error),
        }
        for row in table.itertuples(index=False)
    ]


def build_outcome_summary(holdout: pd.DataFrame) -> list[dict[str, object]]:
    validate_holdout_predictions(holdout)
    predicted = holdout[list(OUTCOME_PROBABILITY_COLUMNS.values())].idxmax(axis=1)
    predicted = predicted.map({column: outcome for outcome, column in OUTCOME_PROBABILITY_COLUMNS.items()})
    rows: list[dict[str, object]] = []
    for outcome, probability_column in OUTCOME_PROBABILITY_COLUMNS.items():
        actual = (holdout["Result"] == outcome).astype(int)
        probability = pd.to_numeric(holdout[probability_column], errors="raise")
        rows.append(
            {
                "outcome": outcome,
                "actual_count": int(actual.sum()),
                "predicted_count": int((predicted == outcome).sum()),
                "avg_predicted_probability": _round(probability.mean()),
                "brier_component": _round(((probability - actual) ** 2).mean()),
            }
        )
    return rows


def filter_holdout_value_bets(value_bets: pd.DataFrame, seasons: Iterable[str] = HOLDOUT_SEASONS) -> pd.DataFrame:
    validate_value_bets(value_bets)
    return value_bets[value_bets["Season"].isin(tuple(seasons))].copy().reset_index(drop=True)


def build_value_bet_diagnostics(
    value_bets: pd.DataFrame,
    seasons: Iterable[str] = HOLDOUT_SEASONS,
    bins: Iterable[float] = PROBABILITY_BINS,
) -> list[dict[str, object]]:
    holdout = filter_holdout_value_bets(value_bets, seasons=seasons)
    if holdout.empty:
        return []
    frame = holdout.copy()
    frame["ModelProbability"] = 1.0 / pd.to_numeric(frame["ModelOdds"], errors="raise")
    frame["Bucket"] = frame["ModelProbability"].map(lambda value: probability_bucket(float(value), bins=bins))
    frame["Won"] = frame["Outcome"] == frame["Result"]
    frame["FlatStakeProfit"] = np.where(frame["Won"], pd.to_numeric(frame["BestBookOdds"], errors="raise") - 1.0, -1.0)
    grouped = frame.groupby(["Outcome", "Bucket"], sort=True)
    table = grouped.agg(
        count=("Won", "size"),
        hit_rate=("Won", "mean"),
        avg_model_probability=("ModelProbability", "mean"),
        avg_book_odds=("BestBookOdds", "mean"),
        avg_edge=("Edge", "mean"),
        flat_stake_roi=("FlatStakeProfit", "mean"),
    ).reset_index()
    return [
        {
            "outcome": row.Outcome,
            "bucket": row.Bucket,
            "count": int(row.count),
            "hit_rate": _round(row.hit_rate),
            "avg_model_probability": _round(row.avg_model_probability),
            "avg_book_odds": _round(row.avg_book_odds),
            "avg_edge": _round(row.avg_edge),
            "flat_stake_roi": _round(row.flat_stake_roi),
        }
        for row in table.itertuples(index=False)
    ]


def worst_calibration_bucket(calibration: list[dict[str, object]], min_count: int = 20) -> dict[str, object] | None:
    eligible = [row for row in calibration if int(row["count"]) >= min_count]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (float(row["abs_calibration_error"]), int(row["count"])))


def build_diagnostics(
    holdout_predictions: pd.DataFrame,
    value_bets: pd.DataFrame,
    seasons: Iterable[str] = HOLDOUT_SEASONS,
    bins: Iterable[float] = PROBABILITY_BINS,
) -> dict[str, object]:
    calibration = build_calibration_table(holdout_predictions, bins=bins)
    value_bet_diagnostics = build_value_bet_diagnostics(value_bets, seasons=seasons, bins=bins)
    return {
        "holdout_seasons": list(seasons),
        "probability_bins": list(bins),
        "outcome_summary": build_outcome_summary(holdout_predictions),
        "calibration_by_outcome_bucket": calibration,
        "value_bets_by_outcome_bucket": value_bet_diagnostics,
        "worst_calibration_bucket": worst_calibration_bucket(calibration),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_pipeline(
    holdout_predictions_path: Path = HOLDOUT_PREDICTIONS_PATH,
    value_bets_path: Path = VALUE_BETS_PATH,
    output_path: Path = OUTPUT_PATH,
    seasons: Iterable[str] = HOLDOUT_SEASONS,
) -> DiagnosticsSummary:
    if not holdout_predictions_path.exists():
        raise FileNotFoundError(f"Missing Stage 3 holdout predictions: {holdout_predictions_path}")
    if not value_bets_path.exists():
        raise FileNotFoundError(f"Missing Stage 5 value bets: {value_bets_path}")
    holdout_predictions = pd.read_parquet(holdout_predictions_path)
    value_bets = pd.read_parquet(value_bets_path)
    diagnostics = build_diagnostics(holdout_predictions, value_bets, seasons=seasons)
    write_json(output_path, diagnostics)
    return DiagnosticsSummary(
        output_path=output_path,
        calibration_rows=len(diagnostics["calibration_by_outcome_bucket"]),
        value_bet_rows=len(diagnostics["value_bets_by_outcome_bucket"]),
        worst_calibration_bucket=diagnostics["worst_calibration_bucket"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model calibration and value-bet diagnostics.")
    parser.add_argument("--holdout-predictions-path", type=Path, default=HOLDOUT_PREDICTIONS_PATH)
    parser.add_argument("--value-bets-path", type=Path, default=VALUE_BETS_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seasons", nargs="+", default=list(HOLDOUT_SEASONS))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_pipeline(
        holdout_predictions_path=args.holdout_predictions_path,
        value_bets_path=args.value_bets_path,
        output_path=args.output_path,
        seasons=args.seasons,
    )
    print(summary.line())
