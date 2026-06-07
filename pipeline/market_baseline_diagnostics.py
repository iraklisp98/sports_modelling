from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from pipeline.model_diagnostics import PROBABILITY_BINS, probability_bucket
    from pipeline.stage3_train import LABELS, evaluate_predictions, multiclass_brier_score
    from pipeline.stage5_compare import (
        FOOTBALL_DATA_DIR,
        MODEL_ODDS_PATH,
        best_odds_for_outcome,
        load_football_data_odds,
        match_model_to_bookmaker_odds,
    )
except ModuleNotFoundError:
    from model_diagnostics import PROBABILITY_BINS, probability_bucket
    from stage3_train import LABELS, evaluate_predictions, multiclass_brier_score
    from stage5_compare import (
        FOOTBALL_DATA_DIR,
        MODEL_ODDS_PATH,
        best_odds_for_outcome,
        load_football_data_odds,
        match_model_to_bookmaker_odds,
    )

OUTPUT_PATH = Path("data/model_artifacts/market_baseline_diagnostics.json")
DEFAULT_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23")
OUTCOMES = ("H", "D", "A")
RESULT_TO_CODE = {"H": 0, "D": 1, "A": 2}
MODEL_ODDS_COLUMNS = {"H": "ModelOdds_Home", "D": "ModelOdds_Draw", "A": "ModelOdds_Away"}
BEST_ODDS_COLUMNS = {"H": "BestOdds_Home", "D": "BestOdds_Draw", "A": "BestOdds_Away"}


@dataclass(frozen=True)
class MarketBaselineSummary:
    output_path: Path
    rows: int
    seasons: list[str] | None

    def line(self) -> str:
        return f"market_baseline={self.output_path}, rows={self.rows}, seasons={self.seasons}"


def _round(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("Expected a probability matrix with three outcome columns")
    if not np.isfinite(matrix).all() or (matrix <= 0.0).any():
        raise ValueError("Probability inputs must be finite and strictly positive")
    return matrix / matrix.sum(axis=1, keepdims=True)


def build_market_table(model_odds: pd.DataFrame, bookmaker_odds: pd.DataFrame) -> pd.DataFrame:
    matched = match_model_to_bookmaker_odds(model_odds, bookmaker_odds)
    if matched.empty:
        raise ValueError("No model/bookmaker rows matched for market diagnostics")
    table = matched.copy().reset_index(drop=True)
    for outcome, column in MODEL_ODDS_COLUMNS.items():
        table[f"ModelProb_{outcome}"] = 1.0 / pd.to_numeric(table[column], errors="raise")
    for outcome in OUTCOMES:
        odds_values = []
        bookmaker_values = []
        for _, row in table.iterrows():
            best_odds, bookmaker = best_odds_for_outcome(row, outcome)
            odds_values.append(best_odds)
            bookmaker_values.append(bookmaker)
        table[BEST_ODDS_COLUMNS[outcome]] = odds_values
        table[f"BestBookmaker_{outcome}"] = bookmaker_values
    table = table.dropna(subset=[BEST_ODDS_COLUMNS[outcome] for outcome in OUTCOMES]).reset_index(drop=True)
    if table.empty:
        raise ValueError("No rows have a complete 1X2 bookmaker market")
    for outcome in OUTCOMES:
        table[f"MarketRawProb_{outcome}"] = 1.0 / pd.to_numeric(table[BEST_ODDS_COLUMNS[outcome]], errors="raise")
    model_probs = _normalize_rows(table[[f"ModelProb_{outcome}" for outcome in OUTCOMES]].to_numpy())
    market_probs = _normalize_rows(table[[f"MarketRawProb_{outcome}" for outcome in OUTCOMES]].to_numpy())
    for idx, outcome in enumerate(OUTCOMES):
        table[f"ModelProb_{outcome}"] = model_probs[:, idx]
        table[f"MarketProb_{outcome}"] = market_probs[:, idx]
    table["BookmakerMargin"] = table[[f"MarketRawProb_{outcome}" for outcome in OUTCOMES]].sum(axis=1) - 1.0
    return table


def filter_seasons(table: pd.DataFrame, seasons: Iterable[str] | None = DEFAULT_SEASONS) -> pd.DataFrame:
    if seasons is None:
        return table.copy().reset_index(drop=True)
    return table[table["Season"].isin(tuple(seasons))].copy().reset_index(drop=True)


def probability_matrix(table: pd.DataFrame, prefix: str) -> np.ndarray:
    return table[[f"{prefix}_{outcome}" for outcome in OUTCOMES]].to_numpy(dtype=float)


def model_metrics(table: pd.DataFrame, prefix: str) -> dict[str, object]:
    y = table["Result"].map(RESULT_TO_CODE).astype("int64")
    proba = probability_matrix(table, prefix)
    metrics = evaluate_predictions(y, proba)
    return {key: _round(value) if isinstance(value, float) else value for key, value in metrics.items()}


def calibration_rows(table: pd.DataFrame, prefix: str, bins: Iterable[float] = PROBABILITY_BINS) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outcome in OUTCOMES:
        frame = pd.DataFrame(
            {
                "outcome": outcome,
                "probability": table[f"{prefix}_{outcome}"].astype(float),
                "actual": (table["Result"] == outcome).astype(int),
            }
        )
        frame["bucket"] = frame["probability"].map(lambda value: probability_bucket(float(value), bins=bins))
        grouped = frame.groupby(["outcome", "bucket"], sort=True)
        summary = grouped.agg(
            count=("actual", "size"),
            avg_probability=("probability", "mean"),
            empirical_rate=("actual", "mean"),
        ).reset_index()
        summary["calibration_error"] = summary["empirical_rate"] - summary["avg_probability"]
        summary["abs_calibration_error"] = summary["calibration_error"].abs()
        for row in summary.itertuples(index=False):
            rows.append(
                {
                    "outcome": row.outcome,
                    "bucket": row.bucket,
                    "count": int(row.count),
                    "avg_probability": _round(row.avg_probability),
                    "empirical_rate": _round(row.empirical_rate),
                    "calibration_error": _round(row.calibration_error),
                    "abs_calibration_error": _round(row.abs_calibration_error),
                }
            )
    return rows


def edge_diagnostics(table: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for outcome in ("H", "A"):
        frame = table.copy()
        frame["ModelEdgeProb"] = frame[f"ModelProb_{outcome}"] - frame[f"MarketProb_{outcome}"]
        frame["Won"] = frame["Result"] == outcome
        frame["FlatStakeProfit"] = np.where(frame["Won"], pd.to_numeric(frame[BEST_ODDS_COLUMNS[outcome]], errors="raise") - 1.0, -1.0)
        frame["EdgeBucket"] = pd.cut(
            frame["ModelEdgeProb"],
            bins=[-1.0, -0.10, -0.05, 0.0, 0.05, 0.10, 1.0],
            labels=["<-10%", "-10%--5%", "-5%-0%", "0%-5%", "5%-10%", ">10%"],
            include_lowest=True,
        ).astype(str)
        grouped = frame.groupby("EdgeBucket", sort=True)
        summary = grouped.agg(
            count=("Won", "size"),
            hit_rate=("Won", "mean"),
            profit=("FlatStakeProfit", "sum"),
            roi=("FlatStakeProfit", "mean"),
            avg_model_probability=(f"ModelProb_{outcome}", "mean"),
            avg_market_probability=(f"MarketProb_{outcome}", "mean"),
            avg_book_odds=(BEST_ODDS_COLUMNS[outcome], "mean"),
        ).reset_index()
        for row in summary.itertuples(index=False):
            rows.append(
                {
                    "outcome": outcome,
                    "edge_bucket": row.EdgeBucket,
                    "count": int(row.count),
                    "hit_rate": _round(row.hit_rate),
                    "profit": _round(row.profit, 2),
                    "roi": _round(row.roi),
                    "avg_model_probability": _round(row.avg_model_probability),
                    "avg_market_probability": _round(row.avg_market_probability),
                    "avg_book_odds": _round(row.avg_book_odds),
                }
            )
    return rows


def build_diagnostics(
    model_odds: pd.DataFrame,
    bookmaker_odds: pd.DataFrame,
    seasons: Iterable[str] | None = DEFAULT_SEASONS,
) -> dict[str, object]:
    table = filter_seasons(build_market_table(model_odds, bookmaker_odds), seasons=seasons)
    if table.empty:
        raise ValueError("No rows remain after season filtering")
    return {
        "seasons": list(seasons) if seasons is not None else None,
        "rows": int(len(table)),
        "avg_bookmaker_margin": _round(table["BookmakerMargin"].mean()),
        "model_metrics": model_metrics(table, "ModelProb"),
        "market_metrics": model_metrics(table, "MarketProb"),
        "model_calibration": calibration_rows(table, "ModelProb"),
        "market_calibration": calibration_rows(table, "MarketProb"),
        "model_minus_market_edge": edge_diagnostics(table),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_pipeline(
    model_odds_path: Path = MODEL_ODDS_PATH,
    football_data_dir: Path = FOOTBALL_DATA_DIR,
    output_path: Path = OUTPUT_PATH,
    seasons: Iterable[str] | None = DEFAULT_SEASONS,
) -> MarketBaselineSummary:
    if not model_odds_path.exists():
        raise FileNotFoundError(f"Missing model odds file: {model_odds_path}")
    model_odds = pd.read_parquet(model_odds_path)
    bookmaker_odds = load_football_data_odds(football_data_dir)
    payload = build_diagnostics(model_odds, bookmaker_odds, seasons=seasons)
    write_json(output_path, payload)
    return MarketBaselineSummary(output_path=output_path, rows=int(payload["rows"]), seasons=payload["seasons"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model probabilities against bookmaker-implied market probabilities.")
    parser.add_argument("--model-odds-path", type=Path, default=MODEL_ODDS_PATH)
    parser.add_argument("--football-data-dir", type=Path, default=FOOTBALL_DATA_DIR)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seasons", nargs="*", default=list(DEFAULT_SEASONS), help="Seasons to include; omit values for all seasons")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seasons = tuple(args.seasons) if args.seasons else None
    summary = run_pipeline(
        model_odds_path=args.model_odds_path,
        football_data_dir=args.football_data_dir,
        output_path=args.output_path,
        seasons=seasons,
    )
    print(summary.line())
