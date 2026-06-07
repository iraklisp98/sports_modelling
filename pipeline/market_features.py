from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from pipeline.stage5_compare import (
        FOOTBALL_DATA_DIR,
        add_match_keys,
        best_odds_for_outcome,
        load_football_data_odds,
        validate_football_data_columns,
    )
except ModuleNotFoundError:
    from stage5_compare import (
        FOOTBALL_DATA_DIR,
        add_match_keys,
        best_odds_for_outcome,
        load_football_data_odds,
        validate_football_data_columns,
    )

MARKET_FEATURE_COLUMNS = (
    "MarketProb_H",
    "MarketProb_D",
    "MarketProb_A",
    "MarketHomeAwayProbDiff",
    "MarketFavoriteProb",
    "MarketBookmakerMargin",
)
OUTCOMES = ("H", "D", "A")
BEST_ODDS_COLUMNS = {"H": "MarketBestOdds_H", "D": "MarketBestOdds_D", "A": "MarketBestOdds_A"}


@dataclass(frozen=True)
class MarketFeatureSummary:
    input_rows: int
    output_rows: int
    dropped_rows: int


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("Expected a 2D matrix with H/D/A columns")
    if not np.isfinite(matrix).all() or (matrix <= 0.0).any():
        raise ValueError("Market odds must produce finite positive implied probabilities")
    return matrix / matrix.sum(axis=1, keepdims=True)


def _best_market_odds_columns(table: pd.DataFrame) -> pd.DataFrame:
    enriched = table.copy().reset_index(drop=True)
    for outcome in OUTCOMES:
        odds_values = []
        bookmaker_values = []
        for _, row in enriched.iterrows():
            best_odds, bookmaker = best_odds_for_outcome(row, outcome)
            odds_values.append(best_odds)
            bookmaker_values.append(bookmaker)
        enriched[BEST_ODDS_COLUMNS[outcome]] = odds_values
        enriched[f"MarketBestBookmaker_{outcome}"] = bookmaker_values
    return enriched


def add_market_features_from_bookmaker_odds(features: pd.DataFrame, bookmaker_odds: pd.DataFrame) -> tuple[pd.DataFrame, MarketFeatureSummary]:
    required_feature_columns = ["RBallID", "Date", "HomeTeam", "AwayTeam"]
    missing_features = [column for column in required_feature_columns if column not in features.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns for market features: {missing_features}")
    validate_football_data_columns(bookmaker_odds)

    input_rows = len(features)
    feature_keyed = add_match_keys(features)
    bookmaker_keyed = add_match_keys(bookmaker_odds)
    join_columns = ["Date", "HomeTeamKey", "AwayTeamKey"]
    bookmaker_value_columns = [
        column
        for column in bookmaker_keyed.columns
        if any(str(column).endswith(f"_{outcome}") for outcome in OUTCOMES)
    ]
    if "League" in bookmaker_keyed.columns:
        bookmaker_value_columns.append("League")

    merged = feature_keyed.merge(
        bookmaker_keyed[join_columns + bookmaker_value_columns],
        on=join_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_Bookmaker"),
    )
    if merged.empty:
        raise ValueError("No feature rows matched bookmaker odds for market-aware features")

    merged = _best_market_odds_columns(merged)
    complete = merged.dropna(subset=[BEST_ODDS_COLUMNS[outcome] for outcome in OUTCOMES]).copy().reset_index(drop=True)
    if complete.empty:
        raise ValueError("No matched rows have a complete H/D/A bookmaker market")

    raw = np.column_stack([1.0 / pd.to_numeric(complete[BEST_ODDS_COLUMNS[outcome]], errors="raise") for outcome in OUTCOMES])
    normalized = _normalize_rows(raw)
    for index, outcome in enumerate(OUTCOMES):
        complete[f"MarketProb_{outcome}"] = normalized[:, index]

    complete["MarketBookmakerMargin"] = raw.sum(axis=1) - 1.0
    complete["MarketHomeAwayProbDiff"] = complete["MarketProb_H"] - complete["MarketProb_A"]
    complete["MarketFavoriteProb"] = complete[["MarketProb_H", "MarketProb_D", "MarketProb_A"]].max(axis=1)
    complete = complete.drop(columns=["HomeTeamKey", "AwayTeamKey"])
    output_rows = len(complete)
    return complete, MarketFeatureSummary(input_rows=input_rows, output_rows=output_rows, dropped_rows=input_rows - output_rows)


def add_market_features(features: pd.DataFrame, football_data_dir: Path = FOOTBALL_DATA_DIR) -> tuple[pd.DataFrame, MarketFeatureSummary]:
    bookmaker_odds = load_football_data_odds(football_data_dir)
    return add_market_features_from_bookmaker_odds(features, bookmaker_odds)


def has_market_features(df: pd.DataFrame, columns: Iterable[str] = MARKET_FEATURE_COLUMNS) -> bool:
    return set(columns).issubset(df.columns)
