from __future__ import annotations

from typing import Iterable

import pandas as pd

LEAGUES = ("ENG", "SPA", "FRA")
LEAGUE_FEATURE_COLUMNS = tuple(f"League_{league}" for league in LEAGUES)

BASE_FEATURE_COLUMNS = (
    "HomeElo",
    "AwayElo",
    "EloDiff",
    "AbsEloDiff",
    "HomeGoals_Last5",
    "AwayGoals_Last5",
    "AbsGoalsLast5Diff",
    "HomeCorners_Last5",
    "AwayCorners_Last5",
    "HomePoints_Last5",
    "AwayPoints_Last5",
    "AbsPointsLast5Diff",
    "HomeDrawRate_Last5",
    "AwayDrawRate_Last5",
    "AvgDrawRateLast5",
    "AbsDrawRateLast5Diff",
    "HomeWinRate_Season",
    "AwayWinRate_Season",
)

FEATURE_COLUMNS = (*BASE_FEATURE_COLUMNS, *LEAGUE_FEATURE_COLUMNS)


def add_league_indicator_features(df: pd.DataFrame, leagues: Iterable[str] = LEAGUES) -> pd.DataFrame:
    if "League" not in df.columns:
        raise ValueError("Missing League column for league indicator features")
    league_values = tuple(leagues)
    unknown_leagues = sorted(set(df["League"].dropna()) - set(league_values))
    if unknown_leagues:
        raise ValueError(f"Unexpected league labels: {unknown_leagues}")
    featured = df.copy()
    for league in league_values:
        featured[f"League_{league}"] = (featured["League"] == league).astype("float64")
    return featured
