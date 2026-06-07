from __future__ import annotations

from typing import Iterable

import pandas as pd

LEAGUES = ("ENG", "SPA", "FRA", "GER", "ITA")
LEAGUE_FEATURE_COLUMNS = tuple(f"League_{league}" for league in LEAGUES)

BASE_FEATURE_COLUMNS = (
    "HomeElo",
    "AwayElo",
    "EloDiff",
    "AbsEloDiff",
    "HomeGoals_Last5",
    "AwayGoals_Last5",
    "AbsGoalsLast5Diff",
    "HomeGoalsAgainst_Last5",
    "AwayGoalsAgainst_Last5",
    "GoalsAgainstLast5Diff",
    "HomeCorners_Last5",
    "AwayCorners_Last5",
    "HomeCornersAgainst_Last5",
    "AwayCornersAgainst_Last5",
    "CornerForLast5Diff",
    "CornerAgainstLast5Diff",
    "HomeShotsOnTargetFor_Last5",
    "AwayShotsOnTargetFor_Last5",
    "HomeShotsOnTargetAgainst_Last5",
    "AwayShotsOnTargetAgainst_Last5",
    "ShotsOnTargetForLast5Diff",
    "ShotsOnTargetAgainstLast5Diff",
    "HomeFoulsFor_Last5",
    "AwayFoulsFor_Last5",
    "FoulsForLast5Diff",
    "HomeOffsidesFor_Last5",
    "AwayOffsidesFor_Last5",
    "OffsidesForLast5Diff",
    "HomePoints_Last5",
    "AwayPoints_Last5",
    "AbsPointsLast5Diff",
    "HomeDrawRate_Last5",
    "AwayDrawRate_Last5",
    "AvgDrawRateLast5",
    "AbsDrawRateLast5Diff",
    "HomeWinRate_Season",
    "AwayWinRate_Season",
    "HomeVenuePoints_Last5",
    "AwayVenuePoints_Last5",
    "VenuePointsLast5Diff",
    "HomeVenueGoalsFor_Last5",
    "AwayVenueGoalsFor_Last5",
    "HomeVenueGoalsAgainst_Last5",
    "AwayVenueGoalsAgainst_Last5",
    "HomeVenueWinRate_Season",
    "AwayVenueWinRate_Season",
    "HomeRestDays",
    "AwayRestDays",
    "RestDaysDiff",
    "HomeMatchesLast14Days",
    "AwayMatchesLast14Days",
    "CongestionDiff",
)

MARKET_FEATURE_COLUMNS = (
    "MarketProb_H",
    "MarketProb_D",
    "MarketProb_A",
    "MarketHomeAwayProbDiff",
    "MarketFavoriteProb",
    "MarketBookmakerMargin",
)

FEATURE_COLUMNS = (*BASE_FEATURE_COLUMNS, *LEAGUE_FEATURE_COLUMNS)
MARKET_AWARE_FEATURE_COLUMNS = (*BASE_FEATURE_COLUMNS, *MARKET_FEATURE_COLUMNS, *LEAGUE_FEATURE_COLUMNS)


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
