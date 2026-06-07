from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
LEAGUES = ("ENG", "SPA", "FRA", "GER", "ITA")

STARTING_ELO = 1500.0
LEAGUE_MEAN_ELO = 1500.0
PROMOTED_TEAM_ELO = LEAGUE_MEAN_ELO - 100.0
HOME_ADVANTAGE = 80.0
K_FACTOR = 30.0
SEASON_REGRESSION = 0.20
ROLLING_WINDOW = 5

REQUIRED_STAGE1_COLUMNS = [
    "RBallID",
    "HomeTeam",
    "AwayTeam",
    "Date",
    "Season",
    "HomeGoals",
    "AwayGoals",
    "HomeCorners",
    "AwayCorners",
    "HomeShotsOnTarget",
    "AwayShotsOnTarget",
    "HomeFouls",
    "AwayFouls",
    "HomeOffsides",
    "AwayOffsides",
]

STAGE2_FEATURE_COLUMNS = [
    "Result",
    "ResultCode",
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
]

RESULT_CODE = {"H": 0, "D": 1, "A": 2}


@dataclass(frozen=True)
class FeatureRunSummary:
    league: str
    rows: int
    output_path: Path
    result_counts: dict[str, int]

    def line(self) -> str:
        return (
            f"[{self.league}] rows={self.rows}, "
            f"results={self.result_counts}, output={self.output_path}"
        )


def validate_stage1_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_STAGE1_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Stage 1 columns: {missing}")


def sort_matches(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.copy()
    sorted_df["Date"] = pd.to_datetime(sorted_df["Date"])
    return sorted_df.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True)


def add_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    conditions = [
        featured["HomeGoals"] > featured["AwayGoals"],
        featured["HomeGoals"] < featured["AwayGoals"],
    ]
    featured["Result"] = np.select(conditions, ["H", "A"], default="D")
    featured["ResultCode"] = featured["Result"].map(RESULT_CODE).astype("int64")
    return featured


def apply_season_regression(
    ratings: dict[str, float],
    league_mean: float = LEAGUE_MEAN_ELO,
    factor: float = SEASON_REGRESSION,
) -> dict[str, float]:
    return {team: rating + factor * (league_mean - rating) for team, rating in ratings.items()}


def expected_home_score(
    home_rating: float,
    away_rating: float,
    home_advantage: float = HOME_ADVANTAGE,
) -> float:
    return 1.0 / (1.0 + 10.0 ** ((away_rating - (home_rating + home_advantage)) / 400.0))


def actual_scores(result: str) -> tuple[float, float]:
    if result == "H":
        return 1.0, 0.0
    if result == "A":
        return 0.0, 1.0
    return 0.5, 0.5


def add_elo_features(
    df: pd.DataFrame,
    starting_elo: float = STARTING_ELO,
    league_mean: float = LEAGUE_MEAN_ELO,
    promoted_team_elo: float = PROMOTED_TEAM_ELO,
    k_factor: float = K_FACTOR,
) -> pd.DataFrame:
    featured = sort_matches(df)
    if "Result" not in featured.columns:
        featured = add_result_columns(featured)

    ratings: dict[str, float] = {}
    first_season = featured["Season"].iloc[0] if not featured.empty else None
    current_season = first_season
    home_elos: list[float] = []
    away_elos: list[float] = []

    for row in featured.itertuples(index=False):
        if row.Season != current_season:
            ratings = apply_season_regression(ratings, league_mean=league_mean)
            current_season = row.Season

        if row.HomeTeam not in ratings:
            ratings[row.HomeTeam] = starting_elo if row.Season == first_season else promoted_team_elo
        if row.AwayTeam not in ratings:
            ratings[row.AwayTeam] = starting_elo if row.Season == first_season else promoted_team_elo

        home_rating = ratings[row.HomeTeam]
        away_rating = ratings[row.AwayTeam]
        home_elos.append(home_rating)
        away_elos.append(away_rating)

        expected_home = expected_home_score(home_rating, away_rating)
        expected_away = 1.0 - expected_home
        actual_home, actual_away = actual_scores(row.Result)

        ratings[row.HomeTeam] = home_rating + k_factor * (actual_home - expected_home)
        ratings[row.AwayTeam] = away_rating + k_factor * (actual_away - expected_away)

    featured["HomeElo"] = home_elos
    featured["AwayElo"] = away_elos
    featured["EloDiff"] = featured["HomeElo"] - featured["AwayElo"]
    return featured


def _team_match_history(df: pd.DataFrame) -> pd.DataFrame:
    home_points = df["Result"].map({"H": 3.0, "D": 1.0, "A": 0.0})
    away_points = df["Result"].map({"H": 0.0, "D": 1.0, "A": 3.0})
    draw_flag = (df["Result"] == "D").astype(float)

    home_history = pd.DataFrame(
        {
            "RBallID": df["RBallID"],
            "Date": df["Date"],
            "Season": df["Season"],
            "Team": df["HomeTeam"],
            "Venue": "H",
            "Goals": df["HomeGoals"],
            "GoalsAgainst": df["AwayGoals"],
            "Corners": df["HomeCorners"],
            "CornersAgainst": df["AwayCorners"],
            "ShotsOnTarget": df["HomeShotsOnTarget"],
            "ShotsOnTargetAgainst": df["AwayShotsOnTarget"],
            "Fouls": df["HomeFouls"],
            "FoulsAgainst": df["AwayFouls"],
            "Offsides": df["HomeOffsides"],
            "OffsidesAgainst": df["AwayOffsides"],
            "Points": home_points,
            "Draw": draw_flag,
        }
    )
    away_history = pd.DataFrame(
        {
            "RBallID": df["RBallID"],
            "Date": df["Date"],
            "Season": df["Season"],
            "Team": df["AwayTeam"],
            "Venue": "A",
            "Goals": df["AwayGoals"],
            "GoalsAgainst": df["HomeGoals"],
            "Corners": df["AwayCorners"],
            "CornersAgainst": df["HomeCorners"],
            "ShotsOnTarget": df["AwayShotsOnTarget"],
            "ShotsOnTargetAgainst": df["HomeShotsOnTarget"],
            "Fouls": df["AwayFouls"],
            "FoulsAgainst": df["HomeFouls"],
            "Offsides": df["AwayOffsides"],
            "OffsidesAgainst": df["HomeOffsides"],
            "Points": away_points,
            "Draw": draw_flag,
        }
    )

    history = pd.concat([home_history, away_history], ignore_index=True)
    return history.sort_values(["Team", "Date", "RBallID"], kind="mergesort").reset_index(drop=True)


def _rolling_mean_by_team(history: pd.DataFrame, source: str, target: str, window: int) -> None:
    grouped = history.groupby("Team", sort=False)
    history[target] = grouped[source].transform(
        lambda values: values.shift(1).rolling(window=window, min_periods=1).mean()
    )


def _rolling_count_by_team(history: pd.DataFrame, days: int = 14) -> pd.Series:
    counts = pd.Series(0.0, index=history.index)
    for _, group in history.groupby("Team", sort=False):
        dates = pd.to_datetime(group["Date"]).reset_index(drop=True)
        values = []
        for idx, current_date in enumerate(dates):
            previous = dates.iloc[:idx]
            values.append(float(((current_date - previous).dt.days <= days).sum()))
        counts.loc[group.index] = values
    return counts


def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    featured = sort_matches(df)
    if "Result" not in featured.columns:
        featured = add_result_columns(featured)

    history = _team_match_history(featured)
    rolling_specs = [
        ("Goals", "Goals_Last5"),
        ("GoalsAgainst", "GoalsAgainst_Last5"),
        ("Corners", "Corners_Last5"),
        ("CornersAgainst", "CornersAgainst_Last5"),
        ("ShotsOnTarget", "ShotsOnTarget_Last5"),
        ("ShotsOnTargetAgainst", "ShotsOnTargetAgainst_Last5"),
        ("Fouls", "Fouls_Last5"),
        ("FoulsAgainst", "FoulsAgainst_Last5"),
        ("Offsides", "Offsides_Last5"),
        ("OffsidesAgainst", "OffsidesAgainst_Last5"),
        ("Points", "Points_Last5"),
        ("Draw", "DrawRate_Last5"),
    ]
    for source, target in rolling_specs:
        _rolling_mean_by_team(history, source, target, window)

    rolling_columns = [target for _, target in rolling_specs]
    history[rolling_columns] = history[rolling_columns].fillna(0.0)
    history["RestDays"] = history.groupby("Team", sort=False)["Date"].diff().dt.days.clip(lower=0).fillna(14.0)
    history["MatchesLast14Days"] = _rolling_count_by_team(history, days=14)

    feature_history = history[["RBallID", "Team", *rolling_columns, "RestDays", "MatchesLast14Days"]]

    home_features = feature_history.rename(
        columns={
            "Team": "HomeTeam",
            "Goals_Last5": "HomeGoals_Last5",
            "GoalsAgainst_Last5": "HomeGoalsAgainst_Last5",
            "Corners_Last5": "HomeCorners_Last5",
            "CornersAgainst_Last5": "HomeCornersAgainst_Last5",
            "ShotsOnTarget_Last5": "HomeShotsOnTargetFor_Last5",
            "ShotsOnTargetAgainst_Last5": "HomeShotsOnTargetAgainst_Last5",
            "Fouls_Last5": "HomeFoulsFor_Last5",
            "FoulsAgainst_Last5": "HomeFoulsAgainst_Last5",
            "Offsides_Last5": "HomeOffsidesFor_Last5",
            "OffsidesAgainst_Last5": "HomeOffsidesAgainst_Last5",
            "Points_Last5": "HomePoints_Last5",
            "DrawRate_Last5": "HomeDrawRate_Last5",
            "RestDays": "HomeRestDays",
            "MatchesLast14Days": "HomeMatchesLast14Days",
        }
    )
    away_features = feature_history.rename(
        columns={
            "Team": "AwayTeam",
            "Goals_Last5": "AwayGoals_Last5",
            "GoalsAgainst_Last5": "AwayGoalsAgainst_Last5",
            "Corners_Last5": "AwayCorners_Last5",
            "CornersAgainst_Last5": "AwayCornersAgainst_Last5",
            "ShotsOnTarget_Last5": "AwayShotsOnTargetFor_Last5",
            "ShotsOnTargetAgainst_Last5": "AwayShotsOnTargetAgainst_Last5",
            "Fouls_Last5": "AwayFoulsFor_Last5",
            "FoulsAgainst_Last5": "AwayFoulsAgainst_Last5",
            "Offsides_Last5": "AwayOffsidesFor_Last5",
            "OffsidesAgainst_Last5": "AwayOffsidesAgainst_Last5",
            "Points_Last5": "AwayPoints_Last5",
            "DrawRate_Last5": "AwayDrawRate_Last5",
            "RestDays": "AwayRestDays",
            "MatchesLast14Days": "AwayMatchesLast14Days",
        }
    )

    featured = featured.merge(home_features, on=["RBallID", "HomeTeam"], how="left")
    featured = featured.merge(away_features, on=["RBallID", "AwayTeam"], how="left")
    return featured


def add_venue_form_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    featured = sort_matches(df)
    if "Result" not in featured.columns:
        featured = add_result_columns(featured)

    history = _team_match_history(featured)
    history = history.sort_values(["Team", "Venue", "Date", "RBallID"], kind="mergesort").reset_index(drop=True)
    grouped = history.groupby(["Team", "Venue"], sort=False)
    for source, target in [
        ("Points", "VenuePoints_Last5"),
        ("Goals", "VenueGoalsFor_Last5"),
        ("GoalsAgainst", "VenueGoalsAgainst_Last5"),
    ]:
        history[target] = grouped[source].transform(
            lambda values: values.shift(1).rolling(window=window, min_periods=1).mean()
        )
    history[["VenuePoints_Last5", "VenueGoalsFor_Last5", "VenueGoalsAgainst_Last5"]] = history[
        ["VenuePoints_Last5", "VenueGoalsFor_Last5", "VenueGoalsAgainst_Last5"]
    ].fillna(0.0)

    history["Win"] = (history["Points"] == 3.0).astype(float)
    season_grouped = history.groupby(["Team", "Venue", "Season"], sort=False)
    previous_wins = season_grouped["Win"].cumsum() - history["Win"]
    previous_matches = season_grouped.cumcount()
    history["VenueWinRate_Season"] = (previous_wins / previous_matches.replace(0, np.nan)).fillna(0.0)

    venue_features = history[
        [
            "RBallID",
            "Team",
            "Venue",
            "VenuePoints_Last5",
            "VenueGoalsFor_Last5",
            "VenueGoalsAgainst_Last5",
            "VenueWinRate_Season",
        ]
    ]
    home_features = venue_features[venue_features["Venue"] == "H"].drop(columns="Venue").rename(
        columns={
            "Team": "HomeTeam",
            "VenuePoints_Last5": "HomeVenuePoints_Last5",
            "VenueGoalsFor_Last5": "HomeVenueGoalsFor_Last5",
            "VenueGoalsAgainst_Last5": "HomeVenueGoalsAgainst_Last5",
            "VenueWinRate_Season": "HomeVenueWinRate_Season",
        }
    )
    away_features = venue_features[venue_features["Venue"] == "A"].drop(columns="Venue").rename(
        columns={
            "Team": "AwayTeam",
            "VenuePoints_Last5": "AwayVenuePoints_Last5",
            "VenueGoalsFor_Last5": "AwayVenueGoalsFor_Last5",
            "VenueGoalsAgainst_Last5": "AwayVenueGoalsAgainst_Last5",
            "VenueWinRate_Season": "AwayVenueWinRate_Season",
        }
    )

    featured = featured.merge(home_features, on=["RBallID", "HomeTeam"], how="left")
    featured = featured.merge(away_features, on=["RBallID", "AwayTeam"], how="left")
    return featured


def add_season_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    featured = sort_matches(df)
    if "Result" not in featured.columns:
        featured = add_result_columns(featured)

    history = _team_match_history(featured)
    history["Win"] = (history["Points"] == 3.0).astype(float)
    history = history.sort_values(["Team", "Season", "Date", "RBallID"], kind="mergesort")
    grouped = history.groupby(["Team", "Season"], sort=False)
    previous_wins = grouped["Win"].cumsum() - history["Win"]
    previous_matches = grouped.cumcount()
    history["WinRate_Season"] = (previous_wins / previous_matches.replace(0, np.nan)).fillna(0.0)

    rates = history[["RBallID", "Team", "WinRate_Season"]]
    home_rates = rates.rename(columns={"Team": "HomeTeam", "WinRate_Season": "HomeWinRate_Season"})
    away_rates = rates.rename(columns={"Team": "AwayTeam", "WinRate_Season": "AwayWinRate_Season"})

    featured = featured.merge(home_rates, on=["RBallID", "HomeTeam"], how="left")
    featured = featured.merge(away_rates, on=["RBallID", "AwayTeam"], how="left")
    return featured


def add_draw_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["AbsEloDiff"] = featured["EloDiff"].abs()
    featured["AbsGoalsLast5Diff"] = (featured["HomeGoals_Last5"] - featured["AwayGoals_Last5"]).abs()
    featured["GoalsAgainstLast5Diff"] = featured["HomeGoalsAgainst_Last5"] - featured["AwayGoalsAgainst_Last5"]
    featured["CornerForLast5Diff"] = featured["HomeCorners_Last5"] - featured["AwayCorners_Last5"]
    featured["CornerAgainstLast5Diff"] = featured["HomeCornersAgainst_Last5"] - featured["AwayCornersAgainst_Last5"]
    featured["ShotsOnTargetForLast5Diff"] = featured["HomeShotsOnTargetFor_Last5"] - featured["AwayShotsOnTargetFor_Last5"]
    featured["ShotsOnTargetAgainstLast5Diff"] = (
        featured["HomeShotsOnTargetAgainst_Last5"] - featured["AwayShotsOnTargetAgainst_Last5"]
    )
    featured["FoulsForLast5Diff"] = featured["HomeFoulsFor_Last5"] - featured["AwayFoulsFor_Last5"]
    featured["OffsidesForLast5Diff"] = featured["HomeOffsidesFor_Last5"] - featured["AwayOffsidesFor_Last5"]
    featured["AbsPointsLast5Diff"] = (featured["HomePoints_Last5"] - featured["AwayPoints_Last5"]).abs()
    featured["AvgDrawRateLast5"] = (featured["HomeDrawRate_Last5"] + featured["AwayDrawRate_Last5"]) / 2.0
    featured["AbsDrawRateLast5Diff"] = (featured["HomeDrawRate_Last5"] - featured["AwayDrawRate_Last5"]).abs()
    featured["VenuePointsLast5Diff"] = featured["HomeVenuePoints_Last5"] - featured["AwayVenuePoints_Last5"]
    featured["RestDaysDiff"] = featured["HomeRestDays"] - featured["AwayRestDays"]
    featured["CongestionDiff"] = featured["HomeMatchesLast14Days"] - featured["AwayMatchesLast14Days"]
    return featured


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    validate_stage1_columns(df)
    featured = sort_matches(df)
    featured = add_result_columns(featured)
    featured = add_elo_features(featured)
    featured = add_rolling_features(featured)
    featured = add_venue_form_features(featured)
    featured = add_season_win_rates(featured)
    featured = add_draw_signal_features(featured)
    return featured


def process_league(
    league: str,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> FeatureRunSummary:
    input_path = input_path or PROCESSED_DIR / f"{league}.parquet"
    output_path = output_path or FEATURES_DIR / f"{league}_features.parquet"

    df = pd.read_parquet(input_path)
    featured = build_feature_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(output_path, index=False)

    result_counts = {key: int(value) for key, value in featured["Result"].value_counts().sort_index().items()}
    return FeatureRunSummary(
        league=league,
        rows=len(featured),
        output_path=output_path,
        result_counts=result_counts,
    )


def run_pipeline(leagues: Iterable[str] = LEAGUES) -> list[FeatureRunSummary]:
    summaries = [process_league(league) for league in leagues]
    for summary in summaries:
        print(summary.line())
    return summaries


if __name__ == "__main__":
    run_pipeline()
