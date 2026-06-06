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
            "Goals": df["HomeGoals"],
            "Corners": df["HomeCorners"],
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
            "Goals": df["AwayGoals"],
            "Corners": df["AwayCorners"],
            "Points": away_points,
            "Draw": draw_flag,
        }
    )

    history = pd.concat([home_history, away_history], ignore_index=True)
    return history.sort_values(["Team", "Date", "RBallID"], kind="mergesort").reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    featured = sort_matches(df)
    if "Result" not in featured.columns:
        featured = add_result_columns(featured)

    history = _team_match_history(featured)
    grouped = history.groupby("Team", sort=False)
    for source, target in [
        ("Goals", "Goals_Last5"),
        ("Corners", "Corners_Last5"),
        ("Points", "Points_Last5"),
        ("Draw", "DrawRate_Last5"),
    ]:
        history[target] = grouped[source].transform(
            lambda values: values.shift(1).rolling(window=window, min_periods=1).mean()
        )
    history[["Goals_Last5", "Corners_Last5", "Points_Last5", "DrawRate_Last5"]] = history[
        ["Goals_Last5", "Corners_Last5", "Points_Last5", "DrawRate_Last5"]
    ].fillna(0.0)

    feature_history = history[["RBallID", "Team", "Goals_Last5", "Corners_Last5", "Points_Last5", "DrawRate_Last5"]]

    home_features = feature_history.rename(
        columns={
            "Team": "HomeTeam",
            "Goals_Last5": "HomeGoals_Last5",
            "Corners_Last5": "HomeCorners_Last5",
            "Points_Last5": "HomePoints_Last5",
            "DrawRate_Last5": "HomeDrawRate_Last5",
        }
    )
    away_features = feature_history.rename(
        columns={
            "Team": "AwayTeam",
            "Goals_Last5": "AwayGoals_Last5",
            "Corners_Last5": "AwayCorners_Last5",
            "Points_Last5": "AwayPoints_Last5",
            "DrawRate_Last5": "AwayDrawRate_Last5",
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
    featured["AbsPointsLast5Diff"] = (featured["HomePoints_Last5"] - featured["AwayPoints_Last5"]).abs()
    featured["AvgDrawRateLast5"] = (featured["HomeDrawRate_Last5"] + featured["AwayDrawRate_Last5"]) / 2.0
    featured["AbsDrawRateLast5Diff"] = (featured["HomeDrawRate_Last5"] - featured["AwayDrawRate_Last5"]).abs()
    return featured


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    validate_stage1_columns(df)
    featured = sort_matches(df)
    featured = add_result_columns(featured)
    featured = add_elo_features(featured)
    featured = add_rolling_features(featured)
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
