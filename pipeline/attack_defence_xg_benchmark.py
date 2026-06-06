from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from pipeline.model_features import LEAGUES
    from pipeline.poisson_goal_model import (
        DEFAULT_MAX_GOALS,
        MIN_LAMBDA,
        build_poisson_odds_frame,
        lambdas_to_outcome_probabilities,
    )
    from pipeline.stage3_train import (
        HOLDOUT_SEASON,
        compact_benchmark_metrics,
        evaluate_predictions,
        season_start_year,
    )
except ModuleNotFoundError:
    from model_features import LEAGUES
    from poisson_goal_model import (
        DEFAULT_MAX_GOALS,
        MIN_LAMBDA,
        build_poisson_odds_frame,
        lambdas_to_outcome_probabilities,
    )
    from stage3_train import HOLDOUT_SEASON, compact_benchmark_metrics, evaluate_predictions, season_start_year

FEATURES_DIR = Path("data/features")
OUTPUT_PATH = Path("data/output/attack_defence_xg_model_odds.parquet")
VALUE_BETS_PATH = Path("data/output/attack_defence_xg_value_bets.parquet")
ARTIFACTS_DIR = Path("data/model_artifacts/attack_defence_xg")
STAGE3_BENCHMARKS_PATH = Path("data/model_artifacts/stage3/model_benchmarks.json")

REQUIRED_COLUMNS = [
    "RBallID",
    "League",
    "Date",
    "Season",
    "HomeTeam",
    "AwayTeam",
    "HomeGoals",
    "AwayGoals",
    "Result",
    "ResultCode",
]
STRENGTH_COLUMNS = [
    "League",
    "Team",
    "matches",
    "goals_for",
    "goals_against",
    "attack_strength",
    "defence_strength",
    "attack_strength_shrunk",
    "defence_strength_shrunk",
]
DEFAULT_SHRINKAGE_MATCHES = 12.0


@dataclass(frozen=True)
class AttackDefenceXgSummary:
    train_rows: int
    holdout_rows: int
    metrics: dict[str, float]
    output_path: Path
    artifacts_dir: Path

    def line(self) -> str:
        return (
            f"train_rows={self.train_rows}, holdout_rows={self.holdout_rows}, "
            f"log_loss={self.metrics['holdout_log_loss']:.4f}, "
            f"brier={self.metrics['holdout_brier_score']:.4f}, "
            f"accuracy={self.metrics['holdout_accuracy']:.4f}, "
            f"output={self.output_path}, artifacts={self.artifacts_dir}"
        )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_feature_data(
    leagues: Iterable[str] = LEAGUES,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    frames = []
    for league in leagues:
        path = features_dir / f"{league}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 2 feature file: {path}")
        frame = pd.read_parquet(path).copy()
        frame["League"] = league
        frames.append(frame)
    if not frames:
        raise ValueError("At least one league is required for the attack/defence xG benchmark")
    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    return combined.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True)


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required attack/defence xG columns: {missing}")
    goals = df[["HomeGoals", "AwayGoals"]].apply(pd.to_numeric, errors="raise")
    if goals.isna().any().any() or (goals < 0).any().any():
        raise ValueError("Goal columns must be non-null, non-negative counts")


def split_train_holdout(
    df: pd.DataFrame,
    holdout_season: str = HOLDOUT_SEASON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_input_columns(df)
    holdout_start = season_start_year(holdout_season)
    season_starts = df["Season"].map(season_start_year)
    train = df[season_starts < holdout_start].copy()
    holdout = df[df["Season"] == holdout_season].copy()
    if train.empty:
        raise ValueError(f"No training rows found before {holdout_season}")
    if holdout.empty:
        raise ValueError(f"No holdout rows found for season: {holdout_season}")
    return (
        train.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True),
        holdout.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True),
    )


def build_team_match_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    validate_input_columns(train_df)
    home = train_df[["League", "HomeTeam", "HomeGoals", "AwayGoals"]].rename(
        columns={"HomeTeam": "Team", "HomeGoals": "goals_for", "AwayGoals": "goals_against"}
    )
    away = train_df[["League", "AwayTeam", "AwayGoals", "HomeGoals"]].rename(
        columns={"AwayTeam": "Team", "AwayGoals": "goals_for", "HomeGoals": "goals_against"}
    )
    return pd.concat([home, away], ignore_index=True)


def fit_league_baselines(train_df: pd.DataFrame) -> pd.DataFrame:
    validate_input_columns(train_df)
    baselines = (
        train_df.groupby("League", sort=True)
        .agg(
            matches=("RBallID", "size"),
            home_goal_baseline=("HomeGoals", "mean"),
            away_goal_baseline=("AwayGoals", "mean"),
        )
        .reset_index()
    )
    baselines["team_goal_baseline"] = (
        baselines["home_goal_baseline"] + baselines["away_goal_baseline"]
    ) / 2.0
    if (baselines[["home_goal_baseline", "away_goal_baseline", "team_goal_baseline"]] <= 0.0).any().any():
        raise ValueError("League goal baselines must be positive")
    return baselines


def fit_team_strengths(
    train_df: pd.DataFrame,
    shrinkage_matches: float = DEFAULT_SHRINKAGE_MATCHES,
) -> pd.DataFrame:
    if shrinkage_matches < 0:
        raise ValueError("shrinkage_matches must be non-negative")
    team_rows = build_team_match_rows(train_df)
    baselines = fit_league_baselines(train_df)[["League", "team_goal_baseline"]]
    strengths = (
        team_rows.groupby(["League", "Team"], sort=True)
        .agg(matches=("Team", "size"), goals_for=("goals_for", "sum"), goals_against=("goals_against", "sum"))
        .reset_index()
        .merge(baselines, on="League", how="left", validate="many_to_one")
    )
    strengths["attack_strength"] = (strengths["goals_for"] / strengths["matches"]) / strengths["team_goal_baseline"]
    strengths["defence_strength"] = (strengths["goals_against"] / strengths["matches"]) / strengths["team_goal_baseline"]
    weight = strengths["matches"] / (strengths["matches"] + float(shrinkage_matches))
    strengths["attack_strength_shrunk"] = 1.0 + (strengths["attack_strength"] - 1.0) * weight
    strengths["defence_strength_shrunk"] = 1.0 + (strengths["defence_strength"] - 1.0) * weight
    return strengths[STRENGTH_COLUMNS].sort_values(["League", "Team"], kind="mergesort").reset_index(drop=True)


def _lookup_strength(strengths: pd.DataFrame, league: object, team: object, column: str) -> float:
    match = strengths[(strengths["League"] == league) & (strengths["Team"] == team)]
    if match.empty:
        return 1.0
    value = float(match.iloc[0][column])
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    return value


def predict_lambdas_from_strengths(
    holdout_df: pd.DataFrame,
    league_baselines: pd.DataFrame,
    team_strengths: pd.DataFrame,
    min_lambda: float = MIN_LAMBDA,
) -> pd.DataFrame:
    validate_input_columns(holdout_df)
    if min_lambda <= 0.0:
        raise ValueError("min_lambda must be positive")

    baseline_map = league_baselines.set_index("League").to_dict(orient="index")
    pooled_home = float(league_baselines["home_goal_baseline"].mean())
    pooled_away = float(league_baselines["away_goal_baseline"].mean())

    rows: list[dict[str, float]] = []
    for row in holdout_df.itertuples(index=False):
        league_baseline = baseline_map.get(row.League, {})
        home_baseline = float(league_baseline.get("home_goal_baseline", pooled_home))
        away_baseline = float(league_baseline.get("away_goal_baseline", pooled_away))
        home_attack = _lookup_strength(team_strengths, row.League, row.HomeTeam, "attack_strength_shrunk")
        away_attack = _lookup_strength(team_strengths, row.League, row.AwayTeam, "attack_strength_shrunk")
        home_defence = _lookup_strength(team_strengths, row.League, row.HomeTeam, "defence_strength_shrunk")
        away_defence = _lookup_strength(team_strengths, row.League, row.AwayTeam, "defence_strength_shrunk")
        rows.append(
            {
                "Lambda_Home": max(min_lambda, home_baseline * home_attack * away_defence),
                "Lambda_Away": max(min_lambda, away_baseline * away_attack * home_defence),
                "HomeAttackStrength": home_attack,
                "AwayAttackStrength": away_attack,
                "HomeDefenceStrength": home_defence,
                "AwayDefenceStrength": away_defence,
            }
        )
    lambdas = pd.DataFrame(rows)
    if not np.isfinite(lambdas.to_numpy(dtype=float)).all():
        raise ValueError("Predicted lambdas and strengths must be finite")
    if (lambdas[["Lambda_Home", "Lambda_Away"]] <= 0.0).any().any():
        raise ValueError("Predicted lambdas must be strictly positive")
    return lambdas


def build_attack_defence_odds_frame(
    holdout_df: pd.DataFrame,
    lambda_frame: pd.DataFrame,
    proba: np.ndarray,
) -> pd.DataFrame:
    return build_poisson_odds_frame(
        holdout_df,
        lambda_frame[["Lambda_Home", "Lambda_Away"]].to_numpy(dtype=float),
        proba,
    )


def build_holdout_predictions_frame(
    holdout_df: pd.DataFrame,
    lambda_frame: pd.DataFrame,
    proba: np.ndarray,
) -> pd.DataFrame:
    predictions = build_attack_defence_odds_frame(holdout_df, lambda_frame, proba)
    predictions.insert(1, "League", holdout_df["League"].to_numpy())
    for offset, column in enumerate(
        [
            "Lambda_Home",
            "Lambda_Away",
            "HomeAttackStrength",
            "AwayAttackStrength",
            "HomeDefenceStrength",
            "AwayDefenceStrength",
        ],
        start=7,
    ):
        predictions.insert(offset, column, lambda_frame[column].to_numpy())
    return predictions


def load_existing_benchmark_rows(path: Path = STAGE3_BENCHMARKS_PATH) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("benchmarks", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid benchmark artifact shape: {path}")
    return rows


def build_benchmark_rows(
    y_holdout: pd.Series | np.ndarray,
    proba: np.ndarray,
    existing_benchmarks_path: Path = STAGE3_BENCHMARKS_PATH,
) -> list[dict[str, object]]:
    rows = [row for row in load_existing_benchmark_rows(existing_benchmarks_path) if row.get("model") != "attack_defence_xg"]
    rows.append(compact_benchmark_metrics("attack_defence_xg", y_holdout, proba))
    return rows


def build_value_bet_roi(value_bets: pd.DataFrame, holdout_season: str = HOLDOUT_SEASON) -> dict[str, object]:
    required = ["Season", "Outcome", "Result", "BestBookOdds", "Edge"]
    missing = [column for column in required if column not in value_bets.columns]
    if missing:
        raise ValueError(f"Missing value-bet ROI columns: {missing}")
    holdout = value_bets[value_bets["Season"] == holdout_season].copy()
    if holdout.empty:
        return {
            "holdout_season": holdout_season,
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "hit_rate": 0.0,
            "flat_stake_profit": 0.0,
            "flat_stake_roi": 0.0,
            "average_bookmaker_odds": None,
            "average_edge": None,
        }
    holdout["Won"] = holdout["Outcome"] == holdout["Result"]
    holdout["FlatStakeProfit"] = np.where(
        holdout["Won"], pd.to_numeric(holdout["BestBookOdds"], errors="raise") - 1.0, -1.0
    )
    total_bets = int(len(holdout))
    wins = int(holdout["Won"].sum())
    profit = float(holdout["FlatStakeProfit"].sum())
    return {
        "holdout_season": holdout_season,
        "total_bets": total_bets,
        "wins": wins,
        "losses": int(total_bets - wins),
        "hit_rate": float(wins / total_bets),
        "flat_stake_profit": profit,
        "flat_stake_roi": float(profit / total_bets),
        "average_bookmaker_odds": float(pd.to_numeric(holdout["BestBookOdds"], errors="raise").mean()),
        "average_edge": float(pd.to_numeric(holdout["Edge"], errors="raise").mean()),
    }


def maybe_write_value_bet_roi(
    value_bets_path: Path = VALUE_BETS_PATH,
    output_path: Path = ARTIFACTS_DIR / "value_bet_roi.json",
    holdout_season: str = HOLDOUT_SEASON,
) -> dict[str, object] | None:
    if not value_bets_path.exists():
        return None
    roi = build_value_bet_roi(pd.read_parquet(value_bets_path), holdout_season=holdout_season)
    write_json(output_path, roi)
    return roi


def run_pipeline(
    leagues: Iterable[str] = LEAGUES,
    features_dir: Path = FEATURES_DIR,
    output_path: Path = OUTPUT_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    existing_benchmarks_path: Path = STAGE3_BENCHMARKS_PATH,
    value_bets_path: Path = VALUE_BETS_PATH,
    holdout_season: str = HOLDOUT_SEASON,
    max_goals: int = DEFAULT_MAX_GOALS,
    shrinkage_matches: float = DEFAULT_SHRINKAGE_MATCHES,
) -> AttackDefenceXgSummary:
    df = load_feature_data(leagues=leagues, features_dir=features_dir)
    validate_input_columns(df)
    train_df, holdout_df = split_train_holdout(df, holdout_season=holdout_season)

    league_baselines = fit_league_baselines(train_df)
    team_strengths = fit_team_strengths(train_df, shrinkage_matches=shrinkage_matches)
    lambda_frame = predict_lambdas_from_strengths(holdout_df, league_baselines, team_strengths)
    proba = lambdas_to_outcome_probabilities(
        lambda_frame[["Lambda_Home", "Lambda_Away"]].to_numpy(dtype=float),
        max_goals=max_goals,
    )
    y_holdout = holdout_df["ResultCode"].astype("int64")
    metrics = evaluate_predictions(y_holdout, proba)
    benchmarks = build_benchmark_rows(y_holdout, proba, existing_benchmarks_path=existing_benchmarks_path)

    odds_df = build_attack_defence_odds_frame(holdout_df, lambda_frame, proba)
    predictions_df = build_holdout_predictions_frame(holdout_df, lambda_frame, proba)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    odds_df.to_parquet(output_path, index=False)
    predictions_df.to_parquet(artifacts_dir / "holdout_predictions.parquet", index=False)
    team_strengths.to_parquet(artifacts_dir / "team_strengths.parquet", index=False)
    write_json(
        artifacts_dir / "metrics.json",
        {
            **metrics,
            "model": "attack_defence_xg",
            "train_rows": len(train_df),
            "holdout_rows": len(holdout_df),
            "holdout_season": holdout_season,
            "max_goals": max_goals,
            "shrinkage_matches": shrinkage_matches,
            "lambda_formula": "league_baseline * team_attack_strength * opponent_defence_strength",
            "home_advantage": "encoded by separate league home_goal_baseline and away_goal_baseline",
            "leakage_guard": f"all strengths and baselines fit only on seasons before {holdout_season}",
            "league_baselines": league_baselines.to_dict(orient="records"),
        },
    )
    write_json(artifacts_dir / "model_benchmarks.json", {"benchmarks": benchmarks})
    maybe_write_value_bet_roi(
        value_bets_path=value_bets_path,
        output_path=artifacts_dir / "value_bet_roi.json",
        holdout_season=holdout_season,
    )

    return AttackDefenceXgSummary(
        train_rows=len(train_df),
        holdout_rows=len(holdout_df),
        metrics=metrics,
        output_path=output_path,
        artifacts_dir=artifacts_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an explainable attack/defence xG benchmark.")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--existing-benchmarks-path", type=Path, default=STAGE3_BENCHMARKS_PATH)
    parser.add_argument("--value-bets-path", type=Path, default=VALUE_BETS_PATH)
    parser.add_argument("--holdout-season", default=HOLDOUT_SEASON)
    parser.add_argument("--max-goals", type=int, default=DEFAULT_MAX_GOALS)
    parser.add_argument("--shrinkage-matches", type=float, default=DEFAULT_SHRINKAGE_MATCHES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_pipeline(
        features_dir=args.features_dir,
        output_path=args.output_path,
        artifacts_dir=args.artifacts_dir,
        existing_benchmarks_path=args.existing_benchmarks_path,
        value_bets_path=args.value_bets_path,
        holdout_season=args.holdout_season,
        max_goals=args.max_goals,
        shrinkage_matches=args.shrinkage_matches,
    )
    print(summary.line())
