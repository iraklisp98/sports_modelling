from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from pipeline.model_features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, LEAGUES, add_league_indicator_features
    from pipeline.stage3_train import (
        HOLDOUT_SEASON,
        LABELS,
        build_model_benchmarks,
        compact_benchmark_metrics,
        evaluate_predictions,
        season_start_year,
        split_train_holdout,
    )
    from pipeline.stage4_odds_gen import OUTPUT_COLUMNS
except ModuleNotFoundError:
    from model_features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, LEAGUES, add_league_indicator_features
    from stage3_train import (
        HOLDOUT_SEASON,
        LABELS,
        build_model_benchmarks,
        compact_benchmark_metrics,
        evaluate_predictions,
        season_start_year,
        split_train_holdout,
    )
    from stage4_odds_gen import OUTPUT_COLUMNS

FEATURES_DIR = Path("data/features")
OUTPUT_PATH = Path("data/output/poisson_model_odds.parquet")
ARTIFACTS_DIR = Path("data/model_artifacts/poisson_goal_model")
STAGE3_BENCHMARKS_PATH = Path("data/model_artifacts/stage3/model_benchmarks.json")

GOAL_TARGET_COLUMNS = ("HomeGoals", "AwayGoals")
LAMBDA_COLUMNS = ("Lambda_Home", "Lambda_Away")
DEFAULT_MAX_GOALS = 10
MIN_LAMBDA = 0.01
MIN_PROBABILITY = 1e-12


@dataclass(frozen=True)
class PoissonGoalModelSummary:
    train_rows: int
    holdout_rows: int
    metrics: dict[str, float]
    output_path: Path
    artifacts_dir: Path

    def line(self) -> str:
        return (
            f"train_rows={self.train_rows}, holdout_rows={self.holdout_rows}, "
            f"log_loss={self.metrics['holdout_log_loss']:.4f}, "
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
        raise ValueError("At least one league is required for the Poisson benchmark")

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = add_league_indicator_features(combined, leagues=leagues)
    return combined.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True)


def validate_goal_model_columns(df: pd.DataFrame) -> None:
    required = [
        "RBallID",
        "HomeTeam",
        "AwayTeam",
        "Date",
        "Season",
        "League",
        "Result",
        "ResultCode",
        *GOAL_TARGET_COLUMNS,
        *BASE_FEATURE_COLUMNS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Poisson goal-model columns: {missing}")


def select_goal_features_and_targets(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    validate_goal_model_columns(df)
    featured = add_league_indicator_features(df)
    columns = list(feature_columns)
    missing = [column for column in columns if column not in featured.columns]
    if missing:
        raise ValueError(f"Missing required model feature columns: {missing}")

    X = featured[columns].apply(pd.to_numeric, errors="raise").astype("float64")
    y_home = df["HomeGoals"].astype("float64")
    y_away = df["AwayGoals"].astype("float64")
    if (y_home < 0).any() or (y_away < 0).any():
        raise ValueError("Goal targets must be non-negative counts")
    return X, y_home, y_away


def train_goal_models(
    train_df: pd.DataFrame,
    alpha: float = 1.0,
    max_iter: int = 1000,
) -> tuple[object, object]:
    X_train, y_home, y_away = select_goal_features_and_targets(train_df)
    home_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=max_iter))
    away_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=max_iter))
    home_model.fit(X_train, y_home)
    away_model.fit(X_train, y_away)
    return home_model, away_model


def predict_lambdas(
    holdout_df: pd.DataFrame,
    home_model: object,
    away_model: object,
) -> np.ndarray:
    X_holdout, _, _ = select_goal_features_and_targets(holdout_df)
    lambdas = np.column_stack([home_model.predict(X_holdout), away_model.predict(X_holdout)])
    if not np.isfinite(lambdas).all():
        raise ValueError("Predicted goal lambdas must be finite")
    return np.clip(lambdas, MIN_LAMBDA, None)


def poisson_pmf_values(lam: float, max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("Poisson lambda must be finite and strictly positive")

    pmf = np.empty(max_goals + 1, dtype=float)
    pmf[0] = np.exp(-float(lam))
    for goals in range(1, max_goals + 1):
        pmf[goals] = pmf[goals - 1] * float(lam) / goals
    return pmf


def scoreline_outcome_probabilities(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    home_pmf = poisson_pmf_values(lambda_home, max_goals=max_goals)
    away_pmf = poisson_pmf_values(lambda_away, max_goals=max_goals)
    score_grid = np.outer(home_pmf, away_pmf)

    home_goals = np.arange(max_goals + 1)[:, None]
    away_goals = np.arange(max_goals + 1)[None, :]
    probabilities = np.array(
        [
            score_grid[home_goals > away_goals].sum(),
            score_grid[home_goals == away_goals].sum(),
            score_grid[home_goals < away_goals].sum(),
        ],
        dtype=float,
    )
    return normalize_outcome_probabilities(probabilities.reshape(1, -1))[0]


def lambdas_to_outcome_probabilities(
    lambdas: np.ndarray,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    lambda_matrix = np.asarray(lambdas, dtype=float)
    if lambda_matrix.ndim != 2 or lambda_matrix.shape[1] != 2:
        raise ValueError("Expected a 2D lambda matrix with home and away columns")
    if not np.isfinite(lambda_matrix).all() or (lambda_matrix <= 0.0).any():
        raise ValueError("Goal lambdas must be finite and strictly positive")

    rows = [
        scoreline_outcome_probabilities(lambda_home, lambda_away, max_goals=max_goals)
        for lambda_home, lambda_away in lambda_matrix
    ]
    return validate_outcome_probabilities(np.vstack(rows))


def normalize_outcome_probabilities(proba: np.ndarray) -> np.ndarray:
    matrix = np.asarray(proba, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != len(LABELS):
        raise ValueError(f"Expected probability matrix with {len(LABELS)} outcome columns")
    if not np.isfinite(matrix).all():
        raise ValueError("Probability matrix must contain only finite values")
    if (matrix < 0.0).any():
        raise ValueError("Probability values must be non-negative")

    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Probability rows must have a positive sum")
    return matrix / row_sums


def validate_outcome_probabilities(proba: np.ndarray, tolerance: float = 0.001) -> np.ndarray:
    matrix = normalize_outcome_probabilities(proba)
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        max_delta = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f"Probability rows must sum to 1.0 ± {tolerance}; max delta={max_delta:.6f}")
    return matrix


def probabilities_to_odds(proba: np.ndarray) -> np.ndarray:
    matrix = validate_outcome_probabilities(proba)
    clipped = np.clip(matrix, MIN_PROBABILITY, 1.0)
    odds = 1.0 / clipped
    if not np.isfinite(odds).all():
        raise ValueError("Decimal odds must be finite")
    return odds


def build_poisson_odds_frame(
    holdout_df: pd.DataFrame,
    lambdas: np.ndarray,
    proba: np.ndarray,
) -> pd.DataFrame:
    validate_goal_model_columns(holdout_df)
    probability_matrix = validate_outcome_probabilities(proba)
    lambda_matrix = np.asarray(lambdas, dtype=float)
    if len(holdout_df) != len(probability_matrix) or len(holdout_df) != len(lambda_matrix):
        raise ValueError("Holdout rows, lambdas, and probabilities must have matching lengths")

    odds = probabilities_to_odds(probability_matrix)
    output = holdout_df[["RBallID", "HomeTeam", "AwayTeam", "Date", "Season", "Result"]].copy()
    output["P_Home"] = probability_matrix[:, 0]
    output["P_Draw"] = probability_matrix[:, 1]
    output["P_Away"] = probability_matrix[:, 2]
    output["ModelOdds_Home"] = odds[:, 0]
    output["ModelOdds_Draw"] = odds[:, 1]
    output["ModelOdds_Away"] = odds[:, 2]
    return output[OUTPUT_COLUMNS]


def build_holdout_predictions_frame(
    holdout_df: pd.DataFrame,
    lambdas: np.ndarray,
    proba: np.ndarray,
) -> pd.DataFrame:
    predictions = build_poisson_odds_frame(holdout_df, lambdas, proba)
    predictions.insert(6, "Lambda_Home", np.asarray(lambdas, dtype=float)[:, 0])
    predictions.insert(7, "Lambda_Away", np.asarray(lambdas, dtype=float)[:, 1])
    if "League" in holdout_df.columns:
        predictions.insert(1, "League", holdout_df["League"].to_numpy())
    return predictions


def load_existing_benchmark_rows(path: Path = STAGE3_BENCHMARKS_PATH) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("benchmarks", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid benchmark artifact shape: {path}")
    return rows


def build_poisson_benchmarks(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    poisson_proba: np.ndarray,
    existing_benchmarks_path: Path = STAGE3_BENCHMARKS_PATH,
) -> list[dict[str, object]]:
    y_train = train_df["ResultCode"].astype("int64")
    y_holdout = holdout_df["ResultCode"].astype("int64")
    existing_rows = load_existing_benchmark_rows(existing_benchmarks_path)

    if existing_rows:
        benchmarks = existing_rows
    else:
        benchmarks = build_model_benchmarks(y_train, holdout_df, y_holdout, poisson_proba)
        benchmarks = [row for row in benchmarks if row["model"] != "calibrated_xgboost"]

    benchmarks = [row for row in benchmarks if row.get("model") != "poisson_goal_model"]
    benchmarks.append(compact_benchmark_metrics("poisson_goal_model", y_holdout, poisson_proba))
    return benchmarks


def run_pipeline(
    leagues: Iterable[str] = LEAGUES,
    features_dir: Path = FEATURES_DIR,
    output_path: Path = OUTPUT_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    existing_benchmarks_path: Path = STAGE3_BENCHMARKS_PATH,
    holdout_season: str = HOLDOUT_SEASON,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> PoissonGoalModelSummary:
    df = load_feature_data(leagues=leagues, features_dir=features_dir)
    validate_goal_model_columns(df)
    train_df, holdout_df = split_train_holdout(df, holdout_season=holdout_season)
    holdout_start = season_start_year(holdout_season)
    forward_df = df[df["Season"].map(season_start_year) >= holdout_start].copy().reset_index(drop=True)
    if forward_df.empty:
        raise ValueError(f"No forward rows found from {holdout_season} onward")

    home_model, away_model = train_goal_models(train_df)
    holdout_lambdas = predict_lambdas(holdout_df, home_model, away_model)
    holdout_proba = lambdas_to_outcome_probabilities(holdout_lambdas, max_goals=max_goals)
    y_holdout = holdout_df["ResultCode"].astype("int64")
    metrics = evaluate_predictions(y_holdout, holdout_proba)

    forward_lambdas = predict_lambdas(forward_df, home_model, away_model)
    forward_proba = lambdas_to_outcome_probabilities(forward_lambdas, max_goals=max_goals)
    odds_df = build_poisson_odds_frame(forward_df, forward_lambdas, forward_proba)
    predictions_df = build_holdout_predictions_frame(holdout_df, holdout_lambdas, holdout_proba)
    benchmarks = build_poisson_benchmarks(
        train_df,
        holdout_df,
        holdout_proba,
        existing_benchmarks_path=existing_benchmarks_path,
    )
    metrics["forward_rows"] = int(len(forward_df))
    metrics["forward_seasons"] = sorted(forward_df["Season"].dropna().unique().tolist(), key=season_start_year)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    odds_df.to_parquet(output_path, index=False)
    predictions_df.to_parquet(artifacts_dir / "holdout_predictions.parquet", index=False)
    write_json(artifacts_dir / "metrics.json", metrics)
    write_json(artifacts_dir / "model_benchmarks.json", {"benchmarks": benchmarks})

    return PoissonGoalModelSummary(
        train_rows=len(train_df),
        holdout_rows=len(holdout_df),
        metrics=metrics,
        output_path=output_path,
        artifacts_dir=artifacts_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Poisson goal-model benchmark and write Stage-4-compatible odds.")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--existing-benchmarks-path", type=Path, default=STAGE3_BENCHMARKS_PATH)
    parser.add_argument("--holdout-season", default=HOLDOUT_SEASON)
    parser.add_argument("--max-goals", type=int, default=DEFAULT_MAX_GOALS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_pipeline(
        features_dir=args.features_dir,
        output_path=args.output_path,
        artifacts_dir=args.artifacts_dir,
        existing_benchmarks_path=args.existing_benchmarks_path,
        holdout_season=args.holdout_season,
        max_goals=args.max_goals,
    )
    print(summary.line())
