from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

FEATURES_DIR = Path("data/features")
ARTIFACTS_DIR = Path("data/model_artifacts/stage3")
LEAGUES = ("ENG", "SPA", "FRA")

TRAIN_SEASONS = ("2017-18", "2018-19")
HOLDOUT_SEASON = "2019-20"
TARGET_COLUMN = "ResultCode"
LABELS = (0, 1, 2)
LABEL_NAMES = {0: "home", 1: "draw", 2: "away"}

FEATURE_COLUMNS = [
    "HomeElo",
    "AwayElo",
    "EloDiff",
    "HomeGoals_Last5",
    "AwayGoals_Last5",
    "HomeCorners_Last5",
    "AwayCorners_Last5",
    "HomePoints_Last5",
    "AwayPoints_Last5",
    "HomeWinRate_Season",
    "AwayWinRate_Season",
]

DEFAULT_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_estimators": 150,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "random_state": 42,
    "n_jobs": 1,
}


@dataclass(frozen=True)
class TrainingRunSummary:
    train_rows: int
    holdout_rows: int
    metrics: dict[str, float]
    model_path: Path
    metrics_path: Path
    mlflow_run_id: str | None
    registered_model_version: str | None

    def line(self) -> str:
        run = self.mlflow_run_id or "not logged"
        return (
            f"train_rows={self.train_rows}, holdout_rows={self.holdout_rows}, "
            f"log_loss={self.metrics['holdout_log_loss']:.4f}, "
            f"accuracy={self.metrics['holdout_accuracy']:.4f}, "
            f"model={self.model_path}, mlflow_run_id={run}, "
            f"registered_model_version={self.registered_model_version or 'not registered'}"
        )


def load_feature_data(
    leagues: Iterable[str] = LEAGUES,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    frames = []
    for league in leagues:
        path = features_dir / f"{league}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 2 feature file: {path}")

        frame = pd.read_parquet(path)
        frame = frame.copy()
        frame["League"] = league
        frames.append(frame)

    if not frames:
        raise ValueError("At least one league is required to train Stage 3")

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    return combined.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True)


def validate_training_columns(df: pd.DataFrame) -> None:
    required = ["RBallID", "Date", "Season", TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Stage 2 columns: {missing}")


def split_train_holdout(
    df: pd.DataFrame,
    train_seasons: Iterable[str] = TRAIN_SEASONS,
    holdout_season: str = HOLDOUT_SEASON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_training_columns(df)

    train = df[df["Season"].isin(tuple(train_seasons))].copy()
    holdout = df[df["Season"] == holdout_season].copy()
    if train.empty:
        raise ValueError(f"No training rows found for seasons: {tuple(train_seasons)}")
    if holdout.empty:
        raise ValueError(f"No holdout rows found for season: {holdout_season}")

    return (
        train.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True),
        holdout.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True),
    )


def select_features_and_target(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    validate_training_columns(df)

    columns = list(feature_columns)
    X = df[columns].apply(pd.to_numeric, errors="raise").astype("float64")
    y = df[target_column].astype("int64")

    unknown_labels = sorted(set(y.unique()) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected target labels: {unknown_labels}")

    return X, y


def normalize_probabilities(y_proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != len(LABELS):
        raise ValueError(f"Expected probability matrix with {len(LABELS)} columns")

    row_sums = proba.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Probability rows must have a positive sum")
    return proba / row_sums


def multiclass_brier_score(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> float:
    y_array = np.asarray(y_true, dtype=int)
    proba = normalize_probabilities(y_proba)
    one_hot = np.eye(len(LABELS))[y_array]
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def evaluate_predictions(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    proba = normalize_probabilities(y_proba)
    predicted = np.argmax(proba, axis=1)
    f1_values = f1_score(y_true, predicted, labels=list(LABELS), average=None, zero_division=0)
    metrics = {
        "holdout_log_loss": float(log_loss(y_true, proba, labels=list(LABELS))),
        "holdout_brier_score": multiclass_brier_score(y_true, proba),
        "holdout_accuracy": float(accuracy_score(y_true, predicted)),
    }
    for label, value in zip(LABELS, f1_values):
        metrics[f"holdout_f1_{LABEL_NAMES[label]}"] = float(value)
    return metrics


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    trials: int,
    random_state: int = 42,
) -> dict[str, object]:
    if trials <= 0:
        return dict(DEFAULT_XGB_PARAMS)

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required when --trials is greater than 0") from exc

    splits = min(3, max(2, len(X_train) // 250))
    tscv = TimeSeriesSplit(n_splits=splits)

    def objective(trial: object) -> float:
        params = {
            **DEFAULT_XGB_PARAMS,
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "random_state": random_state,
        }
        scores = []
        for train_index, val_index in tscv.split(X_train):
            model = XGBClassifier(**params)
            model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
            proba = model.predict_proba(X_train.iloc[val_index])
            scores.append(log_loss(y_train.iloc[val_index], proba, labels=list(LABELS)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {**DEFAULT_XGB_PARAMS, **study.best_params, "random_state": random_state}


def train_classifier(X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, object]) -> XGBClassifier:
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_feature_importance(model: XGBClassifier, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot.barh(ax=ax)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_confusion_matrix(y_true: pd.Series, y_proba: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted = np.argmax(y_proba, axis=1)
    matrix = confusion_matrix(y_true, predicted, labels=list(LABELS))

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), ["H", "D", "A"])
    ax.set_yticks(range(len(LABELS)), ["H", "D", "A"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Holdout Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def log_mlflow_run(
    model: XGBClassifier,
    params: dict[str, object],
    metrics: dict[str, float],
    artifact_paths: Iterable[Path],
    tracking_uri: str | None = None,
) -> tuple[str | None, str | None]:
    try:
        import mlflow
        import mlflow.xgboost
    except ImportError:
        return None, None

    mlflow.set_tracking_uri(tracking_uri or "file:mlruns")
    mlflow.set_experiment("match_outcome_prediction")
    with mlflow.start_run(run_name="stage3_xgboost_multiclass"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in artifact_paths:
            mlflow.log_artifact(str(path))
        model_info = mlflow.xgboost.log_model(model, artifact_path="model")
        return mlflow.active_run().info.run_id, model_info.model_uri


def register_production_model(
    model_uri: str | None,
    model_name: str = "match_outcome_xgb",
    tracking_uri: str | None = None,
) -> str | None:
    if model_uri is None:
        return None

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None

    mlflow.set_tracking_uri(tracking_uri or "file:mlruns")
    result = mlflow.register_model(model_uri, model_name)
    client = MlflowClient()
    client.transition_model_version_stage(
        name=result.name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True,
    )
    return str(result.version)


def run_pipeline(
    leagues: Iterable[str] = LEAGUES,
    features_dir: Path = FEATURES_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    trials: int = 0,
    tracking_uri: str | None = None,
) -> TrainingRunSummary:
    df = load_feature_data(leagues=leagues, features_dir=features_dir)
    train_df, holdout_df = split_train_holdout(df)
    X_train, y_train = select_features_and_target(train_df)
    X_holdout, y_holdout = select_features_and_target(holdout_df)

    params = tune_hyperparameters(X_train, y_train, trials=trials)
    model = train_classifier(X_train, y_train, params=params)
    holdout_proba = normalize_probabilities(model.predict_proba(X_holdout))
    metrics = evaluate_predictions(y_holdout, holdout_proba)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "xgb_match_outcome.json"
    metrics_path = artifacts_dir / "metrics.json"
    predictions_path = artifacts_dir / "holdout_predictions.parquet"
    feature_importance_path = artifacts_dir / "feature_importance.png"
    confusion_matrix_path = artifacts_dir / "confusion_matrix.png"

    model.save_model(model_path)
    write_json(metrics_path, metrics)
    write_feature_importance(model, feature_importance_path)
    write_confusion_matrix(y_holdout, holdout_proba, confusion_matrix_path)

    predictions = holdout_df[["RBallID", "League", "Date", "Season", "HomeTeam", "AwayTeam", "Result"]].copy()
    predictions["P_Home"] = holdout_proba[:, 0]
    predictions["P_Draw"] = holdout_proba[:, 1]
    predictions["P_Away"] = holdout_proba[:, 2]
    predictions.to_parquet(predictions_path, index=False)

    mlflow_run_id, model_uri = log_mlflow_run(
        model=model,
        params=params,
        metrics=metrics,
        artifact_paths=[metrics_path, predictions_path, feature_importance_path, confusion_matrix_path],
        tracking_uri=tracking_uri,
    )
    registered_model_version = register_production_model(
        model_uri=model_uri,
        tracking_uri=tracking_uri,
    )

    return TrainingRunSummary(
        train_rows=len(train_df),
        holdout_rows=len(holdout_df),
        metrics=metrics,
        model_path=model_path,
        metrics_path=metrics_path,
        mlflow_run_id=mlflow_run_id,
        registered_model_version=registered_model_version,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 3 XGBoost match outcome model.")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--trials", type=int, default=0, help="Optional Optuna trials. Default keeps the run fast.")
    parser.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_pipeline(
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        trials=args.trials,
        tracking_uri=args.tracking_uri,
    )
    print(summary.line())
