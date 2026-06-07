from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
try:
    from pipeline.model_features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, LEAGUE_FEATURE_COLUMNS, LEAGUES, MARKET_AWARE_FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS, add_league_indicator_features
    from pipeline.market_features import FOOTBALL_DATA_DIR, add_market_features
except ModuleNotFoundError:
    from model_features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, LEAGUE_FEATURE_COLUMNS, LEAGUES, MARKET_AWARE_FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS, add_league_indicator_features
    from market_features import FOOTBALL_DATA_DIR, add_market_features
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

FEATURES_DIR = Path("data/features")
ARTIFACTS_DIR = Path("data/model_artifacts/stage3")

HOLDOUT_SEASON = "2019-20"
TARGET_COLUMN = "ResultCode"
LABELS = (0, 1, 2)
LABEL_NAMES = {0: "home", 1: "draw", 2: "away"}
DRAW_CLASS_WEIGHT_MULTIPLIER = 1.25
DRAW_OVERLAY_WEIGHT = 0.20
DRAW_OVERLAY_WEIGHT_CANDIDATES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
CALIBRATION_METHODS = ("sigmoid", "isotonic")
DEFAULT_CALIBRATION_METHOD = "isotonic"

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

DEFAULT_DRAW_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
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
    calibration_rows: int
    holdout_rows: int
    metrics: dict[str, float]
    model_path: Path
    metrics_path: Path
    mlflow_run_id: str | None
    registered_model_version: str | None

    def line(self) -> str:
        run = self.mlflow_run_id or "not logged"
        return (
            f"train_rows={self.train_rows}, calibration_rows={self.calibration_rows}, holdout_rows={self.holdout_rows}, "
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
    combined = add_league_indicator_features(combined, leagues=leagues)
    return combined.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True)


def validate_training_columns(df: pd.DataFrame) -> None:
    required = ["RBallID", "Date", "Season", "League", TARGET_COLUMN, *BASE_FEATURE_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Stage 2 columns: {missing}")


def season_start_year(season: str) -> int:
    try:
        return int(str(season).split("-", maxsplit=1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid season label: {season!r}") from exc


def split_train_holdout(
    df: pd.DataFrame,
    train_seasons: Iterable[str] | None = None,
    holdout_season: str = HOLDOUT_SEASON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_training_columns(df)

    if train_seasons is None:
        holdout_start = season_start_year(holdout_season)
        season_starts = df["Season"].map(season_start_year)
        train = df[season_starts < holdout_start].copy()
        train_scope = f"before {holdout_season}"
    else:
        selected_train_seasons = tuple(train_seasons)
        train = df[df["Season"].isin(selected_train_seasons)].copy()
        train_scope = str(selected_train_seasons)

    holdout = df[df["Season"] == holdout_season].copy()
    if train.empty:
        raise ValueError(f"No training rows found for seasons: {train_scope}")
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
    encoded = add_league_indicator_features(df)
    X = encoded[columns].apply(pd.to_numeric, errors="raise").astype("float64")
    y = df[target_column].astype("int64")

    unknown_labels = sorted(set(y.unique()) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected target labels: {unknown_labels}")

    return X, y


def normalize_probabilities(y_proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != len(LABELS):
        raise ValueError(f"Expected probability matrix with {len(LABELS)} columns")
    if not np.isfinite(proba).all():
        raise ValueError("Probability matrix must contain only finite values")
    if (proba < 0).any():
        raise ValueError("Probability values must be non-negative")

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
    y_array = np.asarray(y_true, dtype=int)
    predicted = np.argmax(proba, axis=1)
    f1_values = f1_score(y_array, predicted, labels=list(LABELS), average=None, zero_division=0)
    metrics = {
        "holdout_log_loss": float(log_loss(y_array, proba, labels=list(LABELS))),
        "holdout_brier_score": multiclass_brier_score(y_array, proba),
        "holdout_accuracy": float(accuracy_score(y_array, predicted)),
    }
    for label, value in zip(LABELS, f1_values):
        name = LABEL_NAMES[label]
        metrics[f"holdout_f1_{name}"] = float(value)
        metrics[f"holdout_actual_{name}"] = int(np.sum(y_array == label))
        metrics[f"holdout_predicted_{name}"] = int(np.sum(predicted == label))
    return metrics


def draw_binary_labels(y: pd.Series | np.ndarray) -> pd.Series:
    y_series = pd.Series(np.asarray(y, dtype=int))
    unknown_labels = sorted(set(y_series.unique()) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected target labels for draw binary labels: {unknown_labels}")
    return (y_series == 1).astype("int64")


def compute_binary_sample_weights(y_binary: pd.Series | np.ndarray) -> np.ndarray:
    y_array = np.asarray(y_binary, dtype=int)
    unknown_labels = sorted(set(np.unique(y_array)) - {0, 1})
    if unknown_labels:
        raise ValueError(f"Unexpected binary labels for sample weights: {unknown_labels}")
    counts = {label: int(np.sum(y_array == label)) for label in (0, 1)}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Both binary classes are required for sample weighting: {counts}")
    total = len(y_array)
    class_weights = {label: total / (2 * count) for label, count in counts.items()}
    weights = np.asarray([class_weights[label] for label in y_array], dtype=float)
    return weights / weights.mean()


def blend_draw_probability(
    multiclass_proba: np.ndarray,
    draw_proba: pd.Series | np.ndarray,
    blend_weight: float = DRAW_OVERLAY_WEIGHT,
) -> np.ndarray:
    if blend_weight < 0.0 or blend_weight > 1.0:
        raise ValueError("blend_weight must be between 0 and 1")
    base = normalize_probabilities(multiclass_proba)
    draw = np.asarray(draw_proba, dtype=float).reshape(-1)
    if len(draw) != len(base):
        raise ValueError("draw_proba must have one value per probability row")
    if not np.isfinite(draw).all() or (draw < 0).any() or (draw > 1).any():
        raise ValueError("draw_proba values must be finite probabilities between 0 and 1")

    blended_draw = ((1.0 - blend_weight) * base[:, 1]) + (blend_weight * draw)
    remaining = 1.0 - blended_draw
    old_non_draw = base[:, 0] + base[:, 2]
    home_share = np.divide(base[:, 0], old_non_draw, out=np.full(len(base), 0.5), where=old_non_draw > 0.0)
    away_share = 1.0 - home_share
    blended = np.column_stack([remaining * home_share, blended_draw, remaining * away_share])
    return normalize_probabilities(blended)


def select_draw_overlay_weight(
    y_validation: pd.Series | np.ndarray,
    multiclass_proba: np.ndarray,
    draw_proba: pd.Series | np.ndarray,
    candidate_weights: Iterable[float] = DRAW_OVERLAY_WEIGHT_CANDIDATES,
) -> tuple[float, list[dict[str, float]]]:
    y_array = np.asarray(y_validation, dtype=int)
    unknown_labels = sorted(set(np.unique(y_array)) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected target labels for draw-overlay selection: {unknown_labels}")

    candidates = [float(weight) for weight in candidate_weights]
    if not candidates:
        raise ValueError("At least one draw-overlay candidate weight is required")

    results = []
    for weight in candidates:
        if weight < 0.0 or weight > 1.0:
            raise ValueError("draw-overlay candidate weights must be between 0 and 1")
        blended = blend_draw_probability(multiclass_proba, draw_proba, blend_weight=weight)
        results.append(
            {
                "draw_overlay_weight": weight,
                "validation_log_loss": float(log_loss(y_array, blended, labels=list(LABELS))),
            }
        )

    selected = min(results, key=lambda row: (row["validation_log_loss"], row["draw_overlay_weight"]))
    return float(selected["draw_overlay_weight"]), results


class DrawAdjustedModel:
    def __init__(self, multiclass_model: object, draw_model: object, blend_weight: float, feature_names: Iterable[str]):
        self.multiclass_model = multiclass_model
        self.draw_model = draw_model
        self.blend_weight = float(blend_weight)
        self.classes_ = np.asarray(LABELS)
        self.feature_names_in_ = np.asarray(list(feature_names), dtype=object)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        base_proba = normalize_probabilities(self.multiclass_model.predict_proba(X))
        draw_matrix = np.asarray(self.draw_model.predict_proba(X), dtype=float)
        if draw_matrix.ndim != 2 or draw_matrix.shape[1] != 2:
            raise ValueError("Draw model must return a two-column probability matrix")
        return blend_draw_probability(base_proba, draw_matrix[:, 1], blend_weight=self.blend_weight)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def class_prior_probabilities(y_train: pd.Series | np.ndarray, rows: int) -> np.ndarray:
    y_array = np.asarray(y_train, dtype=int)
    counts = np.asarray([np.sum(y_array == label) for label in LABELS], dtype=float)
    if np.any(counts == 0):
        raise ValueError("All classes are required to build class-prior probabilities")
    probabilities = counts / counts.sum()
    return np.tile(probabilities, (rows, 1))


def hard_class_predictions(label: int, rows: int) -> np.ndarray:
    if label not in LABELS:
        raise ValueError(f"Unexpected hard baseline label: {label}")
    if rows < 0:
        raise ValueError("rows must be non-negative")
    return np.full(rows, label, dtype=int)


def majority_class_predictions(y_train: pd.Series | np.ndarray, rows: int) -> np.ndarray:
    y_array = np.asarray(y_train, dtype=int)
    if len(y_array) == 0:
        raise ValueError("At least one training label is required for the majority-class baseline")
    unknown_labels = sorted(set(np.unique(y_array)) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected target labels for majority-class baseline: {unknown_labels}")
    counts = {label: int(np.sum(y_array == label)) for label in LABELS}
    majority_label = max(LABELS, key=lambda label: (counts[label], -label))
    return hard_class_predictions(majority_label, rows)


def always_home_predictions(rows: int) -> np.ndarray:
    return hard_class_predictions(0, rows)


def majority_class_probabilities(
    y_train: pd.Series | np.ndarray,
    rows: int,
    confidence: float = 0.98,
) -> np.ndarray:
    return hard_predictions_to_probabilities(majority_class_predictions(y_train, rows), confidence=confidence)


def always_home_probabilities(rows: int, confidence: float = 0.98) -> np.ndarray:
    return hard_predictions_to_probabilities(always_home_predictions(rows), confidence=confidence)


def hard_predictions_to_probabilities(predicted: pd.Series | np.ndarray, confidence: float = 0.98) -> np.ndarray:
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    predicted_array = np.asarray(predicted, dtype=int)
    unknown_labels = sorted(set(np.unique(predicted_array)) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected hard baseline labels: {unknown_labels}")
    remainder = (1.0 - confidence) / (len(LABELS) - 1)
    proba = np.full((len(predicted_array), len(LABELS)), remainder, dtype=float)
    for row_index, label in enumerate(predicted_array):
        proba[row_index, label] = confidence
    return normalize_probabilities(proba)


def elo_heuristic_probabilities(holdout_df: pd.DataFrame, y_train: pd.Series | np.ndarray) -> np.ndarray:
    required = ["HomeElo", "AwayElo"]
    missing = [column for column in required if column not in holdout_df.columns]
    if missing:
        raise ValueError(f"Missing ELO columns for benchmark: {missing}")

    y_array = np.asarray(y_train, dtype=int)
    draw_rate = float(np.mean(y_array == 1))
    home_binary = 1.0 / (1.0 + 10.0 ** ((holdout_df["AwayElo"].astype(float) - holdout_df["HomeElo"].astype(float)) / 400.0))
    p_draw = np.full(len(holdout_df), draw_rate, dtype=float)
    non_draw = 1.0 - p_draw
    proba = np.column_stack([non_draw * home_binary, p_draw, non_draw * (1.0 - home_binary)])
    return normalize_probabilities(proba)


def compact_benchmark_metrics(name: str, y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> dict[str, object]:
    metrics = evaluate_predictions(y_true, y_proba)
    return {
        "model": name,
        "type": "probabilistic",
        "log_loss": round(metrics["holdout_log_loss"], 6),
        "brier_score": round(metrics["holdout_brier_score"], 6),
        "accuracy": round(metrics["holdout_accuracy"], 6),
        "f1_home": round(metrics["holdout_f1_home"], 6),
        "f1_draw": round(metrics["holdout_f1_draw"], 6),
        "f1_away": round(metrics["holdout_f1_away"], 6),
        "predicted_home": int(metrics["holdout_predicted_home"]),
        "predicted_draw": int(metrics["holdout_predicted_draw"]),
        "predicted_away": int(metrics["holdout_predicted_away"]),
    }


def compact_hard_baseline_metrics(name: str, y_true: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, object]:
    y_array = np.asarray(y_true, dtype=int)
    predicted_array = np.asarray(predicted, dtype=int)
    f1_values = f1_score(y_array, predicted_array, labels=list(LABELS), average=None, zero_division=0)
    row = {
        "model": name,
        "type": "hard_class",
        "accuracy": round(float(accuracy_score(y_array, predicted_array)), 6),
        "log_loss": None,
        "brier_score": None,
    }
    for label, value in zip(LABELS, f1_values):
        name_suffix = LABEL_NAMES[label]
        row[f"f1_{name_suffix}"] = round(float(value), 6)
        row[f"predicted_{name_suffix}"] = int(np.sum(predicted_array == label))
    return row


def build_model_benchmarks(
    y_train: pd.Series | np.ndarray,
    holdout_df: pd.DataFrame,
    y_holdout: pd.Series | np.ndarray,
    model_proba: np.ndarray,
    draw_overlay_proba: np.ndarray | None = None,
    draw_overlay_weight: float | None = None,
) -> list[dict[str, object]]:
    rows = len(holdout_df)
    benchmarks = [
        compact_benchmark_metrics("historical_class_prior", y_holdout, class_prior_probabilities(y_train, rows)),
        compact_hard_baseline_metrics("majority_class", y_holdout, majority_class_predictions(y_train, rows)),
        compact_hard_baseline_metrics("always_home", y_holdout, always_home_predictions(rows)),
        compact_benchmark_metrics("elo_heuristic", y_holdout, elo_heuristic_probabilities(holdout_df, y_train)),
        compact_benchmark_metrics("calibrated_xgboost", y_holdout, model_proba),
    ]
    if draw_overlay_proba is not None:
        row = compact_benchmark_metrics("calibrated_xgboost_draw_overlay", y_holdout, draw_overlay_proba)
        row["draw_overlay_weight"] = float(draw_overlay_weight if draw_overlay_weight is not None else DRAW_OVERLAY_WEIGHT)
        benchmarks.append(row)
    return benchmarks


def compute_class_sample_weights(
    y: pd.Series | np.ndarray,
    draw_multiplier: float = DRAW_CLASS_WEIGHT_MULTIPLIER,
) -> np.ndarray:
    y_array = np.asarray(y, dtype=int)
    unknown_labels = sorted(set(np.unique(y_array)) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected target labels for sample weights: {unknown_labels}")
    counts = {label: int(np.sum(y_array == label)) for label in LABELS}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"All classes are required for sample weighting: {counts}")
    total = len(y_array)
    class_weights = {label: total / (len(LABELS) * count) for label, count in counts.items()}
    class_weights[1] *= float(draw_multiplier)
    weights = np.asarray([class_weights[label] for label in y_array], dtype=float)
    return weights / weights.mean()


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    trials: int,
    sample_weight: np.ndarray | None = None,
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
            y_fold_train = y_train.iloc[train_index]
            fit_weight = compute_class_sample_weights(y_fold_train) if sample_weight is not None else None
            model.fit(X_train.iloc[train_index], y_fold_train, sample_weight=fit_weight)
            proba = model.predict_proba(X_train.iloc[val_index])
            scores.append(log_loss(y_train.iloc[val_index], proba, labels=list(LABELS)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {**DEFAULT_XGB_PARAMS, **study.best_params, "random_state": random_state}


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, object],
    sample_weight: np.ndarray | None = None,
) -> XGBClassifier:
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_draw_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, object] | None = None,
) -> XGBClassifier:
    draw_labels = draw_binary_labels(y_train)
    sample_weight = compute_binary_sample_weights(draw_labels)
    model = XGBClassifier(**(params or DEFAULT_DRAW_XGB_PARAMS))
    model.fit(X_train, draw_labels, sample_weight=sample_weight)
    return model


def split_model_calibration_data(
    train_df: pd.DataFrame,
    calibration_fraction: float = 0.2,
    min_calibration_rows: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train_df.sort_values(["Date", "RBallID"], kind="mergesort").reset_index(drop=True)
    calibration_rows = max(min_calibration_rows, int(round(len(ordered) * calibration_fraction)))
    if len(ordered) <= calibration_rows:
        raise ValueError("Training data is too small to reserve a calibration split")
    model_df = ordered.iloc[:-calibration_rows].copy().reset_index(drop=True)
    calibration_df = ordered.iloc[-calibration_rows:].copy().reset_index(drop=True)
    if set(calibration_df[TARGET_COLUMN].unique()) != set(LABELS):
        raise ValueError("Calibration split must contain all result classes")
    return model_df, calibration_df


def calibrate_classifier(
    base_model: XGBClassifier,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    method: str = "sigmoid",
):
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unsupported calibration method: {method}")
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")
    calibrated.fit(X_calibration, y_calibration)
    return calibrated


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_feature_importance(model: XGBClassifier, output_path: Path, feature_columns: Iterable[str] = FEATURE_COLUMNS) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance = pd.Series(model.feature_importances_, index=list(feature_columns)).sort_values()
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
    model: object,
    params: dict[str, object],
    metrics: dict[str, float],
    artifact_paths: Iterable[Path],
    tracking_uri: str | None = None,
) -> tuple[str | None, str | None]:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return None, None

    mlflow.set_tracking_uri(tracking_uri or "file:mlruns")
    mlflow.set_experiment("match_outcome_prediction")
    with mlflow.start_run(run_name="stage3_xgboost_calibrated"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in artifact_paths:
            mlflow.log_artifact(str(path))
        model_info = mlflow.sklearn.log_model(model, artifact_path="model")
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
    football_data_dir: Path = FOOTBALL_DATA_DIR,
    trials: int = 0,
    tracking_uri: str | None = None,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
    use_market_features: bool = True,
) -> TrainingRunSummary:
    if calibration_method not in CALIBRATION_METHODS:
        raise ValueError(f"Unsupported calibration method: {calibration_method}")
    df = load_feature_data(leagues=leagues, features_dir=features_dir)
    market_summary = None
    selected_feature_columns = FEATURE_COLUMNS
    if use_market_features:
        df, market_summary = add_market_features(df, football_data_dir=football_data_dir)
        selected_feature_columns = MARKET_AWARE_FEATURE_COLUMNS
    train_df, holdout_df = split_train_holdout(df)
    model_train_df, calibration_df = split_model_calibration_data(train_df)
    X_train, y_train = select_features_and_target(model_train_df, feature_columns=selected_feature_columns)
    X_calibration, y_calibration = select_features_and_target(calibration_df, feature_columns=selected_feature_columns)
    X_holdout, y_holdout = select_features_and_target(holdout_df, feature_columns=selected_feature_columns)

    train_sample_weight = compute_class_sample_weights(y_train)
    params = tune_hyperparameters(X_train, y_train, trials=trials, sample_weight=train_sample_weight)
    model = train_classifier(X_train, y_train, params=params, sample_weight=train_sample_weight)
    calibrated_model = calibrate_classifier(model, X_calibration, y_calibration, method=calibration_method)
    draw_model = train_draw_classifier(X_train, y_train)
    calibrated_draw_model = calibrate_classifier(draw_model, X_calibration, draw_binary_labels(y_calibration), method=calibration_method)
    calibration_base_proba = normalize_probabilities(calibrated_model.predict_proba(X_calibration))
    calibration_draw_proba = np.asarray(calibrated_draw_model.predict_proba(X_calibration), dtype=float)[:, 1]
    selected_draw_overlay_weight, draw_overlay_weight_results = select_draw_overlay_weight(
        y_calibration,
        calibration_base_proba,
        calibration_draw_proba,
    )
    selected_overlay_model = DrawAdjustedModel(
        multiclass_model=calibrated_model,
        draw_model=calibrated_draw_model,
        blend_weight=selected_draw_overlay_weight,
        feature_names=selected_feature_columns,
    )
    base_holdout_proba = normalize_probabilities(calibrated_model.predict_proba(X_holdout))
    selected_overlay_holdout_proba = normalize_probabilities(selected_overlay_model.predict_proba(X_holdout))
    base_metrics = evaluate_predictions(y_holdout, base_holdout_proba)
    selected_overlay_metrics = evaluate_predictions(y_holdout, selected_overlay_holdout_proba)
    overlay_accepted = selected_overlay_metrics["holdout_log_loss"] < base_metrics["holdout_log_loss"]
    production_model = selected_overlay_model if overlay_accepted else calibrated_model
    holdout_proba = selected_overlay_holdout_proba if overlay_accepted else base_holdout_proba
    metrics = selected_overlay_metrics if overlay_accepted else base_metrics

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "xgb_match_outcome.json"
    metrics_path = artifacts_dir / "metrics.json"
    predictions_path = artifacts_dir / "holdout_predictions.parquet"
    feature_importance_path = artifacts_dir / "feature_importance.png"
    confusion_matrix_path = artifacts_dir / "confusion_matrix.png"
    benchmarks_path = artifacts_dir / "model_benchmarks.json"
    draw_overlay_selection_path = artifacts_dir / "draw_overlay_weight_selection.json"

    _, y_benchmark = select_features_and_target(train_df, feature_columns=selected_feature_columns)
    benchmarks = build_model_benchmarks(
        y_benchmark,
        holdout_df,
        y_holdout,
        base_holdout_proba,
        draw_overlay_proba=selected_overlay_holdout_proba,
        draw_overlay_weight=selected_draw_overlay_weight,
    )

    model.save_model(model_path)
    write_json(
        metrics_path,
        {
            **metrics,
            "use_market_features": use_market_features,
            "market_feature_columns": list(MARKET_FEATURE_COLUMNS) if use_market_features else [],
            "market_feature_input_rows": market_summary.input_rows if market_summary else None,
            "market_feature_output_rows": market_summary.output_rows if market_summary else None,
            "market_feature_dropped_rows": market_summary.dropped_rows if market_summary else None,
            "draw_overlay_weight": selected_draw_overlay_weight if overlay_accepted else 0.0,
            "selected_draw_overlay_weight": selected_draw_overlay_weight,
            "draw_overlay_accepted_as_production": overlay_accepted,
            "calibration_method": calibration_method,
        },
    )
    write_json(benchmarks_path, {"benchmarks": benchmarks})
    write_json(
        draw_overlay_selection_path,
        {
            "selection_data": "pre_holdout_calibration_split",
            "selection_metric": "multiclass_log_loss",
            "selected_draw_overlay_weight": selected_draw_overlay_weight,
            "selected_holdout_log_loss": selected_overlay_metrics["holdout_log_loss"],
            "base_holdout_log_loss": base_metrics["holdout_log_loss"],
            "accepted_as_production": overlay_accepted,
            "candidates": draw_overlay_weight_results,
        },
    )
    write_feature_importance(model, feature_importance_path, feature_columns=selected_feature_columns)
    write_confusion_matrix(y_holdout, holdout_proba, confusion_matrix_path)

    predictions = holdout_df[["RBallID", "League", "Date", "Season", "HomeTeam", "AwayTeam", "Result"]].copy()
    predictions["P_Home"] = holdout_proba[:, 0]
    predictions["P_Draw"] = holdout_proba[:, 1]
    predictions["P_Away"] = holdout_proba[:, 2]
    predictions.to_parquet(predictions_path, index=False)

    logged_params = {
        **params,
        "selected_draw_overlay_weight": selected_draw_overlay_weight,
        "production_draw_overlay_weight": selected_draw_overlay_weight if overlay_accepted else 0.0,
        "draw_overlay_accepted_as_production": overlay_accepted,
        "calibration_method": calibration_method,
        "use_market_features": use_market_features,
    }
    mlflow_run_id, model_uri = log_mlflow_run(
        model=production_model,
        params=logged_params,
        metrics=metrics,
        artifact_paths=[
            metrics_path,
            predictions_path,
            feature_importance_path,
            confusion_matrix_path,
            benchmarks_path,
            draw_overlay_selection_path,
        ],
        tracking_uri=tracking_uri,
    )
    registered_model_version = register_production_model(
        model_uri=model_uri,
        tracking_uri=tracking_uri,
    )

    return TrainingRunSummary(
        train_rows=len(model_train_df),
        calibration_rows=len(calibration_df),
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
    parser.add_argument("--football-data-dir", type=Path, default=FOOTBALL_DATA_DIR)
    parser.add_argument("--trials", type=int, default=0, help="Optional Optuna trials. Default keeps the run fast.")
    parser.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI.")
    parser.add_argument("--calibration-method", choices=CALIBRATION_METHODS, default=DEFAULT_CALIBRATION_METHOD)
    parser.add_argument("--no-market-features", action="store_true", help="Train the original team-stat-only model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_pipeline(
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
        football_data_dir=args.football_data_dir,
        trials=args.trials,
        tracking_uri=args.tracking_uri,
        calibration_method=args.calibration_method,
        use_market_features=not args.no_market_features,
    )
    print(summary.line())
