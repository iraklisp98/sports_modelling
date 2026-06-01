# Stage 3 — Model Training & Experiment Tracking

**Status:** Complete  
**Script:** `pipeline/stage3_train.py`  
**Input:** Feature Parquet files from Stage 2 (`data/features/`)  
**Output:** XGBoost model artifact, holdout predictions, local artifacts, MLflow run, and registered MLflow model

---

## What

Train an XGBoost multiclass classifier to predict `H` / `D` / `A` match outcomes from the Stage 2 feature set. Track the run in MLflow, log evaluation artifacts, and register the model as `match_outcome_xgb` with the latest version promoted to `Production`.

---

## Why This Approach

### Why MLflow?
Without experiment tracking, you cannot answer which model was trained, with which parameters, on which split, and with what metrics. Stage 3 logs parameters, metrics, artifacts, holdout predictions, and the model itself so the run can be inspected and reproduced.

### Why XGBoost?
The current model input is tabular: ELO ratings, rolling form, corners, points, and season win rates. XGBoost is a strong baseline for tabular classification and gives feature importance artifacts that are easy to explain in an interview.

### Why time-based splitting?
Football data is chronological. A random split can train on future matches and validate on earlier matches, which leaks future information. Stage 3 trains on every available season before `2019-20`, then evaluates on the `2019-20` holdout snapshot.

### Why Optuna is optional by CLI
The script supports Optuna with `--trials N`, but defaults to `--trials 0` for a fast reproducible baseline. This keeps local development quick while still allowing proper hyperparameter search when needed.

---

## Implemented Contract

### Inputs

- `data/features/ENG_features.parquet`
- `data/features/SPA_features.parquet`
- `data/features/FRA_features.parquet`

### Feature columns

```python
[
    "HomeElo", "AwayElo", "EloDiff",
    "HomeGoals_Last5", "AwayGoals_Last5",
    "HomeCorners_Last5", "AwayCorners_Last5",
    "HomePoints_Last5", "AwayPoints_Last5",
    "HomeWinRate_Season", "AwayWinRate_Season",
]
```

### Target

`ResultCode`, where `H=0`, `D=1`, `A=2`.

### Split

- Train: every available season before `2019-20`
- Holdout: `2019-20`

### Metrics

- Holdout log loss
- Multiclass Brier score: mean squared error between one-hot labels and 3-class probabilities
- Holdout accuracy
- F1 per class: home, draw, away

ROI/backtest metrics are intentionally deferred until Stage 5, because bookmaker odds and flagged value bets do not exist in Stage 3.

### Outputs

Ignored local artifacts under `data/model_artifacts/stage3/`:

- `xgb_match_outcome.json`
- `metrics.json`
- `holdout_predictions.parquet`
- `feature_importance.png`
- `confusion_matrix.png`

MLflow outputs under ignored `mlruns/`:

- Experiment: `match_outcome_prediction`
- Logged params and metrics
- Logged artifacts
- Logged XGBoost model
- Registered model: `match_outcome_xgb`
- Latest version promoted to `Production`

---

## How To Run

Fast baseline:

```bash
python pipeline/stage3_train.py --trials 0
```

Small tuning run:

```bash
python pipeline/stage3_train.py --trials 10
```

Full intended tuning run:

```bash
python pipeline/stage3_train.py --trials 50
```

Load the registered Production model:

```python
import mlflow

mlflow.set_tracking_uri("file:mlruns")
model = mlflow.xgboost.load_model("models:/match_outcome_xgb/Production")
```

---

## Acceptance Criteria

### Engineering Criteria

- [x] Script runs without errors: `python pipeline/stage3_train.py --trials 0`
- [x] Focused tests pass: `python -m pytest tests/test_stage3_train.py`
- [x] MLflow experiment `match_outcome_prediction` is created under `mlruns/`
- [x] Run logs parameters, metrics, feature importance, confusion matrix, holdout predictions, and model artifact
- [x] Model is registered as `match_outcome_xgb` in the local MLflow Model Registry
- [x] Latest registered version is promoted to `Production`
- [x] `mlflow.xgboost.load_model("models:/match_outcome_xgb/Production")` works without error

### Model Quality Targets

- [x] Holdout accuracy target `> 55%` met in a 30-trial tuning run: `0.5583`
- [ ] Holdout log-loss target `< 0.95` not yet met; best checked 30-trial run: `0.9750`

The log-loss gap is a model-quality improvement item, not a pipeline implementation blocker. Likely next improvements are calibration, richer features, and cleaner betting-market filters.

---

## Interview Q&A

**Q: How do you track and reproduce model experiments?**  
A: "I use MLflow. Each run logs the parameters, metrics, feature list, artifacts, holdout predictions, and model artifact. The model is registered as `match_outcome_xgb` so downstream stages can load a stable Production model URI."

**Q: Why is log loss the primary metric?**  
A: "For odds modelling, calibrated probabilities matter more than just picking the most likely class. Log loss penalizes confident wrong predictions, which is exactly the kind of error that hurts a betting model."

**Q: Why is ROI not computed in Stage 3?**  
A: "ROI needs bookmaker odds and flagged value bets. Those are produced later by the odds comparison stage, so Stage 3 logs model probabilities and leaves ROI/backtesting to Stage 5 and the dashboard export."

**Q: Why use a chronological holdout?**  
A: "The model will be used on future matches, so evaluation must mimic that. Training on earlier seasons and holding out the latest available season avoids future leakage."
