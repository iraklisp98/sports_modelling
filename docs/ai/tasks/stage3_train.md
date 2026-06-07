# Stage 3 — Model Training & Experiment Tracking

**Status:** Complete  
**Script:** `pipeline/stage3_train.py`  
**Input:** Feature Parquet files from Stage 2 (`data/features/`)  
**Output:** raw XGBoost artifact, calibrated sklearn Production model, holdout predictions, local artifacts, MLflow run, and registered MLflow model

---

## What

Train an XGBoost multiclass classifier to predict `H` / `D` / `A` match outcomes from the Stage 2 feature set, add a calibrated draw-vs-not-draw overlay, compare the final probabilities against simple holdout baselines, track the run in MLflow, and register the composite sklearn-style wrapper as `match_outcome_xgb` with the latest version promoted to `Production`.

---

## Why This Approach

### Why MLflow?
Without experiment tracking, you cannot answer which model was trained, with which parameters, on which split, and with what metrics. Stage 3 logs parameters, metrics, artifacts, holdout predictions, and the model itself so the run can be inspected and reproduced.

### Why XGBoost?
The current model input is tabular: ELO ratings, rolling form, corners, points, and season win rates. XGBoost is a strong baseline for tabular classification and gives feature importance artifacts that are easy to explain in an interview.

### Why time-based splitting?
Football data is chronological. A random split can train on future matches and validate on earlier matches, which leaks future information. Stage 3 trains on every available season before `2019-20`, reserves the latest part of that training window for probability calibration, then evaluates on the untouched `2019-20` holdout snapshot. The default calibration method is isotonic regression, exposed through `--calibration-method`, so calibration is an explicit experiment setting rather than a hidden post-processing step.


### Why league indicator features?
The model trains one global classifier across Premier League, La Liga, Ligue 1, Bundesliga, and Serie A. League one-hot features let the model learn league-level differences in home advantage, draw tendency, and match environment without training three separate small models. Stage 3 and Stage 4 share the same feature contract through `pipeline/model_features.py` so training and inference cannot silently drift.

### Why draw-aware features and weighting?
Draws are not simply a minority class; they usually happen when team strength and recent form are close. Stage 2 adds symmetric closeness features such as absolute ELO gap, recent points gap, and rolling draw rates. Stage 3 also uses class-balanced sample weights with a modest draw multiplier so draw mistakes matter during XGBoost fitting without turning the model into a forced draw predictor.


### Why a draw-vs-not-draw overlay?
The multiclass model still under-predicts draws. Stage 3 now trains a second calibrated binary classifier for `draw` vs `not draw`, then blends its draw probability into the H/D/A matrix. The home-vs-away ratio still comes from the multiclass model:

```text
p_draw = 0.20 * binary_draw_probability + 0.80 * multiclass_draw_probability
non_draw = 1 - p_draw
p_home = non_draw * multiclass_home / (multiclass_home + multiclass_away)
p_away = non_draw * multiclass_away / (multiclass_home + multiclass_away)
```

This is deliberately conservative. The overlay improved draw F1 slightly in the latest generated run, but it did not improve log loss versus the earlier tuned model, so it should be treated as a draw-diagnostics improvement rather than a full betting-model breakthrough.

### Why probability calibration?
Betting uses probabilities directly, not just predicted classes. Stage 3 wraps the fitted XGBoost model with `CalibratedClassifierCV` using sigmoid calibration, then logs that calibrated sklearn model to MLflow. This keeps Stage 4 simple: it loads the Production model and calls `predict_proba()`, receiving calibrated probabilities.


### Why compare against simple baselines?
A model is only useful if it beats simpler alternatives on the same holdout. Probabilistic baselines report log loss and Brier score; hard-class baselines report accuracy and F1 only, because assigning fake confidence would distort probability metrics. Stage 3 now writes a benchmark table comparing calibrated XGBoost with:

- `historical_class_prior`: constant probabilities from the training class distribution
- `majority_class`: a hard-class baseline that always predicts the training majority class
- `always_home`: a hard-class home-win baseline
- `elo_heuristic`: an ELO-only probability heuristic with the training draw rate

This gives an interview-ready answer to: "Does XGBoost add signal, or could a simpler model do the same job?"

### Why Optuna is optional by CLI
The script supports Optuna with `--trials N`, but defaults to `--trials 0` for a fast reproducible baseline. This keeps local development quick while still allowing proper hyperparameter search when needed.

---

## Implemented Contract

### Inputs

- `data/features/ENG_features.parquet`
- `data/features/SPA_features.parquet`
- `data/features/FRA_features.parquet`
- `data/features/GER_features.parquet`
- `data/features/ITA_features.parquet`

### Feature columns

```python
[
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
    "League_ENG",
    "League_SPA",
    "League_FRA",
    "League_GER",
    "League_ITA",
]
```

### Target

`ResultCode`, where `H=0`, `D=1`, `A=2`.

### Split

- Base train: earlier rows before `2019-20`
- Calibration: latest rows before `2019-20`
- Holdout: untouched `2019-20`

### Metrics

- Holdout log loss
- Benchmark comparison versus simple baselines
- Multiclass Brier score: mean squared error between one-hot labels and 3-class probabilities
- Holdout accuracy
- F1 per class: home, draw, away
- Calibration by outcome/probability bucket is produced by `pipeline/model_diagnostics.py` from Stage 3 holdout predictions

ROI/backtest metrics are intentionally deferred until Stage 5, because bookmaker odds and flagged value bets do not exist in Stage 3.

### Outputs

Ignored local artifacts under `data/model_artifacts/stage3/`:

- `xgb_match_outcome.json`
- `metrics.json`
- `holdout_predictions.parquet`
- `feature_importance.png`
- `confusion_matrix.png`
- `model_diagnostics.json` after Stage 5.5 runs
- `model_benchmarks.json` comparing calibrated XGBoost against simple holdout baselines

MLflow outputs under ignored `mlruns/`:

- Experiment: `match_outcome_prediction`
- Logged params and metrics
- Logged artifacts
- Logged composite sklearn-style wrapper around the calibrated multiclass XGBoost model and calibrated binary draw model
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
import mlflow.sklearn

mlflow.set_tracking_uri("file:mlruns")
model = mlflow.sklearn.load_model("models:/match_outcome_xgb/Production")
```

---

## Acceptance Criteria

### Engineering Criteria

- [x] Script runs without errors: `python pipeline/stage3_train.py --trials 0`
- [x] Focused tests pass: `.venv/bin/python -m unittest tests.test_stage3_train`
- [x] MLflow experiment `match_outcome_prediction` is created under `mlruns/`
- [x] Run logs parameters, metrics, feature importance, confusion matrix, holdout predictions, model benchmark comparison, and model artifact
- [x] Model is registered as `match_outcome_xgb` in the local MLflow Model Registry
- [x] Latest registered version is promoted to `Production`
- [x] `mlflow.sklearn.load_model("models:/match_outcome_xgb/Production")` works without error

### Model Quality Targets

- [ ] Holdout accuracy target `> 55%` is not met by the latest generated Production run: `0.5188`
- [ ] Holdout log-loss target `< 0.95` is not met by the latest generated Production run: `0.9964`

The log-loss gap is a model-quality improvement item, not a pipeline implementation blocker. Stage 3 now defaults to market-aware training: Football-Data bookmaker odds are converted into overround-normalized market probabilities, then used as explicit model features alongside team-form features. The latest isolated market-aware experiment (`data/model_artifacts/stage3_market/`) improved the expanded XGBoost holdout log loss to `0.9986` and accuracy to `0.5289`, but still did not beat the raw market baseline over 2019-20 through 2025-26 (`0.9851` model log loss vs `0.9778` market log loss in `market_baseline_diagnostics.json`).

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
