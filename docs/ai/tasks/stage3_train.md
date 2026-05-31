# Stage 3 — Model Training & Experiment Tracking

**Status:** Not started  
**Script:** `pipeline/stage3_train.py`  
**Input:** Feature Parquet files from Stage 2 (`data/features/`)  
**Output:** Trained model artifact + MLflow experiment run logged to `mlruns/`

---

## What

Train an XGBoost classifier to predict match outcomes (Home win / Draw / Away win). Track every training run with MLflow — hyperparameters, metrics, feature importance, and the model artifact. Tune hyperparameters with Optuna. Register the best model in the MLflow Model Registry.

---

## Why This Approach

### Why MLflow?
Without experiment tracking you have no way to answer: "which model am I running in production and why was it chosen?" MLflow records every run so you can reproduce any experiment, compare runs side by side, and promote a specific run to production. It's the industry standard — used at every company that does ML seriously.

If a hiring manager asks "how do you manage model versions?", this is your answer.

### Why Optuna instead of GridSearchCV?
GridSearchCV tries every combination of hyperparameters. If you have 5 parameters with 5 values each, that's 5^5 = 3125 combinations. Optuna uses **Bayesian optimisation** — it learns from previous trials to focus on promising regions of the search space, finding good hyperparameters in 50–100 trials instead of thousands.

### Why XGBoost for this problem?
Football match outcomes are a tabular classification problem. ELO ratings, rolling form, win rates — these are structured numerical features. XGBoost consistently outperforms other algorithms on tabular data. It handles missing values natively (important because early-season matches have no rolling history), and it's fast to train.

### Why time-series cross-validation instead of random split?
A random 80/20 split would let the model train on 2019-2020 data and test on 2017-2018 data — which means it "knows the future." For time-series data, validation folds must always come after training data chronologically. We use the last N months of each season as validation folds.

---

## New Concepts to Learn Before Building

### MLflow concepts
MLflow has four main components. You'll use three of them:

1. **Tracking** — logs parameters, metrics, and artifacts for each run
2. **Model Registry** — stores versioned model artifacts with lifecycle stages (None → Staging → Production)
3. **Projects** — (not used here)
4. **Models** — standard model format for serving (not used until Phase 2)

Key objects:
```python
import mlflow

mlflow.set_experiment("match_outcome_prediction")  # group runs under this name

with mlflow.start_run():               # creates a run
    mlflow.log_param("n_estimators", 200)   # log one hyperparameter
    mlflow.log_metric("log_loss", 0.94)     # log one metric
    mlflow.log_artifact("feature_importance.png")  # log a file
    mlflow.xgboost.log_model(model, "model")        # log the model itself
```

### Optuna concepts
Optuna works by defining an **objective function** that takes a `trial` object. Optuna calls this function many times, each time with different hyperparameter suggestions. After N trials it returns the best result.

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    }
    # train with these params, return the metric to minimise
    return log_loss_score

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
best_params = study.best_params
```

### Log Loss vs Accuracy
Accuracy tells you what % of predictions were correct. It doesn't tell you *how confident* the model was. Log loss penalises confident wrong predictions heavily. A model that says "90% chance Home wins" and the away team wins incurs much higher log loss than a model that said "55% Home". For a betting model, calibrated probabilities matter more than raw accuracy.

---

## How to Build It (Step by Step)

### Step 1 — Create the script file
Create `pipeline/stage3_train.py`.

### Step 2 — Load and combine feature data
```python
import pandas as pd
import numpy as np

leagues = ["ENG", "FRA", "SPA"]
dfs = [pd.read_parquet(f"data/features/{l}_features.parquet") for l in leagues]
df = pd.concat(dfs, ignore_index=True).sort_values("Date")
```

Combining all three leagues triples the training data and lets the model learn cross-league patterns.

### Step 3 — Define features and target
```python
FEATURES = [
    "HomeElo", "AwayElo", "EloDiff",
    "HomeGoals_Last5", "AwayGoals_Last5",
    "HomeCorners_Last5", "AwayCorners_Last5",
    "HomePoints_Last5", "AwayPoints_Last5",
    "HomeWinRate_Season", "AwayWinRate_Season",
]
TARGET = "ResultCode"  # 0=H, 1=D, 2=A

X = df[FEATURES]
y = df[TARGET]
```

### Step 4 — Train/test split by date
The holdout is the 2019–2020 season. All earlier data is for training.

```python
train_mask = df["Season"].isin(["2017-18", "2018-19"])
test_mask = df["Season"] == "2019-20"

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
```

### Step 5 — Define time-series cross-validation folds
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=3)
```

`TimeSeriesSplit` ensures each validation fold is always chronologically after its training fold.

### Step 6 — Define the Optuna objective
```python
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
    }

    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)
        scores.append(log_loss(y_val, preds))

    return np.mean(scores)
```

### Step 7 — Run Optuna tuning
```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
print(f"Best log loss: {study.best_value:.4f}")
print(f"Best params: {best_params}")
```

### Step 8 — Train final model and log to MLflow
```python
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, f1_score, confusion_matrix
import seaborn as sns

mlflow.set_experiment("match_outcome_prediction")

best_params.update({
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
})

with mlflow.start_run(run_name="xgboost_optuna_tuned"):
    # Train on all training data
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # Evaluate on holdout
    y_pred_proba = final_model.predict_proba(X_test)
    y_pred = final_model.predict(X_test)

    ll = log_loss(y_test, y_pred_proba)
    bs = brier_score_loss(y_test == 0, y_pred_proba[:, 0])  # home win
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=None)

    # Log params
    mlflow.log_params(best_params)

    # Log metrics
    mlflow.log_metric("test_log_loss", ll)
    mlflow.log_metric("test_brier_score", bs)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("f1_home", f1[0])
    mlflow.log_metric("f1_draw", f1[1])
    mlflow.log_metric("f1_away", f1[2])

    # Log feature importance chart
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.Series(final_model.feature_importances_, index=FEATURES).sort_values().plot.barh(ax=ax)
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["H","D","A"], yticklabels=["H","D","A"], ax=ax2)
    ax2.set_title("Confusion Matrix — Holdout")
    fig2.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log the model
    mlflow.xgboost.log_model(final_model, artifact_path="model")

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    print(f"Log Loss: {ll:.4f} | Accuracy: {acc:.4f}")
```

### Step 9 — Register the model
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_uri = f"runs:/{run_id}/model"

# Register to the Model Registry
result = mlflow.register_model(model_uri, "match_outcome_xgb")

# Transition to Production
client.transition_model_version_stage(
    name="match_outcome_xgb",
    version=result.version,
    stage="Production"
)
print(f"Model version {result.version} promoted to Production")
```

### Step 10 — Verify you can load the Production model
```python
model = mlflow.xgboost.load_model("models:/match_outcome_xgb/Production")
test_preds = model.predict_proba(X_test[:5])
print(test_preds)
```

If this prints probabilities, Stage 3 is complete.

---

## Acceptance Criteria

- [ ] Script runs without errors: `python pipeline/stage3_train.py`
- [ ] MLflow experiment `match_outcome_prediction` visible at `mlruns/`
- [ ] Run logs all params, metrics (log loss, brier, accuracy, F1 per class), and two artifacts (feature importance chart, confusion matrix)
- [ ] Test log loss < 0.95
- [ ] Test accuracy > 55%
- [ ] Model registered as `match_outcome_xgb` in MLflow Model Registry with stage = Production
- [ ] `mlflow.xgboost.load_model("models:/match_outcome_xgb/Production")` works without error

---

## Interview Q&A

**Q: How do you track and reproduce your model experiments?**  
A: "I use MLflow. Every training run logs the hyperparameters, all evaluation metrics, the feature importance chart, and the model artifact. I also register the best model in the MLflow Model Registry and transition it to Production. That means I can always reload the exact model that's running, reproduce any past experiment by run ID, and compare runs side by side."

**Q: Why Optuna over GridSearchCV?**  
A: "GridSearch is exhaustive — it tries every combination. With 6 hyperparameters and 5 values each, that's 15,000+ fits. Optuna uses Bayesian optimisation — it learns from previous trials and focuses on promising regions. 50 Optuna trials typically finds a better result than 500 GridSearch combinations."

**Q: What is log loss and why did you choose it as your primary metric?**  
A: "Log loss measures how well-calibrated the probability outputs are, not just whether the top prediction was correct. For a betting model this matters more than accuracy — the whole system depends on reliable win probabilities. A model that says 90% confidence and gets it wrong is much worse than one that says 55% and gets it wrong. Log loss captures that distinction; accuracy doesn't."

**Q: Why use TimeSeriesSplit for cross-validation?**  
A: "Standard K-fold randomly splits data, which means a validation fold could contain earlier matches than some training data. For time-series, that's data leakage — the model would have seen 'future' information. TimeSeriesSplit ensures each validation fold is always chronologically after its training fold, which mirrors how the model will be used in production."
