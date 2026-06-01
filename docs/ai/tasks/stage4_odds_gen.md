# Stage 4 — Odds Generation

**Status:** Complete  
**Script:** `pipeline/stage4_odds_gen.py`  
**Input:** Feature Parquet files from Stage 2 + Production model from MLflow  
**Output:** `data/output/model_odds.parquet` — one row per match with model-implied decimal odds

---

## What

Load the calibrated Production model from MLflow, run inference on the full feature dataset, convert the calibrated probability outputs to decimal odds, validate them, and write the results. Stage 4 imports the shared model feature contract from `pipeline/model_features.py` so inference columns stay aligned with Stage 3 training.

---

## Why This Approach

### Why load from MLflow instead of a local file?
Loading the calibrated sklearn model via `"models:/match_outcome_xgb/Production"` means this stage always uses whatever is currently in Production. If you retrain and promote a new version in Stage 3, Stage 4 picks it up automatically without any code change. This is how ML systems work in production: stages are decoupled from specific model versions.

### Why convert probabilities to decimal odds?
Bookmakers express their prices as decimal odds (e.g. 2.50 = $2.50 back for every $1 staked including stake). The formula is:
```
decimal_odds = 1 / probability
```
If the model says P(Home win) = 0.55, the implied decimal odds are 1/0.55 = 1.82. If the bookmaker offers 2.10, that's a value bet.

### Why validate that probabilities sum to 1?
The model outputs three probabilities: P(H), P(D), P(A). They must sum to 1.0 — otherwise something went wrong in inference (wrong number of classes, wrong model loaded, etc.). This is a contract check, not optional.

---

## How to Build It (Step by Step)

### Step 1 — Create the script file
Create `pipeline/stage4_odds_gen.py`.

### Step 2 — Load the Production model
```python
import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/match_outcome_xgb/Production")
```

### Step 3 — Load feature data
```python
import pandas as pd

leagues = ["ENG", "FRA", "SPA"]
dfs = [pd.read_parquet(f"data/features/{l}_features.parquet") for l in leagues]
df = pd.concat(dfs, ignore_index=True).sort_values("Date")
```

### Step 4 — Run inference
```python
FEATURES = [
    "HomeElo", "AwayElo", "EloDiff",
    "HomeGoals_Last5", "AwayGoals_Last5",
    "HomeCorners_Last5", "AwayCorners_Last5",
    "HomePoints_Last5", "AwayPoints_Last5",
    "HomeWinRate_Season", "AwayWinRate_Season",
]

proba = model.predict_proba(df[FEATURES])  # shape: (n_matches, 3)

df["P_Home"] = proba[:, 0]
df["P_Draw"] = proba[:, 1]
df["P_Away"] = proba[:, 2]
```

### Step 5 — Validate probabilities
```python
prob_sum = df[["P_Home", "P_Draw", "P_Away"]].sum(axis=1)
assert (prob_sum - 1.0).abs().max() < 0.001, "Probabilities do not sum to 1.0"
print("Probability validation passed.")
```

If this assertion fails, stop and investigate. Do not proceed with invalid probabilities.

### Step 6 — Convert to decimal odds
```python
df["ModelOdds_Home"] = 1 / df["P_Home"]
df["ModelOdds_Draw"] = 1 / df["P_Draw"]
df["ModelOdds_Away"] = 1 / df["P_Away"]
```

### Step 7 — Select output columns and write
```python
import os

output_cols = [
    "RBallID", "HomeTeam", "AwayTeam", "Date", "Season", "Result",
    "P_Home", "P_Draw", "P_Away",
    "ModelOdds_Home", "ModelOdds_Draw", "ModelOdds_Away"
]

os.makedirs("data/output", exist_ok=True)
df[output_cols].to_parquet("data/output/model_odds.parquet", index=False)
print(f"Written {len(df)} rows to data/output/model_odds.parquet")
```

---

## Acceptance Criteria

- [x] Script runs without errors: `python pipeline/stage4_odds_gen.py`
- [x] `data/output/model_odds.parquet` created
- [x] All probability columns (`P_Home`, `P_Draw`, `P_Away`) are between 0 and 1
- [x] Probabilities sum to 1.0 ± 0.001 for every row (assertion passes)
- [x] Decimal odds are `1 / probability` — verify a few rows manually
- [x] No rows have null odds values

---

## Interview Q&A

**Q: How do you convert a model's probability output to betting odds?**  
A: "Decimal odds are simply the reciprocal of probability: `odds = 1 / p`. If the model says there's a 55% chance of a home win, the implied odds are 1/0.55 = 1.82. If the bookmaker is offering 2.10 for the same outcome, the model is saying that bet is underpriced — that's the edge we're looking for."

**Q: Why validate that probabilities sum to 1?**  
A: "It's a contract check. The three outcomes are mutually exclusive and exhaustive, so their probabilities must sum to 1. If they don't, something is wrong — wrong model loaded, wrong feature set, class encoding mismatch. Catching it here prevents silent errors propagating to the odds comparison stage where the arbitrage logic would produce nonsense."
