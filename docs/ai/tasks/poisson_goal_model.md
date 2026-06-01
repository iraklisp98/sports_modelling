# Poisson Goal Model Benchmark

**Status:** Complete  
**Script:** `pipeline/poisson_goal_model.py`  
**Input:** Stage 2 feature Parquet files from `data/features/`  
**Output:** Poisson 1X2 probabilities, implied odds, benchmark metrics, and an optional MLflow run

---

## What

Build a transparent goal-scoring model that estimates expected home and away goals, converts those expected goals into a scoreline probability matrix with independent Poisson distributions, then aggregates scorelines into `P_Home`, `P_Draw`, and `P_Away`.

This is the first-best modeling improvement because the current calibrated XGBoost model predicts match outcome directly. A Poisson score model uses football structure explicitly: matches are decided by goals, draws are naturally represented by equal scorelines, and the output can be compared against XGBoost on the exact same 2019-20 holdout.

---

## Why This Approach

### Why a Poisson model?

Football scores are low-count events. A Poisson model gives an interpretable probability for each possible goal count, such as 0, 1, 2, 3, and so on. From there, `P(Home)`, `P(Draw)`, and `P(Away)` are not guessed directly; they are derived from the full scoreline distribution.

### Why compare it with calibrated XGBoost?

XGBoost may pick up useful nonlinear patterns, but it can also become hard to explain and may under-handle draws. The Poisson model is easier to reason about: if expected goals are close, draw probability rises naturally. If it beats XGBoost on log loss or Brier score, it becomes a strong candidate for Stage 4. If it does not, it is still a valuable benchmark and interview signal.

### Why keep the same Stage 4 odds contract?

Downstream stages only need `P_Home`, `P_Draw`, `P_Away`, and reciprocal model odds. The Poisson model should write the same probability/odds columns as Stage 4, preferably to a separate file such as `data/output/poisson_model_odds.parquet`, so Stage 5 comparison can be run against either model without changing the value-bet logic.

---

## Implementation Contract

### Inputs

- `data/features/ENG_features.parquet`
- `data/features/SPA_features.parquet`
- `data/features/FRA_features.parquet`

The script should reuse the same chronological split convention as Stage 3:

- Train: seasons before `2019-20`
- Holdout: `2019-20`

### Model Shape

Train two count models:

- Home-goals model predicting `HomeGoals`
- Away-goals model predicting `AwayGoals`

Use a simple, explainable first version before considering more complex variants:

- `sklearn.linear_model.PoissonRegressor`
- Features may start from the existing Stage 2 feature contract, especially ELO, recent goals, recent points, draw-rate/closeness features, season win rates, and league indicators
- Predicted expected goals must be finite and strictly positive after any clipping

### Probability Derivation

For each match:

1. Predict `lambda_home` and `lambda_away`.
2. Build a scoreline probability grid from `0..max_goals` for both teams, defaulting to `max_goals=10`.
3. Compute independent score probabilities:
   ```text
   P(score h-a) = PoissonPMF(h, lambda_home) * PoissonPMF(a, lambda_away)
   ```
4. Aggregate:
   ```text
   P_Home = sum P(h > a)
   P_Draw = sum P(h = a)
   P_Away = sum P(h < a)
   ```
5. Normalize the three probabilities so they sum to 1.0 within tolerance.
6. Convert to decimal odds with `1 / probability`.

### Outputs

Write ignored local artifacts under `data/model_artifacts/poisson_goal_model/`:

- `metrics.json`
- `holdout_predictions.parquet`
- `model_benchmarks.json`

Write a model-odds file with the Stage 4-compatible schema:

- Recommended path: `data/output/poisson_model_odds.parquet`
- Required columns: `RBallID`, `HomeTeam`, `AwayTeam`, `Date`, `Season`, `Result`, `P_Home`, `P_Draw`, `P_Away`, `ModelOdds_Home`, `ModelOdds_Draw`, `ModelOdds_Away`
- Optional diagnostic columns: `Lambda_Home`, `Lambda_Away`

Add the Poisson row to the existing benchmark comparison so it can be read next to:

- `historical_class_prior`
- `majority_class`
- `always_home`
- `elo_heuristic`
- `calibrated_xgboost`
- `calibrated_xgboost_draw_overlay`
- `poisson_goal_model`

MLflow logging is useful but not required for the first implementation. If added, use a separate run name such as `poisson_goal_model` and do not replace `match_outcome_xgb` Production unless a later task explicitly chooses it.

---

## Acceptance Criteria

- [x] `pipeline/poisson_goal_model.py` trains separate home and away goal models without using any 2019-20 rows for fitting
- [x] The scoreline grid derives valid `P_Home`, `P_Draw`, and `P_Away` probabilities for each holdout match
- [x] Probabilities are finite, non-negative, and sum to 1.0 +/- 0.001 per row
- [x] Decimal odds are reciprocal probabilities and contain no null or infinite values
- [x] Holdout metrics include log loss, multiclass Brier score, accuracy, and F1 per class
- [x] Benchmark output compares `poisson_goal_model` against the existing calibrated XGBoost benchmark on the same holdout rows
- [x] Focused unit tests cover scoreline aggregation, probability normalization, odds conversion, and chronological split behavior
- [x] Narrowest useful test command passes, for example `.venv/bin/python -m unittest tests.test_poisson_goal_model`
- [x] `docs/ai/TASKS.md` status is updated when implementation is complete

---

## Result

The Poisson goal model is useful as an interpretable benchmark, but it should not replace the current XGBoost model.

| Metric | Calibrated XGBoost draw overlay | Poisson goal model | Delta |
|---|---:|---:|---:|
| `holdout_log_loss` | `0.997383` | `1.000892` | `+0.003509` |
| `holdout_brier_score` | `0.595766` | `0.598405` | `+0.002639` |
| `holdout_accuracy` | `0.520693` | `0.516843` | `-0.003850` |
| `holdout_f1_draw` | `0.042105` | `0.000000` | `-0.042105` |
| `holdout_predicted_draw` | `18` | `0` | `-18` |

Decision: benchmark complete, not promoted. The next model experiment should target an observed probability-combination or calibration issue rather than adding more raw form features.

---

## Interview Q&A

**Q: Why add a Poisson model when you already have XGBoost?**  
A: "XGBoost predicts outcomes directly from tabular features. The Poisson model predicts expected goals first, then derives the 1X2 market from the score distribution. That gives me a more interpretable football-specific benchmark and a better way to reason about draws."

**Q: How do you get draw probability from a Poisson model?**  
A: "I compute the probability of each home-away scoreline. Draw probability is the sum of all equal-score cells: 0-0, 1-1, 2-2, and so on. Home and away probabilities are the sums above and below that diagonal."

**Q: What decides whether the Poisson model should replace XGBoost?**  
A: "It has to be compared on the same chronological holdout using probability metrics first: log loss and Brier score. ROI only comes after the probabilities pass that check and are run through the same bookmaker-odds comparison logic."
