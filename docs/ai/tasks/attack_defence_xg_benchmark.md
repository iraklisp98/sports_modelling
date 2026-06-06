# Attack/Defence xG Benchmark

**Status:** Not started  
**Suggested script:** `pipeline/attack_defence_xg_benchmark.py`  
**Input:** Stage 2 feature Parquet files from `data/features/`  
**Output:** separate benchmark odds, explainability artifacts, calibration diagnostics, and value-bet ROI comparison

---

## What

Build a narrow, explainable expected-goals benchmark that estimates team attack and defence strengths from historical goals, derives expected home and away goals for each holdout match, converts those expected goals into a scoreline probability grid, and aggregates that grid into `P_Home`, `P_Draw`, and `P_Away`.

This is not a replacement for the current MLflow Production XGBoost model. It is a benchmark task: the goal is to learn whether a football-native, interpretable model gives better probability quality or betting behaviour than the current calibrated XGBoost output.

---

## Why

The completed `pipeline/poisson_goal_model.py` proves that a scoreline-grid benchmark can run, but it trains generic home-goal and away-goal regressors on the Stage 2 feature matrix. That gives probability metrics, but it does not give the clean football explanation a hiring manager will expect:

- "Team A has above-average attacking strength."
- "Team B concedes more than league average."
- "The predicted home xG comes from league baseline, home advantage, home attack, and away defence."

The hiring signal is model comparison discipline. A stronger MLE answer is not "XGBoost is more advanced"; it is "I compared a flexible tabular model against an interpretable domain model on the same holdout, using probability metrics first and ROI second."

---

## First-Version Scope

Keep this deliberately small.

- Fit league-specific or pooled team attack/defence strengths from completed pre-holdout matches only.
- Use goals only for the first version: `HomeGoals`, `AwayGoals`, `League`, `Season`, `Date`, `HomeTeam`, `AwayTeam`, `Result`, `ResultCode`, `RBallID`.
- Include a fixed or learned home-advantage term, documented in the metrics artifact.
- Apply simple shrinkage toward league average so teams with fewer observations do not get extreme ratings.
- Derive `lambda_home` and `lambda_away` from attack and defence strengths.
- Convert lambdas into 1X2 probabilities via a scoreline grid, default `max_goals=10`.
- Write benchmark outputs to separate paths; do not overwrite `data/output/model_odds.parquet`, `data/output/value_bets.parquet`, dashboard JSON, Stage 3 artifacts, or MLflow Production.

Out of scope for version 1:

- Dixon-Coles correlation adjustment.
- Time-decayed fitting.
- In-play features, shots, corners, bookmaker odds, or any holdout result data.
- Promoting this model to Production.
- Rebuilding the dashboard around this model.

---

## Model Contract

### Training Split

Use the same chronological convention as Stage 3:

- Train: all rows with season start year before `2019-20`.
- Holdout: exactly `2019-20`.

No `2019-20` row may contribute to attack strength, defence strength, league baseline, home advantage, shrinkage priors, calibration choices, threshold choices, or model selection. This is the main leakage rule.

### Inputs

Required Stage 2 columns:

- `RBallID`
- `League`
- `Date`
- `Season`
- `HomeTeam`
- `AwayTeam`
- `HomeGoals`
- `AwayGoals`
- `Result`
- `ResultCode`

Stage 2 already supplies these from Stage 1 Football-Data match rows and the result encoding.

### Strength Estimates

The artifact must expose team-level estimates, not just predictions.

Minimum required columns for team strengths:

- `League`
- `Team`
- `matches`
- `goals_for`
- `goals_against`
- `attack_strength`
- `defence_strength`
- `attack_strength_shrunk`
- `defence_strength_shrunk`

Use a documented convention, for example:

- `attack_strength > 1.0` means the team scores above league average.
- `defence_strength < 1.0` means the team concedes fewer goals than league average.

For new or low-sample teams, shrink toward `1.0` rather than allowing unstable extremes.

### Expected Goals

For each holdout match, produce:

- `Lambda_Home`
- `Lambda_Away`

A valid first-version formula is:

```text
lambda_home = league_home_goal_baseline * home_attack_strength * away_defence_strength
lambda_away = league_away_goal_baseline * away_attack_strength * home_defence_strength
```

Clip lambdas to a small positive floor, for example `0.05`, and validate that all lambdas are finite and strictly positive.

### Probability Derivation

For each holdout match:

1. Build independent Poisson PMFs for home and away goals from `0..max_goals`.
2. Compute the scoreline grid.
3. Aggregate scoreline cells:
   - `P_Home = sum P(home_goals > away_goals)`
   - `P_Draw = sum P(home_goals == away_goals)`
   - `P_Away = sum P(home_goals < away_goals)`
4. Normalize the three probabilities to sum to `1.0 +/- 0.001`.
5. Convert probabilities to decimal odds with `1 / probability`.

### Output Paths

Recommended separate outputs:

- `data/output/attack_defence_xg_model_odds.parquet`
- `data/output/attack_defence_xg_value_bets.parquet`
- `data/model_artifacts/attack_defence_xg/metrics.json`
- `data/model_artifacts/attack_defence_xg/holdout_predictions.parquet`
- `data/model_artifacts/attack_defence_xg/team_strengths.parquet`
- `data/model_artifacts/attack_defence_xg/model_benchmarks.json`
- `data/model_artifacts/attack_defence_xg/model_diagnostics.json`
- `data/model_artifacts/attack_defence_xg/value_bet_roi.json`

The model-odds file must follow the Stage 4-compatible schema:

- `RBallID`
- `HomeTeam`
- `AwayTeam`
- `Date`
- `Season`
- `Result`
- `P_Home`
- `P_Draw`
- `P_Away`
- `ModelOdds_Home`
- `ModelOdds_Draw`
- `ModelOdds_Away`

Optional but recommended columns in `holdout_predictions.parquet`:

- `League`
- `Lambda_Home`
- `Lambda_Away`
- `HomeAttackStrength`
- `AwayAttackStrength`
- `HomeDefenceStrength`
- `AwayDefenceStrength`

---

## Benchmark Contract

Compare against calibrated XGBoost on the same 2019-20 holdout rows.

Required probability metrics:

- `holdout_log_loss`
- `holdout_brier_score`
- `holdout_accuracy`
- `holdout_f1_home`
- `holdout_f1_draw`
- `holdout_f1_away`
- predicted class counts for H/D/A

Required calibration diagnostics:

- outcome summary by H/D/A
- calibration by outcome/probability bucket using the same bucket logic as `pipeline/model_diagnostics.py`
- worst calibration bucket among buckets with enough samples

Required value-bet ROI metrics:

- run Stage 5 logic against `data/output/attack_defence_xg_model_odds.parquet`
- write value bets to `data/output/attack_defence_xg_value_bets.parquet`
- report holdout-only:
  - total bets
  - wins
  - losses
  - hit rate
  - flat-stake profit where stake is 1 unit
  - flat-stake ROI
  - average bookmaker odds
  - average edge

Use the existing Stage 5 risk policy and 10% edge threshold unless the task explicitly documents a different comparison. Do not tune thresholds on the holdout.

---

## Acceptance Criteria

- [ ] New task implementation trains attack/defence strengths using only pre-`2019-20` rows.
- [ ] Team strength artifact is written and is human-readable enough to inspect top/bottom attacks and defences.
- [ ] Holdout predictions contain finite positive `Lambda_Home` and `Lambda_Away`.
- [ ] 1X2 probabilities are derived from a scoreline grid, are finite/non-negative, and sum to `1.0 +/- 0.001`.
- [ ] Decimal odds are reciprocal probabilities and contain no null or infinite values.
- [ ] Benchmark metrics compare `attack_defence_xg` against the existing calibrated XGBoost rows on the exact same holdout.
- [ ] Calibration diagnostics are written under `data/model_artifacts/attack_defence_xg/` without overwriting Stage 3 diagnostics.
- [ ] Stage 5 value-bet comparison is run using the benchmark odds path and writes only benchmark-specific value-bet outputs.
- [ ] ROI artifact reports holdout-only flat-stake betting performance.
- [ ] Focused tests cover no-holdout-leakage split behavior, strength shrinkage defaults for low-sample/new teams, scoreline aggregation, probability/odds validation, and ROI calculation.
- [ ] `docs/ai/TASKS.md` is updated when implementation is complete.

---

## Commands For Coder-Tester

Assume the `.venv` is active before running these.

Narrow tests after implementation:

```bash
.venv/bin/python -m unittest tests.test_attack_defence_xg_benchmark
```

Useful integration checks:

```bash
.venv/bin/python pipeline/attack_defence_xg_benchmark.py
.venv/bin/python pipeline/stage5_compare.py \
  --model-odds-path data/output/attack_defence_xg_model_odds.parquet \
  --output-path data/output/attack_defence_xg_value_bets.parquet \
  --dashboard-json-path data/model_artifacts/attack_defence_xg/value_bets.json
.venv/bin/python pipeline/model_diagnostics.py \
  --holdout-predictions-path data/model_artifacts/attack_defence_xg/holdout_predictions.parquet \
  --value-bets-path data/output/attack_defence_xg_value_bets.parquet \
  --output-path data/model_artifacts/attack_defence_xg/model_diagnostics.json
```

If the implementation reuses existing helper functions touched by Stage 3, Stage 4, or Stage 5, also run:

```bash
.venv/bin/python -m unittest tests.test_stage3_train tests.test_stage4_odds_gen tests.test_stage5_compare tests.test_model_diagnostics
```

---

## Metrics To Report

When coder-tester finishes, report this table:

| Metric | Calibrated XGBoost | Attack/Defence xG | Delta |
|---|---:|---:|---:|
| `holdout_log_loss` |  |  |  |
| `holdout_brier_score` |  |  |  |
| `holdout_accuracy` |  |  |  |
| `holdout_f1_draw` |  |  |  |
| `holdout_predicted_draw` |  |  |  |
| `holdout_value_bets` |  |  |  |
| `holdout_flat_stake_roi` |  |  |  |

Also report:

- Top 5 attack strengths by league.
- Top 5 defence strengths by league.
- Worst calibration bucket.
- Whether the model is only a benchmark or deserves a later promotion task.

Do not promote this model automatically, even if it beats XGBoost. Promotion requires a separate reviewer-approved task because Stage 4 currently loads the MLflow Production model.

---

## Interview Q&A

**Q: Why benchmark against an attack/defence model after XGBoost?**  
A: "XGBoost is a strong tabular baseline, but football scores come from goal counts. The attack/defence model gives me interpretable expected goals and lets me compare a domain model against a flexible ML model on the same holdout."

**Q: How did you prevent leakage?**  
A: "All team strengths, league baselines, home advantage, and shrinkage priors were fit only on seasons before 2019-20. The 2019-20 rows were used only for final probability, calibration, and ROI evaluation."

**Q: Why use log loss and Brier score before ROI?**  
A: "ROI can be noisy because it depends on a smaller set of flagged bets and the bookmaker odds available. Log loss and Brier score evaluate whether the probabilities themselves are good. Betting performance only makes sense after that."
