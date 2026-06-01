# XGBoost Two-Feature Improvement Experiment

**Status:** Rejected — log loss worsened in the `--trials 0` holdout run  
**Role:** `coder-tester`  
**Scope:** Add exactly two Stage 2 feature columns, retrain Stage 3, and report the metric delta versus the immediately previous XGBoost run.

---

## Goal

Test whether adding a compact shots-on-target form signal improves the current XGBoost holdout performance without widening the experiment into a general feature-engineering refactor.

## Add Exactly These Two Feature Columns

1. `HomeShotsOnTarget_Last5`
2. `AwayShotsOnTarget_Last5`

Do not add a shots-on-target difference column in this task. Keeping only the two side-specific columns makes the experiment easy to attribute: did recent chance creation for each team help the model beyond ELO, goals, corners, points, draw rates, and season win rates?

## Rationale

Shots on target is already present in the Stage 1 contract as `HomeShotsOnTarget` and `AwayShotsOnTarget`, but Stage 2 does not currently use it for training. It is a strong next candidate because it is closer to attacking chance quality than raw goals, less outcome-dependent than points, and still available before a future match when computed as a shifted rolling average.

For an interview, the important point is leakage control: these columns must describe each team's previous five matches, not the match being predicted.

## Implementation Contract

- Extend Stage 2 team history to carry a per-team `ShotsOnTarget` value.
- Compute `ShotsOnTarget_Last5` with the same `shift(1).rolling(window=5, min_periods=1).mean()` pattern used for goals, corners, points, and draw rate.
- Merge the values back as `HomeShotsOnTarget_Last5` and `AwayShotsOnTarget_Last5`.
- Add both columns to `STAGE2_FEATURE_COLUMNS` and `pipeline/model_features.py::BASE_FEATURE_COLUMNS`.
- Update Stage 2/3/4/Poisson tests that build synthetic `BASE_FEATURE_COLUMNS` rows.
- Do not change model hyperparameters, split policy, draw-overlay weight, or holdout season in this task.

## Acceptance Criteria

- Stage 2 outputs include `HomeShotsOnTarget_Last5` and `AwayShotsOnTarget_Last5` for all three leagues.
- The first historical match for a team has `0.0` for its shots-on-target rolling feature.
- A later match uses only previous matches for that team, proving the current row is excluded.
- Focused tests pass:

```bash
.venv/bin/python -m unittest tests.test_stage2_features tests.test_stage3_train tests.test_stage4_odds_gen tests.test_poisson_goal_model
```

- The full feature/train smoke run completes:

```bash
.venv/bin/python pipeline/stage2_features.py
MLFLOW_TRACKING_URI=file:/tmp/mlruns .venv/bin/python pipeline/stage3_train.py --trials 0 --tracking-uri file:/tmp/mlruns
```

- Report the before/after deltas for at least:
  - `holdout_log_loss`
  - `holdout_brier_score`
  - `holdout_accuracy`
  - `holdout_f1_draw`

## Improvement Rule

Treat the experiment as improved only if `holdout_log_loss` decreases versus the immediately previous `--trials 0` XGBoost baseline. Accuracy and draw F1 are secondary; they can explain the change, but they do not override worse log loss.

If log loss does not improve, keep the implementation only if Iraklis wants the extra model signal anyway; otherwise revert the production-code part and record the result as a rejected experiment.

## Result

The experiment was implemented and evaluated, then the production-code feature addition was reverted because it failed the log-loss rule.

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `holdout_log_loss` | 0.9973826083 | 0.9978337829 | +0.0004511746 |
| `holdout_brier_score` | 0.5957660475 | 0.5961088864 | +0.0003428389 |
| `holdout_accuracy` | 0.5206929740 | 0.5110683349 | -0.0096246391 |
| `holdout_f1_draw` | 0.0421052632 | 0.0342465753 | -0.0078586878 |

Decision: rejected. The two shots-on-target rolling features made probability quality slightly worse, so they should not be kept in the main XGBoost feature contract.

