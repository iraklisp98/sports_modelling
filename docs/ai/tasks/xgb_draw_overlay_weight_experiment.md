# XGBoost Draw Overlay Weight Experiment

**Status:** Ready  
**Role:** `coder-tester`  
**Scope:** Replace the hard-coded draw-overlay blend weight with a weight selected on pre-holdout data, then report holdout metric deltas.

---

## Goal

Test whether the existing binary draw overlay helps probability quality when its blend weight is selected from evidence instead of fixed at `0.20`.

This is the next narrow experiment because the latest artifacts show:

- Base calibrated XGBoost beats the draw overlay on log loss by a tiny margin: `0.997354` vs `0.997383`.
- The overlay improves draw F1 only slightly: `0.041522` to `0.042105`.
- The Poisson benchmark does not beat XGBoost and predicts zero draws as top class.
- The rejected shots-on-target features worsened log loss and Brier score.

The failure mode is therefore not "add another random feature." It is: the draw-specific model may contain signal, but the current fixed blend weight is not justified by validation evidence.

---

## Implementation Contract

- Keep the current Stage 2 feature contract unchanged.
- Keep the holdout season unchanged: `2019-20` must remain final evaluation only.
- Keep the existing multiclass XGBoost model, calibrated multiclass wrapper, binary draw model, and `blend_draw_probability` formula.
- Add a small helper that evaluates candidate blend weights on pre-holdout data using multiclass log loss.
- Candidate weights should include `0.0` so the experiment can choose "no overlay" if that is best.
- A reasonable first grid is:

```python
[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
```

- Select the lowest-log-loss candidate using pre-holdout predictions only.
- Use the selected weight when constructing the production `DrawOverlayClassifier`.
- Log and write the selected weight, for example as `draw_overlay_weight`, and include a small weight-selection artifact if practical.
- Do not change Optuna search space, class weights, feature columns, Poisson code, Stage 4, or Stage 5 in this task.

The cleanest implementation is to compute candidate probabilities from:

1. calibrated multiclass probabilities,
2. calibrated binary draw probabilities,
3. `blend_draw_probability(..., blend_weight=candidate)`,

then score each candidate with `sklearn.metrics.log_loss` against the same pre-holdout labels used for selection. The 2019-20 holdout must be scored only after the weight has been chosen.

---

## Acceptance Criteria

- Focused unit tests cover:
  - the weight selector includes `0.0` as a valid candidate,
  - the selector chooses the candidate with the lowest validation log loss,
  - ties are deterministic, preferably choosing the smaller weight,
  - invalid candidate weights outside `[0.0, 1.0]` are rejected.
- Focused tests pass:

```bash
.venv/bin/python -m unittest tests.test_stage3_train
```

- The fast Stage 3 smoke run completes:

```bash
MLFLOW_TRACKING_URI=file:/tmp/mlruns .venv/bin/python pipeline/stage3_train.py --trials 0 --tracking-uri file:/tmp/mlruns
```

- `data/model_artifacts/stage3/metrics.json` and `model_benchmarks.json` are refreshed from the run.
- `docs/ai/TASKS.md` status is updated after evaluation.

---

## Metrics To Report

Report these before/after metrics against the current fixed-weight XGBoost draw-overlay run:

| Metric | Current fixed `0.20` overlay |
|---|---:|
| `holdout_log_loss` | `0.9973826083` |
| `holdout_brier_score` | `0.5957660475` |
| `holdout_accuracy` | `0.5206929740` |
| `holdout_f1_draw` | `0.0421052632` |
| `holdout_predicted_draw` | `18` |

Also report against the same-run base calibrated XGBoost row from `model_benchmarks.json`:

| Metric | Current base calibrated XGBoost |
|---|---:|
| `log_loss` | `0.997354` |
| `brier_score` | `0.595771` |
| `accuracy` | `0.518768` |
| `f1_draw` | `0.041522` |
| `predicted_draw` | `22` |

The experiment is accepted only if selected-weight overlay holdout `log_loss` is lower than both:

- the current fixed `0.20` overlay log loss: `0.9973826083`
- the current base calibrated XGBoost benchmark log loss: `0.997354`

Accuracy and draw F1 are explanatory secondary metrics. They do not override worse log loss.

---

## If The Experiment Fails

If the selected-weight overlay does not beat both log-loss baselines, do not keep it as the Production behavior. Record the result here and leave the model contract on the better probability source, likely base calibrated XGBoost or a zero-weight overlay.

The hiring signal is the discipline: a model component that sounds plausible still has to win on a chronological validation and holdout protocol before it earns its place in the pipeline.
