# Stage 5.5 — Model Diagnostics

**Status:** Complete
**Script:** `pipeline/model_diagnostics.py`
**Input:** Stage 3 holdout predictions + Stage 5 value bets
**Output:** `data/model_artifacts/stage3/model_diagnostics.json`

---

## What

Build a calibration report before changing the model. The diagnostic expands each holdout match into one row per outcome (`H`, `D`, `A`), buckets predicted probabilities into 0.1-wide bands, and compares predicted probability against empirical hit rate.

It also summarizes value bets by outcome and model-probability bucket on the holdout season only.

---

## Why

Accuracy tells us how often the top class is right. Betting needs calibrated probabilities. If the model says an outcome has a 60% chance, that outcome should happen close to 60% of the time across similar cases.

This script tells us where the probabilities are wrong before we add features, tune hyperparameters, or change betting thresholds.

---

## Data Contract

Top-level JSON keys:

- `holdout_seasons`
- `probability_bins`
- `outcome_summary`
- `calibration_by_outcome_bucket`
- `value_bets_by_outcome_bucket`
- `worst_calibration_bucket`

Each calibration row contains:

| Field | Meaning |
|---|---|
| `outcome` | `H`, `D`, or `A` |
| `bucket` | Probability range, e.g. `0.6-0.7` |
| `count` | Number of one-vs-rest observations in the bucket |
| `avg_predicted_probability` | Mean model probability in that bucket |
| `empirical_rate` | Actual hit rate for that outcome in the bucket |
| `calibration_error` | `empirical_rate - avg_predicted_probability` |
| `abs_calibration_error` | Absolute calibration error |

---

## How To Run

```bash
python pipeline/model_diagnostics.py
```

The full pipeline now runs it after Stage 5 and before dashboard export.

---

## Acceptance Criteria

- [x] Validates holdout prediction columns and probability rows
- [x] Validates Stage 5 value-bet columns
- [x] Writes deterministic JSON diagnostics
- [x] Focused tests cover buckets, calibration math, empty value bets, and invalid probabilities

---

## Interview Q&A

**Q: Why add diagnostics before improving the model?**
A: "Because betting models fail through probability calibration, not only classification accuracy. The diagnostic tells me which outcome and probability range is miscalibrated, so the next model change targets an observed failure instead of guessing."
