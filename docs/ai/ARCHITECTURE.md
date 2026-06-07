# Architecture

## Overview

This repository is a batch ML engineering project for football odds research. It downloads historical Football-Data.co.uk match and bookmaker-odds CSVs, turns them into typed Parquet datasets, engineers leakage-safe pre-match features, trains and tracks model candidates, compares model-implied odds against bookmaker odds, and publishes the results through a static dashboard.

The key architectural choice is file-based stage isolation. Each stage reads a documented input artifact and writes a documented output artifact. That makes the pipeline reproducible, debuggable, and easy to extend without hiding logic inside notebooks.

## Final Model Story

The completed project should be presented as a **sports betting model research pipeline**, not as a guaranteed profitable betting system.

Final candidate:

- Model: market-aware XGBoost value strategy.
- Training policy: expanding annual walk-forward retraining.
- League policy: all five leagues retained.
- Bet scope: home and away wins only; draws remain probability outputs but are not value-bet actions.
- Production interpretation: retrain on all completed matches available up to the current date, then score the next fixture batch. The season-level walk-forward experiment is the historical proxy for that production policy.

Validation summary:

| Experiment | Result | Decision |
|---|---:|---|
| Static pre-2019 split | Useful but too narrow | Baseline only |
| Frozen XGBoost simulator | +0.19% ROI | Comparison only |
| Full forward Poisson benchmark | -6.92% ROI | Benchmark only |
| Full forward mispricing model | -16.52% ROI | Rejected as lead strategy |
| Recent-window walk-forward | -3.29% ROI | Rejected training policy |
| Expanding annual walk-forward | +5.47% ROI, 3/5 positive folds | Final dashboard default |
| League subset audit | all_five best ROI | Keep full five-league training pool |

## System Flow

```text
Football-Data.co.uk CSVs
        |
        v
Stage 1: Ingest & Clean
        -> data/processed/{ENG,SPA,FRA,GER,ITA}.parquet
        |
        v
Stage 2: Feature Engineering
        -> data/features/{ENG,SPA,FRA,GER,ITA}_features.parquet
        |
        v
Stage 3: XGBoost Training + MLflow
        -> mlruns/ and data/model_artifacts/stage3/
        |
        +--> Benchmark/experiment layers
        |    -> poisson_goal_model.py
        |    -> mispricing_model.py
        |    -> market_baseline_diagnostics.py
        |    -> training_window_experiments.py
        |
        v
Stage 4: Odds Generation
        -> data/output/model_odds.parquet
        |
        v
Stage 5: Odds Comparison
        -> data/output/value_bets.parquet
        |
        v
Dashboard Export
        -> dashboard/data/*.json
        |
        v
Static Dashboard
        -> dashboard/index.html served directly or by nginx
```

## Stage Contracts

### Stage 1 - Ingest & Clean

Script: `pipeline/stage1_ingest.py`

Inputs:

- Cached or downloaded Football-Data CSVs for `ENG`, `SPA`, `FRA`, `GER`, `ITA`.
- Default season range: `2010-11` through `2025-26`.

Outputs:

- `data/processed/{league}.parquet`
- Match-level schema with `RBallID`, teams, date, season, goals, result, corners, shots, fouls, offsides, and bookmaker odds where available.

Purpose:

- Normalize raw CSV differences into one contract.
- Deduplicate fixtures and assign deterministic IDs.
- Keep raw-source ingestion separate from modelling.

### Stage 2 - Feature Engineering

Script: `pipeline/stage2_features.py`

Outputs:

- `data/features/{league}_features.parquet`

Feature families:

- Pre-match ELO.
- Rolling 5-match goals, goals conceded, points, corners, shots on target, fouls, offsides.
- Venue-specific form.
- Rest days and fixture congestion.
- Season-to-date win rates.
- League indicators.

Leakage rule:

Every rolling feature is shifted before the current match. The row describes what was known before kick-off, not what happened in the match.

### Stage 3 - Model Training

Script: `pipeline/stage3_train.py`

Tools:

- XGBoost for tabular classification.
- Optuna for hyperparameter search.
- MLflow for experiment tracking and model artifacts.

Outputs:

- `mlruns/`
- `data/model_artifacts/stage3/metrics.json`
- `data/model_artifacts/stage3/holdout_predictions.parquet`

Main point:

The pipeline tracks probability quality, not just accuracy. Log loss and Brier score matter because betting decisions depend on calibrated probabilities.

### Benchmark And Diagnostic Layers

These scripts are not decorative. They explain why the final model was selected.

| Script | Role |
|---|---|
| `pipeline/poisson_goal_model.py` | Football-specific expected-goals benchmark converted into 1X2 odds |
| `pipeline/mispricing_model.py` | Second-stage market-disagreement bet selector, retained as a rejected benchmark |
| `pipeline/market_baseline_diagnostics.py` | Compares model probabilities against normalized bookmaker-implied probabilities |
| `pipeline/model_diagnostics.py` | Calibration buckets, odds buckets, actual result rates, flat-stake ROI |
| `pipeline/compare_value_bet_models.py` | Strategy-level comparison across XGBoost, Poisson, and mispricing outputs |
| `pipeline/training_window_experiments.py` | Split policy, expanding walk-forward, recent-window, and league-subset experiments |

The most important artifact from this layer is:

- `data/model_artifacts/expanding_walk_forward_training_window.json`

That artifact is what turns the model story from "one backtest looked good" into "we tested a production-like retraining policy across seasons".

### Stage 4 - Odds Generation

Script: `pipeline/stage4_odds_gen.py`

Outputs:

- `data/output/model_odds.parquet`

Contract:

- `P_Home`, `P_Draw`, `P_Away`
- `ModelOdds_Home`, `ModelOdds_Draw`, `ModelOdds_Away`

Probabilities are validated to sum to 1 within tolerance. Decimal odds are derived as `1 / probability` after numerical hygiene.

### Stage 5 - Odds Comparison

Script: `pipeline/stage5_compare.py`

Outputs:

- `data/output/value_bets.parquet`
- strategy-specific benchmark value-bet files when generated.

Value-bet rule:

```text
edge = (best_bookmaker_odds / model_implied_odds) - 1
value bet = edge >= 10%
```

Current policy:

- Only home and away outcomes are surfaced as actionable bets.
- Draws are excluded from value-bet selection because they were unstable and less frequent, but their probabilities remain visible in model odds.
- Sanity filters remove extreme long-shot odds and unrealistic edge outliers.

### Dashboard Export

Script: `pipeline/export_dashboard_data.py`

Outputs:

- `dashboard/data/league_analytics.json`
- `dashboard/data/backtest.json`
- `dashboard/data/value_bets.json`
- `dashboard/data/simulator.json`
- `dashboard/data/strategy_comparison.json`
- `dashboard/data/training_policy.json`
- `dashboard/data/project_summary.json`
- `dashboard/data/diagnostics.json`

The export stage is the dashboard boundary. The browser does not import Python, load Parquet, call MLflow, or run models.

### Stage 6 - Dashboard

Directory: `dashboard/`

Tabs:

- Project Story: the executive validation narrative.
- League Analytics: league and team summaries.
- Backtest: model metrics and MLflow comparison.
- Odds Inspector: filterable value-bet table and modal.
- Calibration: odds bucket diagnostics.
- Simulator: bankroll replay, strategy comparison, and expanding retraining robustness.

The dashboard is deliberately static. For a portfolio project this is stronger than a backend that adds moving parts without adding value.

### Stage 7 - Docker

Directory: `docker/`

Services:

- `pipeline`: one-shot Python batch container.
- `dashboard`: nginx static server.

Run:

```bash
docker compose up --build
```

Dashboard URL:

```text
http://localhost:8080
```

MLflow file-store handling:

The Docker setup keeps MLflow inside the container path instead of bind-mounting a host `mlruns/` directory, because file-store artifact metadata can contain absolute paths that break across host/container boundaries.

## Data Contracts

### Match Feature Contract

Important columns flowing from Stage 2 into model training:

| Column | Meaning |
|---|---|
| `RBallID` | Stable match identifier |
| `Date`, `Season`, `League` | Time and competition identity |
| `HomeTeam`, `AwayTeam` | Fixture teams |
| `Result`, `ResultCode` | Target label |
| `HomeElo`, `AwayElo`, `EloDiff` | Pre-match team strength |
| `*_Last5` | Rolling recent form, shifted to avoid leakage |
| `HomeRestDays`, `AwayRestDays` | Recovery proxy |
| `HomeMatchesLast14Days`, `AwayMatchesLast14Days` | Congestion proxy |
| `League_*` | League indicators |

### Value Bet Contract

| Column | Meaning |
|---|---|
| `RBallID` | Match identifier |
| `League`, `Season`, `Date` | Context |
| `HomeTeam`, `AwayTeam` | Fixture |
| `Result` | Actual result |
| `Outcome` | Flagged outcome, `H` or `A` |
| `ModelOdds` | Model-implied decimal odds |
| `BestBookOdds` | Best available bookmaker odds |
| `Edge` | `(BestBookOdds / ModelOdds) - 1` |
| `BestBookmaker` | Book offering the best price |

## Testing Strategy

The test suite protects contracts rather than only checking implementation details.

Coverage areas:

- CSV ingestion and season handling.
- Leakage-safe feature generation.
- Model feature lists and market feature construction.
- Stage 3 training utilities.
- Odds generation and comparison rules.
- Calibration and market diagnostics.
- Poisson, mispricing, and value-bet comparison artifacts.
- Training-window experiments.
- Dashboard JSON export contracts.
- Static dashboard wiring.
- Docker configuration.

Final verification command:

```bash
python -m unittest discover tests
```

Latest result: 153 tests OK, 1 skipped. `pytest` can run the same tests if installed, but the repo does not require it for the current suite.

## Why This Is Portfolio-Ready

A hiring manager should be able to see these signals quickly:

- The notebooks are no longer the product; the product is a staged pipeline.
- Parquet contracts separate data engineering from modelling.
- MLflow makes experiments reproducible.
- Calibration and market-baseline diagnostics prevent false confidence.
- Benchmarks are kept even when they lose.
- Walk-forward validation tests production-like retraining behavior.
- The dashboard tells the model story honestly instead of hiding weak results.
- Docker gives a one-command runtime path.

## Phase 2 Evolution

The next realistic extension is live ingestion and scheduled retraining.

Phase 1:

```text
Football-Data historical CSVs -> local batch pipeline -> static dashboard
```

Phase 2:

```text
Live football/odds API -> scheduled ingestion -> database or object storage -> retraining job -> static or API-backed dashboard
```

The architecture is ready for that because Stage 2 and later depend on schemas, not on where Stage 1 got the data.
