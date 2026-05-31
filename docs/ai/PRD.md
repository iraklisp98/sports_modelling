# PRD: Sports Betting Odds Arbitrage Pipeline

**Author:** Iraklis Papageorgiou  
**Status:** In Progress  
**Phase:** 1 of 2  
**Last Updated:** 2026-05-31

---

## Problem Statement

Sportsbooks set odds using proprietary models with inherent inefficiencies. When a data-driven model assigns higher win probability to an outcome than the implied probability in the sportsbook's offered odds, there is a quantifiable edge. This project builds the full system to detect, validate, and track those edges — from raw historical data through to a live dashboard surfacing actionable value bets.

The core arbitrage condition is:

```
sportsbook_odds >= 1.10 × model_implied_odds
```

A 10% threshold filters out noise and model uncertainty, retaining only high-confidence discrepancies.

---

## Goals

- Build a reproducible, containerised end-to-end pipeline from raw CSV data to odds comparison output
- Train and track a match outcome prediction model that produces calibrated win probabilities
- Use Football-Data.co.uk historical odds CSVs for bookmaker comparison
- Surface results through a dashboard with four views: league analytics, backtest performance, odds inspection, and a betting simulator
- Demonstrate production-grade engineering practices: modular pipeline stages, experiment tracking with MLflow, and Docker deployment

## Non-Goals (Phase 1)

- Live data ingestion from an external API (Phase 2)
- Automated bet placement
- Real-money tracking
- Leagues beyond Premier League, La Liga, and Ligue 1

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Raw CSVs (ENG / FRA / SPA)  →  PySpark cleaning & merging     │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                     FEATURE ENGINEERING                         │
│  ELO ratings · rolling form · home/away splits · season index   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                      MODEL TRAINING                             │
│  XGBoost classifier · experiment tracking via MLflow            │
│  Outputs: P(Home) · P(Draw) · P(Away) → implied odds            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                     ODDS COMPARISON                             │
│  Football-Data CSVs → historical bookmaker odds               │
│  Flag bets where bookmaker_odds >= 1.10 × model_odds            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                       DASHBOARD                                 │
│  Static HTML/CSS/JS + native SVG/CSS charts                                  │
│  Tabs: League Analytics | Backtest | Odds | Betting Simulator    │
└─────────────────────────────────────────────────────────────────┘
```

All stages are containerised with Docker. Each stage is independently runnable and writes its output to a well-defined location so any stage can be re-run in isolation.

---

## Pipeline Stages

### Stage 1 — Data Ingestion & Cleaning

**Input:** Raw event-level CSVs in `data/ENG/`, `data/FRA/`, `data/SPA/`  
**Output:** Parquet files in `data/processed/` (one per league)

**Responsibilities:**
- Merge per-match CSVs into a single dataset per league using PySpark
- Parse and standardise timestamps; assign season labels (Aug–Jul)
- Pivot event-level rows into match-level feature columns (goals, corners, shots, fouls, etc.)
- Validate schema and emit a data quality report (null rates, row counts, date ranges)
- Write Parquet output for efficient downstream reads

**Key engineering decisions:**
- PySpark handles the ~180MB combined dataset and will scale to larger datasets in Phase 2 without code changes
- Parquet over CSV: columnar format cuts downstream read time and enforces schema

---

### Stage 2 — Feature Engineering

**Input:** Processed Parquet files  
**Output:** Feature-enriched dataset ready for training

**Features produced:**
| Feature | Description |
|---|---|
| `HomeElo`, `AwayElo` | Dynamic ELO rating at kick-off |
| `EloDiff` | Home minus Away ELO |
| `HomeGoals_Last5` | Rolling 5-match goals scored (home) |
| `AwayGoals_Last5` | Rolling 5-match goals scored (away) |
| `HomeCorners_Last5` | Rolling 5-match corners (home) |
| `AwayCorners_Last5` | Rolling 5-match corners (away) |
| `HomePoints_Last5` | Rolling 5-match points (home) |
| `AwayPoints_Last5` | Rolling 5-match points (away) |
| `HomeWinRate_Season` | Season win rate at time of match (home) |
| `AwayWinRate_Season` | Season win rate at time of match (away) |
| `Result` | Target: H / D / A |

**ELO implementation:**
- Starting ELO: 1500 for all teams
- Home advantage: +80 points
- K-factor: 30
- Season regression: 20% regression to league mean between seasons
- Newly promoted teams: initialised at league mean − 100

---

### Stage 3 — Model Training & Experiment Tracking

**Input:** Feature dataset from Stage 2  
**Output:** Serialised model artifact + MLflow experiment run

**Model:** XGBoost multi-class classifier (H / D / A)

**Train/test split:** Seasons 2017–2018 and 2018–2019 for training; available 2019–2020 rows are held out for evaluation

**Evaluation metrics:**
- Log Loss (primary — measures probability calibration)
- Multiclass Brier Score (secondary — measures probability sharpness across H/D/A)
- Accuracy
- F1 per class (Home / Draw / Away)
- Holdout prediction export for later ROI simulation after Stage 5 odds comparison

**MLflow tracking:**
- Each training run logs: hyperparameters, all metrics above, feature importance chart, confusion matrix
- Model artifact registered to MLflow Model Registry
- Best run promoted to "Production" stage

**Hyperparameter tuning:** Optuna with 50 trials, optimising for log loss on a 3-fold time-series cross-validation

---

### Stage 4 — Odds Generation

**Input:** Trained model + feature dataset (or live feature snapshot)  
**Output:** DataFrame with columns `[RBallID, HomeTeam, AwayTeam, Date, Season, Result, P_Home, P_Draw, P_Away, ModelOdds_Home, ModelOdds_Draw, ModelOdds_Away]`

**Process:**
1. Load the Production model from MLflow registry
2. Run inference on the feature dataset
3. Convert probabilities to decimal odds: `odds = 1 / probability`
4. Validate: probabilities sum to 1.0 ± 0.001 per match

---

### Stage 5 — Odds Comparison & Value Bet Flagging

**Input:** Model odds (Stage 4) + historical bookmaker odds from Football-Data.co.uk CSVs  
**Output:** Flagged value bets table

**Football-Data.co.uk integration:**
- Source pattern: `https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv`
- Seasons: 2017-18, 2018-19, 2019-20
- League codes: `E0` Premier League, `SP1` La Liga, `F1` Ligue 1
- Bookmaker odds columns: `B365*`, `PS*`, `WH*`, and other supported 1X2 prefixes
- Match on normalised home team, away team, and match date

**Flagging logic:**
```python
value_bet = best_bookmaker_odds >= 1.10 * model_odds
```

Where `best_bookmaker_odds` is the maximum offered across available Football-Data bookmaker columns. Because decimal model odds are fair odds, the bookmaker price must be higher than the model fair price to be value.

**Output schema:**
| Column | Description |
|---|---|
| `RBallID` | Unique match identifier |
| `HomeTeam`, `AwayTeam` | Team names |
| `Date` | Match date |
| `Season` | Season label |
| `League` | ENG / SPA / FRA from the matched Football-Data source |
| `Outcome` | H / D / A |
| `ModelOdds` | Model-implied decimal odds |
| `BestBookOdds` | Best available bookmaker odds |
| `Edge` | `(BestBookOdds / ModelOdds) - 1` as % |
| `ValueBet` | Boolean flag |
| `BestBookmaker` | Source of best odds |

---

### Stage 6 — Dashboard

**Stack:** Pure HTML / CSS / JavaScript (native SVG/CSS visualisations)  
**Data contract:** Pipeline stages write pre-computed JSON files to `dashboard/data/`. The dashboard reads them at load time — no backend server required.  
**Deployment:** Docker container running nginx, served on port 8080

**Four tabs:**

#### Tab 1 — League Analytics
- League selector (PL / La Liga / Ligue 1)
- Season selector
- KPI cards: avg goals/match, home win %, draw %, away win %
- Time series: goals, corners, and shots per month
- Team leaderboard: points, GD, goals scored/conceded
- Home vs Away performance split (stacked bar per team)

#### Tab 2 — Backtest Performance
- Model performance table: Log Loss, Brier Score, Accuracy, F1 per class
- Equity curve: cumulative P&L on flagged value bets over the holdout period
- Value bet hit rate over time (rolling 20-bet window)
- Confusion matrix heatmap
- MLflow run comparison table (top-5 runs by log loss)

#### Tab 3 — Odds Inspector
- Date-range filter + league filter
- Table of all flagged value bets with edge % highlighted green
- Sortable by edge, date, or league
- Match detail modal: available Stage 5 odds breakdown plus model probability bar chart
- Export to CSV button

#### Tab 4 — Betting Simulator
- Fixed stake input (default: $10 per bet) with a slider to adjust ($1–$100)
- Simulates placing that stake on every green-flagged value bet in the backtest period
- **KPI cards:**
  - Starting bankroll (stake × number of bets)
  - Ending bankroll
  - Total profit / loss ($)
  - ROI %
  - Hit rate (% of flagged bets that won)
  - Max drawdown ($)
- **Equity curve:** Bankroll over time, with drawdown shaded in red beneath the peak line
- **Bet-by-bet log table:** Date · Match · Outcome · Bet side · Odds · Stake · Return · Running bankroll
  - Winning rows highlighted green, losing rows red
- **Summary stats panel:**
  - Total bets placed
  - Wins / Losses / Draws (void)
  - Longest winning streak
  - Longest losing streak
  - Average odds on flagged bets
  - Average edge %

---

## Technical Stack

| Layer | Technology | Reason |
|---|---|---|
| Data processing | PySpark | Scales to Phase 2 volumes; columnar processing; industry standard for DE roles |
| Feature engineering | Python / Pandas | Familiar, fast for tabular feature logic |
| Model training | XGBoost + scikit-learn | Strong baseline for tabular data; well-understood |
| Hyperparameter tuning | Optuna | Modern, efficient; better than GridSearch for this scale |
| Experiment tracking | MLflow | Industry standard MLOps tool; signals ML engineering depth |
| Odds data | Football-Data.co.uk CSVs | Historical bookmaker odds are available for the backtest seasons |
| Dashboard | HTML / CSS / JavaScript with native SVG/CSS charts | No framework dependency; pipeline writes JSON, browser reads it — no backend needed |
| Dashboard server | nginx (Docker) | Serves static files; zero app-server complexity |
| Containerisation | Docker + Docker Compose | Reproducible environment; signals production awareness |
| Notebooks | Jupyter | EDA only — not part of the pipeline |

---

## Planned Project Structure

```
sports_modelling/
├── data/
│   ├── ENG/                     # Raw per-match CSVs
│   ├── FRA/
│   ├── SPA/
│   └── processed/               # Stage 1 Parquet output target
├── pipeline/
│   ├── stage1_ingest.py         # PySpark merge + clean
│   ├── stage2_features.py       # Feature engineering (planned)
│   ├── stage3_train.py          # Model training + MLflow logging (planned)
│   ├── stage4_odds_gen.py       # Probability → implied odds (planned)
│   ├── stage5_compare.py        # Football-Data odds comparison + value bet flagging
│   ├── export_dashboard_data.py # Dashboard JSON export (planned)
│   └── run_pipeline.py          # Orchestrates all stages end-to-end (planned)
├── dashboard/                   # Planned
│   ├── index.html               # Single-page app entry point
│   ├── css/
│   ├── js/
│   │   ├── main.js              # Static dashboard renderers
│   │   ├── simulator.js         # Betting simulator logic
│   │   └── main.js              # Tab routing + data loading
│   └── data/                    # Pre-computed JSON written by pipeline
│       ├── league_analytics.json
│       ├── backtest.json
│       ├── value_bets.json
│       └── simulator.json
├── analysis/
│   └── descriptive.ipynb        # EDA only — not part of pipeline
├── models/
│   └── training.ipynb           # Original model exploration
├── mlruns/                      # Planned MLflow tracking store
├── docker/                      # Planned
│   ├── Dockerfile.pipeline
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
├── config/                      # Planned
│   └── settings.yaml            # API keys, league config, thresholds
├── tests/                       # Planned
│   ├── test_features.py
│   ├── test_odds_comparison.py
│   └── test_pipeline_integration.py
├── docs/
│   └── ai/
│       ├── PRD.md               # This document
│       ├── ARCHITECTURE.md      # Architecture notes
│       ├── TASKS.md             # Canonical task status
│       └── tasks/               # Detailed stage guides
├── README.md
├── LICENSE                      # MIT
└── requirements.txt
```

---

## Success Criteria

| Metric | Target |
|---|---|
| Pipeline runs end-to-end from raw CSVs | Required |
| Log Loss on 2019–2020 holdout | Target < 0.95; current best checked Stage 3 run 0.9750 |
| Brier Score on 2019–2020 holdout | < 0.55 |
| Prediction accuracy | > 55%; current best checked Stage 3 run 55.83% |
| Value bet edge threshold enforced | Exactly 10% |
| MLflow experiment logged per training run | Required |
| Dashboard loads all four tabs without error | Required |
| Full environment reproducible via Docker | Required |
| Unit tests passing | > 80% coverage on pipeline logic |

---

## Timeline (2 Weeks)

| Day | Milestone |
|---|---|
| 1–2 | Stage 1: PySpark ingestion + Parquet output + data quality report |
| 3–4 | Stage 2: Feature engineering module + unit tests |
| 5–6 | Stage 3: MLflow integration + Optuna tuning + model registry |
| 7 | Stage 4 & 5: Odds generation + Football-Data odds comparison + value bet flagging |
| 8–10 | Stage 6: Dashboard — Tab 1 (analytics) + Tab 2 (backtest) |
| 11–12 | Stage 6: Dashboard — Tab 3 (odds inspector) + polish |
| 13 | Docker Compose setup + end-to-end integration test |
| 14 | README update, cleanup, final review |

---

## Phase 2 Preview (Out of Scope Now)

Phase 2 replaces the static CSV source with a live data ingestion layer:

- **Data source:** Football-Data API or StatsBomb open data
- **Storage:** PostgreSQL (structured match data) + possible S3 for raw event logs
- **Orchestration:** Prefect or Airflow DAG replacing `run_pipeline.py`
- **Scheduling:** Daily pipeline runs ahead of match days
- **Model retraining:** Triggered automatically when new season data reaches a threshold

This is where the Data Engineering profile becomes the primary signal.

---

## Risk & Mitigations

| Risk | Mitigation |
|---|---|
| Football-Data CSV schema changes | Validate required match columns and supported bookmaker odds prefixes before comparison |
| Team name mismatches between datasets | Normalise team names before exact date/team matching; add explicit mappings if mismatches remain |
| Model not calibrated (probabilities don't sum cleanly) | Add calibration step (Platt scaling or isotonic regression) post-training |
| PySpark overhead on local machine | Use local\[*\] Spark session; acceptable for this data size |
| MLflow server not running | Default to file-based tracking store (no server needed) |

---

## Disclaimer

This project is for educational and portfolio purposes. It does not constitute financial advice. Sports betting carries inherent risk and past model performance does not guarantee future returns.
