# Architecture

## Overview

This project is a **batch ML pipeline** that processes historical football match data, trains a match outcome prediction model, compares model-implied odds against bookmaker odds, and surfaces the results through a static dashboard.

The build is organised into seven stages. Pipeline stages communicate exclusively through files — no stage calls another stage's code directly. This means any stage can be rerun, replaced, or debugged in isolation. Stage 6 is the static dashboard and Stage 7 is Docker packaging.

---

## System Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                │
│                                                                      │
│   data/ENG/*.csv          data/FRA/*.csv          data/SPA/*.csv     │
│   (event-level,           (event-level,           (event-level,      │
│    ~950 files)             ~940 files)             ~927 files)        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Ingest & Clean                  pipeline/stage1_ingest.py │
│                                                                      │
│  Tool: PySpark                                                       │
│  • Merge per-match CSVs into one dataset per league                  │
│  • Filter noise (Minute = 0 rows)                                    │
│  • Parse timestamps, assign season labels                            │
│  • Pivot event rows → one row per match                              │
│  • Validate schema, emit data quality report                         │
│                                                                      │
│  Output: data/processed/{ENG,FRA,SPA}.parquet                        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  Parquet (match-level, fixed schema)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Feature Engineering            pipeline/stage2_features.py│
│                                                                      │
│  Tool: pandas                                                        │
│  • Compute dynamic ELO ratings per team (updated after each match)   │
│  • Rolling 5-match form features (goals, corners, points)            │
│  • Season win rates (no data leakage — always computed before match) │
│  • Encode target: H=0, D=1, A=2                                      │
│                                                                      │
│  Output: data/features/{ENG,FRA,SPA}_features.parquet                │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  Parquet (feature-enriched)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Model Training                   pipeline/stage3_train.py │
│                                                                      │
│  Tools: XGBoost · Optuna · MLflow                                    │
│  • Combine all three leagues into one training dataset               │
│  • Time-series cross-validation (train on 2017–19, test on 2019–20) │
│  • Optuna tunes hyperparameters (50 trials, minimise log loss)       │
│  • Final model trained, evaluated (log loss, Brier, accuracy, F1)   │
│  • All params, metrics, and artifacts logged to MLflow               │
│  • Best model registered in MLflow Model Registry → "Production"    │
│                                                                      │
│  Output: mlruns/ (experiment tracking + model artifact)              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  MLflow Model Registry
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — Odds Generation                pipeline/stage4_odds_gen.py│
│                                                                      │
│  Tools: MLflow (load) · pandas                                       │
│  • Load Production model from MLflow registry                        │
│  • Run inference on feature dataset                                  │
│  • Convert P(H), P(D), P(A) → decimal odds (1/p)                    │
│  • Validate: probabilities sum to 1.0 ± 0.001                        │
│                                                                      │
│  Output: data/output/model_odds.parquet                              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  Parquet (model-implied odds)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — Odds Comparison                pipeline/stage5_compare.py │
│                                                                      │
│  Tools: The Odds API · rapidfuzz · requests                          │
│  • Fetch historical bookmaker odds per match (cached to disk)        │
│  • Fuzzy-match team names between dataset and API                    │
│  • For each outcome: edge = (model_odds / best_book_odds) - 1        │
│  • Flag value bets where edge >= 10%                                 │
│                                                                      │
│  Output: data/output/value_bets.parquet                              │
│          dashboard/data/value_bets.json                              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  JSON
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  EXPORT                     pipeline/export_dashboard_data.py        │
│                                                                      │
│  • Reads all Parquet outputs and mlruns/                             │
│  • Writes four JSON files to dashboard/data/                         │
│                                                                      │
│  Output: dashboard/data/league_analytics.json                        │
│          dashboard/data/backtest.json                                │
│          dashboard/data/value_bets.json   (already written above)    │
│          dashboard/data/simulator.json                               │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  JSON (pre-computed, static)
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — Dashboard                                   dashboard/    │
│                                                                      │
│  Stack: HTML · CSS · JavaScript · Chart.js · nginx (Docker)          │
│  • Tab 1: League Analytics — trends, team leaderboards               │
│  • Tab 2: Backtest Performance — metrics, equity curve, conf. matrix │
│  • Tab 3: Odds Inspector — filterable value bet table + modal        │
│  • Tab 4: Betting Simulator — configurable stake, P&L, drawdown      │
│                                                                      │
│  Served at: http://localhost:8080                                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Contracts

Each stage communicates with the next via a file with a defined schema. No stage imports code from another stage.

### Stage 1 → Stage 2: Match-level Parquet

| Column | Type | Description |
|---|---|---|
| `RBallID` | string | Unique match ID |
| `HomeTeam` | string | Home team name |
| `AwayTeam` | string | Away team name |
| `Date` | date | Match date |
| `Season` | string | e.g. `"2017-18"` |
| `HomeGoals` | int | Goals scored by home team |
| `AwayGoals` | int | Goals scored by away team |
| `HomeCorners` | int | Corners won by home team |
| `AwayCorners` | int | Corners won by away team |
| `HomeShotsOnTarget` | int | |
| `AwayShotsOnTarget` | int | |
| `HomeFouls` | int | |
| `AwayFouls` | int | |
| `HomeOffsides` | int | |
| `AwayOffsides` | int | |

### Stage 2 → Stages 3 & 4: Feature-enriched Parquet

All columns from Stage 1 output, plus:

| Column | Type | Description |
|---|---|---|
| `Result` | string | `H` / `D` / `A` |
| `ResultCode` | int | `0` / `1` / `2` |
| `HomeElo` | float | Home team ELO before kick-off |
| `AwayElo` | float | Away team ELO before kick-off |
| `EloDiff` | float | `HomeElo - AwayElo` |
| `HomeGoals_Last5` | float | Rolling 5-match avg goals (home) |
| `AwayGoals_Last5` | float | Rolling 5-match avg goals (away) |
| `HomeCorners_Last5` | float | Rolling 5-match avg corners (home) |
| `AwayCorners_Last5` | float | Rolling 5-match avg corners (away) |
| `HomePoints_Last5` | float | Rolling 5-match avg points (home) |
| `AwayPoints_Last5` | float | Rolling 5-match avg points (away) |
| `HomeWinRate_Season` | float | Season win rate to date (home) |
| `AwayWinRate_Season` | float | Season win rate to date (away) |

### Stage 4 → Stage 5: Model odds Parquet

| Column | Type | Description |
|---|---|---|
| `RBallID` | string | Match ID |
| `HomeTeam` | string | |
| `AwayTeam` | string | |
| `Date` | date | |
| `Season` | string | |
| `Result` | string | Actual result |
| `P_Home` | float | Model probability of home win |
| `P_Draw` | float | Model probability of draw |
| `P_Away` | float | Model probability of away win |
| `ModelOdds_Home` | float | `1 / P_Home` |
| `ModelOdds_Draw` | float | `1 / P_Draw` |
| `ModelOdds_Away` | float | `1 / P_Away` |

### Stage 5 → Dashboard: Value bets Parquet + JSON

| Column | Type | Description |
|---|---|---|
| `HomeTeam` | string | |
| `AwayTeam` | string | |
| `Date` | date | |
| `Season` | string | |
| `Result` | string | Actual result |
| `Outcome` | string | The flagged outcome (`H` / `D` / `A`) |
| `ModelOdds` | float | Model-implied odds for this outcome |
| `BestBookOdds` | float | Best available bookmaker odds |
| `Edge` | float | `(ModelOdds / BestBookOdds) - 1` |
| `ValueBet` | bool | Always `True` in this file |
| `BestBookmaker` | string | Bookmaker offering best odds |

---

## Technology Choices

| Component | Technology | Alternative considered | Reason for choice |
|---|---|---|---|
| Ingestion | PySpark | pandas | Scales to Phase 2 API volumes without code changes |
| Intermediate storage | Parquet | CSV | Columnar, schema-enforced, faster reads, industry standard |
| Feature engineering | pandas | PySpark | ELO is sequential and stateful; pandas is simpler for this |
| Model | XGBoost | LightGBM, Logistic Regression | Best tabular baseline; handles NaN natively |
| Hyperparameter tuning | Optuna | GridSearchCV | Bayesian optimisation; far fewer trials needed |
| Experiment tracking | MLflow | Weights & Biases | Open source, local-first, no account required |
| Odds data | The Odds API | Scraping | Clean REST API; free tier sufficient; multi-bookmaker |
| Team name matching | rapidfuzz | Exact match | Team names differ between data sources |
| Dashboard | HTML / CSS / Chart.js | Flask + Jinja | No backend needed; pipeline pre-computes everything |
| Dashboard server | nginx | Python http.server | Production-grade static file server |
| Containerisation | Docker + Compose | bare Python scripts | One-command reproducibility on any machine |

---

## Key Design Principles

### 1. Stages communicate via files, not function calls
No stage imports code from another stage. Each reads its input file, does its work, and writes its output file. This means:
- Any stage can be rerun independently
- A failed stage doesn't corrupt earlier outputs
- Swapping a stage's implementation only requires keeping the same output schema

### 2. Data contracts are the interface
The Parquet schema between stages is the only coupling between them. As long as Stage 1 writes the same column names and types, Stage 2 doesn't care whether the source was CSVs, an API, or a database. This is how Phase 2 replaces Stage 1 without touching anything else.

### 3. No data leakage by construction
All rolling features are computed with `.shift(1)` before the window. ELO ratings stored at row `i` reflect ratings before match `i`. Win rates are computed from matches strictly before the current match date. This isn't a convention — it's enforced in the feature engineering logic.

### 4. The dashboard is decoupled from the pipeline
The dashboard reads static JSON files. It does not call the pipeline, query a database, or load a model. This means the dashboard works without any Python environment — just nginx. It also means the pipeline can be rerun without restarting the dashboard.

### 5. Secrets never touch the codebase
The Odds API key lives in a `.env` file that is in `.gitignore`. Docker Compose reads it as an environment variable. Nothing is hardcoded.

---

## Planned Directory Layout

```
sports_modelling/
│
├── data/
│   ├── ENG/                        # Raw per-match CSVs (~950 files)
│   ├── FRA/                        # Raw per-match CSVs (~940 files)
│   ├── SPA/                        # Raw per-match CSVs (~927 files)
│   ├── processed/                  # Stage 1 output — Parquet per league
│   ├── features/                   # Stage 2 output — feature-enriched Parquet
│   ├── output/                     # Stage 4 & 5 output — odds + value bets
│   └── odds_cache/                 # Cached Odds API responses (JSON)
│
├── pipeline/
│   ├── stage1_ingest.py            # PySpark CSV → Parquet
│   ├── stage2_features.py          # ELO + rolling features (planned)
│   ├── stage3_train.py             # XGBoost + Optuna + MLflow (planned)
│   ├── stage4_odds_gen.py          # Model inference → implied odds (planned)
│   ├── stage5_compare.py           # Odds API + value bet flagging (planned)
│   ├── export_dashboard_data.py    # Parquet → JSON for dashboard (planned)
│   └── run_pipeline.py             # Orchestrates all stages in order (planned)
│
├── dashboard/                   # Planned
│   ├── index.html                  # Single-page app
│   ├── css/
│   ├── js/
│   │   ├── main.js                 # Tab routing + data loading
│   │   ├── charts.js               # Chart.js wrappers
│   │   └── simulator.js            # Betting simulator logic
│   └── data/                       # Pre-computed JSON (written by pipeline)
│
├── mlruns/                         # Planned MLflow tracking store
│
├── docker/                      # Planned
│   ├── Dockerfile.pipeline
│   ├── Dockerfile.dashboard
│   ├── nginx.conf
│   └── docker-compose.yml
│
├── config/                      # Planned
│   └── settings.yaml               # Thresholds, API config, league keys
│
├── tests/                       # Planned
│   ├── test_features.py
│   ├── test_odds_comparison.py
│   └── test_pipeline_integration.py
│
├── analysis/
│   └── descriptive.ipynb           # EDA only — not part of the pipeline
│
├── models/
│   └── training.ipynb              # Original model exploration
│
├── docs/
│   └── ai/
│       ├── ARCHITECTURE.md         # This file
│       ├── PRD.md                  # Product requirements
│       ├── TASKS.md                # Canonical task status
│       └── tasks/                  # Step-by-step build guides
├── .venv/                          # Virtual environment (not committed)
├── .env                            # Secrets (not committed)
├── .gitignore
├── requirements.txt
├── AGENTS.md                       # Tutor instructions
├── README.md
└── LICENSE
```

---

## Phase 2 Evolution

Phase 2 replaces the static CSV source with live API ingestion. The diagram below shows what changes (highlighted) vs what stays the same.

```
Phase 1 (current)              Phase 2 (future)
─────────────────              ────────────────
Raw CSVs                  →    Football Data API (daily pull)
Stage 1: PySpark CSV read →    Stage 1: API client + DB write
                               + PostgreSQL / S3 storage
                               + Airflow DAG (replaces run_pipeline.py)
                               + Scheduled daily runs

Stages 2–6: unchanged     →    Stages 2–6: unchanged
```

Only Stage 1 changes. Everything from Stage 2 onwards reads the same Parquet schema and is completely unaffected by the source format change. This is the direct consequence of the data contract design.
