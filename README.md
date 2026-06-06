# Sports Modelling

End-to-end sports betting odds arbitrage pipeline for European football. The project is being rebuilt from exploratory notebooks into production-style pipeline stages: Football-Data CSV download -> match-level Parquet -> feature engineering -> XGBoost + MLflow -> odds comparison -> static dashboard -> Docker packaging.

## Current Scope

Phase 1 covers Premier League, La Liga, Ligue 1, Bundesliga, and Serie A historical Football-Data.co.uk seasons from 2010-11 through 2022-23. The goal is to identify value bets where:

```text
best_bookmaker_odds >= 1.10 * model_implied_odds
```

Stage 5 also applies odds sanity filters so extreme long-shot prices and unrealistic edge outliers do not become dashboard bets.

This repository is for education and portfolio development. It does not place bets and does not provide financial advice.

## Documentation

- Product requirements: `docs/ai/PRD.md`
- Architecture notes: `docs/ai/ARCHITECTURE.md`
- Canonical task status: `docs/ai/TASKS.md`
- Detailed stage guides: `docs/ai/tasks/`
- Tutor/agent working agreement: `AGENTS.md`

## Current Repo State

- Pipeline stages 1-5 are implemented under `pipeline/`.
- Stage 6 dashboard is implemented under `dashboard/` and reads precomputed JSON from `dashboard/data/`.
- Stage 7 Docker packaging is implemented under `docker/`.
- `analysis/descriptive.ipynb` and `models/training.ipynb` are EDA/reference notebooks, not production pipeline code.

## Planned Pipeline Stages

| Stage | Output |
|---|---|
| 1 - Ingest & Clean | Match-level Parquet files in `data/processed/` |
| 2 - Feature Engineering | ELO, rolling form, draw signals, and season features |
| 3 - Model Training | Calibrated/tuned XGBoost model tracked with MLflow and baseline benchmarks |
| 4 - Odds Generation | Model probabilities and implied odds |
| 5 - Odds Comparison | Flagged value bets |
| 5.5 - Model Diagnostics | Calibration and value-bet bucket diagnostics |
| 6 - Dashboard | Static HTML/CSS/JS dashboard using JSON outputs |
| 7 - Docker | Reproducible local run environment |

## Environment

Use the project virtual environment before running scripts:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full pipeline with the venv active:

```bash
python pipeline/run_pipeline.py
```

Preview the stage order without executing the pipeline:

```bash
python pipeline/run_pipeline.py --dry-run
```

## Docker Run

From the project root:

```bash
docker compose up --build
```

When the pipeline container finishes successfully, open:

```text
http://localhost:8080
```

The pipeline is a one-shot batch container. The dashboard is an nginx container serving the static files and mounted `dashboard/data/` JSON output. Docker keeps MLflow inside the pipeline container at `/app/mlruns` instead of bind-mounting the host `mlruns/` directory, because MLflow file-store metadata contains absolute artifact paths that are not portable across host and container filesystems.

## License

MIT. See `LICENSE`.
