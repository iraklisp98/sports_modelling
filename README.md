# Sports Modelling

End-to-end sports betting odds arbitrage pipeline for European football. The project is being rebuilt from exploratory notebooks into production-style pipeline stages: raw event CSVs -> PySpark cleaning -> feature engineering -> XGBoost + MLflow -> odds comparison -> static dashboard -> Docker packaging.

## Current Scope

Phase 1 covers Premier League, La Liga, and Ligue 1 event-level data from 2017-2020. The goal is to identify value bets where:

```text
model_implied_odds >= 1.10 * best_bookmaker_odds
```

This repository is for education and portfolio development. It does not place bets and does not provide financial advice.

## Documentation

- Product requirements: `docs/ai/PRD.md`
- Architecture notes: `docs/ai/ARCHITECTURE.md`
- Canonical task status: `docs/ai/TASKS.md`
- Detailed stage guides: `docs/ai/tasks/`
- Tutor/agent working agreement: `AGENTS.md`

## Current Repo State

- `pipeline/stage1_ingest.py` exists and writes Parquet outputs under `data/processed/`.
- Stage 1 is still marked in progress because tests are not present yet.
- `analysis/descriptive.ipynb` and `models/training.ipynb` are EDA/reference notebooks, not production pipeline code.
- `dashboard/`, `docker/`, `config/`, and `tests/` are planned but not created yet.

## Planned Pipeline Stages

| Stage | Output |
|---|---|
| 1 - Ingest & Clean | Match-level Parquet files in `data/processed/` |
| 2 - Feature Engineering | ELO, rolling form, and season features |
| 3 - Model Training | XGBoost model tracked with MLflow |
| 4 - Odds Generation | Model probabilities and implied odds |
| 5 - Odds Comparison | Flagged value bets |
| 6 - Dashboard | Static HTML/CSS/JS dashboard using JSON outputs |
| 7 - Docker | Reproducible local run environment |

## Environment

Use the project virtual environment before running scripts:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Run the current Stage 1 script with the venv active:

```bash
python pipeline/stage1_ingest.py
```

## License

MIT. See `LICENSE`.
