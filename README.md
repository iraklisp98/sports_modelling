# Sports Modelling

End-to-end sports betting odds research pipeline for European football. The project converts Football-Data.co.uk historical CSVs into a reproducible ML workflow: ingest -> Parquet -> feature engineering -> MLflow-tracked models -> odds comparison -> calibration diagnostics -> strategy simulation -> static dashboard -> Docker packaging.

This is a portfolio project, not betting advice. The value of the repo is the engineering story and the honest model validation: it shows how to test whether a model has a durable edge instead of only showing a lucky backtest.

## Final Project Story

The final candidate is a **market-aware XGBoost value-bet model** trained with an **expanding annual walk-forward policy** across all five leagues.

| Result | Evidence |
|---|---|
| Final model candidate | Market-aware XGBoost value strategy |
| Data scope | Premier League, La Liga, Ligue 1, Bundesliga, Serie A; 2010-11 through 2025-26 |
| Actionable bet scope | Home and away wins only; draws are modelled but not selected as value bets |
| Best validation protocol | Expanding annual retraining: train through the last completed season, test the next season |
| Walk-forward ROI | +5.47% across 767 bets, 3/5 positive folds |
| Dashboard default simulator | Expanding walk-forward XGBoost +5.47%; frozen XGBoost remains visible at +0.19% |
| League subset audit | All five leagues remained best; filtering leagues did not improve the headline result |

The important modelling conclusion is narrow and defensible: **the strongest candidate is not a globally superior football prediction model; it is a value-bet selection workflow that only becomes interesting under expanding retraining validation.** The project keeps the weaker benchmarks in the dashboard because rejected models are part of the story.

## Dashboard

The dashboard is a static HTML/CSS/JS app served by nginx in Docker or opened locally from `dashboard/index.html` after the pipeline exports JSON.

Tabs:

- **Project Story**: final model decision, validation path, rejected benchmarks, league subset audit, and test summary.
- **League Analytics**: match trends, standings, home/away performance.
- **Backtest**: ML metrics, equity curve, confusion matrix, MLflow run comparison.
- **Odds Inspector**: filterable value-bet table with per-match probability/odds modal.
- **Calibration**: odds buckets, actual outcomes, hit rate, ROI by range.
- **Simulator**: fixed-stake replay, bankroll curve, strategy comparison, expanding retraining robustness.

## Pipeline Stages

| Stage | Script | Output |
|---|---|---|
| 1 - Ingest & Clean | `pipeline/stage1_ingest.py` | Match-level Parquet in `data/processed/` |
| 2 - Feature Engineering | `pipeline/stage2_features.py` | ELO, rolling form, rest, venue, and pressure features in `data/features/` |
| 3 - Model Training | `pipeline/stage3_train.py` | XGBoost model, metrics, MLflow artifacts |
| 3.6 - Poisson Benchmark | `pipeline/poisson_goal_model.py` | Interpretable expected-goals benchmark odds |
| 4 - Odds Generation | `pipeline/stage4_odds_gen.py` | Model probabilities and implied odds |
| 5 - Odds Comparison | `pipeline/stage5_compare.py` | Home/away value bets against bookmaker odds |
| 5.5 - Diagnostics | `pipeline/model_diagnostics.py` and related scripts | Calibration, market baseline, benchmark comparison artifacts |
| 6 - Dashboard Export | `pipeline/export_dashboard_data.py` | Static dashboard JSON, including `project_summary.json` |
| 7 - Docker | `docker/` and `docker-compose.yml` | One-command local runtime |

## Run Locally

Activate the venv first:

```bash
source .venv/bin/activate
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python pipeline/run_pipeline.py
```

Regenerate dashboard JSON only:

```bash
python pipeline/export_dashboard_data.py
```

Preview the stage order without executing work:

```bash
python pipeline/run_pipeline.py --dry-run
```

## Docker Run

From the project root:

```bash
docker compose up --build
```

When the pipeline container finishes, open:

```text
http://localhost:8080
```

The pipeline container is a one-shot batch job. The dashboard container serves static files and mounted `dashboard/data/` JSON. MLflow is kept inside the pipeline container at `/app/mlruns` to avoid non-portable host artifact paths.

## Tests

Verification command used for the final pass:

```bash
python -m unittest discover tests
```

Latest result: **153 tests OK, 1 skipped**. The final pass adds dashboard/export contract coverage for the new `project_summary.json` story tab. `pytest` is optional; it was not installed in the current `.venv`, so the unittest discovery command is the documented reproducible test path.

## Documentation

- Product requirements: `docs/ai/PRD.md`
- Architecture notes: `docs/ai/ARCHITECTURE.md`
- Canonical task status: `docs/ai/TASKS.md`
- Stage guides: `docs/ai/tasks/`
- Working agreement: `AGENTS.md`
- Session findings: `ai.log`

## Portfolio Talking Point

If asked what this project proves, say:

> It proves I can build a reproducible ML engineering pipeline, evaluate models against market odds, reject weak strategies honestly, and communicate the final candidate with walk-forward validation. It does not claim a guaranteed betting edge.

## License

MIT. See `LICENSE`.
