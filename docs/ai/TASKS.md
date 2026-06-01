# Task Index

This is the canonical task status file for the repo. Detailed stage guides live in `docs/ai/tasks/`.

Coder-tester should pick the next incomplete task from this table, implement only that task, add or update relevant tests, run the narrowest useful test command, and update the status here.

| Stage | Guide | Main output | Status |
|---|---|---|---|
| 0 — Environment Setup | `docs/ai/tasks/setup.md` | `.venv/` and installed dependencies | In progress |
| 1 — Data Ingestion & Cleaning | `docs/ai/tasks/stage1_ingest.md` | `pipeline/stage1_ingest.py` | Complete |
| 2 — Feature Engineering | `docs/ai/tasks/stage2_features.md` | `pipeline/stage2_features.py` | Complete |
| 3 — Model Training | `docs/ai/tasks/stage3_train.md` | `pipeline/stage3_train.py` | Complete |
| 3.1 — XGBoost Two-Feature Improvement Experiment | `docs/ai/tasks/xgb_two_feature_experiment.md` | metric delta report | Rejected — no log-loss improvement |
| 3.2 — XGBoost Draw Overlay Weight Experiment | `docs/ai/tasks/xgb_draw_overlay_weight_experiment.md` | metric delta report | Rejected — selected overlay worsened holdout log loss; production remains base calibrated XGBoost |
| 3.6 — Poisson Goal Model Benchmark | `docs/ai/tasks/poisson_goal_model.md` | `pipeline/poisson_goal_model.py` + benchmark artifacts | Complete |
| 4 — Odds Generation | `docs/ai/tasks/stage4_odds_gen.md` | `pipeline/stage4_odds_gen.py` | Complete |
| 5 — Odds Comparison | `docs/ai/tasks/stage5_compare.md` | `pipeline/stage5_compare.py` | Complete |
| 5.5 — Model Diagnostics | `docs/ai/tasks/model_diagnostics.md` | `pipeline/model_diagnostics.py` | Complete |
| 6 — Dashboard | `docs/ai/tasks/stage6_dashboard.md` | `dashboard/` | Complete |
| 7 — Docker | `docs/ai/tasks/stage7_docker.md` | `docker/` | Runtime smoke pending |

Update this table when a stage moves from not started to in progress, blocked, or complete.
