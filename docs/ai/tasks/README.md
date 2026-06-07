# Tasks

One file per pipeline stage. Each file explains what to build, why it's built that way, and how to build it step by step.

Work through them in order. Each stage depends on the output of the previous one.

**Before starting any stage:** complete [setup.md](setup.md) — venv creation and dependency installation.

| Stage | File | Status |
|---|---|---|
| 0 — Environment Setup | [setup.md](setup.md) | In progress |
| 1 — Data Ingestion & Cleaning | [stage1_ingest.md](stage1_ingest.md) | Complete |
| 2 — Feature Engineering | [stage2_features.md](stage2_features.md) | Complete |
| 3 — Model Training | [stage3_train.md](stage3_train.md) | Complete |
| 3.5 — Poisson Goal Benchmark | [poisson_goal_model.md](poisson_goal_model.md) | Complete |
| 3.6 — Market Mispricing Model | `pipeline/mispricing_model.py` | Complete |
| 4 — Odds Generation | [stage4_odds_gen.md](stage4_odds_gen.md) | Complete |
| 5 — Odds Comparison | [stage5_compare.md](stage5_compare.md) | Complete |
| 6 — Dashboard | [stage6_dashboard.md](stage6_dashboard.md) | Complete |
| 7 — Docker | [stage7_docker.md](stage7_docker.md) | Runtime smoke pending |

Update the status column as you complete each stage.
