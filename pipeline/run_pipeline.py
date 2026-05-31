from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineStage:
    name: str
    script: Path
    args: tuple[str, ...] = ()


TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:mlruns")

STAGES = (
    PipelineStage("stage1_ingest", Path("pipeline/stage1_ingest.py")),
    PipelineStage("stage2_features", Path("pipeline/stage2_features.py")),
    PipelineStage("stage3_train", Path("pipeline/stage3_train.py"), ("--tracking-uri", TRACKING_URI)),
    PipelineStage("stage4_odds_gen", Path("pipeline/stage4_odds_gen.py"), ("--tracking-uri", TRACKING_URI)),
    PipelineStage("stage5_compare", Path("pipeline/stage5_compare.py")),
    PipelineStage("export_dashboard_data", Path("pipeline/export_dashboard_data.py")),
)


def _stage_index(stage_name: str) -> int:
    names = [stage.name for stage in STAGES]
    if stage_name not in names:
        raise ValueError(f"Unknown stage '{stage_name}'. Expected one of: {', '.join(names)}")
    return names.index(stage_name)


def select_stages(from_stage: str | None = None, to_stage: str | None = None) -> list[PipelineStage]:
    start = _stage_index(from_stage) if from_stage else 0
    end = _stage_index(to_stage) if to_stage else len(STAGES) - 1
    if start > end:
        raise ValueError("--from-stage must come before or equal --to-stage")
    return list(STAGES[start : end + 1])


def run_stage(stage: PipelineStage) -> None:
    banner = "=" * 72
    print(f"\n{banner}\nRunning {stage.name}: {stage.script}\n{banner}", flush=True)
    subprocess.run([sys.executable, str(stage.script), *stage.args], check=True)
    print(f"Completed {stage.name}", flush=True)


def run_pipeline(from_stage: str | None = None, to_stage: str | None = None, dry_run: bool = False) -> list[PipelineStage]:
    selected = select_stages(from_stage=from_stage, to_stage=to_stage)
    if dry_run:
        for stage in selected:
            suffix = f" {' '.join(stage.args)}" if stage.args else ""
            print(f"DRY RUN: {stage.name} -> {stage.script}{suffix}")
        return selected

    for stage in selected:
        run_stage(stage)
    print("\nPipeline complete.", flush=True)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sports modelling pipeline stages in order.")
    parser.add_argument("--from-stage", choices=[stage.name for stage in STAGES], help="First stage to run.")
    parser.add_argument("--to-stage", choices=[stage.name for stage in STAGES], help="Last stage to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected stages without executing them.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(from_stage=args.from_stage, to_stage=args.to_stage, dry_run=args.dry_run)
