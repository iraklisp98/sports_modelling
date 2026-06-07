from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23")
DEFAULT_OUTPUT_PATH = Path("data/model_artifacts/value_bet_model_comparison.json")


@dataclass(frozen=True)
class ComparisonSummary:
    output_path: Path
    models: int

    def line(self) -> str:
        return f"comparison={self.output_path}, models={self.models}"


def _round(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def load_value_bets(path: Path, seasons: Iterable[str] | None = DEFAULT_SEASONS) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing value-bet file: {path}")
    df = pd.read_parquet(path).copy()
    required = ["Season", "League", "Outcome", "Result", "BestBookOdds"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing value-bet columns in {path}: {missing}")
    if seasons is not None:
        df = df[df["Season"].isin(tuple(seasons))].copy()
    df["Won"] = df["Outcome"] == df["Result"]
    df["FlatStakeProfit"] = df.apply(lambda row: float(row.BestBookOdds) - 1.0 if row.Won else -1.0, axis=1)
    return df.reset_index(drop=True)


def summarize_group(df: pd.DataFrame, group_columns: list[str]) -> list[dict[str, object]]:
    if df.empty:
        return []
    grouped = df.groupby(group_columns, sort=True)
    table = grouped.agg(
        bets=("Won", "size"),
        wins=("Won", "sum"),
        hit_rate=("Won", "mean"),
        profit=("FlatStakeProfit", "sum"),
        roi=("FlatStakeProfit", "mean"),
        avg_book_odds=("BestBookOdds", "mean"),
    ).reset_index()
    rows: list[dict[str, object]] = []
    for row in table.itertuples(index=False):
        item = {column: getattr(row, column) for column in group_columns}
        item.update(
            {
                "bets": int(row.bets),
                "wins": int(row.wins),
                "hit_rate": _round(row.hit_rate),
                "profit": _round(row.profit, 2),
                "roi": _round(row.roi),
                "avg_book_odds": _round(row.avg_book_odds),
            }
        )
        rows.append(item)
    return rows


def summarize_model(name: str, path: Path, seasons: Iterable[str] | None = DEFAULT_SEASONS) -> dict[str, object]:
    df = load_value_bets(path, seasons=seasons)
    total_profit = float(df["FlatStakeProfit"].sum()) if not df.empty else 0.0
    total_bets = int(len(df))
    return {
        "model": name,
        "path": str(path),
        "seasons": list(seasons) if seasons is not None else None,
        "overall": {
            "bets": total_bets,
            "wins": int(df["Won"].sum()) if not df.empty else 0,
            "hit_rate": _round(df["Won"].mean()) if not df.empty else 0.0,
            "profit": _round(total_profit, 2),
            "roi": _round(total_profit / total_bets) if total_bets else 0.0,
        },
        "by_outcome": summarize_group(df, ["Outcome"]),
        "by_league": summarize_group(df, ["League"]),
        "by_season": summarize_group(df, ["Season"]),
    }


def build_comparison(model_specs: list[tuple[str, Path]], seasons: Iterable[str] | None = DEFAULT_SEASONS) -> dict[str, object]:
    return {"models": [summarize_model(name, path, seasons=seasons) for name, path in model_specs]}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_model_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Model specs must use name=path format")
    name, value = raw.split("=", 1)
    if not name.strip() or not value.strip():
        raise argparse.ArgumentTypeError("Model specs must use name=path format")
    return name.strip(), Path(value.strip())


def run_pipeline(
    model_specs: list[tuple[str, Path]],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    seasons: Iterable[str] | None = DEFAULT_SEASONS,
) -> ComparisonSummary:
    payload = build_comparison(model_specs, seasons=seasons)
    write_json(output_path, payload)
    return ComparisonSummary(output_path=output_path, models=len(model_specs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare value-bet strategy outputs from multiple model odds pipelines.")
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True, help="name=path to a value-bets parquet file")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seasons", nargs="*", default=list(DEFAULT_SEASONS), help="Seasons to include; omit values for all seasons")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seasons = tuple(args.seasons) if args.seasons else None
    summary = run_pipeline(args.model, output_path=args.output_path, seasons=seasons)
    print(summary.line())
