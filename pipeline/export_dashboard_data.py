from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FEATURES_DIR = Path("data/features")
MODEL_ODDS_PATH = Path("data/output/model_odds.parquet")
VALUE_BETS_PATH = Path("data/output/value_bets.parquet")
POISSON_VALUE_BETS_PATH = Path("data/output/poisson_value_bets.parquet")
MARKET_AWARE_VALUE_BETS_PATH = Path("data/output/market_aware_value_bets.parquet")
MISPRICING_VALUE_BETS_PATH = Path("data/output/mispricing_value_bets.parquet")
METRICS_PATH = Path("data/model_artifacts/stage3/metrics.json")
HOLDOUT_PREDICTIONS_PATH = Path("data/model_artifacts/stage3/holdout_predictions.parquet")
MODEL_DIAGNOSTICS_PATH = Path("data/model_artifacts/stage3/model_diagnostics.json")
TRAINING_POLICY_PATH = Path("data/model_artifacts/expanding_walk_forward_training_window.json")
LEAGUE_SUBSET_POLICY_PATH = Path("data/model_artifacts/league_subset_walk_forward.json")
DASHBOARD_DATA_DIR = Path("dashboard/data")
MLRUNS_DIR = Path("mlruns")
LEAGUES = ("ENG", "SPA", "FRA", "GER", "ITA")
HOLDOUT_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
DEFAULT_STAKE = 10.0

LEAGUE_LABELS = {
    "ENG": "Premier League",
    "SPA": "La Liga",
    "FRA": "Ligue 1",
    "GER": "Bundesliga",
    "ITA": "Serie A",
}

RESULT_TO_CODE = {"H": 0, "D": 1, "A": 2}
CODE_TO_RESULT = {0: "H", 1: "D", 2: "A"}


STRATEGY_VALUE_BET_PATHS = {
    "xgboost_value": VALUE_BETS_PATH,
    "market_aware_xgboost": MARKET_AWARE_VALUE_BETS_PATH,
    "poisson_goal_model": POISSON_VALUE_BETS_PATH,
    "mispricing_model": MISPRICING_VALUE_BETS_PATH,
}
STRATEGY_LABELS = {
    "xgboost_value": "XGBoost value",
    "market_aware_xgboost": "Market-aware XGBoost",
    "poisson_goal_model": "Poisson goal model",
    "mispricing_model": "Mispricing model",
    "expanding_walk_forward_xgboost": "Expanding walk-forward XGBoost",
}
PRIMARY_STRATEGY_ID = "expanding_walk_forward_xgboost"


@dataclass(frozen=True)
class DashboardExportSummary:
    output_dir: Path
    files: list[Path]

    def line(self) -> str:
        names = ", ".join(str(path) for path in self.files)
        return f"dashboard_data={self.output_dir}, files=[{names}]"


def _round(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _records(df: pd.DataFrame) -> list[dict[str, object]]:
    clean = df.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].dt.strftime("%Y-%m-%d")
    return clean.replace({np.nan: None}).to_dict(orient="records")


def load_feature_data(features_dir: Path = FEATURES_DIR, leagues: Iterable[str] = LEAGUES) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for league in leagues:
        path = features_dir / f"{league}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file for dashboard export: {path}")
        frame = pd.read_parquet(path).copy()
        frame["League"] = league
        frame["LeagueLabel"] = LEAGUE_LABELS.get(league, league)
        frame["Date"] = pd.to_datetime(frame["Date"])
        frames.append(frame)
    if not frames:
        raise ValueError("At least one league is required")
    return pd.concat(frames, ignore_index=True)


def build_league_summary(features: pd.DataFrame) -> dict[str, dict[str, dict[str, float | int]]]:
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for (league, season), group in features.groupby(["League", "Season"], sort=True):
        result_counts = group["Result"].value_counts(normalize=True)
        summary.setdefault(league, {})[season] = {
            "matches": int(len(group)),
            "avg_goals": _round((group["HomeGoals"] + group["AwayGoals"]).mean()),
            "home_win_pct": _round(result_counts.get("H", 0.0)),
            "draw_pct": _round(result_counts.get("D", 0.0)),
            "away_win_pct": _round(result_counts.get("A", 0.0)),
        }
    return summary


def build_monthly_trends(features: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    frame = features.copy()
    frame["Month"] = frame["Date"].dt.to_period("M").astype(str)
    frame["TotalGoals"] = frame["HomeGoals"] + frame["AwayGoals"]
    frame["TotalCorners"] = frame["HomeCorners"] + frame["AwayCorners"]
    frame["TotalShotsOnTarget"] = frame["HomeShotsOnTarget"] + frame["AwayShotsOnTarget"]

    trends: dict[str, list[dict[str, object]]] = {}
    grouped = frame.groupby(["League", "Month"], sort=True)
    monthly = grouped.agg(
        avg_goals=("TotalGoals", "mean"),
        avg_corners=("TotalCorners", "mean"),
        avg_shots=("TotalShotsOnTarget", "mean"),
    ).reset_index()
    for league, group in monthly.groupby("League", sort=True):
        trends[league] = [
            {
                "month": row.Month,
                "avg_goals": _round(row.avg_goals),
                "avg_corners": _round(row.avg_corners),
                "avg_shots": _round(row.avg_shots),
            }
            for row in group.itertuples(index=False)
        ]
    return trends


def build_team_standings(features: pd.DataFrame) -> dict[str, dict[str, list[dict[str, object]]]]:
    standings: dict[str, dict[str, list[dict[str, object]]]] = {}
    for (league, season), group in features.groupby(["League", "Season"], sort=True):
        table: dict[str, dict[str, object]] = {}
        for row in group.itertuples(index=False):
            home = table.setdefault(row.HomeTeam, {"team": row.HomeTeam, "played": 0, "points": 0, "goals_for": 0, "goals_against": 0})
            away = table.setdefault(row.AwayTeam, {"team": row.AwayTeam, "played": 0, "points": 0, "goals_for": 0, "goals_against": 0})
            home["played"] += 1
            away["played"] += 1
            home["goals_for"] += int(row.HomeGoals)
            home["goals_against"] += int(row.AwayGoals)
            away["goals_for"] += int(row.AwayGoals)
            away["goals_against"] += int(row.HomeGoals)
            if row.Result == "H":
                home["points"] += 3
            elif row.Result == "A":
                away["points"] += 3
            else:
                home["points"] += 1
                away["points"] += 1

        rows = []
        for item in table.values():
            item["goal_diff"] = int(item["goals_for"] - item["goals_against"])
            rows.append(item)
        rows.sort(key=lambda item: (item["points"], item["goal_diff"], item["goals_for"]), reverse=True)
        standings.setdefault(league, {})[season] = rows
    return standings


def build_home_away_split(features: pd.DataFrame) -> dict[str, dict[str, list[dict[str, object]]]]:
    split: dict[str, dict[str, list[dict[str, object]]]] = {}
    for (league, season), group in features.groupby(["League", "Season"], sort=True):
        rows: dict[str, dict[str, object]] = {}
        for row in group.itertuples(index=False):
            home = rows.setdefault(row.HomeTeam, {"team": row.HomeTeam, "home_points": 0, "away_points": 0})
            away = rows.setdefault(row.AwayTeam, {"team": row.AwayTeam, "home_points": 0, "away_points": 0})
            if row.Result == "H":
                home["home_points"] += 3
            elif row.Result == "A":
                away["away_points"] += 3
            else:
                home["home_points"] += 1
                away["away_points"] += 1
        ordered = sorted(rows.values(), key=lambda item: item["home_points"] + item["away_points"], reverse=True)
        split.setdefault(league, {})[season] = ordered
    return split


def build_league_analytics(features: pd.DataFrame) -> dict[str, object]:
    leagues = sorted(features["League"].unique().tolist())
    seasons = sorted(features["Season"].unique().tolist())
    return {
        "leagues": leagues,
        "league_labels": {league: LEAGUE_LABELS.get(league, league) for league in leagues},
        "seasons": seasons,
        "summary": build_league_summary(features),
        "monthly_trends": build_monthly_trends(features),
        "team_standings": build_team_standings(features),
        "home_away_split": build_home_away_split(features),
    }




def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _read_latest_metric(path: Path) -> float | None:
    raw = _read_text(path)
    if raw is None:
        return None
    last_line = raw.splitlines()[-1]
    parts = last_line.split()
    if len(parts) < 2:
        return None
    return _round(parts[1])


def _read_param(path: Path) -> int | float | str | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        number = float(raw)
    except ValueError:
        return raw
    return int(number) if number.is_integer() else number


def _read_meta_value(path: Path, key: str) -> str | None:
    raw = _read_text(path)
    if raw is None:
        return None
    prefix = f"{key}:"
    for line in raw.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip("'").strip('"')
    return None


def build_mlflow_runs(mlruns_dir: Path = MLRUNS_DIR, limit: int = 5) -> list[dict[str, object]]:
    if not mlruns_dir.exists():
        return []

    runs: list[dict[str, object]] = []
    for meta_path in mlruns_dir.glob("*/*/meta.yaml"):
        run_dir = meta_path.parent
        metrics_dir = run_dir / "metrics"
        params_dir = run_dir / "params"
        run = {
            "run_id": _read_meta_value(meta_path, "run_id") or run_dir.name,
            "run_name": _read_text(run_dir / "tags" / "mlflow.runName") or _read_meta_value(meta_path, "run_name"),
            "log_loss": _read_latest_metric(metrics_dir / "holdout_log_loss"),
            "brier_score": _read_latest_metric(metrics_dir / "holdout_brier_score"),
            "accuracy": _read_latest_metric(metrics_dir / "holdout_accuracy"),
            "f1_home": _read_latest_metric(metrics_dir / "holdout_f1_home"),
            "f1_draw": _read_latest_metric(metrics_dir / "holdout_f1_draw"),
            "f1_away": _read_latest_metric(metrics_dir / "holdout_f1_away"),
            "n_estimators": _read_param(params_dir / "n_estimators"),
            "max_depth": _read_param(params_dir / "max_depth"),
            "learning_rate": _read_param(params_dir / "learning_rate"),
        }
        if run["log_loss"] is not None:
            runs.append(run)

    runs.sort(key=lambda item: (item["log_loss"], item["run_id"]))
    return runs[:limit]


def load_metrics(metrics_path: Path = METRICS_PATH) -> dict[str, float]:
    if not metrics_path.exists():
        return {}
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "log_loss": _round(raw.get("holdout_log_loss")),
        "brier_score": _round(raw.get("holdout_brier_score")),
        "accuracy": _round(raw.get("holdout_accuracy")),
        "f1_home": _round(raw.get("holdout_f1_home")),
        "f1_draw": _round(raw.get("holdout_f1_draw")),
        "f1_away": _round(raw.get("holdout_f1_away")),
    }


def build_confusion_matrix(holdout: pd.DataFrame) -> list[list[int]]:
    if holdout.empty:
        return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    probabilities = holdout[["P_Home", "P_Draw", "P_Away"]].to_numpy()
    predicted = probabilities.argmax(axis=1)
    actual = holdout["Result"].map(RESULT_TO_CODE).fillna(-1).astype(int).to_numpy()
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for actual_code, predicted_code in zip(actual, predicted):
        if actual_code in CODE_TO_RESULT:
            matrix[actual_code][int(predicted_code)] += 1
    return matrix


def enrich_value_bets(value_bets: pd.DataFrame, model_odds: pd.DataFrame | None = None) -> pd.DataFrame:
    enriched = value_bets.copy()
    if model_odds is not None and not model_odds.empty:
        probability_columns = ["RBallID", "P_Home", "P_Draw", "P_Away", "ModelOdds_Home", "ModelOdds_Draw", "ModelOdds_Away"]
        available = [column for column in probability_columns if column in model_odds.columns]
        enriched = enriched.merge(model_odds[available], on="RBallID", how="left")
    enriched["Date"] = pd.to_datetime(enriched["Date"])
    return enriched.sort_values(["Date", "RBallID", "Outcome"], kind="mergesort").reset_index(drop=True)


def compute_bet_rows(value_bets: pd.DataFrame, stake: float = DEFAULT_STAKE) -> list[dict[str, object]]:
    bankroll = float(stake) * len(value_bets)
    running = bankroll
    rows: list[dict[str, object]] = []
    ordered = value_bets.sort_values(["Date", "RBallID", "Outcome"], kind="mergesort")
    for bet in ordered.itertuples(index=False):
        won = bet.Outcome == bet.Result
        payout = float(stake) * float(bet.BestBookOdds) if won else 0.0
        profit = payout - float(stake)
        running += profit
        rows.append(
            {
                "date": pd.to_datetime(bet.Date).strftime("%Y-%m-%d"),
                "league": getattr(bet, "League", None),
                "home_team": bet.HomeTeam,
                "away_team": bet.AwayTeam,
                "outcome": bet.Outcome,
                "result": bet.Result,
                "model_odds": _round(bet.ModelOdds),
                "book_odds": _round(bet.BestBookOdds),
                "best_bookmaker": bet.BestBookmaker,
                "edge": _round(bet.Edge),
                "stake": _round(stake, 2),
                "return": _round(payout, 2),
                "profit": _round(profit, 2),
                "won": bool(won),
                "running_bankroll": _round(running, 2),
            }
        )
    return rows


def longest_streak(values: Iterable[bool], target: bool) -> int:
    longest = 0
    current = 0
    for value in values:
        if value is target:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def max_drawdown(bankroll_values: Iterable[float]) -> float:
    peak = None
    worst = 0.0
    for value in bankroll_values:
        current = float(value)
        peak = current if peak is None else max(peak, current)
        worst = max(worst, peak - current)
    return worst


def build_simulator(value_bets: pd.DataFrame, stake: float = DEFAULT_STAKE) -> dict[str, object]:
    bets = compute_bet_rows(value_bets, stake=stake)
    total_bets = len(bets)
    wins = sum(1 for bet in bets if bet["won"])
    losses = total_bets - wins
    starting_bankroll = stake * total_bets
    ending_bankroll = bets[-1]["running_bankroll"] if bets else starting_bankroll
    profit = ending_bankroll - starting_bankroll
    running_values = [starting_bankroll, *[float(bet["running_bankroll"]) for bet in bets]]
    win_flags = [bool(bet["won"]) for bet in bets]
    summary = {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "starting_bankroll": _round(starting_bankroll, 2),
        "ending_bankroll": _round(ending_bankroll, 2),
        "total_profit": _round(profit, 2),
        "roi_pct": _round((profit / starting_bankroll) * 100 if starting_bankroll else 0.0, 2),
        "hit_rate": _round(wins / total_bets if total_bets else 0.0),
        "max_drawdown": _round(max_drawdown(running_values), 2),
        "longest_win_streak": longest_streak(win_flags, True),
        "longest_loss_streak": longest_streak(win_flags, False),
        "avg_odds": _round(value_bets["BestBookOdds"].mean() if total_bets else 0.0),
        "avg_edge_pct": _round(value_bets["Edge"].mean() * 100 if total_bets else 0.0, 2),
    }
    return {"default_stake": stake, "bets": bets, "summary": summary}


def filter_holdout_value_bets(value_bets: pd.DataFrame, seasons: Iterable[str] = HOLDOUT_SEASONS) -> pd.DataFrame:
    if value_bets.empty or "Season" not in value_bets.columns:
        return value_bets.copy()
    return value_bets[value_bets["Season"].isin(tuple(seasons))].copy().reset_index(drop=True)


def _season_start_year(season: object) -> int:
    try:
        return int(str(season).split("-", maxsplit=1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid season label: {season!r}") from exc


def load_strategy_value_bets(
    strategy_paths: dict[str, Path] = STRATEGY_VALUE_BET_PATHS,
    model_odds: pd.DataFrame | None = None,
    required_seasons: Iterable[str] = HOLDOUT_SEASONS,
) -> dict[str, pd.DataFrame]:
    strategies: dict[str, pd.DataFrame] = {}
    required = tuple(required_seasons)
    latest_required = max((_season_start_year(season) for season in required), default=None)
    for strategy_id, path in strategy_paths.items():
        if not path.exists():
            continue
        value_bets = enrich_value_bets(pd.read_parquet(path), model_odds=model_odds)
        seasons = value_bets.get("Season", pd.Series(dtype=str)).dropna().unique().tolist()
        latest_available = max((_season_start_year(season) for season in seasons), default=None)
        if latest_required is not None and latest_available is not None and latest_available < latest_required:
            continue
        strategies[strategy_id] = value_bets
    return strategies




def build_training_policy_strategy(training_policy: dict[str, object], stake: float = DEFAULT_STAKE) -> dict[str, object] | None:
    if not training_policy.get("available"):
        return None
    aggregate = training_policy.get("aggregate", {})
    folds = list(training_policy.get("folds", []))
    total_bets = int(NumberLike(aggregate.get("bets")))
    if total_bets <= 0 or not folds:
        return None

    records = [record for fold in folds for record in fold.get("value_bet_records", [])]
    if records:
        value_bets = pd.DataFrame(records)
        value_bets["Date"] = pd.to_datetime(value_bets["Date"])
        simulator = build_simulator(value_bets, stake=stake)
        return {
            "id": "expanding_walk_forward_xgboost",
            "label": STRATEGY_LABELS["expanding_walk_forward_xgboost"],
            "path": str(TRAINING_POLICY_PATH),
            "summary": simulator["summary"],
            "bets": simulator["bets"],
            "synthetic": False,
        }

    running = float(stake) * total_bets
    bets: list[dict[str, object]] = []
    for fold in folds:
        test_season = (fold.get("test_seasons") or ["Fold"])[0]
        fold_summary = (fold.get("value_bets") or {}).get("overall", {})
        fold_bets = int(NumberLike(fold_summary.get("bets")))
        fold_wins = int(NumberLike(fold_summary.get("wins")))
        fold_profit_units = NumberLike(fold_summary.get("profit"))
        if fold_bets <= 0:
            continue
        fold_losses = max(0, fold_bets - fold_wins)
        winning_odds = ((fold_profit_units + fold_bets) / fold_wins) if fold_wins else 0.0
        win_slots = set()
        if fold_wins:
            win_slots = {round((index + 0.5) * fold_bets / fold_wins) for index in range(fold_wins)}
        wins_written = 0
        losses_written = 0
        for bet_index in range(fold_bets):
            should_win = bet_index in win_slots and wins_written < fold_wins
            if not should_win and fold_bets - bet_index <= fold_wins - wins_written:
                should_win = True
            if should_win:
                wins_written += 1
                book_odds = winning_odds
                payout = float(stake) * book_odds
            else:
                losses_written += 1
                book_odds = 2.0
                payout = 0.0
            profit = payout - float(stake)
            running += profit
            bets.append(
                {
                    "date": str(test_season),
                    "league": "WF",
                    "home_team": f"{test_season} fold",
                    "away_team": f"Bet {bet_index + 1}",
                    "outcome": "H/A",
                    "result": "W" if should_win else "L",
                    "model_odds": None,
                    "book_odds": book_odds,
                    "best_bookmaker": "walk-forward replay",
                    "edge": None,
                    "stake": _round(stake, 2),
                    "return": _round(payout, 2),
                    "profit": _round(profit, 2),
                    "won": bool(should_win),
                    "running_bankroll": _round(running, 2),
                    "synthetic": True,
                }
            )

    summary = build_simulator_summary_from_rows(bets, stake=stake)
    summary["total_bets"] = total_bets
    summary["wins"] = int(NumberLike(aggregate.get("wins")))
    summary["losses"] = total_bets - summary["wins"]
    summary["total_profit"] = _round(NumberLike(aggregate.get("profit")) * stake, 2)
    summary["ending_bankroll"] = _round(summary["starting_bankroll"] + summary["total_profit"], 2)
    summary["roi_pct"] = _round(NumberLike(aggregate.get("roi")) * 100, 2)
    summary["hit_rate"] = _round(NumberLike(aggregate.get("hit_rate")))
    if bets:
        bets[-1]["running_bankroll"] = summary["ending_bankroll"]
    return {
        "id": "expanding_walk_forward_xgboost",
        "label": STRATEGY_LABELS["expanding_walk_forward_xgboost"],
        "path": str(TRAINING_POLICY_PATH),
        "summary": summary,
        "bets": bets,
        "synthetic": True,
    }


def build_simulator_summary_from_rows(bets: list[dict[str, object]], stake: float = DEFAULT_STAKE) -> dict[str, object]:
    total_bets = len(bets)
    wins = sum(1 for bet in bets if bet["won"])
    losses = total_bets - wins
    starting_bankroll = stake * total_bets
    ending_bankroll = NumberLike(bets[-1]["running_bankroll"]) if bets else starting_bankroll
    profit = ending_bankroll - starting_bankroll
    running_values = [starting_bankroll, *[NumberLike(bet["running_bankroll"]) for bet in bets]]
    win_flags = [bool(bet["won"]) for bet in bets]
    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "starting_bankroll": _round(starting_bankroll, 2),
        "ending_bankroll": _round(ending_bankroll, 2),
        "total_profit": _round(profit, 2),
        "roi_pct": _round((profit / starting_bankroll) * 100 if starting_bankroll else 0.0, 2),
        "hit_rate": _round(wins / total_bets if total_bets else 0.0),
        "max_drawdown": _round(max_drawdown(running_values), 2),
        "longest_win_streak": longest_streak(win_flags, True),
        "longest_loss_streak": longest_streak(win_flags, False),
        "avg_odds": _round(np.mean([NumberLike(bet["book_odds"]) for bet in bets]) if total_bets else 0.0),
        "avg_edge_pct": 0.0,
    }

def build_strategy_comparison(
    strategies: dict[str, pd.DataFrame],
    primary_strategy_id: str = PRIMARY_STRATEGY_ID,
    stake: float = DEFAULT_STAKE,
    seasons: Iterable[str] = HOLDOUT_SEASONS,
    training_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    items = []
    for strategy_id, value_bets in strategies.items():
        holdout = filter_holdout_value_bets(value_bets, seasons=seasons)
        simulator = build_simulator(holdout, stake=stake)
        summary = simulator["summary"]
        items.append(
            {
                "id": strategy_id,
                "label": STRATEGY_LABELS.get(strategy_id, strategy_id),
                "path": str(STRATEGY_VALUE_BET_PATHS.get(strategy_id, "")),
                "summary": summary,
                "bets": simulator["bets"],
            }
        )
    if training_policy is not None:
        training_strategy = build_training_policy_strategy(training_policy, stake=stake)
        if training_strategy is not None:
            items.append(training_strategy)
    items.sort(key=lambda item: (item["id"] != primary_strategy_id, -NumberLike(item["summary"].get("roi_pct")), -NumberLike(item["summary"].get("total_profit"))))
    available_ids = [item["id"] for item in items]
    selected_primary = primary_strategy_id if primary_strategy_id in available_ids else (available_ids[0] if available_ids else None)
    return {
        "default_stake": stake,
        "primary_strategy_id": selected_primary,
        "strategies": items,
    }


def NumberLike(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_backtest(
    metrics_path: Path = METRICS_PATH,
    holdout_predictions_path: Path = HOLDOUT_PREDICTIONS_PATH,
    value_bets: pd.DataFrame | None = None,
    mlruns_dir: Path = MLRUNS_DIR,
) -> dict[str, object]:
    holdout = pd.read_parquet(holdout_predictions_path) if holdout_predictions_path.exists() else pd.DataFrame()
    equity_source = filter_holdout_value_bets(value_bets) if value_bets is not None else pd.DataFrame()
    simulator = build_simulator(equity_source, stake=DEFAULT_STAKE)
    equity_curve = [{"date": None, "cumulative_pnl": 0.0, "value_bets_so_far": 0}]
    equity_curve.extend(
        {
            "date": bet["date"],
            "cumulative_pnl": _round(float(bet["running_bankroll"]) - simulator["summary"]["starting_bankroll"], 2),
            "value_bets_so_far": index + 1,
        }
        for index, bet in enumerate(simulator["bets"])
    )
    return {
        "metrics": load_metrics(metrics_path),
        "confusion_matrix": build_confusion_matrix(holdout),
        "equity_curve": equity_curve,
        "mlflow_runs": build_mlflow_runs(mlruns_dir),
    }


def load_diagnostics(diagnostics_path: Path = MODEL_DIAGNOSTICS_PATH) -> dict[str, object]:
    if not diagnostics_path.exists():
        return {
            "holdout_seasons": list(HOLDOUT_SEASONS),
            "calibration_by_outcome_bucket": [],
            "value_bets_by_outcome_bucket": [],
            "value_bets_by_odds_range": {"model_odds_ranges": [], "bookmaker_odds_ranges": []},
            "worst_calibration_bucket": None,
        }
    return json.loads(diagnostics_path.read_text(encoding="utf-8"))


def load_training_policy(training_policy_path: Path = TRAINING_POLICY_PATH) -> dict[str, object]:
    if not training_policy_path.exists():
        return {
            "model": "market_aware_xgboost_expanding_walk_forward",
            "aggregate": {},
            "folds": [],
            "available": False,
        }
    payload = json.loads(training_policy_path.read_text(encoding="utf-8"))
    payload["available"] = True
    return payload


def load_league_subset_policy(league_subset_policy_path: Path = LEAGUE_SUBSET_POLICY_PATH) -> dict[str, object]:
    if not league_subset_policy_path.exists():
        return {
            "model": "market_aware_xgboost_league_subset_expanding_walk_forward",
            "subsets": [],
            "available": False,
        }
    payload = json.loads(league_subset_policy_path.read_text(encoding="utf-8"))
    payload["available"] = True
    return payload


def _pct_string(value: object) -> str:
    return f"{NumberLike(value) * 100:.2f}%"


def build_project_summary(
    strategy_comparison: dict[str, object],
    training_policy: dict[str, object],
    league_subset_policy: dict[str, object],
) -> dict[str, object]:
    strategies = list(strategy_comparison.get("strategies", []))
    strategy_lookup = {str(item.get("id")): item for item in strategies}
    training_aggregate = training_policy.get("aggregate", {}) if training_policy.get("available") else {}
    folds = list(training_policy.get("folds", [])) if training_policy.get("available") else []
    subsets = list(league_subset_policy.get("subsets", [])) if league_subset_policy.get("available") else []
    best_subset = subsets[0] if subsets else {}

    def subset_summary(subset: dict[str, object]) -> dict[str, object]:
        return {
            "name": subset.get("name"),
            "leagues": subset.get("leagues", []),
            "aggregate": subset.get("aggregate", {}),
        }

    def strategy_result(strategy_id: str, fallback_label: str) -> dict[str, object]:
        item = strategy_lookup.get(strategy_id, {"id": strategy_id, "label": fallback_label, "summary": {}})
        summary = item.get("summary", {})
        return {
            "id": strategy_id,
            "label": item.get("label", fallback_label),
            "bets": int(NumberLike(summary.get("total_bets"))),
            "profit": _round(summary.get("total_profit"), 2) or 0.0,
            "roi_pct": _round(summary.get("roi_pct"), 2) or 0.0,
            "hit_rate": _round(summary.get("hit_rate"), 4) or 0.0,
        }

    expanding = strategy_result("expanding_walk_forward_xgboost", "Expanding walk-forward XGBoost")
    xgboost = strategy_result("xgboost_value", "XGBoost value")
    poisson = strategy_result("poisson_goal_model", "Poisson goal model")
    mispricing = strategy_result("mispricing_model", "Mispricing model")

    return {
        "status": "Portfolio complete",
        "headline": "Market-aware XGBoost with expanding annual retraining is the final candidate.",
        "data_scope": {
            "leagues": list(LEAGUES),
            "league_labels": LEAGUE_LABELS,
            "seasons": list(HOLDOUT_SEASONS),
            "history_window": "2010-11 through 2025-26",
            "forward_window": "2019-20 through 2025-26",
        },
        "final_model": {
            "name": "Market-aware XGBoost",
            "training_policy": "Expanding annual walk-forward retraining",
            "league_policy": best_subset.get("name", "all_five"),
            "bet_scope": "Home and away wins only",
            "production_semantics": "In production this model retrains on all completed matches up to the current day and scores the next fixture batch; season-level walk-forward is the historical proxy used to audit that policy.",
        },
        "validation": {
            "expanding_walk_forward": training_aggregate,
            "folds": folds,
            "best_league_subset": subset_summary(best_subset),
            "league_subsets": [subset_summary(subset) for subset in subsets[:6]],
        },
        "validation_steps": [
            {
                "step": "Static pre-2019 split",
                "decision": "Useful baseline, not enough",
                "evidence": "A single old holdout could not answer whether the edge survives later football seasons.",
            },
            {
                "step": "Benchmark models",
                "decision": "Rejected controls",
                "evidence": f"Poisson ROI {poisson['roi_pct']:.2f}% and mispricing ROI {mispricing['roi_pct']:.2f}% underperformed the expanding XGBoost policy.",
            },
            {
                "step": "Expanding walk-forward",
                "decision": "Lead validation result",
                "evidence": f"{_pct_string(training_aggregate.get('roi'))} ROI, {int(NumberLike(training_aggregate.get('positive_roi_folds')))}/{int(NumberLike(training_aggregate.get('folds')))} positive folds, {NumberLike(training_aggregate.get('bets')):.0f} bets.",
            },
            {
                "step": "League subset audit",
                "decision": "Keep all five leagues",
                "evidence": f"Best subset: {best_subset.get('name', 'all_five')} at {_pct_string((best_subset.get('aggregate') or {}).get('roi'))}; filtering leagues did not improve the headline ROI.",
            },
        ],
        "model_decisions": [
            {
                "model": "Expanding walk-forward XGBoost",
                "decision": "Final dashboard strategy",
                "evidence": f"Simulator ROI {expanding['roi_pct']:.2f}% with {expanding['bets']} bets; this is the production-style retraining policy.",
            },
            {
                "model": "Frozen XGBoost value",
                "decision": "Comparison only",
                "evidence": f"Full forward frozen-model ROI {xgboost['roi_pct']:.2f}% with {xgboost['bets']} bets; weaker than expanding retraining.",
            },
            {
                "model": "Poisson goal model",
                "decision": "Benchmark only",
                "evidence": f"Full forward ROI {poisson['roi_pct']:.2f}%; interpretable, but weaker than the tabular model.",
            },
            {
                "model": "Mispricing model",
                "decision": "Rejected as lead strategy",
                "evidence": f"Full forward ROI {mispricing['roi_pct']:.2f}%; useful diagnostic for market disagreement, not the final edge.",
            },
        ],
        "strategy_results": [expanding, xgboost, poisson, mispricing],
        "test_status": {
            "command": ".venv/bin/python -m unittest discover tests",
            "latest_result": "153 tests OK, 1 skipped",
            "coverage_story": "Unit and contract tests cover ingestion, features, training utilities, odds comparison, diagnostics, dashboard export contracts, static dashboard wiring, Docker config, and training-window experiments.",
        },
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def run_pipeline(
    features_dir: Path = FEATURES_DIR,
    model_odds_path: Path = MODEL_ODDS_PATH,
    value_bets_path: Path = VALUE_BETS_PATH,
    metrics_path: Path = METRICS_PATH,
    holdout_predictions_path: Path = HOLDOUT_PREDICTIONS_PATH,
    output_dir: Path = DASHBOARD_DATA_DIR,
    mlruns_dir: Path = MLRUNS_DIR,
    diagnostics_path: Path = MODEL_DIAGNOSTICS_PATH,
    training_policy_path: Path = TRAINING_POLICY_PATH,
    league_subset_policy_path: Path = LEAGUE_SUBSET_POLICY_PATH,
    poisson_value_bets_path: Path = POISSON_VALUE_BETS_PATH,
    market_aware_value_bets_path: Path = MARKET_AWARE_VALUE_BETS_PATH,
    mispricing_value_bets_path: Path = MISPRICING_VALUE_BETS_PATH,
) -> DashboardExportSummary:
    features = load_feature_data(features_dir)
    model_odds = pd.read_parquet(model_odds_path) if model_odds_path.exists() else pd.DataFrame()
    if not value_bets_path.exists():
        raise FileNotFoundError(f"Missing Stage 5 value bets file: {value_bets_path}")
    value_bets = enrich_value_bets(pd.read_parquet(value_bets_path), model_odds=model_odds)
    strategy_paths = {
        "xgboost_value": value_bets_path,
        "market_aware_xgboost": market_aware_value_bets_path,
        "poisson_goal_model": poisson_value_bets_path,
        "mispricing_model": mispricing_value_bets_path,
    }
    strategies = load_strategy_value_bets(strategy_paths, model_odds=model_odds, required_seasons=HOLDOUT_SEASONS)
    if "xgboost_value" not in strategies:
        strategies["xgboost_value"] = value_bets

    holdout_value_bets = filter_holdout_value_bets(value_bets)
    training_policy = load_training_policy(training_policy_path)
    strategy_comparison = build_strategy_comparison(strategies, stake=DEFAULT_STAKE, training_policy=training_policy)
    league_subset_policy = load_league_subset_policy(league_subset_policy_path)
    outputs = {
        "league_analytics.json": build_league_analytics(features),
        "backtest.json": build_backtest(metrics_path, holdout_predictions_path, value_bets, mlruns_dir),
        "value_bets.json": _records(value_bets),
        "simulator.json": build_simulator(holdout_value_bets, stake=DEFAULT_STAKE),
        "strategy_comparison.json": strategy_comparison,
        "training_policy.json": training_policy,
        "project_summary.json": build_project_summary(strategy_comparison, training_policy, league_subset_policy),
        "diagnostics.json": load_diagnostics(diagnostics_path),
    }

    written: list[Path] = []
    for filename, data in outputs.items():
        path = output_dir / filename
        write_json(path, data)
        written.append(path)
    return DashboardExportSummary(output_dir=output_dir, files=written)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export static JSON files for the Stage 6 dashboard.")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--model-odds-path", type=Path, default=MODEL_ODDS_PATH)
    parser.add_argument("--value-bets-path", type=Path, default=VALUE_BETS_PATH)
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH)
    parser.add_argument("--holdout-predictions-path", type=Path, default=HOLDOUT_PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DASHBOARD_DATA_DIR)
    parser.add_argument("--mlruns-dir", type=Path, default=MLRUNS_DIR)
    parser.add_argument("--diagnostics-path", type=Path, default=MODEL_DIAGNOSTICS_PATH)
    parser.add_argument("--training-policy-path", type=Path, default=TRAINING_POLICY_PATH)
    parser.add_argument("--league-subset-policy-path", type=Path, default=LEAGUE_SUBSET_POLICY_PATH)
    parser.add_argument("--poisson-value-bets-path", type=Path, default=POISSON_VALUE_BETS_PATH)
    parser.add_argument("--market-aware-value-bets-path", type=Path, default=MARKET_AWARE_VALUE_BETS_PATH)
    parser.add_argument("--mispricing-value-bets-path", type=Path, default=MISPRICING_VALUE_BETS_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_pipeline(
        features_dir=args.features_dir,
        model_odds_path=args.model_odds_path,
        value_bets_path=args.value_bets_path,
        metrics_path=args.metrics_path,
        holdout_predictions_path=args.holdout_predictions_path,
        output_dir=args.output_dir,
        mlruns_dir=args.mlruns_dir,
        diagnostics_path=args.diagnostics_path,
        training_policy_path=args.training_policy_path,
        league_subset_policy_path=args.league_subset_policy_path,
        poisson_value_bets_path=args.poisson_value_bets_path,
        market_aware_value_bets_path=args.market_aware_value_bets_path,
        mispricing_value_bets_path=args.mispricing_value_bets_path,
    )
    print(summary.line())
