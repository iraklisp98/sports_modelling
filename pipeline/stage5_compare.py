from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import unicodedata

import pandas as pd

MODEL_ODDS_PATH = Path("data/output/model_odds.parquet")
FOOTBALL_DATA_DIR = Path("data/bookmaker_odds/football_data")
VALUE_BETS_PATH = Path("data/output/value_bets.parquet")
DASHBOARD_JSON_PATH = Path("dashboard/data/value_bets.json")
EDGE_THRESHOLD = 0.10
MIN_MODEL_PROBABILITY = 0.35
MIN_BOOKMAKER_ODDS = 1.20
MAX_BOOKMAKER_ODDS = 8.0
MAX_EDGE = 0.30
FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
SEASON_CODES = ("1718", "1819", "1920")
FOOTBALL_DATA_LEAGUE_CODES = {"ENG": "E0", "SPA": "SP1", "FRA": "F1"}

LEAGUE_FILE_PATTERNS = {
    "ENG": ("E0*.csv", "eng*.csv", "premier*.csv"),
    "SPA": ("SP1*.csv", "spa*.csv", "la_liga*.csv"),
    "FRA": ("F1*.csv", "fra*.csv", "ligue*.csv"),
}

BOOKMAKER_PREFIXES = {
    "B365": "Bet365",
    "BW": "Bet&Win",
    "IW": "Interwetten",
    "LB": "Ladbrokes",
    "PS": "Pinnacle",
    "SB": "Sportingbet",
    "SJ": "Stan James",
    "VC": "VC Bet",
    "WH": "William Hill",
}

REQUIRED_MODEL_ODDS_COLUMNS = [
    "RBallID",
    "HomeTeam",
    "AwayTeam",
    "Date",
    "Season",
    "Result",
    "ModelOdds_Home",
    "ModelOdds_Draw",
    "ModelOdds_Away",
]

REQUIRED_FOOTBALL_DATA_COLUMNS = ["Date", "HomeTeam", "AwayTeam"]

OUTPUT_COLUMNS = [
    "RBallID",
    "HomeTeam",
    "AwayTeam",
    "Date",
    "Season",
    "League",
    "Result",
    "Outcome",
    "ModelOdds",
    "BestBookOdds",
    "Edge",
    "ValueBet",
    "BestBookmaker",
]
VALUE_BET_COLUMNS = OUTPUT_COLUMNS
MATCH_COLUMNS = ["RBallID"]
NORMALIZED_BOOKMAKER_COLUMNS = [
    "RBallID",
    "Bookmaker",
    "Odds_Home",
    "Odds_Draw",
    "Odds_Away",
]



TEAM_NAME_ALIASES = {
    "afc bournemouth": "bournemouth",
    "as monaco fc": "monaco",
    "as monaco": "monaco",
    "as saint etienne": "st etienne",
    "athletic club bilbao": "ath bilbao",
    "atletico de madrid": "ath madrid",
    "brighton and hove albion fc": "brighton",
    "brighton and hove albion": "brighton",
    "ca osasuna": "osasuna",
    "cd leganes": "leganes",
    "deportivo alaves": "alaves",
    "deportivo la coruna": "la coruna",
    "ea guingamp": "guingamp",
    "fc barcelona": "barcelona",
    "fc girondins de bordeaux": "bordeaux",
    "fc metz": "metz",
    "fc nantes": "nantes",
    "man city": "manchester city",
    "man united": "manchester united",
    "manchester city fc": "manchester city",
    "manchester united fc": "manchester united",
    "nimes olympique": "nimes",
    "olympique lyonnais": "lyon",
    "olympique de marseille": "marseille",
    "paris saint germain fc": "paris sg",
    "paris saint germain": "paris sg",
    "rc celta de vigo": "celta",
    "rcd espanyol barcelona": "espanyol",
    "rcd espanyol": "espanyol",
    "rcd mallorca": "mallorca",
    "rcs strasbourg": "strasbourg",
    "real betis balompie": "betis",
    "real madrid cf": "real madrid",
    "real sociedad": "sociedad",
    "real valladolid cf": "valladolid",
    "sd eibar": "eibar",
    "sd huesca": "huesca",
    "stade brestois 29": "brest",
    "stade rennais": "rennes",
    "stade de reims": "reims",
    "tottenham hotspur fc": "tottenham",
    "ud las palmas": "las palmas",
    "west bromwich albion fc": "west brom",
    "west ham united fc": "west ham",
    "wolverhampton wanderers fc": "wolves",
}

@dataclass(frozen=True)
class ValueBetRiskPolicy:
    min_model_probability: float = MIN_MODEL_PROBABILITY
    min_bookmaker_odds: float = MIN_BOOKMAKER_ODDS
    max_bookmaker_odds: float = MAX_BOOKMAKER_ODDS
    max_edge: float = MAX_EDGE


@dataclass(frozen=True)
class OddsComparisonSummary:
    model_rows: int
    matched_rows: int
    value_bets: int
    output_path: Path
    dashboard_json_path: Path

    def line(self) -> str:
        return (
            f"model_rows={self.model_rows}, matched_rows={self.matched_rows}, "
            f"value_bets={self.value_bets}, output={self.output_path}, "
            f"dashboard_json={self.dashboard_json_path}"
        )


def _clean_team_name(name: object) -> str:
    normalised = unicodedata.normalize("NFKD", str(name))
    ascii_name = normalised.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.casefold().replace("&", " and ")
    ascii_name = re.sub(r"[^a-z0-9]+", " ", ascii_name)
    return " ".join(ascii_name.split())


def normalise_team_name(name: object) -> str:
    cleaned = _clean_team_name(name)
    if cleaned in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[cleaned]

    tokens = cleaned.split()
    while tokens and tokens[-1] in {"fc", "cf", "afc", "sc", "sco", "fco", "ac", "ud"}:
        tokens.pop()
    simplified = " ".join(tokens)
    return TEAM_NAME_ALIASES.get(simplified, simplified)


def parse_football_data_date(values: pd.Series) -> pd.Series:
    first_pass = pd.to_datetime(values, dayfirst=True, errors="coerce", format="mixed")
    second_pass = pd.to_datetime(values, errors="coerce", format="mixed")
    return first_pass.fillna(second_pass).dt.normalize()


def season_from_date(dates: pd.Series) -> pd.Series:
    years = dates.dt.year
    starts = years.where(dates.dt.month >= 8, years - 1)
    ends = (starts + 1).astype(str).str[-2:]
    return starts.astype(str) + "-" + ends


def validate_model_odds_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_MODEL_ODDS_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Stage 4 model odds columns: {missing}")


def validate_football_data_columns(df: pd.DataFrame, source: Path | str = "Football-Data CSV") -> None:
    missing = [column for column in REQUIRED_FOOTBALL_DATA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Football-Data columns in {source}: {missing}")


def bookmaker_columns(columns: Iterable[str]) -> dict[str, dict[str, str]]:
    available = set(columns)
    mapping: dict[str, dict[str, str]] = {}
    for prefix in BOOKMAKER_PREFIXES:
        required = {outcome: f"{prefix}{suffix}" for outcome, suffix in {"H": "H", "D": "D", "A": "A"}.items()}
        if set(required.values()).issubset(available):
            mapping[prefix] = required
    return mapping


def load_football_data_csv(path: Path, league: str | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    validate_football_data_columns(raw, source=path)

    odds_columns = bookmaker_columns(raw.columns)
    if not odds_columns:
        raise ValueError(f"No supported bookmaker odds columns found in {path}")

    normalised = raw[["Date", "HomeTeam", "AwayTeam"]].copy()
    normalised["Date"] = parse_football_data_date(normalised["Date"])
    normalised["League"] = league or infer_league_from_file(path)
    normalised["Season"] = season_from_date(normalised["Date"])
    normalised["HomeTeamKey"] = normalised["HomeTeam"].map(normalise_team_name)
    normalised["AwayTeamKey"] = normalised["AwayTeam"].map(normalise_team_name)

    for prefix, columns in odds_columns.items():
        for outcome, source_column in columns.items():
            normalised[f"{prefix}_{outcome}"] = pd.to_numeric(raw[source_column], errors="coerce")

    normalised = normalised.dropna(subset=["Date", "HomeTeamKey", "AwayTeamKey"]).reset_index(drop=True)
    return normalised


def infer_league_from_file(path: Path) -> str:
    name = path.name.casefold()
    if name.startswith("e0") or "premier" in name or "eng" in name:
        return "ENG"
    if name.startswith("sp1") or "liga" in name or "spa" in name:
        return "SPA"
    if name.startswith("f1") or "ligue" in name or "fra" in name:
        return "FRA"
    return "UNKNOWN"


def football_data_url(season_code: str, league: str) -> str:
    league_code = FOOTBALL_DATA_LEAGUE_CODES[league]
    return f"{FOOTBALL_DATA_BASE_URL}/{season_code}/{league_code}.csv"


def football_data_cache_path(odds_dir: Path, season_code: str, league: str) -> Path:
    league_code = FOOTBALL_DATA_LEAGUE_CODES[league]
    return odds_dir / f"{league_code}_{season_code}.csv"


def download_football_data_odds(
    odds_dir: Path = FOOTBALL_DATA_DIR,
    season_codes: Iterable[str] = SEASON_CODES,
    leagues: Iterable[str] = tuple(FOOTBALL_DATA_LEAGUE_CODES),
) -> list[Path]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required to download Football-Data odds CSVs") from exc

    odds_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for season_code in season_codes:
        for league in leagues:
            output_path = football_data_cache_path(odds_dir, season_code, league)
            if output_path.exists():
                downloaded.append(output_path)
                continue

            response = requests.get(football_data_url(season_code, league), timeout=30)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            downloaded.append(output_path)
    return downloaded


def ensure_football_data_files(odds_dir: Path = FOOTBALL_DATA_DIR) -> None:
    if odds_dir.is_file():
        return

    expected_paths = [
        football_data_cache_path(odds_dir, season_code, league)
        for season_code in SEASON_CODES
        for league in FOOTBALL_DATA_LEAGUE_CODES
    ]
    if all(path.exists() for path in expected_paths):
        return
    download_football_data_odds(odds_dir)


def resolve_football_data_files(odds_dir: Path = FOOTBALL_DATA_DIR) -> list[tuple[Path, str]]:
    if odds_dir.is_file():
        return [(odds_dir, infer_league_from_file(odds_dir))]
    if not odds_dir.exists():
        raise FileNotFoundError(f"Missing Football-Data odds directory: {odds_dir}")

    resolved: list[tuple[Path, str]] = []
    for league, patterns in LEAGUE_FILE_PATTERNS.items():
        league_files: list[Path] = []
        for pattern in patterns:
            league_files.extend(odds_dir.glob(pattern))
        resolved.extend((path, league) for path in sorted(set(league_files)))

    if not resolved:
        resolved = [(path, infer_league_from_file(path)) for path in sorted(odds_dir.glob("*.csv"))]
    if not resolved:
        raise FileNotFoundError(f"No Football-Data CSV files found in {odds_dir}")
    return resolved


def load_football_data_odds(odds_dir: Path = FOOTBALL_DATA_DIR) -> pd.DataFrame:
    ensure_football_data_files(odds_dir)
    frames = [load_football_data_csv(path, league=league) for path, league in resolve_football_data_files(odds_dir)]
    if not frames:
        raise ValueError("At least one Football-Data odds CSV is required")
    return pd.concat(frames, ignore_index=True)


def add_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    keyed["Date"] = pd.to_datetime(keyed["Date"]).dt.normalize()
    keyed["HomeTeamKey"] = keyed["HomeTeam"].map(normalise_team_name)
    keyed["AwayTeamKey"] = keyed["AwayTeam"].map(normalise_team_name)
    return keyed


def best_odds_for_outcome(row: pd.Series, outcome: str) -> tuple[float | None, str | None]:
    candidates: list[tuple[float, str]] = []
    suffix = f"_{outcome}"
    for column, value in row.items():
        if not str(column).endswith(suffix) or pd.isna(value):
            continue
        price = float(value)
        if price <= 1.0:
            continue
        prefix = str(column).removesuffix(suffix)
        candidates.append((price, BOOKMAKER_PREFIXES.get(prefix, prefix)))

    if not candidates:
        return None, None
    price, bookmaker = max(candidates, key=lambda item: item[0])
    return price, bookmaker


def validate_risk_policy(policy: ValueBetRiskPolicy) -> None:
    values = {
        "min_model_probability": policy.min_model_probability,
        "min_bookmaker_odds": policy.min_bookmaker_odds,
        "max_bookmaker_odds": policy.max_bookmaker_odds,
        "max_edge": policy.max_edge,
    }
    non_finite = [name for name, value in values.items() if not math.isfinite(float(value))]
    if non_finite:
        raise ValueError(f"Risk policy values must be finite: {non_finite}")
    if policy.min_model_probability <= 0.0 or policy.min_model_probability >= 1.0:
        raise ValueError("min_model_probability must be between 0 and 1")
    if policy.min_bookmaker_odds <= 1.0:
        raise ValueError("min_bookmaker_odds must be greater than 1.0")
    if policy.max_bookmaker_odds <= policy.min_bookmaker_odds:
        raise ValueError("max_bookmaker_odds must be greater than min_bookmaker_odds")
    if policy.max_edge < 0.0:
        raise ValueError("max_edge must be non-negative")


def model_probability_from_odds(model_odds: float) -> float:
    if model_odds <= 0.0:
        raise ValueError("model_odds must be positive")
    return 1.0 / model_odds


def passes_value_bet_risk_policy(
    model_odds: float,
    best_book_odds: float,
    edge: float,
    policy: ValueBetRiskPolicy,
) -> bool:
    validate_risk_policy(policy)
    if model_probability_from_odds(model_odds) < policy.min_model_probability:
        return False
    if best_book_odds < policy.min_bookmaker_odds or best_book_odds > policy.max_bookmaker_odds:
        return False
    if edge > policy.max_edge:
        return False
    return True

def validate_normalized_bookmaker_columns(df: pd.DataFrame, match_columns: Iterable[str]) -> None:
    required = [*match_columns, *NORMALIZED_BOOKMAKER_COLUMNS[1:]]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required normalized bookmaker odds columns: {missing}")


def _best_normalized_offer(group: pd.DataFrame, odds_column: str) -> pd.Series:
    odds = pd.to_numeric(group[odds_column], errors="coerce")
    valid = group.loc[odds.notna() & (odds > 0)].copy()
    if valid.empty:
        return pd.Series({"odds": pd.NA, "bookmaker": pd.NA})

    valid["_odds"] = pd.to_numeric(valid[odds_column], errors="raise")
    best_idx = valid["_odds"].idxmax()
    return pd.Series(
        {
            "odds": float(valid.loc[best_idx, "_odds"]),
            "bookmaker": valid.loc[best_idx, "Bookmaker"],
        }
    )


def build_best_bookmaker_odds_frame(
    bookmaker_odds_df: pd.DataFrame,
    match_columns: Iterable[str] = MATCH_COLUMNS,
) -> pd.DataFrame:
    match_columns = list(match_columns)
    validate_normalized_bookmaker_columns(bookmaker_odds_df, match_columns)

    rows: list[dict[str, object]] = []
    outcome_specs = [
        ("Odds_Home", "BestOdds_Home", "BestBookmaker_Home"),
        ("Odds_Draw", "BestOdds_Draw", "BestBookmaker_Draw"),
        ("Odds_Away", "BestOdds_Away", "BestBookmaker_Away"),
    ]
    for match_key, group in bookmaker_odds_df.groupby(match_columns, dropna=False, sort=False):
        if len(match_columns) == 1:
            value = match_key[0] if isinstance(match_key, tuple) else match_key
            output_row = {match_columns[0]: value}
        else:
            output_row = dict(zip(match_columns, match_key))

        for odds_column, best_odds_column, best_bookmaker_column in outcome_specs:
            best = _best_normalized_offer(group, odds_column)
            output_row[best_odds_column] = best["odds"]
            output_row[best_bookmaker_column] = best["bookmaker"]
        rows.append(output_row)

    return pd.DataFrame(rows)


def compare_model_to_bookmaker_odds(
    model_odds_df: pd.DataFrame,
    bookmaker_odds_df: pd.DataFrame,
    edge_threshold: float = EDGE_THRESHOLD,
    match_columns: Iterable[str] = MATCH_COLUMNS,
    risk_policy: ValueBetRiskPolicy = ValueBetRiskPolicy(),
) -> pd.DataFrame:
    match_columns = list(match_columns)
    validate_model_odds_columns(model_odds_df)
    missing_match_columns = [column for column in match_columns if column not in model_odds_df.columns]
    if missing_match_columns:
        raise ValueError(f"Missing required model odds match columns: {missing_match_columns}")
    if edge_threshold < 0:
        raise ValueError("edge_threshold must be non-negative")
    validate_risk_policy(risk_policy)

    best_odds = build_best_bookmaker_odds_frame(bookmaker_odds_df, match_columns=match_columns)
    merged = model_odds_df.merge(best_odds, on=match_columns, how="inner")

    value_bets: list[dict[str, object]] = []
    outcome_specs = [
        ("H", "ModelOdds_Home", "BestOdds_Home", "BestBookmaker_Home"),
        ("D", "ModelOdds_Draw", "BestOdds_Draw", "BestBookmaker_Draw"),
        ("A", "ModelOdds_Away", "BestOdds_Away", "BestBookmaker_Away"),
    ]
    for _, row in merged.iterrows():
        for outcome, model_column, best_odds_column, best_bookmaker_column in outcome_specs:
            model_odds = pd.to_numeric(row[model_column], errors="coerce")
            best_book_odds = pd.to_numeric(row[best_odds_column], errors="coerce")
            if pd.isna(model_odds) or pd.isna(best_book_odds) or model_odds <= 0 or best_book_odds <= 0:
                continue

            model_odds = float(model_odds)
            best_book_odds = float(best_book_odds)
            edge = (best_book_odds / model_odds) - 1.0
            if edge >= edge_threshold and passes_value_bet_risk_policy(model_odds, best_book_odds, edge, risk_policy):
                value_bets.append(
                    {
                        "RBallID": row["RBallID"],
                        "HomeTeam": row["HomeTeam"],
                        "AwayTeam": row["AwayTeam"],
                        "Date": row["Date"],
                        "Season": row["Season"],
                        "League": row.get("League", pd.NA),
                        "Result": row["Result"],
                        "Outcome": outcome,
                        "ModelOdds": model_odds,
                        "BestBookOdds": best_book_odds,
                        "Edge": edge,
                        "ValueBet": True,
                        "BestBookmaker": row[best_bookmaker_column],
                    }
                )

    output = pd.DataFrame(value_bets, columns=VALUE_BET_COLUMNS)
    if not output.empty:
        outcome_order = {"H": 0, "D": 1, "A": 2}
        output["_OutcomeOrder"] = output["Outcome"].map(outcome_order)
        output = (
            output.sort_values(["Date", "RBallID", "_OutcomeOrder"], kind="mergesort")
            .drop(columns=["_OutcomeOrder"])
            .reset_index(drop=True)
        )
    return output


def match_model_to_bookmaker_odds(model_odds: pd.DataFrame, bookmaker_odds: pd.DataFrame) -> pd.DataFrame:
    validate_model_odds_columns(model_odds)
    validate_football_data_columns(bookmaker_odds)

    model_keyed = add_match_keys(model_odds)
    bookmaker_keyed = add_match_keys(bookmaker_odds)

    join_columns = ["Date", "HomeTeamKey", "AwayTeamKey"]
    bookmaker_value_columns = [
        column
        for column in bookmaker_keyed.columns
        if any(column.endswith(f"_{outcome}") for outcome in ("H", "D", "A"))
    ]
    if "League" in bookmaker_keyed.columns:
        bookmaker_value_columns.append("League")
    merged = model_keyed.merge(
        bookmaker_keyed[join_columns + bookmaker_value_columns],
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    return merged.drop(columns=["HomeTeamKey", "AwayTeamKey"])


def compute_value_bets(
    matched_odds: pd.DataFrame,
    edge_threshold: float = EDGE_THRESHOLD,
    risk_policy: ValueBetRiskPolicy = ValueBetRiskPolicy(),
) -> pd.DataFrame:
    if edge_threshold < 0:
        raise ValueError("edge_threshold must be non-negative")
    validate_risk_policy(risk_policy)

    rows: list[dict[str, object]] = []
    for row in matched_odds.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        for outcome, model_column in [
            ("H", "ModelOdds_Home"),
            ("D", "ModelOdds_Draw"),
            ("A", "ModelOdds_Away"),
        ]:
            best_odds, bookmaker = best_odds_for_outcome(row_series, outcome)
            if best_odds is None:
                continue

            model_odds = float(row_series[model_column])
            if model_odds <= 0.0:
                continue
            edge = (best_odds / model_odds) - 1.0
            if edge >= edge_threshold and passes_value_bet_risk_policy(model_odds, best_odds, edge, risk_policy):
                rows.append(
                    {
                        "RBallID": row_series["RBallID"],
                        "HomeTeam": row_series["HomeTeam"],
                        "AwayTeam": row_series["AwayTeam"],
                        "Date": row_series["Date"],
                        "Season": row_series["Season"],
                        "League": row_series.get("League", pd.NA),
                        "Result": row_series["Result"],
                        "Outcome": outcome,
                        "ModelOdds": model_odds,
                        "BestBookOdds": best_odds,
                        "Edge": edge,
                        "ValueBet": True,
                        "BestBookmaker": bookmaker,
                    }
                )

    value_bets = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if not value_bets.empty:
        value_bets["_OutcomeOrder"] = value_bets["Outcome"].map({"H": 0, "D": 1, "A": 2})
        value_bets = value_bets.sort_values(["Date", "RBallID", "_OutcomeOrder"], kind="mergesort").drop(columns=["_OutcomeOrder"]).reset_index(drop=True)
    return value_bets


def write_outputs(value_bets: pd.DataFrame, output_path: Path, dashboard_json_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    value_bets.to_parquet(output_path, index=False)

    dashboard_json_path.parent.mkdir(parents=True, exist_ok=True)
    value_bets.to_json(dashboard_json_path, orient="records", date_format="iso")


def run_pipeline(
    model_odds_path: Path = MODEL_ODDS_PATH,
    football_data_dir: Path = FOOTBALL_DATA_DIR,
    output_path: Path = VALUE_BETS_PATH,
    dashboard_json_path: Path = DASHBOARD_JSON_PATH,
    edge_threshold: float = EDGE_THRESHOLD,
    risk_policy: ValueBetRiskPolicy = ValueBetRiskPolicy(),
) -> OddsComparisonSummary:
    if not model_odds_path.exists():
        raise FileNotFoundError(f"Missing Stage 4 model odds file: {model_odds_path}")

    model_odds = pd.read_parquet(model_odds_path)
    bookmaker_odds = load_football_data_odds(football_data_dir)
    matched = match_model_to_bookmaker_odds(model_odds, bookmaker_odds)
    if matched.empty:
        raise ValueError("Stage 5 matched zero model rows to Football-Data odds; check team names, dates, and input seasons")
    value_bets = compute_value_bets(matched, edge_threshold=edge_threshold, risk_policy=risk_policy)
    write_outputs(value_bets, output_path=output_path, dashboard_json_path=dashboard_json_path)

    return OddsComparisonSummary(
        model_rows=len(model_odds),
        matched_rows=len(matched),
        value_bets=len(value_bets),
        output_path=output_path,
        dashboard_json_path=dashboard_json_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model odds to Football-Data historical bookmaker odds.")
    parser.add_argument("--model-odds-path", type=Path, default=MODEL_ODDS_PATH)
    parser.add_argument("--football-data-dir", type=Path, default=FOOTBALL_DATA_DIR)
    parser.add_argument("--output-path", type=Path, default=VALUE_BETS_PATH)
    parser.add_argument("--dashboard-json-path", type=Path, default=DASHBOARD_JSON_PATH)
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD)
    parser.add_argument("--min-model-probability", type=float, default=MIN_MODEL_PROBABILITY)
    parser.add_argument("--min-bookmaker-odds", type=float, default=MIN_BOOKMAKER_ODDS)
    parser.add_argument("--max-bookmaker-odds", type=float, default=MAX_BOOKMAKER_ODDS)
    parser.add_argument("--max-edge", type=float, default=MAX_EDGE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_pipeline(
        model_odds_path=args.model_odds_path,
        football_data_dir=args.football_data_dir,
        output_path=args.output_path,
        dashboard_json_path=args.dashboard_json_path,
        edge_threshold=args.edge_threshold,
        risk_policy=ValueBetRiskPolicy(
            min_model_probability=args.min_model_probability,
            min_bookmaker_odds=args.min_bookmaker_odds,
            max_bookmaker_odds=args.max_bookmaker_odds,
            max_edge=args.max_edge,
        ),
    )
    print(summary.line())
