from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import shutil
import unicodedata

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


DEFAULT_LEAGUES = (
    ("ENG", "data/ENG", "data/processed/ENG.parquet"),
    ("SPA", "data/SPA", "data/processed/SPA.parquet"),
    ("FRA", "data/FRA", "data/processed/FRA.parquet"),
    ("GER", "data/GER", "data/processed/GER.parquet"),
    ("ITA", "data/ITA", "data/processed/ITA.parquet"),
)

FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_DATA_DIR = Path("data/bookmaker_odds/football_data")
PROCESSED_DIR = Path("data/processed")
FOOTBALL_DATA_LEAGUE_CODES = {"ENG": "E0", "SPA": "SP1", "FRA": "F1", "GER": "D1", "ITA": "I1"}
DEFAULT_SEASON_CODES = tuple(f"{year:02d}{year + 1:02d}" for year in range(10, 26))

INCIDENT_MAP = {
    "GOAL1": "HomeGoals",
    "GOAL2": "AwayGoals",
    "CR1": "HomeCorners",
    "CR2": "AwayCorners",
    "SHG1": "HomeShotsOnTarget",
    "SHG2": "AwayShotsOnTarget",
    "SF1": "HomeFouls",
    "SF2": "AwayFouls",
    "OS1": "HomeOffsides",
    "OS2": "AwayOffsides",
}

MATCH_KEYS = ["RBallID", "HomeTeam", "AwayTeam", "Date", "Season"]
METRIC_COLUMNS = list(INCIDENT_MAP.values())
OUTPUT_COLUMNS = MATCH_KEYS + METRIC_COLUMNS
REQUIRED_RAW_COLUMNS = [
    "RBallID",
    "HomeTeam",
    "AwayTeam",
    "Timestamp",
    "Incident",
    "Minute",
]


@dataclass(frozen=True)
class QualityReport:
    league: str
    row_count: int
    null_counts: dict[str, int]
    min_date: object
    max_date: object

    def lines(self) -> list[str]:
        return [
            f"[{self.league}] Rows: {self.row_count}",
            f"[{self.league}] Null counts: {self.null_counts}",
            f"[{self.league}] Date range: {self.min_date} to {self.max_date}",
        ]


@dataclass(frozen=True)
class FootballDataIngestSummary:
    league: str
    rows: int
    seasons: list[str]
    output_path: Path
    raw_files: int

    def lines(self) -> list[str]:
        return [
            f"[{self.league}] Rows: {self.rows}",
            f"[{self.league}] Seasons: {self.seasons}",
            f"[{self.league}] Raw Football-Data files: {self.raw_files}",
            f"[{self.league}] Output: {self.output_path}",
        ]


def football_data_url(season_code: str, league: str) -> str:
    league_code = FOOTBALL_DATA_LEAGUE_CODES[league]
    return f"{FOOTBALL_DATA_BASE_URL}/{season_code}/{league_code}.csv"


def football_data_cache_path(cache_dir: Path, season_code: str, league: str) -> Path:
    league_code = FOOTBALL_DATA_LEAGUE_CODES[league]
    return cache_dir / f"{league_code}_{season_code}.csv"


def download_football_data_csvs(
    cache_dir: Path = FOOTBALL_DATA_DIR,
    season_codes: Iterable[str] = DEFAULT_SEASON_CODES,
    leagues: Iterable[str] = tuple(FOOTBALL_DATA_LEAGUE_CODES),
) -> list[Path]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required to download Football-Data CSVs") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for season_code in season_codes:
        for league in leagues:
            output_path = football_data_cache_path(cache_dir, season_code, league)
            if output_path.exists():
                downloaded.append(output_path)
                continue
            response = requests.get(football_data_url(season_code, league), timeout=30)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            downloaded.append(output_path)
    return downloaded


def parse_football_data_date(values: pd.Series) -> pd.Series:
    first_pass = pd.to_datetime(values, dayfirst=True, errors="coerce", format="mixed")
    second_pass = pd.to_datetime(values, errors="coerce", format="mixed")
    return first_pass.fillna(second_pass).dt.normalize()


def season_from_date(dates: pd.Series) -> pd.Series:
    years = dates.dt.year
    starts = years.where(dates.dt.month >= 8, years - 1)
    ends = (starts + 1).astype(str).str[-2:]
    return starts.astype(str) + "-" + ends


def _clean_identifier(value: object) -> str:
    normalised = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalised.encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.casefold().replace("&", " and ")
    ascii_value = re.sub(r"[^a-z0-9]+", "_", ascii_value)
    return ascii_value.strip("_")


def build_rball_id(league: str, season: pd.Series, date: pd.Series, home_team: pd.Series, away_team: pd.Series) -> pd.Series:
    date_part = pd.to_datetime(date).dt.strftime("%Y%m%d")
    home = home_team.map(_clean_identifier)
    away = away_team.map(_clean_identifier)
    return league + "_" + season.astype(str) + "_" + date_part + "_" + home + "_" + away


FOOTBALL_DATA_REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
FOOTBALL_DATA_STAT_COLUMNS = {
    "HomeGoals": "FTHG",
    "AwayGoals": "FTAG",
    "HomeCorners": "HC",
    "AwayCorners": "AC",
    "HomeShotsOnTarget": "HST",
    "AwayShotsOnTarget": "AST",
    "HomeFouls": "HF",
    "AwayFouls": "AF",
    "HomeOffsides": "HO",
    "AwayOffsides": "AO",
}


def validate_football_data_columns(df: pd.DataFrame, source: Path | str = "Football-Data CSV") -> None:
    missing = [column for column in FOOTBALL_DATA_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Football-Data columns in {source}: {missing}")


def normalise_football_data_frame(raw: pd.DataFrame, league: str, source: Path | str = "Football-Data CSV") -> pd.DataFrame:
    validate_football_data_columns(raw, source=source)
    base = raw[["Date", "HomeTeam", "AwayTeam"]].copy()
    base["Date"] = parse_football_data_date(base["Date"])
    valid = base.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    source_rows = raw.loc[valid.index].copy()
    output = valid.reset_index(drop=True)
    source_rows = source_rows.reset_index(drop=True)
    output["Season"] = season_from_date(output["Date"])
    output["RBallID"] = build_rball_id(league, output["Season"], output["Date"], output["HomeTeam"], output["AwayTeam"])

    for target_column, source_column in FOOTBALL_DATA_STAT_COLUMNS.items():
        if source_column in source_rows.columns:
            values = pd.to_numeric(source_rows[source_column], errors="coerce").fillna(0)
        else:
            values = pd.Series(0, index=output.index)
        output[target_column] = values.astype("int64")

    output["League"] = league
    return output[[*OUTPUT_COLUMNS, "League"]]


def load_football_data_csv(path: Path, league: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return normalise_football_data_frame(raw, league=league, source=path)


def write_parquet_overwrite(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.is_dir():
            shutil.rmtree(output_path)
        elif output_path.exists():
            output_path.unlink()
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot overwrite {output_path}. It is probably owned by a Docker container user. "
            "Fix ownership once with: sudo chown -R $(id -u):$(id -g) "
            "data/processed data/features data/output data/model_artifacts dashboard/data mlruns"
        ) from exc
    df.to_parquet(output_path, index=False)


def process_football_data_league(
    league: str,
    season_codes: Iterable[str] = DEFAULT_SEASON_CODES,
    cache_dir: Path = FOOTBALL_DATA_DIR,
    output_dir: Path = PROCESSED_DIR,
) -> FootballDataIngestSummary:
    frames = []
    raw_files = 0
    for season_code in season_codes:
        path = football_data_cache_path(cache_dir, season_code, league)
        if not path.exists():
            download_football_data_csvs(cache_dir=cache_dir, season_codes=(season_code,), leagues=(league,))
        frames.append(load_football_data_csv(path, league=league))
        raw_files += 1

    if not frames:
        raise ValueError(f"At least one Football-Data season is required for {league}")
    data = pd.concat(frames, ignore_index=True).sort_values(["Date", "RBallID"], kind="mergesort")
    data = data.drop_duplicates(subset=["RBallID"]).reset_index(drop=True)

    output_path = output_dir / f"{league}.parquet"
    write_parquet_overwrite(data, output_path)
    return FootballDataIngestSummary(
        league=league,
        rows=len(data),
        seasons=sorted(data["Season"].dropna().unique().tolist()),
        output_path=output_path,
        raw_files=raw_files,
    )


def run_football_data_pipeline(
    season_codes: Iterable[str] = DEFAULT_SEASON_CODES,
    leagues: Iterable[str] = tuple(FOOTBALL_DATA_LEAGUE_CODES),
    cache_dir: Path = FOOTBALL_DATA_DIR,
    output_dir: Path = PROCESSED_DIR,
    report_path: Path = PROCESSED_DIR / "quality_report.txt",
) -> list[FootballDataIngestSummary]:
    summaries = [
        process_football_data_league(league, season_codes=season_codes, cache_dir=cache_dir, output_dir=output_dir)
        for league in leagues
    ]
    report_lines = [line for summary in summaries for line in summary.lines()]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    for line in report_lines:
        print(line)
    return summaries


def create_spark_session(app_name: str = "sports_modelling_stage1") -> SparkSession:
    return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()


def resolve_csv_files(input_path: str) -> list[str]:
    path = Path(input_path)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
    else:
        files = sorted(path.parent.glob(path.name))

    if not files:
        raise FileNotFoundError(f"No CSV files found for input path: {input_path}")
    return [str(file_path) for file_path in files]


def validate_raw_schema(df: DataFrame) -> None:
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")


def add_date_and_season(df: DataFrame) -> DataFrame:
    dated = (
        df.withColumn("Date", F.to_date("Timestamp", "MM/dd/yyyy HH:mm:ss"))
        .withColumn("Year", F.year("Date"))
        .withColumn("Month", F.month("Date"))
    )

    return dated.withColumn(
        "Season",
        F.when(
            F.col("Month") >= 8,
            F.concat(
                F.col("Year").cast("string"),
                F.lit("-"),
                (F.col("Year") + 1).cast("string").substr(3, 2),
            ),
        ).otherwise(
            F.concat(
                (F.col("Year") - 1).cast("string"),
                F.lit("-"),
                F.col("Year").cast("string").substr(3, 2),
            )
        ),
    ).drop("Year", "Month")


def clean_event_rows(df: DataFrame) -> DataFrame:
    validate_raw_schema(df)
    return df.filter(F.col("Minute") > 0)


def pivot_events_to_matches(df: DataFrame) -> DataFrame:
    known_incident_codes = list(INCIDENT_MAP.keys())
    counts = (
        df.filter(F.col("Incident").isin(known_incident_codes))
        .groupBy(*MATCH_KEYS, "Incident")
        .agg(F.count("*").alias("Count"))
    )

    match_df = (
        counts.groupBy(*MATCH_KEYS)
        .pivot("Incident", known_incident_codes)
        .agg(F.first("Count"))
        .fillna(0)
    )

    for code, name in INCIDENT_MAP.items():
        match_df = match_df.withColumnRenamed(code, name)

    for column in METRIC_COLUMNS:
        if column not in match_df.columns:
            match_df = match_df.withColumn(column, F.lit(0))

    return match_df.select(*OUTPUT_COLUMNS)


def build_match_level_dataset(raw_df: DataFrame) -> DataFrame:
    clean_df = clean_event_rows(raw_df)
    dated_df = add_date_and_season(clean_df)
    return pivot_events_to_matches(dated_df)


def build_quality_report(league: str, match_df: DataFrame) -> QualityReport:
    row_count = match_df.count()
    null_counts = {
        column: match_df.filter(F.col(column).isNull()).count()
        for column in ["HomeTeam", "AwayTeam", "Date", "Season"]
    }
    date_range = match_df.agg(F.min("Date"), F.max("Date")).collect()[0]

    return QualityReport(
        league=league,
        row_count=row_count,
        null_counts=null_counts,
        min_date=date_range[0],
        max_date=date_range[1],
    )


def process_league(
    spark: SparkSession,
    league: str,
    input_path: str,
    output_path: str,
) -> QualityReport:
    raw_df = spark.read.csv(resolve_csv_files(input_path), header=True, inferSchema=True)
    match_df = build_match_level_dataset(raw_df)
    report = build_quality_report(league, match_df)

    for line in report.lines():
        print(line)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    match_df.write.mode("overwrite").parquet(output_path)
    return report


def write_quality_report(reports: Iterable[QualityReport], output_path: str) -> None:
    report_lines = [line for report in reports for line in report.lines()]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_pipeline(
    leagues: Iterable[tuple[str, str, str]] = DEFAULT_LEAGUES,
    report_path: str = "data/processed/quality_report.txt",
) -> list[QualityReport]:
    spark = create_spark_session()
    try:
        reports = [
            process_league(spark, league, input_path, output_path)
            for league, input_path, output_path in leagues
        ]
        write_quality_report(reports, report_path)
        return reports
    finally:
        spark.stop()


def parse_args() -> object:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Football-Data CSVs into Stage 1 match-level Parquet files.")
    parser.add_argument("--source", choices=("football-data", "event-csv"), default="football-data")
    parser.add_argument("--season-codes", nargs="+", default=list(DEFAULT_SEASON_CODES), help="Football-Data season codes, e.g. 1011 1112 ... 2526")
    parser.add_argument("--leagues", nargs="+", choices=list(FOOTBALL_DATA_LEAGUE_CODES), default=list(FOOTBALL_DATA_LEAGUE_CODES))
    parser.add_argument("--cache-dir", type=Path, default=FOOTBALL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report-path", type=Path, default=PROCESSED_DIR / "quality_report.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.source == "event-csv":
        run_pipeline()
    else:
        run_football_data_pipeline(
            season_codes=args.season_codes,
            leagues=args.leagues,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            report_path=args.report_path,
        )
