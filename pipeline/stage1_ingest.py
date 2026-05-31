from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


DEFAULT_LEAGUES = (
    ("ENG", "data/ENG", "data/processed/ENG.parquet"),
    ("SPA", "data/SPA", "data/processed/SPA.parquet"),
    ("FRA", "data/FRA", "data/processed/FRA.parquet"),
)

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


if __name__ == "__main__":
    run_pipeline()
