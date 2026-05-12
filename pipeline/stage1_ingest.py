from pyspark.sql import SparkSession 
from pyspark.sql import functions as F
import os

spark = SparkSession.builder \
    .appName("sports_modelling_stage") \
    .master("local[*]") \
    .getOrCreate()
leagues = [
    ("ENG", "data/ENG/*.csv", "data/processed/ENG.parquet"),
    ("SPA", "data/SPA/*.csv", "data/processed/SPA.parquet"),
    ("FRA", "data/FRA/*.csv", "data/processed/FRA.parquet"),
]
incident_map = {
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


os.makedirs("data/processed", exist_ok=True)

report_lines = []

for league, path, out_path in leagues:
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.filter(df.Minute > 0)

    df = df.withColumn("Date", F.to_date("Timestamp", "MM/dd/yyyy HH:mm:ss")) \
           .withColumn("Year", F.year("Date")) \
           .withColumn("Month", F.month("Date")) \
           .withColumn("Season", F.when(F.col("Month") >= 8,
                           F.concat(F.col("Year").cast("string"),
                                    F.lit("-"),
                                    (F.col("Year") + 1).cast("string").substr(3, 2)))
                       .otherwise(
                           F.concat((F.col("Year") - 1).cast("string"),
                                    F.lit("-"),
                                    F.col("Year").cast("string").substr(3, 2))))

    counts = df.groupBy("RBallID", "HomeTeam", "AwayTeam", "Date", "Season", "Incident").agg(F.count("*").alias("Count"))
    match_df = counts.groupBy("RBallID", "HomeTeam", "AwayTeam", "Date", "Season").pivot("Incident").agg(F.first("Count")).fillna(0)

    for code, name in incident_map.items():
        if code in match_df.columns:
            match_df = match_df.withColumnRenamed(code, name)

    row_counts = match_df.count()
    null_counts = {col: match_df.filter(F.col(col).isNull()).count() for col in ["HomeTeam", "AwayTeam", "Date", "Season"]}
    date_range = match_df.agg(F.min("Date"), F.max("Date")).collect()[0]

    lines = [
        f"[{league}] Rows: {row_counts}",
        f"[{league}] Null counts: {null_counts}",
        f"[{league}] Date range: {date_range[0]} to {date_range[1]}",
    ]
    for line in lines:
        print(line)
    report_lines.extend(lines)

    match_df.write.mode("overwrite").parquet(out_path)

with open("data/processed/quality_report.txt", "w") as f:
    f.write("\n".join(report_lines) + "\n")

spark.stop()

