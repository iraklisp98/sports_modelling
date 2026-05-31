# Stage 1 — Data Ingestion & Cleaning

**Status:** Complete  
**Script:** `pipeline/stage1_ingest.py`  
**Input:** Raw event-level CSVs in `data/ENG/`, `data/FRA/`, `data/SPA/`  
**Output:** One Parquet file per league in `data/processed/` + a data quality report

---

## What

Read thousands of per-match CSV files, merge them into one dataset per league, clean and validate the data, pivot from event-level rows to match-level rows, and write the result as Parquet.

---

## Why This Approach

### Why PySpark instead of pandas?
pandas loads the entire dataset into memory on a single CPU core. That works fine today at 180MB, but Phase 2 brings in live API data continuously — the volume will grow. PySpark distributes the work across CPU cores and, if needed, across machines. By writing Spark code now, you make zero changes when the data gets bigger.

The other reason: PySpark fluency is a hard requirement on most Data Engineer job descriptions. Hiring managers will ask "have you worked with Spark?" and you want to say yes with a real example.

### Why Parquet instead of CSV?
CSV stores data row by row. When the next stage needs only 5 columns out of 20, it still reads all 20.

Parquet is **columnar** — data is stored column by column. Reading 5 columns from a 20-column Parquet file means reading roughly 25% of the data. It also:
- Enforces schema (no silent type mismatches downstream)
- Compresses significantly better than CSV
- Is the default format in every modern data warehouse (Snowflake, BigQuery, Databricks)

### Why pivot from event-level to match-level here?
Every downstream stage (features, model, odds) thinks in matches, not events. Doing the pivot once in Stage 1 means every other stage gets a clean, flat table. This is the **single responsibility principle** applied to pipeline design.

---

## New Concepts to Learn Before Building

### PySpark basics
PySpark is a Python API for Apache Spark. The key mental model: **nothing actually runs until you call an action** (like `.write()` or `.collect()`). Before that, Spark just builds a plan. This is called **lazy evaluation**.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("sports_modelling_stage1") \
    .master("local[*]") \  # use all available CPU cores on this machine
    .getOrCreate()
```

`local[*]` means "run locally, use all cores." In production this would point to a cluster.

### PySpark vs pandas key differences
| Operation | pandas | PySpark |
|---|---|---|
| Read CSV | `pd.read_csv()` | `spark.read.csv()` |
| Filter rows | `df[df.col > 0]` | `df.filter(df.col > 0)` |
| Group & aggregate | `df.groupby().agg()` | `df.groupBy().agg()` |
| Write output | `df.to_parquet()` | `df.write.parquet()` |

The logic is nearly identical — the API is slightly different.

---

## How to Build It (Step by Step)

### Step 1 — Create the script file
Create `pipeline/stage1_ingest.py`.

### Step 2 — Start a Spark session
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("stage1_ingest") \
    .master("local[*]") \
    .getOrCreate()
```

### Step 3 — Read all CSVs for a league
Each league folder contains hundreds of per-match CSV files. Use a wildcard path to read them all at once:
```python
df = spark.read.csv("data/ENG/*.csv", header=True, inferSchema=True)
```
`inferSchema=True` tells Spark to detect column types. For production you'd define the schema explicitly — but `inferSchema` is fine here.

### Step 4 — Inspect the raw schema
Before cleaning anything, print the schema and a few rows. Understand what you have.
```python
df.printSchema()
df.show(5)
```

The raw event schema:
| Column | Type | Description |
|---|---|---|
| `RBallID` | string | Unique match ID |
| `HomeTeam` | string | Home team name |
| `AwayTeam` | string | Away team name |
| `Timestamp` | string | Match date/time |
| `Incident` | string | Event type code (GOAL1, CR2, SHG1...) |
| `IncidentNumber` | int | Sequential ID within match |
| `Minute` | int | Minute of the event |

### Step 5 — Filter out non-play records
Some rows have `Minute = 0` or null — these are administrative entries, not real match events. Drop them.
```python
df = df.filter(df.Minute > 0)
```

### Step 6 — Parse the Timestamp column
Cast it to a proper date type and extract the season:
- August or later → new season starts
- Season label: "2017-18", "2018-19", "2019-20"

```python
from pyspark.sql import functions as F

df = df.withColumn("Date", F.to_date("Timestamp")) \
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
```

### Step 7 — Pivot events to match-level columns
The goal is one row per match with columns like `HomeGoals`, `AwayGoals`, `HomeCorners`, etc.

The incident codes encode both the event type and which team:
- `GOAL1` = home team goal, `GOAL2` = away team goal
- `CR1` = home corner, `CR2` = away corner
- `SHG1` = home shot on goal, `SHG2` = away shot on goal
- `SF1` = home foul, `SF2` = away foul

Count each incident type per match, then pivot:
```python
from pyspark.sql import functions as F

incident_map = {
    "GOAL1": "HomeGoals",  "GOAL2": "AwayGoals",
    "CR1":   "HomeCorners","CR2":   "AwayCorners",
    "SHG1":  "HomeShotsOnTarget", "SHG2": "AwayShotsOnTarget",
    "SF1":   "HomeFouls",  "SF2":   "AwayFouls",
    "OS1":   "HomeOffsides","OS2":  "AwayOffsides",
}

# Count each incident type per match
counts = df.groupBy("RBallID", "HomeTeam", "AwayTeam", "Date", "Season", "Incident") \
           .agg(F.count("*").alias("Count"))

# Pivot incident types into columns
match_df = counts.groupBy("RBallID", "HomeTeam", "AwayTeam", "Date", "Season") \
                 .pivot("Incident") \
                 .agg(F.first("Count")) \
                 .fillna(0)

# Rename columns using the incident map
for code, name in incident_map.items():
    if code in match_df.columns:
        match_df = match_df.withColumnRenamed(code, name)
```

### Step 8 — Validate the output
Before writing, run basic checks:
```python
row_count = match_df.count()
null_counts = {col: match_df.filter(F.col(col).isNull()).count()
               for col in ["HomeTeam", "AwayTeam", "Date", "Season"]}
date_range = match_df.agg(F.min("Date"), F.max("Date")).collect()[0]

print(f"Rows: {row_count}")
print(f"Nulls: {null_counts}")
print(f"Date range: {date_range[0]} to {date_range[1]}")
```

These four numbers become your **data quality report**. Write them to a text file alongside the Parquet output.

### Step 9 — Write Parquet output
```python
match_df.write.mode("overwrite").parquet("data/processed/ENG.parquet")
```

`mode("overwrite")` means re-running the script replaces the old file cleanly. Never use `append` here — you'd get duplicate data.

### Step 10 — Loop over all three leagues
Wrap everything in a function `process_league(league: str)` and call it for ENG, FRA, SPA.

---

## Acceptance Criteria

- [x] Script runs without errors: `python pipeline/stage1_ingest.py`
- [x] Three Parquet files created: `data/processed/ENG.parquet`, `FRA.parquet`, `SPA.parquet`
- [x] Each file has one row per match (verified by zero duplicate `RBallID` values)
- [x] No nulls in `HomeTeam`, `AwayTeam`, `Date`, `Season`
- [x] Data quality report printed to console and written to `data/processed/quality_report.txt`
- [x] Date range covers the available raw data: ENG 2017-08-11 to 2019-12-29; SPA 2017-08-18 to 2019-12-22; FRA 2017-08-04 to 2019-12-21

---

## Interview Q&A

**Q: Why did you use PySpark for a 180MB dataset? pandas would have been simpler.**  
A: "The pipeline is designed for Phase 2 where data comes from a live API on a daily schedule — the volume grows continuously. Switching from pandas to Spark later would mean rewriting every stage. By writing Spark code now, Phase 2 is just a change to the data source, not the processing logic."

**Q: What is Parquet and why use it over CSV?**  
A: "Parquet is a columnar file format — data is stored column by column rather than row by row. When you only need 5 of 20 columns, Parquet reads roughly 25% of the data. It also enforces schema and compresses better. Every modern data warehouse — BigQuery, Snowflake, Databricks — uses Parquet as its default storage format."

**Q: What is lazy evaluation in Spark?**  
A: "Spark doesn't execute transformations immediately. When you call `.filter()` or `.groupBy()`, Spark just adds those steps to a logical plan. Execution only happens when you call an action like `.write()` or `.count()`. This lets Spark optimise the full execution plan before running anything — it might reorder operations, push filters down, or combine steps."
