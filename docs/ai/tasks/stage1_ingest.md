# Stage 1 — Data Ingestion & Cleaning

**Status:** Complete
**Script:** `pipeline/stage1_ingest.py`
**Default input:** Football-Data.co.uk season CSVs downloaded from `https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv`
**Cache:** `data/bookmaker_odds/football_data/`
**Output:** One Parquet file per league in `data/processed/` + a data quality report

---

## What

Stage 1 now downloads reproducible historical match CSVs directly from Football-Data.co.uk for Premier League, La Liga, Ligue 1, Bundesliga, and Serie A. It normalises those CSVs into the same match-level schema Stage 2 already expects, then writes Parquet files per league.

The old event-level PySpark path is still available with `--source event-csv`, but the default path is Football-Data because it gives us more seasons and uses the same source as the bookmaker odds comparison.

---

## Why This Approach

### Why download in Stage 1?
A pipeline should own its inputs. If a hiring manager clones the repo, they should not need a hidden manual step where someone downloaded CSVs earlier. Stage 1 now fetches missing files, caches them, and creates deterministic outputs.

### Why Football-Data.co.uk?
The project needs historical match results and bookmaker odds for the same fixtures. Football-Data publishes season-level CSVs with results, match stats, team names, dates, and closing odds. Using it for both Stage 1 and Stage 5 removes a major source-matching problem.

### Why still write the old Stage 1 schema?
Stage 2 should not care where the raw data came from. Keeping the contract stable means the rest of the pipeline can keep reading:

`RBallID, HomeTeam, AwayTeam, Date, Season, HomeGoals, AwayGoals, HomeCorners, AwayCorners, HomeShotsOnTarget, AwayShotsOnTarget, HomeFouls, AwayFouls, HomeOffsides, AwayOffsides`

That is the engineering point: a data contract lets you replace the source without rewriting every downstream stage.

---

## Current Download Scope

| League | Football-Data code | Output |
|---|---|---|
| Premier League | `E0` | `data/processed/ENG.parquet` |
| La Liga | `SP1` | `data/processed/SPA.parquet` |
| Ligue 1 | `F1` | `data/processed/FRA.parquet` |
| Bundesliga | `D1` | `data/processed/GER.parquet` |
| Serie A | `I1` | `data/processed/ITA.parquet` |

Default seasons are `1011` through `2223`, which become season labels `2010-11` through `2022-23`. The holdout remains `2019-20`; Stage 3 trains on every available season before that by default.

---

## How It Works

1. Build each source URL: `https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv`.
2. Download missing CSVs into `data/bookmaker_odds/football_data/`.
3. Parse mixed date formats safely with day-first preference.
4. Derive the season label from match date: August or later starts a new season.
5. Map Football-Data columns into the Stage 1 contract:
   - `FTHG`, `FTAG` -> goals
   - `HC`, `AC` -> corners
   - `HST`, `AST` -> shots on target
   - `HF`, `AF` -> fouls
   - `HO`, `AO` -> offsides when available, otherwise `0`
6. Create deterministic `RBallID` values from league, season, date, home team, and away team.
7. Sort, deduplicate by `RBallID`, and write Parquet.

---

## Commands

Run the default Football-Data ingestion:

```bash
python pipeline/stage1_ingest.py
```

Run a tiny sample while developing:

```bash
python pipeline/stage1_ingest.py --season-codes 1920 --leagues ENG
```

Run the old event-level CSV path only if those local files exist:

```bash
python pipeline/stage1_ingest.py --source event-csv
```

---

## Acceptance Criteria

- [x] Script downloads missing Football-Data CSVs into the local cache
- [x] Script writes `data/processed/ENG.parquet`, `SPA.parquet`, `FRA.parquet`, `GER.parquet`, and `ITA.parquet`
- [x] Output keeps the Stage 1 -> Stage 2 schema stable
- [x] Required source columns are validated before writing output
- [x] Missing optional stat columns are handled explicitly, not silently ignored
- [x] Data quality report is written to `data/processed/quality_report.txt`
- [x] Unit tests cover URL construction, cache paths, schema validation, and Football-Data normalization

---

## Interview Q&A

**Q: Why move downloading into Stage 1?**
A: "Because the pipeline should be reproducible from a fresh clone. Stage 1 owns raw data acquisition, caches immutable source files, and emits a clean Parquet contract for downstream stages. That removes manual setup and makes Docker useful."

**Q: Why did changing Stage 1 not break Stage 2?**
A: "Because the boundary between stages is a data contract, not a function call. As long as Stage 1 writes the same columns and types, Stage 2 can stay unchanged even if the raw source changes completely."

**Q: Why keep Parquet?**
A: "Parquet is columnar, compressed, schema-aware, and standard in data engineering stacks. It is the right intermediate format for a multi-stage batch pipeline."
