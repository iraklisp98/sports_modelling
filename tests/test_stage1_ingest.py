import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd

from pyspark.sql import SparkSession

from pipeline.stage1_ingest import (
    DEFAULT_SEASON_CODES,
    FOOTBALL_DATA_LEAGUE_CODES,
    METRIC_COLUMNS,
    OUTPUT_COLUMNS,
    add_date_and_season,
    build_match_level_dataset,
    football_data_cache_path,
    football_data_url,
    normalise_football_data_frame,
    process_football_data_league,
    write_parquet_overwrite,
)


class Stage1IngestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("java") is None:
            raise unittest.SkipTest("Java is required for PySpark Stage 1 tests")
        cls.spark = (
            SparkSession.builder.appName("stage1_ingest_tests")
            .master("local[1]")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_assigns_august_to_july_season_labels(self):
        df = self.spark.createDataFrame(
            [
                ("1", "08/12/2017 15:00:00"),
                ("2", "07/15/2018 15:00:00"),
                ("3", "08/01/2018 15:00:00"),
                ("4", "05/31/2020 15:00:00"),
            ],
            ["RBallID", "Timestamp"],
        )

        rows = {
            row.RBallID: row.Season
            for row in add_date_and_season(df).select("RBallID", "Season").collect()
        }

        self.assertEqual(
            rows,
            {
                "1": "2017-18",
                "2": "2017-18",
                "3": "2018-19",
                "4": "2019-20",
            },
        )

    def test_pivots_events_to_match_level_output_contract(self):
        raw_df = self.spark.createDataFrame(
            [
                ("m1", "Arsenal", "Chelsea", "08/12/2017 15:00:00", "Lineup changed", 0),
                ("m1", "Arsenal", "Chelsea", "08/12/2017 15:05:00", "GOAL1", 5),
                ("m1", "Arsenal", "Chelsea", "08/12/2017 15:10:00", "GOAL1", 10),
                ("m1", "Arsenal", "Chelsea", "08/12/2017 15:20:00", "GOAL2", 20),
                ("m1", "Arsenal", "Chelsea", "08/12/2017 15:30:00", "CR2", 30),
                ("m1", "Arsenal", "Chelsea", "08/12/2017 15:40:00", "SHG1", 40),
                ("m1", "Arsenal", "Chelsea", "08/12/2017 15:45:00", "UNKNOWN", 45),
                ("m2", "Lyon", "Marseille", "01/10/2019 20:00:00", "SF1", 1),
                ("m2", "Lyon", "Marseille", "01/10/2019 20:10:00", "OS2", 10),
            ],
            ["RBallID", "HomeTeam", "AwayTeam", "Timestamp", "Incident", "Minute"],
        )

        match_df = build_match_level_dataset(raw_df)
        rows = {row.RBallID: row.asDict() for row in match_df.collect()}

        self.assertEqual(match_df.columns, OUTPUT_COLUMNS)
        self.assertEqual(set(rows), {"m1", "m2"})

        self.assertEqual(rows["m1"]["HomeGoals"], 2)
        self.assertEqual(rows["m1"]["AwayGoals"], 1)
        self.assertEqual(rows["m1"]["AwayCorners"], 1)
        self.assertEqual(rows["m1"]["HomeShotsOnTarget"], 1)
        self.assertEqual(rows["m1"]["Season"], "2017-18")

        self.assertEqual(rows["m2"]["HomeFouls"], 1)
        self.assertEqual(rows["m2"]["AwayOffsides"], 1)
        self.assertEqual(rows["m2"]["Season"], "2018-19")

        for metric in METRIC_COLUMNS:
            self.assertIsInstance(rows["m1"][metric], int)
            self.assertIsInstance(rows["m2"][metric], int)

        self.assertEqual(rows["m1"]["HomeCorners"], 0)
        self.assertEqual(rows["m2"]["AwayGoals"], 0)
        self.assertNotIn("UNKNOWN", match_df.columns)


class FootballDataStage1Tests(unittest.TestCase):
    def test_normalises_football_data_match_rows_to_stage2_contract(self):
        raw = pd.DataFrame(
            [
                {
                    "Date": "11/08/2017",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Leicester",
                    "FTHG": "4",
                    "FTAG": "3",
                    "HC": "9",
                    "AC": "4",
                    "HST": "10",
                    "AST": "3",
                    "HF": "9",
                    "AF": "12",
                },
                {
                    "Date": "2018-05-13",
                    "HomeTeam": "Man United",
                    "AwayTeam": "Watford",
                    "FTHG": 1,
                    "FTAG": 0,
                    "HC": None,
                    "AC": 5,
                    "HST": 7,
                    "AST": 3,
                    "HF": 6,
                    "AF": 11,
                },
            ]
        )

        normalised = normalise_football_data_frame(raw, league="ENG")

        self.assertEqual(normalised.columns.tolist(), [*OUTPUT_COLUMNS, "League"])
        self.assertEqual(normalised.loc[0, "RBallID"], "ENG_2017-18_20170811_arsenal_leicester")
        self.assertEqual(normalised["Season"].tolist(), ["2017-18", "2017-18"])
        self.assertEqual(normalised["HomeGoals"].tolist(), [4, 1])
        self.assertEqual(normalised["AwayGoals"].tolist(), [3, 0])
        self.assertEqual(normalised["HomeCorners"].tolist(), [9, 0])
        self.assertEqual(normalised["AwayCorners"].tolist(), [4, 5])
        self.assertEqual(normalised["HomeOffsides"].tolist(), [0, 0])
        self.assertEqual(normalised["AwayOffsides"].tolist(), [0, 0])
        self.assertEqual(normalised["League"].tolist(), ["ENG", "ENG"])

    def test_process_football_data_league_uses_cached_csv_and_writes_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            output_dir = root / "processed"
            cache_dir.mkdir()
            football_data_cache_path(cache_dir, "1920", "ENG").write_text(
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC,HST,AST,HF,AF\n"
                "09/08/2019,Liverpool,Norwich,4,1,11,2,7,5,9,9\n",
                encoding="utf-8",
            )

            summary = process_football_data_league(
                "ENG",
                season_codes=("1920",),
                cache_dir=cache_dir,
                output_dir=output_dir,
            )

            output = pd.read_parquet(output_dir / "ENG.parquet")
            self.assertEqual(summary.rows, 1)
            self.assertEqual(summary.seasons, ["2019-20"])
            self.assertEqual(output.loc[0, "HomeTeam"], "Liverpool")
            self.assertEqual(output.loc[0, "HomeGoals"], 4)
            self.assertEqual(output.loc[0, "RBallID"], "ENG_2019-20_20190809_liverpool_norwich")

    def test_process_football_data_league_overwrites_old_spark_parquet_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            output_dir = root / "processed"
            output_path = output_dir / "ENG.parquet"
            cache_dir.mkdir()
            output_path.mkdir(parents=True)
            (output_path / "part-00000.snappy.parquet").write_text("old spark output", encoding="utf-8")
            football_data_cache_path(cache_dir, "1920", "ENG").write_text(
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC,HST,AST,HF,AF\n"
                "09/08/2019,Liverpool,Norwich,4,1,11,2,7,5,9,9\n",
                encoding="utf-8",
            )

            process_football_data_league(
                "ENG",
                season_codes=("1920",),
                cache_dir=cache_dir,
                output_dir=output_dir,
            )

            self.assertTrue(output_path.is_file())
            output = pd.read_parquet(output_path)
            self.assertEqual(output.loc[0, "RBallID"], "ENG_2019-20_20190809_liverpool_norwich")

    def test_write_parquet_overwrite_explains_docker_owned_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "ENG.parquet"
            output_path.mkdir()
            df = pd.DataFrame({"RBallID": ["m1"]})

            with mock.patch("pipeline.stage1_ingest.shutil.rmtree", side_effect=PermissionError("_SUCCESS")):
                with self.assertRaisesRegex(PermissionError, "sudo chown -R"):
                    write_parquet_overwrite(df, output_path)

    def test_normalise_football_data_rejects_missing_required_columns(self):
        raw = pd.DataFrame([{"Date": "11/08/2017", "HomeTeam": "Arsenal"}])

        with self.assertRaisesRegex(ValueError, "Missing required Football-Data columns"):
            normalise_football_data_frame(raw, league="ENG")

    def test_football_data_urls_and_default_download_scope_are_stable(self):
        self.assertEqual(football_data_url("1920", "ENG"), "https://www.football-data.co.uk/mmz4281/1920/E0.csv")
        self.assertEqual(football_data_cache_path(Path("cache"), "1920", "SPA"), Path("cache/SP1_1920.csv"))
        self.assertEqual(FOOTBALL_DATA_LEAGUE_CODES, {"ENG": "E0", "SPA": "SP1", "FRA": "F1"})
        self.assertEqual(DEFAULT_SEASON_CODES[0], "1011")
        self.assertEqual(DEFAULT_SEASON_CODES[-1], "1920")


if __name__ == "__main__":
    unittest.main()
