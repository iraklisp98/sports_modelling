import unittest

from pyspark.sql import SparkSession

from pipeline.stage1_ingest import (
    METRIC_COLUMNS,
    OUTPUT_COLUMNS,
    add_date_and_season,
    build_match_level_dataset,
)


class Stage1IngestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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


if __name__ == "__main__":
    unittest.main()
