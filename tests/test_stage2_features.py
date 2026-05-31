import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pipeline.stage2_features import (
    STAGE2_FEATURE_COLUMNS,
    build_feature_dataset,
    expected_home_score,
    process_league,
)


def sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RBallID": 1,
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Date": "2017-08-01",
                "Season": "2017-18",
                "HomeGoals": 2,
                "AwayGoals": 0,
                "HomeCorners": 5,
                "AwayCorners": 3,
                "HomeShotsOnTarget": 6,
                "AwayShotsOnTarget": 2,
                "HomeFouls": 10,
                "AwayFouls": 11,
                "HomeOffsides": 1,
                "AwayOffsides": 2,
            },
            {
                "RBallID": 2,
                "HomeTeam": "Chelsea",
                "AwayTeam": "Arsenal",
                "Date": "2017-08-08",
                "Season": "2017-18",
                "HomeGoals": 1,
                "AwayGoals": 1,
                "HomeCorners": 4,
                "AwayCorners": 7,
                "HomeShotsOnTarget": 3,
                "AwayShotsOnTarget": 4,
                "HomeFouls": 9,
                "AwayFouls": 8,
                "HomeOffsides": 0,
                "AwayOffsides": 1,
            },
            {
                "RBallID": 3,
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Date": "2017-08-15",
                "Season": "2017-18",
                "HomeGoals": 0,
                "AwayGoals": 1,
                "HomeCorners": 2,
                "AwayCorners": 6,
                "HomeShotsOnTarget": 1,
                "AwayShotsOnTarget": 5,
                "HomeFouls": 7,
                "AwayFouls": 12,
                "HomeOffsides": 0,
                "AwayOffsides": 0,
            },
            {
                "RBallID": 4,
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Date": "2018-08-01",
                "Season": "2018-19",
                "HomeGoals": 3,
                "AwayGoals": 1,
                "HomeCorners": 8,
                "AwayCorners": 2,
                "HomeShotsOnTarget": 7,
                "AwayShotsOnTarget": 2,
                "HomeFouls": 6,
                "AwayFouls": 10,
                "HomeOffsides": 1,
                "AwayOffsides": 1,
            },
        ]
    )


class Stage2FeatureTests(unittest.TestCase):
    def test_result_and_result_code_mapping(self):
        featured = build_feature_dataset(sample_matches())
        results = dict(zip(featured["RBallID"], featured["Result"]))
        codes = dict(zip(featured["RBallID"], featured["ResultCode"]))

        self.assertEqual(results[1], "H")
        self.assertEqual(results[2], "D")
        self.assertEqual(results[3], "A")
        self.assertEqual(codes[1], 0)
        self.assertEqual(codes[2], 1)
        self.assertEqual(codes[3], 2)

    def test_elo_is_stored_before_match_update(self):
        featured = build_feature_dataset(sample_matches())
        first = featured.loc[featured["RBallID"] == 1].iloc[0]
        second = featured.loc[featured["RBallID"] == 2].iloc[0]

        self.assertEqual(first["HomeElo"], 1500.0)
        self.assertEqual(first["AwayElo"], 1500.0)

        expected = expected_home_score(1500.0, 1500.0)
        arsenal_after_win = 1500.0 + 30.0 * (1.0 - expected)
        chelsea_after_loss = 1500.0 + 30.0 * (0.0 - (1.0 - expected))

        self.assertAlmostEqual(second["AwayElo"], arsenal_after_win)
        self.assertAlmostEqual(second["HomeElo"], chelsea_after_loss)
        self.assertLess(second["AwayElo"], 1580.0)

    def test_rolling_features_exclude_current_match(self):
        featured = build_feature_dataset(sample_matches())
        first = featured.loc[featured["RBallID"] == 1].iloc[0]
        second = featured.loc[featured["RBallID"] == 2].iloc[0]
        third = featured.loc[featured["RBallID"] == 3].iloc[0]

        self.assertEqual(first["HomeGoals_Last5"], 0.0)
        self.assertEqual(first["HomePoints_Last5"], 0.0)

        self.assertEqual(second["AwayGoals_Last5"], 2.0)
        self.assertEqual(second["AwayPoints_Last5"], 3.0)

        self.assertEqual(third["HomeGoals_Last5"], 1.5)
        self.assertEqual(third["HomePoints_Last5"], 2.0)
        self.assertNotEqual(third["HomeGoals_Last5"], third["HomeGoals"])

    def test_season_win_rate_resets_and_excludes_current_match(self):
        featured = build_feature_dataset(sample_matches())
        first = featured.loc[featured["RBallID"] == 1].iloc[0]
        second = featured.loc[featured["RBallID"] == 2].iloc[0]
        fourth = featured.loc[featured["RBallID"] == 4].iloc[0]

        self.assertEqual(first["HomeWinRate_Season"], 0.0)
        self.assertEqual(second["AwayWinRate_Season"], 1.0)
        self.assertEqual(fourth["HomeWinRate_Season"], 0.0)

    def test_elo_regresses_between_seasons_and_new_teams_start_below_mean(self):
        raw = pd.concat(
            [
                sample_matches(),
                pd.DataFrame(
                    [
                        {
                            "RBallID": 5,
                            "HomeTeam": "Brentford",
                            "AwayTeam": "Arsenal",
                            "Date": "2018-08-02",
                            "Season": "2018-19",
                            "HomeGoals": 0,
                            "AwayGoals": 2,
                            "HomeCorners": 1,
                            "AwayCorners": 5,
                            "HomeShotsOnTarget": 2,
                            "AwayShotsOnTarget": 6,
                            "HomeFouls": 13,
                            "AwayFouls": 9,
                            "HomeOffsides": 1,
                            "AwayOffsides": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        featured = build_feature_dataset(raw)
        fourth = featured.loc[featured["RBallID"] == 4].iloc[0]
        fifth = featured.loc[featured["RBallID"] == 5].iloc[0]

        self.assertLess(fourth["HomeElo"], 1500.0)
        self.assertGreater(fourth["HomeElo"], 1400.0)
        self.assertEqual(fifth["HomeElo"], 1400.0)

    def test_output_schema_and_row_count(self):
        raw = sample_matches()
        featured = build_feature_dataset(raw)

        for column in STAGE2_FEATURE_COLUMNS:
            self.assertIn(column, featured.columns)
        self.assertEqual(len(featured), len(raw))
        self.assertFalse(featured["ResultCode"].isna().any())

    def test_process_league_writes_feature_parquet(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "ENG.parquet"
            output_path = tmp_path / "ENG_features.parquet"
            sample_matches().to_parquet(input_path, index=False)

            summary = process_league("ENG", input_path=input_path, output_path=output_path)
            written = pd.read_parquet(output_path)

        self.assertEqual(summary.league, "ENG")
        self.assertEqual(summary.rows, 4)
        self.assertEqual(summary.output_path, output_path)
        self.assertEqual(len(written), 4)
        for column in STAGE2_FEATURE_COLUMNS:
            self.assertIn(column, written.columns)


if __name__ == "__main__":
    unittest.main()
