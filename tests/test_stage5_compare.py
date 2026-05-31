import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pipeline.stage5_compare import (
    VALUE_BET_COLUMNS,
    football_data_url,
    load_football_data_csv,
    match_model_to_bookmaker_odds,
    normalise_team_name,
    parse_football_data_date,
    run_pipeline,
    build_best_bookmaker_odds_frame,
    compare_model_to_bookmaker_odds,
)


def sample_model_odds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RBallID": 2,
                "HomeTeam": "Chelsea",
                "AwayTeam": "Arsenal",
                "Date": "2019-08-11",
                "Season": "2019-20",
                "Result": "D",
                "ModelOdds_Home": 2.00,
                "ModelOdds_Draw": 3.40,
                "ModelOdds_Away": 4.20,
            },
            {
                "RBallID": 1,
                "HomeTeam": "Liverpool FC",
                "AwayTeam": "Everton FC",
                "Date": "2019-08-10",
                "Season": "2019-20",
                "Result": "H",
                "ModelOdds_Home": 1.90,
                "ModelOdds_Draw": 3.10,
                "ModelOdds_Away": 4.80,
            },
        ]
    )


def sample_bookmaker_odds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RBallID": 1,
                "Bookmaker": "bet365",
                "Odds_Home": 2.00,
                "Odds_Draw": 3.00,
                "Odds_Away": 5.20,
            },
            {
                "RBallID": 1,
                "Bookmaker": "pinnacle",
                "Odds_Home": 2.10,
                "Odds_Draw": 2.80,
                "Odds_Away": 5.50,
            },
            {
                "RBallID": 2,
                "Bookmaker": "bet365",
                "Odds_Home": 2.10,
                "Odds_Draw": 3.30,
                "Odds_Away": 4.10,
            },
            {
                "RBallID": 2,
                "Bookmaker": "pinnacle",
                "Odds_Home": 2.05,
                "Odds_Draw": None,
                "Odds_Away": 4.20,
            },
        ]
    )


class Stage5CompareTests(unittest.TestCase):
    def test_build_best_bookmaker_odds_frame_takes_best_price_per_outcome(self):
        best = build_best_bookmaker_odds_frame(sample_bookmaker_odds())
        rows = {row.RBallID: row for row in best.itertuples(index=False)}

        self.assertEqual(rows[1].BestOdds_Home, 2.10)
        self.assertEqual(rows[1].BestBookmaker_Home, "pinnacle")
        self.assertEqual(rows[1].BestOdds_Draw, 3.00)
        self.assertEqual(rows[1].BestBookmaker_Draw, "bet365")
        self.assertEqual(rows[1].BestOdds_Away, 5.50)
        self.assertEqual(rows[1].BestBookmaker_Away, "pinnacle")

    def test_compare_model_to_bookmaker_odds_flags_only_edges_at_or_above_threshold(self):
        value_bets = compare_model_to_bookmaker_odds(
            sample_model_odds(),
            sample_bookmaker_odds(),
            edge_threshold=0.10,
        )

        self.assertEqual(value_bets.columns.tolist(), VALUE_BET_COLUMNS)
        self.assertEqual(value_bets[["RBallID", "Outcome"]].values.tolist(), [[1, "H"], [1, "A"]])
        self.assertTrue(value_bets["ValueBet"].all())
        self.assertAlmostEqual(value_bets.loc[0, "ModelOdds"], 1.90)
        self.assertAlmostEqual(value_bets.loc[0, "BestBookOdds"], 2.10)
        self.assertAlmostEqual(value_bets.loc[0, "Edge"], (2.10 / 1.90) - 1.0)
        self.assertEqual(value_bets.loc[0, "BestBookmaker"], "pinnacle")
        self.assertGreaterEqual(value_bets["Edge"].min(), 0.10)

    def test_compare_model_to_bookmaker_odds_returns_empty_contract_when_no_edges(self):
        model_odds = sample_model_odds().assign(
            ModelOdds_Home=3.00,
            ModelOdds_Draw=4.00,
            ModelOdds_Away=7.00,
        )

        value_bets = compare_model_to_bookmaker_odds(model_odds, sample_bookmaker_odds())

        self.assertTrue(value_bets.empty)
        self.assertEqual(value_bets.columns.tolist(), VALUE_BET_COLUMNS)

    def test_compare_model_to_bookmaker_odds_rejects_missing_model_columns(self):
        model_odds = sample_model_odds().drop(columns=["ModelOdds_Away"])

        with self.assertRaisesRegex(ValueError, "Missing required Stage 4 model odds columns"):
            compare_model_to_bookmaker_odds(model_odds, sample_bookmaker_odds())

    def test_compare_model_to_bookmaker_odds_rejects_negative_threshold(self):
        with self.assertRaisesRegex(ValueError, "edge_threshold"):
            compare_model_to_bookmaker_odds(sample_model_odds(), sample_bookmaker_odds(), edge_threshold=-0.01)


    def test_normalise_team_name_handles_stage4_and_football_data_aliases(self):
        self.assertEqual(normalise_team_name("Arsenal FC"), "arsenal")
        self.assertEqual(normalise_team_name("Paris Saint-Germain FC"), "paris sg")
        self.assertEqual(normalise_team_name("Paris SG"), "paris sg")
        self.assertEqual(normalise_team_name("Man United"), "manchester united")

    def test_football_data_url_uses_season_and_league_codes(self):
        self.assertEqual(
            football_data_url("1920", "ENG"),
            "https://www.football-data.co.uk/mmz4281/1920/E0.csv",
        )

    def test_parse_football_data_date_handles_day_first_dates(self):
        parsed = parse_football_data_date(pd.Series(["11/08/2019", "2019-08-12"]))

        self.assertEqual(parsed.tolist(), [pd.Timestamp("2019-08-11"), pd.Timestamp("2019-08-12")])

    def test_load_football_data_csv_normalises_supported_bookmakers(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "E0_1920.csv"
            pd.DataFrame(
                [
                    {
                        "Date": "11/08/2019",
                        "HomeTeam": "Arsenal",
                        "AwayTeam": "Chelsea",
                        "B365H": "2.10",
                        "B365D": "3.60",
                        "B365A": "4.20",
                        "PSH": "2.25",
                        "PSD": "3.70",
                        "PSA": "4.30",
                    }
                ]
            ).to_csv(path, index=False)

            odds = load_football_data_csv(path, league="ENG")

        self.assertEqual(odds.loc[0, "Date"], pd.Timestamp("2019-08-11"))
        self.assertEqual(odds.loc[0, "Season"], "2019-20")
        self.assertEqual(odds.loc[0, "League"], "ENG")
        self.assertAlmostEqual(odds.loc[0, "B365_H"], 2.10)
        self.assertAlmostEqual(odds.loc[0, "PS_A"], 4.30)
        self.assertEqual(odds.loc[0, "HomeTeamKey"], "arsenal")

    def test_run_pipeline_writes_parquet_and_dashboard_json_from_football_data_csv(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_path = tmp_path / "model_odds.parquet"
            odds_file = tmp_path / "E0_1920.csv"
            output_path = tmp_path / "value_bets.parquet"
            json_path = tmp_path / "dashboard" / "value_bets.json"
            sample_model_odds().to_parquet(model_path, index=False)
            pd.DataFrame(
                [
                    {
                        "Date": "10/08/2019",
                        "HomeTeam": "Liverpool",
                        "AwayTeam": "Everton",
                        "B365H": 2.00,
                        "B365D": 3.00,
                        "B365A": 5.20,
                        "PSH": 2.10,
                        "PSD": 2.80,
                        "PSA": 5.50,
                    }
                ]
            ).to_csv(odds_file, index=False)

            summary = run_pipeline(
                model_odds_path=model_path,
                football_data_dir=odds_file,
                output_path=output_path,
                dashboard_json_path=json_path,
                edge_threshold=0.10,
            )
            output_exists = output_path.exists()
            json_exists = json_path.exists()
            written = pd.read_parquet(output_path)

        self.assertTrue(output_exists)
        self.assertTrue(json_exists)
        self.assertEqual(summary.model_rows, 2)
        self.assertEqual(summary.matched_rows, 1)
        self.assertEqual(summary.value_bets, 2)
        self.assertEqual(written[["RBallID", "Outcome"]].values.tolist(), [[1, "H"], [1, "A"]])


if __name__ == "__main__":
    unittest.main()
