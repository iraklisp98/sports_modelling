import unittest

import pandas as pd

from pipeline.training_window_experiments import aggregate_fold_results, expanding_train_window_for_test, recent_train_window_for_test, season_range, summarize_value_bets, validate_league_subset, value_bet_records


class TrainingWindowExperimentTests(unittest.TestCase):
    def test_season_range_filters_inclusive_window(self):
        df = pd.DataFrame({"Season": ["2017-18", "2018-19", "2022-23", "2023-24"]})

        mask = season_range(df, "2018-19", "2022-23")

        self.assertEqual(df[mask]["Season"].tolist(), ["2018-19", "2022-23"])

    def test_recent_train_window_for_test_builds_fixed_prior_window(self):
        self.assertEqual(recent_train_window_for_test("2025-26", window_seasons=5), ("2020-21", "2024-25"))

    def test_expanding_train_window_for_test_uses_all_prior_history(self):
        self.assertEqual(expanding_train_window_for_test("2025-26"), ("2010-11", "2024-25"))
        self.assertEqual(expanding_train_window_for_test("2023-24", start_season="2018-19"), ("2018-19", "2022-23"))

    def test_aggregate_fold_results_sums_profit_and_roi(self):
        folds = [
            {"value_bets": {"overall": {"bets": 10, "wins": 5, "profit": 2.0, "roi": 0.2}}},
            {"value_bets": {"overall": {"bets": 5, "wins": 1, "profit": -1.0, "roi": -0.2}}},
        ]

        aggregate = aggregate_fold_results(folds)

        self.assertEqual(aggregate["bets"], 15)
        self.assertEqual(aggregate["wins"], 6)
        self.assertAlmostEqual(aggregate["profit"], 1.0)
        self.assertAlmostEqual(aggregate["roi"], 0.0667)
        self.assertEqual(aggregate["positive_roi_folds"], 1)
        self.assertEqual(aggregate["negative_roi_folds"], 1)

    def test_validate_league_subset_rejects_unknown_league(self):
        self.assertEqual(validate_league_subset(["ENG", "SPA"]), ("ENG", "SPA"))
        with self.assertRaisesRegex(ValueError, "Unknown leagues"):
            validate_league_subset(["ENG", "NED"])


    def test_value_bet_records_serializes_real_match_rows(self):
        value_bets = pd.DataFrame(
            [
                {
                    "RBallID": "ENG-1",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea",
                    "Date": pd.Timestamp("2021-08-08"),
                    "Season": "2021-22",
                    "League": "ENG",
                    "Result": "H",
                    "Outcome": "H",
                    "ModelOdds": 1.9,
                    "BestBookOdds": 2.2,
                    "Edge": 0.1579,
                    "BestBookmaker": "Bet365",
                }
            ]
        )

        records = value_bet_records(value_bets)

        self.assertEqual(records[0]["Date"], "2021-08-08")
        self.assertEqual(records[0]["League"], "ENG")
        self.assertEqual(records[0]["HomeTeam"], "Arsenal")
        self.assertEqual(records[0]["BestBookOdds"], 2.2)

    def test_summarize_value_bets_returns_overall_and_groups(self):
        value_bets = pd.DataFrame(
            [
                {"Season": "2023-24", "League": "ENG", "Outcome": "H", "Result": "H", "BestBookOdds": 2.5},
                {"Season": "2023-24", "League": "ENG", "Outcome": "A", "Result": "H", "BestBookOdds": 3.0},
                {"Season": "2024-25", "League": "SPA", "Outcome": "A", "Result": "A", "BestBookOdds": 2.2},
            ]
        )

        summary = summarize_value_bets(value_bets)

        self.assertEqual(summary["overall"]["bets"], 3)
        self.assertAlmostEqual(summary["overall"]["profit"], 1.7)
        self.assertEqual({row["Season"] for row in summary["by_season"]}, {"2023-24", "2024-25"})
        self.assertEqual({(row["League"], row["Season"]) for row in summary["by_league_season"]}, {("ENG", "2023-24"), ("SPA", "2024-25")})


if __name__ == "__main__":
    unittest.main()
