import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pipeline.model_diagnostics import (
    build_calibration_table,
    build_diagnostics,
    build_prediction_rows,
    build_value_bet_diagnostics,
    build_odds_range_diagnostics,
    odds_bucket,
    probability_bucket,
    run_pipeline,
)


def sample_holdout() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"RBallID": "m1", "Result": "H", "P_Home": 0.60, "P_Draw": 0.20, "P_Away": 0.20},
            {"RBallID": "m2", "Result": "D", "P_Home": 0.40, "P_Draw": 0.35, "P_Away": 0.25},
            {"RBallID": "m3", "Result": "A", "P_Home": 0.20, "P_Draw": 0.30, "P_Away": 0.50},
        ]
    )


def sample_value_bets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RBallID": "m1",
                "Season": "2019-20",
                "Outcome": "H",
                "Result": "H",
                "ModelOdds": 2.0,
                "BestBookOdds": 2.4,
                "Edge": 0.2,
            },
            {
                "RBallID": "m2",
                "Season": "2019-20",
                "Outcome": "A",
                "Result": "D",
                "ModelOdds": 4.0,
                "BestBookOdds": 5.0,
                "Edge": 0.25,
            },
            {
                "RBallID": "m-old",
                "Season": "2018-19",
                "Outcome": "H",
                "Result": "A",
                "ModelOdds": 2.5,
                "BestBookOdds": 3.0,
                "Edge": 0.2,
            },
        ]
    )


class ModelDiagnosticsTests(unittest.TestCase):
    def test_each_match_contributes_once_per_outcome(self):
        rows = build_prediction_rows(sample_holdout(), bins=(0.0, 0.5, 1.0))

        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[rows["outcome"] == "H"]["actual"].tolist(), [1, 0, 0])
        self.assertEqual(rows[rows["outcome"] == "D"]["actual"].tolist(), [0, 1, 0])
        self.assertEqual(rows[rows["outcome"] == "A"]["actual"].tolist(), [0, 0, 1])

    def test_calibration_table_computes_expected_error_per_outcome_bucket(self):
        table = build_calibration_table(sample_holdout(), bins=(0.0, 0.5, 1.0))
        lookup = {(row["outcome"], row["bucket"]): row for row in table}

        home_low = lookup[("H", "0.0-0.5")]
        self.assertEqual(home_low["count"], 2)
        self.assertEqual(home_low["avg_predicted_probability"], 0.3)
        self.assertEqual(home_low["empirical_rate"], 0.0)
        self.assertEqual(home_low["abs_calibration_error"], 0.3)

        home_high = lookup[("H", "0.5-1.0")]
        self.assertEqual(home_high["count"], 1)
        self.assertEqual(home_high["empirical_rate"], 1.0)

    def test_probability_bucket_boundaries_are_stable(self):
        self.assertEqual(probability_bucket(0.0, bins=(0.0, 0.1, 0.2, 1.0)), "0.0-0.1")
        self.assertEqual(probability_bucket(0.1, bins=(0.0, 0.1, 0.2, 1.0)), "0.0-0.1")
        self.assertEqual(probability_bucket(0.10001, bins=(0.0, 0.1, 0.2, 1.0)), "0.1-0.2")
        self.assertEqual(probability_bucket(1.0, bins=(0.0, 0.1, 0.2, 1.0)), "0.2-1.0")


    def test_odds_bucket_boundaries_are_stable(self):
        self.assertEqual(odds_bucket(1.01, bins=(1.0, 2.0, 3.0)), "1.0-2.0")
        self.assertEqual(odds_bucket(2.0, bins=(1.0, 2.0, 3.0)), "1.0-2.0")
        self.assertEqual(odds_bucket(2.01, bins=(1.0, 2.0, 3.0)), "2.0-3.0")
        self.assertEqual(odds_bucket(5.0, bins=(1.0, 2.0, 3.0)), "3.0+")

    def test_odds_range_diagnostics_include_actual_result_rates_and_roi(self):
        diagnostics = build_odds_range_diagnostics(sample_value_bets(), seasons=("2019-20",), odds_bins=(1.0, 2.5, 4.0, 8.0))
        model_lookup = {(row["outcome"], row["bucket"]): row for row in diagnostics["model_odds_ranges"]}
        book_lookup = {(row["outcome"], row["bucket"]): row for row in diagnostics["bookmaker_odds_ranges"]}

        home_model = model_lookup[("H", "1.0-2.5")]
        self.assertEqual(home_model["count"], 1)
        self.assertEqual(home_model["wins"], 1)
        self.assertEqual(home_model["actual_result_rates"], {"H": 1.0, "D": 0.0, "A": 0.0})

        away_book = book_lookup[("A", "4.0-8.0")]
        self.assertEqual(away_book["count"], 1)
        self.assertEqual(away_book["wins"], 0)
        self.assertEqual(away_book["flat_stake_roi"], -1.0)

    def test_value_bet_diagnostics_are_limited_to_holdout_value_bet_rows(self):
        diagnostics = build_value_bet_diagnostics(sample_value_bets(), seasons=("2019-20",), bins=(0.0, 0.5, 1.0))

        self.assertEqual(sum(row["count"] for row in diagnostics), 2)
        self.assertEqual({row["outcome"] for row in diagnostics}, {"H", "A"})

    def test_empty_value_bets_still_writes_empty_value_bet_section(self):
        empty_value_bets = sample_value_bets().iloc[0:0].copy()

        payload = build_diagnostics(sample_holdout(), empty_value_bets)

        self.assertEqual(payload["value_bets_by_outcome_bucket"], [])
        self.assertGreater(len(payload["calibration_by_outcome_bucket"]), 0)

    def test_run_pipeline_writes_valid_json_contract(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            holdout_path = tmp_path / "holdout.parquet"
            value_bets_path = tmp_path / "value_bets.parquet"
            output_path = tmp_path / "diagnostics.json"
            sample_holdout().to_parquet(holdout_path, index=False)
            sample_value_bets().to_parquet(value_bets_path, index=False)

            summary = run_pipeline(holdout_path, value_bets_path, output_path, seasons=("2019-20",))
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary.output_path, output_path)
        self.assertIn("outcome_summary", payload)
        self.assertIn("calibration_by_outcome_bucket", payload)
        self.assertIn("value_bets_by_outcome_bucket", payload)
        self.assertIn("value_bets_by_odds_range", payload)
        self.assertIn("worst_calibration_bucket", payload)

    def test_missing_required_holdout_columns_raises_clear_error(self):
        holdout = sample_holdout().drop(columns=["RBallID"])

        with self.assertRaisesRegex(ValueError, "Missing holdout prediction columns"):
            build_calibration_table(holdout)

    def test_missing_required_value_bet_columns_raises_clear_error(self):
        value_bets = sample_value_bets().drop(columns=["RBallID"])

        with self.assertRaisesRegex(ValueError, "Missing value bet columns"):
            build_value_bet_diagnostics(value_bets)

    def test_invalid_probabilities_are_rejected(self):
        holdout = sample_holdout()
        holdout.loc[0, "P_Home"] = 1.2

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            build_calibration_table(holdout)

    def test_probability_rows_must_sum_to_one_with_tolerance(self):
        holdout = sample_holdout()
        holdout.loc[0, "P_Home"] = 0.7

        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            build_calibration_table(holdout)


if __name__ == "__main__":
    unittest.main()
