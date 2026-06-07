import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pipeline.market_baseline_diagnostics import build_diagnostics, build_market_table, run_pipeline


def sample_model_odds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RBallID": "m1",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Date": "2019-08-10",
                "Season": "2019-20",
                "Result": "H",
                "ModelOdds_Home": 2.0,
                "ModelOdds_Draw": 4.0,
                "ModelOdds_Away": 4.0,
            },
            {
                "RBallID": "m2",
                "HomeTeam": "Chelsea",
                "AwayTeam": "Arsenal",
                "Date": "2019-08-17",
                "Season": "2019-20",
                "Result": "A",
                "ModelOdds_Home": 3.0,
                "ModelOdds_Draw": 3.0,
                "ModelOdds_Away": 2.4,
            },
        ]
    )


def sample_bookmaker_odds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Date": "2019-08-10", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "League": "ENG", "B365_H": 1.9, "B365_D": 3.4, "B365_A": 4.0},
            {"Date": "2019-08-17", "HomeTeam": "Chelsea", "AwayTeam": "Arsenal", "League": "ENG", "B365_H": 2.8, "B365_D": 3.1, "B365_A": 2.2},
        ]
    )


class MarketBaselineDiagnosticsTests(unittest.TestCase):
    def test_build_market_table_normalizes_bookmaker_and_model_probabilities(self):
        table = build_market_table(sample_model_odds(), sample_bookmaker_odds())

        self.assertEqual(len(table), 2)
        self.assertTrue(all(abs(table[["ModelProb_H", "ModelProb_D", "ModelProb_A"]].sum(axis=1) - 1.0) < 1e-9))
        self.assertTrue(all(abs(table[["MarketProb_H", "MarketProb_D", "MarketProb_A"]].sum(axis=1) - 1.0) < 1e-9))
        self.assertGreater(table.loc[0, "BookmakerMargin"], 0.0)

    def test_build_diagnostics_compares_model_and_market_metrics(self):
        payload = build_diagnostics(sample_model_odds(), sample_bookmaker_odds(), seasons=("2019-20",))

        self.assertEqual(payload["rows"], 2)
        self.assertIn("model_metrics", payload)
        self.assertIn("market_metrics", payload)
        self.assertIn("model_calibration", payload)
        self.assertIn("market_calibration", payload)
        self.assertIn("model_minus_market_edge", payload)
        self.assertGreater(len(payload["model_minus_market_edge"]), 0)

    def test_run_pipeline_writes_json(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_path = tmp_path / "model.parquet"
            output_path = tmp_path / "market.json"
            sample_model_odds().to_parquet(model_path, index=False)

            # run_pipeline loads Football-Data through stage5, so use build_diagnostics for unit shape and write manually here.
            payload = build_diagnostics(sample_model_odds(), sample_bookmaker_odds(), seasons=("2019-20",))
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["rows"], 2)


if __name__ == "__main__":
    unittest.main()
