import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pipeline.compare_value_bet_models import build_comparison, run_pipeline


def sample_value_bets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Season": "2019-20", "League": "ENG", "Outcome": "H", "Result": "H", "BestBookOdds": 2.2},
            {"Season": "2019-20", "League": "ENG", "Outcome": "A", "Result": "H", "BestBookOdds": 3.4},
            {"Season": "2020-21", "League": "SPA", "Outcome": "A", "Result": "A", "BestBookOdds": 2.8},
            {"Season": "2018-19", "League": "ENG", "Outcome": "H", "Result": "A", "BestBookOdds": 2.0},
        ]
    )


class CompareValueBetModelsTests(unittest.TestCase):
    def test_build_comparison_summarizes_overall_and_groups(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "value_bets.parquet"
            sample_value_bets().to_parquet(path, index=False)

            payload = build_comparison([("model", path)], seasons=("2019-20", "2020-21"))

        model = payload["models"][0]
        self.assertEqual(model["overall"]["bets"], 3)
        self.assertEqual(model["overall"]["wins"], 2)
        self.assertAlmostEqual(model["overall"]["profit"], 2.0)
        self.assertEqual({row["Outcome"] for row in model["by_outcome"]}, {"H", "A"})
        self.assertEqual({row["League"] for row in model["by_league"]}, {"ENG", "SPA"})

    def test_run_pipeline_writes_json(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            value_bets_path = tmp_path / "value_bets.parquet"
            output_path = tmp_path / "comparison.json"
            sample_value_bets().to_parquet(value_bets_path, index=False)

            summary = run_pipeline([("model", value_bets_path)], output_path=output_path, seasons=("2019-20",))
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary.models, 1)
        self.assertEqual(summary.output_path, output_path)
        self.assertEqual(payload["models"][0]["overall"]["bets"], 2)


if __name__ == "__main__":
    unittest.main()
