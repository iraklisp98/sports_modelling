import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from pipeline.poisson_goal_model import (
    BASE_FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    build_poisson_benchmarks,
    build_poisson_odds_frame,
    lambdas_to_outcome_probabilities,
    normalize_outcome_probabilities,
    probabilities_to_odds,
    run_pipeline,
    scoreline_outcome_probabilities,
    split_train_holdout,
    validate_outcome_probabilities,
)


def sample_goal_features() -> pd.DataFrame:
    rows = []
    seasons = ["2017-18", "2017-18", "2018-19", "2018-19", "2019-20", "2019-20"]
    results = [("H", 0, 2, 0), ("D", 1, 1, 1), ("A", 2, 0, 2), ("H", 0, 3, 1), ("D", 1, 0, 0), ("A", 2, 1, 2)]
    for idx, season in enumerate(seasons, start=1):
        result, result_code, home_goals, away_goals = results[idx - 1]
        row = {
            "RBallID": idx,
            "HomeTeam": f"Home {idx}",
            "AwayTeam": f"Away {idx}",
            "Date": f"{2016 + idx}-08-01",
            "Season": season,
            "League": "ENG",
            "Result": result,
            "ResultCode": result_code,
            "HomeGoals": home_goals,
            "AwayGoals": away_goals,
        }
        for offset, column in enumerate(BASE_FEATURE_COLUMNS):
            row[column] = float(idx + offset) / 10.0
        rows.append(row)
    return pd.DataFrame(rows)


class PoissonGoalModelTests(unittest.TestCase):
    def test_scoreline_aggregation_favors_home_when_home_lambda_is_higher(self):
        proba = scoreline_outcome_probabilities(lambda_home=2.2, lambda_away=0.8, max_goals=8)

        self.assertEqual(proba.shape, (3,))
        self.assertTrue(np.allclose(proba.sum(), 1.0))
        self.assertGreater(proba[0], proba[2])
        self.assertGreater(proba[1], 0.0)

    def test_lambdas_to_outcome_probabilities_returns_one_row_per_match(self):
        lambdas = np.array([[1.4, 1.4], [0.7, 1.8]])

        proba = lambdas_to_outcome_probabilities(lambdas, max_goals=8)

        self.assertEqual(proba.shape, (2, 3))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))
        self.assertGreater(proba[0, 1], proba[1, 1])
        self.assertGreater(proba[1, 2], proba[1, 0])

    def test_probability_validation_normalizes_finite_non_negative_rows(self):
        proba = normalize_outcome_probabilities(np.array([[2.0, 1.0, 1.0], [0.2, 0.3, 0.5]]))

        self.assertTrue(np.allclose(proba, np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])))
        self.assertTrue(np.allclose(validate_outcome_probabilities(proba).sum(axis=1), 1.0))

    def test_probability_validation_rejects_invalid_rows(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_outcome_probabilities(np.array([[np.nan, 0.3, 0.7]]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalize_outcome_probabilities(np.array([[0.5, -0.1, 0.6]]))
        with self.assertRaisesRegex(ValueError, "positive sum"):
            normalize_outcome_probabilities(np.array([[0.0, 0.0, 0.0]]))

    def test_odds_conversion_uses_reciprocal_probabilities(self):
        odds = probabilities_to_odds(np.array([[0.5, 0.25, 0.25]]))

        self.assertTrue(np.allclose(odds, np.array([[2.0, 4.0, 4.0]])))
        self.assertTrue(np.isfinite(odds).all())

    def test_split_behavior_uses_only_pre_2019_20_rows_for_training(self):
        train, holdout = split_train_holdout(sample_goal_features())

        self.assertEqual(train["Season"].tolist(), ["2017-18", "2017-18", "2018-19", "2018-19"])
        self.assertEqual(holdout["Season"].tolist(), ["2019-20", "2019-20"])
        self.assertLess(train["Date"].max(), holdout["Date"].min())

    def test_build_poisson_odds_frame_matches_stage4_schema(self):
        holdout = sample_goal_features().query("Season == '2019-20'").reset_index(drop=True)
        lambdas = np.array([[1.1, 1.0], [0.8, 1.6]])
        proba = lambdas_to_outcome_probabilities(lambdas, max_goals=8)

        odds = build_poisson_odds_frame(holdout, lambdas, proba)

        self.assertEqual(odds.columns.tolist(), OUTPUT_COLUMNS)
        self.assertEqual(odds["RBallID"].tolist(), [5, 6])
        self.assertTrue(np.allclose(odds[["P_Home", "P_Draw", "P_Away"]].sum(axis=1), 1.0))
        self.assertTrue((odds[["ModelOdds_Home", "ModelOdds_Draw", "ModelOdds_Away"]] > 0).all().all())

    def test_build_poisson_benchmarks_appends_to_existing_xgboost_rows(self):
        with TemporaryDirectory() as tmp_dir:
            benchmark_path = Path(tmp_dir) / "model_benchmarks.json"
            benchmark_path.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {"model": "calibrated_xgboost", "log_loss": 0.99, "accuracy": 0.5},
                            {"model": "calibrated_xgboost_draw_overlay", "log_loss": 0.98, "accuracy": 0.52},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            df = sample_goal_features()
            train, holdout = split_train_holdout(df)
            proba = np.array([[0.3, 0.4, 0.3], [0.2, 0.2, 0.6]])

            benchmarks = build_poisson_benchmarks(train, holdout, proba, existing_benchmarks_path=benchmark_path)

        self.assertEqual([row["model"] for row in benchmarks], ["calibrated_xgboost", "calibrated_xgboost_draw_overlay", "poisson_goal_model"])
        self.assertIn("brier_score", benchmarks[-1])
        self.assertIn("f1_draw", benchmarks[-1])

    def test_run_pipeline_writes_odds_metrics_and_benchmark_artifacts(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            features_dir = tmp_path / "features"
            output_path = tmp_path / "output" / "poisson_model_odds.parquet"
            artifacts_dir = tmp_path / "artifacts"
            existing_benchmarks_path = tmp_path / "stage3" / "model_benchmarks.json"
            features_dir.mkdir(parents=True)
            existing_benchmarks_path.parent.mkdir(parents=True)
            existing_benchmarks_path.write_text(
                json.dumps({"benchmarks": [{"model": "calibrated_xgboost", "log_loss": 0.99, "accuracy": 0.5}]}),
                encoding="utf-8",
            )
            sample_goal_features().to_parquet(features_dir / "ENG_features.parquet", index=False)

            summary = run_pipeline(
                leagues=["ENG"],
                features_dir=features_dir,
                output_path=output_path,
                artifacts_dir=artifacts_dir,
                existing_benchmarks_path=existing_benchmarks_path,
                max_goals=8,
            )

            odds = pd.read_parquet(output_path)
            predictions = pd.read_parquet(artifacts_dir / "holdout_predictions.parquet")
            metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
            benchmarks = json.loads((artifacts_dir / "model_benchmarks.json").read_text(encoding="utf-8"))["benchmarks"]

        self.assertEqual(summary.train_rows, 4)
        self.assertEqual(summary.holdout_rows, 2)
        self.assertEqual(odds.columns.tolist(), OUTPUT_COLUMNS)
        self.assertTrue({"Lambda_Home", "Lambda_Away"}.issubset(predictions.columns))
        self.assertTrue({"holdout_log_loss", "holdout_brier_score", "holdout_accuracy", "holdout_f1_home"}.issubset(metrics))
        self.assertEqual(benchmarks[-1]["model"], "poisson_goal_model")


if __name__ == "__main__":
    unittest.main()
