import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.stage4_odds_gen import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    build_model_odds_frame,
    load_production_model,
    probabilities_to_odds,
    run_pipeline,
    validate_model_class_order,
    validate_probability_matrix,
)


def sample_features() -> pd.DataFrame:
    rows = []
    for idx, season in enumerate(["2017-18", "2018-19"], start=1):
        row = {
            "RBallID": idx,
            "HomeTeam": f"Home {idx}",
            "AwayTeam": f"Away {idx}",
            "Date": f"2017-08-0{idx}",
            "Season": season,
            "Result": ["H", "D"][idx - 1],
        }
        for offset, column in enumerate(FEATURE_COLUMNS):
            row[column] = float(idx + offset)
        rows.append(row)
    return pd.DataFrame(rows[::-1])


class FakeModel:
    def __init__(self, proba: np.ndarray, classes_: np.ndarray | None = None):
        self._proba = proba
        self.classes_ = np.array([0, 1, 2]) if classes_ is None else np.asarray(classes_)
        self.seen_features = None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.seen_features = features.copy()
        return self._proba


class Stage4OddsGenTests(unittest.TestCase):
    def test_validate_probability_matrix_accepts_probabilities_that_sum_to_one(self):
        proba = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])

        validated = validate_probability_matrix(proba)

        self.assertTrue(np.array_equal(validated, proba))

    def test_validate_probability_matrix_rejects_rows_outside_tolerance(self):
        proba = np.array([[0.5, 0.25, 0.2]])

        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            validate_probability_matrix(proba)

    def test_probabilities_to_odds_converts_each_outcome(self):
        proba = np.array([[0.5, 0.25, 0.2]])

        odds = probabilities_to_odds(proba)

        self.assertTrue(np.allclose(odds, np.array([[2.0, 4.0, 5.0]])))

    def test_probabilities_to_odds_rejects_zero_or_negative_values(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            probabilities_to_odds(np.array([[0.5, 0.0, 0.5]]))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            probabilities_to_odds(np.array([[0.5, -0.1, 0.6]]))

    def test_validate_model_class_order_rejects_unexpected_encoding(self):
        with self.assertRaisesRegex(ValueError, "H/D/A"):
            validate_model_class_order(FakeModel(np.array([[0.2, 0.3, 0.5]]), classes_=[2, 1, 0]))

    def test_load_production_model_uses_registry_uri(self):
        fake_model = object()
        with patch("mlflow.xgboost.load_model", return_value=fake_model) as mocked_load:
            loaded = load_production_model(tracking_uri="file:mlruns")

        mocked_load.assert_called_once_with("models:/match_outcome_xgb/Production")
        self.assertIs(loaded, fake_model)

    def test_build_model_odds_frame_sorts_rows_and_adds_odds(self):
        df = sample_features()
        model = FakeModel(np.array([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]]))

        odds_df = build_model_odds_frame(df, model)

        self.assertEqual(odds_df["RBallID"].tolist(), [1, 2])
        self.assertEqual(odds_df.columns.tolist(), OUTPUT_COLUMNS)
        self.assertEqual(odds_df["Result"].tolist(), ["H", "D"])
        self.assertEqual(odds_df["P_Home"].tolist(), [0.2, 0.5])
        self.assertAlmostEqual(odds_df.loc[0, "ModelOdds_Home"], 5.0)
        self.assertAlmostEqual(odds_df.loc[1, "ModelOdds_Draw"], 4.0)
        self.assertEqual(model.seen_features.columns.tolist(), FEATURE_COLUMNS)
        self.assertTrue(odds_df["Date"].is_monotonic_increasing)

    def test_run_pipeline_writes_model_odds_parquet(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            features_dir = tmp_path / "features"
            output_path = tmp_path / "output" / "model_odds.parquet"
            features_dir.mkdir(parents=True, exist_ok=True)

            sample_features().to_parquet(features_dir / "ENG_features.parquet", index=False)
            sample_features().assign(
                RBallID=[3, 4],
                Date=["2017-08-03", "2017-08-04"],
                Result=["A", "H"],
            ).to_parquet(features_dir / "SPA_features.parquet", index=False)

            fake_model = FakeModel(
                np.array(
                    [
                        [0.2, 0.3, 0.5],
                        [0.5, 0.25, 0.25],
                        [0.4, 0.4, 0.2],
                        [0.1, 0.2, 0.7],
                    ]
                )
            )

            with patch("pipeline.stage4_odds_gen.load_production_model", return_value=fake_model):
                summary = run_pipeline(
                    leagues=["ENG", "SPA"],
                    features_dir=features_dir,
                    output_path=output_path,
                    tracking_uri="file:mlruns",
                )

            written = pd.read_parquet(output_path)
            self.assertTrue(output_path.exists())

        self.assertEqual(summary.rows, 4)
        self.assertEqual(summary.output_path, output_path)
        self.assertEqual(written.columns.tolist(), OUTPUT_COLUMNS)
        self.assertEqual(written["RBallID"].tolist(), [1, 2, 3, 4])
        self.assertTrue(np.allclose(written[["P_Home", "P_Draw", "P_Away"]].sum(axis=1), 1.0))
        self.assertTrue((written[["ModelOdds_Home", "ModelOdds_Draw", "ModelOdds_Away"]] > 0).all().all())


if __name__ == "__main__":
    unittest.main()
