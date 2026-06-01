import unittest

import numpy as np
import pandas as pd

from pipeline.stage3_train import (
    FEATURE_COLUMNS,
    evaluate_predictions,
    multiclass_brier_score,
    select_features_and_target,
    split_train_holdout,
)


def sample_feature_rows() -> pd.DataFrame:
    rows = []
    seasons = ["2016-17", "2017-18", "2018-19", "2019-20"]
    for idx, season in enumerate(seasons, start=1):
        row = {
            "RBallID": idx,
            "Date": f"{2016 + idx}-08-01",
            "Season": season,
            "ResultCode": (idx - 1) % 3,
        }
        for offset, column in enumerate(FEATURE_COLUMNS):
            row[column] = float(idx + offset)
        rows.append(row)
    return pd.DataFrame(rows)


class Stage3TrainTests(unittest.TestCase):
    def test_split_train_holdout_uses_all_pre_holdout_seasons_by_default(self):
        train, holdout = split_train_holdout(sample_feature_rows())

        self.assertEqual(train["Season"].tolist(), ["2016-17", "2017-18", "2018-19"])
        self.assertEqual(holdout["Season"].tolist(), ["2019-20"])
        self.assertEqual(train["RBallID"].tolist(), [1, 2, 3])
        self.assertEqual(holdout["RBallID"].tolist(), [4])

    def test_split_train_holdout_accepts_explicit_training_seasons(self):
        train, holdout = split_train_holdout(sample_feature_rows(), train_seasons=("2017-18", "2018-19"))

        self.assertEqual(train["Season"].tolist(), ["2017-18", "2018-19"])
        self.assertEqual(holdout["Season"].tolist(), ["2019-20"])

    def test_select_features_and_target_keeps_expected_numeric_columns_only(self):
        df = sample_feature_rows()
        df["UnusedTextColumn"] = "ignore-me"

        X, y = select_features_and_target(df)

        self.assertEqual(X.columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(y.tolist(), [0, 1, 2, 0])
        self.assertTrue(all(str(dtype) == "float64" for dtype in X.dtypes))

    def test_select_features_and_target_rejects_unknown_labels(self):
        df = sample_feature_rows()
        df.loc[0, "ResultCode"] = 9

        with self.assertRaisesRegex(ValueError, "Unexpected target labels"):
            select_features_and_target(df)

    def test_multiclass_brier_score_uses_all_classes(self):
        y_true = pd.Series([0, 1, 2])
        y_proba = np.array(
            [
                [0.80, 0.10, 0.10],
                [0.20, 0.60, 0.20],
                [0.20, 0.20, 0.60],
            ]
        )

        expected = np.mean(
            [
                (0.80 - 1) ** 2 + 0.10**2 + 0.10**2,
                0.20**2 + (0.60 - 1) ** 2 + 0.20**2,
                0.20**2 + 0.20**2 + (0.60 - 1) ** 2,
            ]
        )
        self.assertAlmostEqual(multiclass_brier_score(y_true, y_proba), expected)

    def test_evaluate_predictions_reports_holdout_metrics(self):
        y_true = pd.Series([0, 1, 2, 0])
        y_proba = np.array(
            [
                [0.90, 0.05, 0.05],
                [0.20, 0.70, 0.10],
                [0.10, 0.20, 0.70],
                [0.60, 0.30, 0.10],
            ]
        )

        metrics = evaluate_predictions(y_true, y_proba)

        self.assertEqual(metrics["holdout_accuracy"], 1.0)
        self.assertLess(metrics["holdout_log_loss"], 0.5)
        self.assertIn("holdout_brier_score", metrics)
        self.assertEqual(metrics["holdout_f1_home"], 1.0)
        self.assertEqual(metrics["holdout_f1_draw"], 1.0)
        self.assertEqual(metrics["holdout_f1_away"], 1.0)


if __name__ == "__main__":
    unittest.main()
