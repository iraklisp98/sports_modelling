import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pipeline.stage3_train import (
    BASE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    LEAGUE_FEATURE_COLUMNS,
    add_league_indicator_features,
    always_home_probabilities,
    build_model_benchmarks,
    calibrate_classifier,
    class_prior_probabilities,
    blend_draw_probability,
    compute_binary_sample_weights,
    compute_class_sample_weights,
    draw_binary_labels,
    elo_heuristic_probabilities,
    evaluate_predictions,
    majority_class_probabilities,
    multiclass_brier_score,
    normalize_probabilities,
    run_pipeline,
    select_draw_overlay_weight,
    select_features_and_target,
    split_model_calibration_data,
    split_train_holdout,
    train_classifier,
)


def sample_feature_rows() -> pd.DataFrame:
    rows = []
    seasons = ["2016-17", "2017-18", "2018-19", "2019-20"]
    for idx, season in enumerate(seasons, start=1):
        row = {
            "RBallID": idx,
            "Date": f"{2016 + idx}-08-01",
            "Season": season,
            "League": ["ENG", "SPA", "FRA", "ENG"][idx - 1],
            "ResultCode": (idx - 1) % 3,
        }
        for offset, column in enumerate(BASE_FEATURE_COLUMNS):
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

    def test_add_league_indicator_features_adds_numeric_one_hot_columns(self):
        df = sample_feature_rows().assign(League=["ENG", "SPA", "FRA", "ENG"])

        featured = add_league_indicator_features(df)

        self.assertEqual(list(LEAGUE_FEATURE_COLUMNS), ["League_ENG", "League_SPA", "League_FRA", "League_GER", "League_ITA"])
        self.assertEqual(featured.loc[0, ["League_ENG", "League_SPA", "League_FRA", "League_GER", "League_ITA"]].tolist(), [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(featured.loc[1, ["League_ENG", "League_SPA", "League_FRA", "League_GER", "League_ITA"]].tolist(), [0.0, 1.0, 0.0, 0.0, 0.0])
        self.assertTrue(all(str(featured[column].dtype) == "float64" for column in LEAGUE_FEATURE_COLUMNS))

    def test_add_league_indicator_features_rejects_unknown_league(self):
        df = sample_feature_rows().assign(League=["ENG", "NED", "FRA", "ENG"])

        with self.assertRaisesRegex(ValueError, "Unexpected league labels"):
            add_league_indicator_features(df)

    def test_select_features_and_target_keeps_expected_numeric_columns_only(self):
        df = sample_feature_rows()
        df["UnusedTextColumn"] = "ignore-me"

        X, y = select_features_and_target(df)

        self.assertEqual(X.columns.tolist(), list(FEATURE_COLUMNS))
        self.assertEqual(y.tolist(), [0, 1, 2, 0])
        self.assertTrue(all(str(dtype) == "float64" for dtype in X.dtypes))

    def test_select_features_and_target_adds_numeric_league_indicators(self):
        X, _ = select_features_and_target(sample_feature_rows())

        self.assertEqual(X[["League_ENG", "League_SPA", "League_FRA", "League_GER", "League_ITA"]].values.tolist(), [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ])
        self.assertTrue(all(str(X[column].dtype) == "float64" for column in ["League_ENG", "League_SPA", "League_FRA", "League_GER", "League_ITA"]))

    def test_select_features_and_target_rejects_unknown_league_labels(self):
        df = sample_feature_rows()
        df.loc[0, "League"] = "NED"

        with self.assertRaisesRegex(ValueError, "Unexpected league labels"):
            select_features_and_target(df)

    def test_select_features_and_target_rejects_unknown_labels(self):
        df = sample_feature_rows()
        df.loc[0, "ResultCode"] = 9

        with self.assertRaisesRegex(ValueError, "Unexpected target labels"):
            select_features_and_target(df)

    def test_normalize_probabilities_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_probabilities(np.array([[np.nan, 0.2, 0.8]]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalize_probabilities(np.array([[0.6, -0.1, 0.5]]))
        with self.assertRaisesRegex(ValueError, "positive sum"):
            normalize_probabilities(np.array([[0.0, 0.0, 0.0]]))

    def test_split_model_calibration_data_uses_latest_training_rows(self):
        base = sample_feature_rows().iloc[0].copy()
        rows = []
        for idx in range(12):
            row = base.copy()
            row["RBallID"] = idx
            row["Date"] = f"2018-09-{idx + 1:02d}"
            row["Season"] = "2018-19"
            row["ResultCode"] = idx % 3
            rows.append(row)
        train_df = pd.DataFrame(rows)

        model_df, calibration_df = split_model_calibration_data(train_df, calibration_fraction=0.25, min_calibration_rows=3)

        self.assertEqual(len(calibration_df), 3)
        self.assertEqual(set(calibration_df["ResultCode"]), {0, 1, 2})
        self.assertGreater(calibration_df["Date"].min(), model_df["Date"].max())

    def test_calibrate_classifier_returns_predict_proba_wrapper(self):
        X_train = pd.DataFrame(
            {
                "a": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
                "b": [0.0, 0.2, 0.1, 0.3, 0.5, 0.4, 1.0, 1.2, 1.1, 1.3, 1.5, 1.4, 2.0, 2.2, 2.1, 2.3, 2.5, 2.4],
            }
        )
        y_train = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        base_model = LogisticRegression(max_iter=200).fit(X_train, y_train)
        X_calibration = X_train.copy()
        y_calibration = y_train.copy()

        calibrated = calibrate_classifier(base_model, X_calibration, y_calibration)
        proba = calibrated.predict_proba(X_calibration)

        self.assertTrue(hasattr(calibrated, "predict_proba"))
        self.assertTrue(np.array_equal(calibrated.classes_, np.array([0, 1, 2])))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))


    def test_calibrate_classifier_rejects_unsupported_method(self):
        X_train = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]})
        y_train = pd.Series([0, 1, 2])
        base_model = LogisticRegression(max_iter=200).fit(X_train, y_train)

        with self.assertRaisesRegex(ValueError, "Unsupported calibration method"):
            calibrate_classifier(base_model, X_train, y_train, method="unknown")

    def test_class_prior_probabilities_repeats_training_distribution(self):
        proba = class_prior_probabilities(pd.Series([0, 0, 1, 2]), rows=3)

        self.assertEqual(proba.shape, (3, 3))
        self.assertTrue(np.allclose(proba[0], [0.5, 0.25, 0.25]))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_majority_class_probabilities_predicts_training_majority(self):
        proba = majority_class_probabilities(pd.Series([0, 0, 0, 1, 2]), rows=3, confidence=0.8)

        self.assertEqual(proba.shape, (3, 3))
        self.assertTrue(np.allclose(proba, np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_always_home_probabilities_uses_configured_confidence(self):
        proba = always_home_probabilities(rows=2, confidence=0.8)

        self.assertTrue(np.allclose(proba, np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])))

    def test_elo_heuristic_probabilities_uses_elo_direction_and_train_draw_rate(self):
        holdout = pd.DataFrame({"HomeElo": [1600.0, 1400.0], "AwayElo": [1400.0, 1600.0]})
        proba = elo_heuristic_probabilities(holdout, pd.Series([0, 0, 1, 2]))

        self.assertEqual(proba.shape, (2, 3))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))
        self.assertAlmostEqual(proba[0, 1], 0.25)
        self.assertGreater(proba[0, 0], proba[0, 2])
        self.assertGreater(proba[1, 2], proba[1, 0])

    def test_build_model_benchmarks_includes_baselines_and_model_metrics(self):
        holdout = pd.DataFrame({"HomeElo": [1600.0, 1400.0, 1500.0], "AwayElo": [1400.0, 1600.0, 1500.0]})
        y_train = pd.Series([0, 0, 1, 2])
        y_holdout = pd.Series([0, 2, 1])
        model_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]])

        benchmarks = build_model_benchmarks(y_train, holdout, y_holdout, model_proba)

        self.assertEqual(
            [row["model"] for row in benchmarks],
            ["historical_class_prior", "majority_class", "always_home", "elo_heuristic", "calibrated_xgboost"],
        )
        for row in benchmarks:
            self.assertIn("log_loss", row)
            self.assertIn("accuracy", row)
            self.assertIn("f1_draw", row)
            self.assertIn("predicted_draw", row)
        self.assertEqual(benchmarks[-1]["accuracy"], 1.0)

    def test_draw_binary_labels_maps_draw_to_one(self):
        labels = draw_binary_labels(pd.Series([0, 1, 2, 1]))

        self.assertEqual(labels.tolist(), [0, 1, 0, 1])

    def test_compute_binary_sample_weights_balances_draw_labels(self):
        labels = pd.Series([0, 0, 0, 1])

        weights = compute_binary_sample_weights(labels)

        self.assertEqual(len(weights), len(labels))
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertGreater(weights[labels == 1].mean(), weights[labels == 0].mean())

    def test_blend_draw_probability_preserves_rows_and_home_away_ratio(self):
        base = np.array([[0.50, 0.20, 0.30], [0.20, 0.50, 0.30]])
        draw = np.array([0.40, 0.30])

        blended = blend_draw_probability(base, draw, blend_weight=0.5)

        self.assertEqual(blended.shape, (2, 3))
        self.assertTrue(np.allclose(blended.sum(axis=1), 1.0))
        self.assertAlmostEqual(blended[0, 1], 0.30)
        self.assertAlmostEqual(blended[1, 1], 0.40)
        self.assertAlmostEqual(blended[0, 0] / blended[0, 2], base[0, 0] / base[0, 2])
        self.assertAlmostEqual(blended[1, 0] / blended[1, 2], base[1, 0] / base[1, 2])

    def test_blend_draw_probability_rejects_invalid_binary_probabilities(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            blend_draw_probability(np.array([[0.4, 0.3, 0.3]]), np.array([1.2]))
        with self.assertRaisesRegex(ValueError, "one value per"):
            blend_draw_probability(np.array([[0.4, 0.3, 0.3]]), np.array([0.2, 0.3]))


    def test_select_draw_overlay_weight_includes_zero_as_valid_candidate(self):
        y_true = pd.Series([0, 1, 2])
        base = np.array([[0.70, 0.20, 0.10], [0.20, 0.60, 0.20], [0.10, 0.20, 0.70]])
        draw = np.array([0.80, 0.10, 0.80])

        selected, results = select_draw_overlay_weight(y_true, base, draw, candidate_weights=[0.0, 0.5])

        self.assertEqual(selected, 0.0)
        self.assertIn(0.0, [row["draw_overlay_weight"] for row in results])

    def test_select_draw_overlay_weight_chooses_lowest_validation_log_loss(self):
        y_true = pd.Series([0, 1, 2])
        base = np.array([[0.70, 0.20, 0.10], [0.20, 0.30, 0.50], [0.10, 0.20, 0.70]])
        draw = np.array([0.05, 0.80, 0.05])

        selected, results = select_draw_overlay_weight(y_true, base, draw, candidate_weights=[0.0, 0.5, 1.0])

        losses = {row["draw_overlay_weight"]: row["validation_log_loss"] for row in results}
        self.assertEqual(selected, min(losses, key=losses.get))
        self.assertLess(losses[selected], losses[0.0])

    def test_select_draw_overlay_weight_breaks_ties_with_smaller_weight(self):
        y_true = pd.Series([0, 1, 2])
        base = np.array([[0.70, 0.20, 0.10], [0.20, 0.60, 0.20], [0.10, 0.20, 0.70]])
        draw = base[:, 1]

        selected, _ = select_draw_overlay_weight(y_true, base, draw, candidate_weights=[0.5, 0.0, 0.25])

        self.assertEqual(selected, 0.0)

    def test_select_draw_overlay_weight_rejects_invalid_candidates(self):
        y_true = pd.Series([0, 1, 2])
        base = np.array([[0.70, 0.20, 0.10], [0.20, 0.60, 0.20], [0.10, 0.20, 0.70]])
        draw = np.array([0.20, 0.60, 0.20])

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            select_draw_overlay_weight(y_true, base, draw, candidate_weights=[0.0, 1.2])

    def test_run_pipeline_selects_weight_from_calibration_labels_not_holdout(self):
        def make_rows(season: str, labels: list[int], start_id: int) -> pd.DataFrame:
            rows = []
            for offset, label in enumerate(labels):
                row = {
                    "RBallID": start_id + offset,
                    "Date": f"2018-09-{offset + 1:02d}" if season != "2019-20" else f"2019-09-{offset + 1:02d}",
                    "Season": season,
                    "League": "ENG",
                    "HomeTeam": f"Home {start_id + offset}",
                    "AwayTeam": f"Away {start_id + offset}",
                    "Result": ["H", "D", "A"][label],
                    "ResultCode": label,
                }
                for index, column in enumerate(BASE_FEATURE_COLUMNS):
                    row[column] = float(start_id + offset + index)
                rows.append(row)
            return pd.DataFrame(rows)

        model_df = make_rows("2018-19", [0, 1, 2], 1)
        calibration_df = make_rows("2018-19", [0, 1, 2], 10)
        holdout_df = make_rows("2019-20", [2, 2, 2], 20)
        base_model = Mock()
        base_model.save_model = Mock()
        calibrated_model = Mock()
        calibrated_model.predict_proba.side_effect = [
            np.array([[0.70, 0.20, 0.10], [0.20, 0.30, 0.50], [0.10, 0.20, 0.70]]),
            np.array([[0.10, 0.20, 0.70], [0.10, 0.20, 0.70], [0.10, 0.20, 0.70]]),
            np.array([[0.10, 0.20, 0.70], [0.10, 0.20, 0.70], [0.10, 0.20, 0.70]]),
        ]
        calibrated_draw_model = Mock()
        calibrated_draw_model.predict_proba.side_effect = [
            np.array([[0.95, 0.05], [0.20, 0.80], [0.95, 0.05]]),
            np.array([[0.50, 0.50], [0.50, 0.50], [0.50, 0.50]]),
        ]

        seen_labels = []

        def selecting_weight(y_validation, multiclass_proba, draw_proba, candidate_weights=None):
            seen_labels.extend(np.asarray(y_validation, dtype=int).tolist())
            return select_draw_overlay_weight(y_validation, multiclass_proba, draw_proba, candidate_weights or [0.0, 0.5, 1.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pipeline.stage3_train.load_feature_data", return_value=pd.concat([model_df, calibration_df, holdout_df])), \
                patch("pipeline.stage3_train.split_train_holdout", return_value=(pd.concat([model_df, calibration_df]), holdout_df)), \
                patch("pipeline.stage3_train.split_model_calibration_data", return_value=(model_df, calibration_df)), \
                patch("pipeline.stage3_train.tune_hyperparameters", return_value={"random_state": 42}), \
                patch("pipeline.stage3_train.train_classifier", return_value=base_model), \
                patch("pipeline.stage3_train.train_draw_classifier", return_value=Mock()), \
                patch("pipeline.stage3_train.calibrate_classifier", side_effect=[calibrated_model, calibrated_draw_model]), \
                patch("pipeline.stage3_train.select_draw_overlay_weight", side_effect=selecting_weight), \
                patch("pipeline.stage3_train.write_feature_importance"), \
                patch("pipeline.stage3_train.write_confusion_matrix"), \
                patch("pipeline.stage3_train.log_mlflow_run", return_value=(None, None)), \
                patch("pipeline.stage3_train.register_production_model", return_value=None):
                run_pipeline(artifacts_dir=Path(tmpdir), trials=0, use_market_features=False)

        self.assertEqual(seen_labels, [0, 1, 2])

    def test_compute_class_sample_weights_upweights_underrepresented_draws(self):
        y = pd.Series([0] * 7 + [1] * 2 + [2] * 5)

        weights = compute_class_sample_weights(y)

        self.assertEqual(len(weights), len(y))
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights > 0).all())
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertGreater(weights[y == 1].mean(), weights[y == 0].mean())
        self.assertGreater(weights[y == 1].mean(), weights[y == 2].mean())

    def test_compute_class_sample_weights_rejects_missing_classes(self):
        with self.assertRaisesRegex(ValueError, "All classes are required"):
            compute_class_sample_weights(pd.Series([0, 0, 1, 1]))

    def test_train_classifier_passes_sample_weight_to_xgboost(self):
        X = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in FEATURE_COLUMNS})
        y = pd.Series([0, 1, 2])
        sample_weight = np.array([0.8, 1.4, 0.9])
        fitted_model = Mock()

        with patch("pipeline.stage3_train.XGBClassifier", return_value=fitted_model) as classifier:
            model = train_classifier(X, y, params={"random_state": 42}, sample_weight=sample_weight)

        classifier.assert_called_once_with(random_state=42)
        fitted_model.fit.assert_called_once()
        _, kwargs = fitted_model.fit.call_args
        self.assertIs(kwargs["sample_weight"], sample_weight)
        self.assertIs(model, fitted_model)

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
        self.assertEqual(metrics["holdout_actual_draw"], 1)
        self.assertEqual(metrics["holdout_predicted_draw"], 1)
        for value in metrics.values():
            self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
