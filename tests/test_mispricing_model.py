import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.model_features import LEAGUE_FEATURE_COLUMNS
from pipeline.mispricing_model import (
    CANDIDATE_FEATURE_COLUMNS,
    build_candidate_bets,
    apply_outcome_threshold_floors,
    predict_candidates,
    rolling_validation_thresholds,
    run_pipeline,
    select_ev_threshold,
    select_ev_threshold_by_outcome,
    select_mispriced_bets,
    select_mispriced_bets_by_outcome,
    split_train_forward,
    split_train_validation_forward,
)


def sample_market_features() -> pd.DataFrame:
    rows = []
    for idx, season in enumerate(["2017-18", "2018-19", "2019-20", "2020-21"], start=1):
        row = {
            "RBallID": idx,
            "Date": pd.Timestamp(f"{2017 + idx}-08-10"),
            "Season": season,
            "League": "ENG",
            "HomeTeam": f"Home {idx}",
            "AwayTeam": f"Away {idx}",
            "Result": ["H", "A", "A", "D"][idx - 1],
            "MarketProb_H": [0.55, 0.45, 0.42, 0.40][idx - 1],
            "MarketProb_A": [0.25, 0.35, 0.36, 0.32][idx - 1],
            "MarketHomeAwayProbDiff": [0.30, 0.10, 0.06, 0.08][idx - 1],
            "MarketFavoriteProb": [0.55, 0.45, 0.42, 0.40][idx - 1],
            "MarketBookmakerMargin": 0.03,
            "MarketBestOdds_H": [1.85, 2.20, 2.40, 2.50][idx - 1],
            "MarketBestOdds_A": [4.10, 3.00, 2.90, 3.20][idx - 1],
            "MarketBestBookmaker_H": "book-h",
            "MarketBestBookmaker_A": "book-a",
            "EloDiff": [120, -40, -20, 10][idx - 1],
            "AbsEloDiff": [120, 40, 20, 10][idx - 1],
            "HomeGoals_Last5": [2.0, 1.1, 1.2, 1.4][idx - 1],
            "AwayGoals_Last5": [1.0, 1.8, 1.7, 1.2][idx - 1],
            "GoalsAgainstLast5Diff": [-0.5, 0.4, 0.2, 0.1][idx - 1],
            "CornerForLast5Diff": [1.0, -0.4, -0.2, 0.2][idx - 1],
            "CornerAgainstLast5Diff": [-0.2, 0.3, 0.2, 0.1][idx - 1],
            "ShotsOnTargetForLast5Diff": [1.1, -0.5, -0.3, 0.4][idx - 1],
            "ShotsOnTargetAgainstLast5Diff": [-0.4, 0.2, 0.1, 0.1][idx - 1],
            "FoulsForLast5Diff": [0.1, -0.2, -0.1, 0.0][idx - 1],
            "OffsidesForLast5Diff": [0.2, -0.1, 0.0, 0.1][idx - 1],
            "HomePoints_Last5": [2.2, 1.0, 1.2, 1.5][idx - 1],
            "AwayPoints_Last5": [1.1, 2.0, 1.7, 1.2][idx - 1],
            "HomeDrawRate_Last5": [0.1, 0.2, 0.25, 0.3][idx - 1],
            "AwayDrawRate_Last5": [0.2, 0.1, 0.15, 0.2][idx - 1],
            "VenuePointsLast5Diff": [0.8, -0.3, -0.2, 0.2][idx - 1],
            "RestDaysDiff": [1.0, -1.0, 0.5, 0.0][idx - 1],
            "CongestionDiff": [-1.0, 1.0, -0.5, 0.0][idx - 1],
        }
        for league_column in LEAGUE_FEATURE_COLUMNS:
            row[league_column] = 1.0 if league_column == "League_ENG" else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


class FakeMispricingModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return np.column_stack([1.0 - self.probabilities[: len(features)], self.probabilities[: len(features)]])


class MispricingModelTests(unittest.TestCase):
    def test_build_candidate_bets_creates_home_and_away_rows_with_profit_target(self):
        candidates = build_candidate_bets(sample_market_features())

        self.assertEqual(len(candidates), 8)
        self.assertTrue(set(CANDIDATE_FEATURE_COLUMNS).issubset(candidates.columns))
        first_home = candidates[(candidates["RBallID"] == 1) & (candidates["Outcome"] == "H")].iloc[0]
        first_away = candidates[(candidates["RBallID"] == 1) & (candidates["Outcome"] == "A")].iloc[0]
        self.assertTrue(first_home["Won"])
        self.assertAlmostEqual(first_home["FlatStakeProfit"], 0.85)
        self.assertFalse(first_away["Won"])
        self.assertAlmostEqual(first_away["FlatStakeProfit"], -1.0)
        self.assertGreater(first_home["SignalEloDiff"], 0)
        self.assertLess(first_away["SignalEloDiff"], 0)

    def test_split_train_forward_uses_pre_2019_rows_for_training(self):
        train, forward = split_train_forward(build_candidate_bets(sample_market_features()))

        self.assertEqual(sorted(train["Season"].unique()), ["2017-18", "2018-19"])
        self.assertEqual(sorted(forward["Season"].unique()), ["2019-20", "2020-21"])


    def test_split_train_validation_forward_reserves_2018_19_for_threshold_selection(self):
        train, validation, forward = split_train_validation_forward(build_candidate_bets(sample_market_features()))

        self.assertEqual(sorted(train["Season"].unique()), ["2017-18"])
        self.assertEqual(sorted(validation["Season"].unique()), ["2018-19"])
        self.assertEqual(sorted(forward["Season"].unique()), ["2019-20", "2020-21"])

    def test_select_ev_threshold_uses_validation_roi_with_minimum_bet_count(self):
        candidates = build_candidate_bets(sample_market_features()).head(4)
        scored = predict_candidates(FakeMispricingModel([0.70, 0.10, 0.45, 0.50]), candidates)

        threshold, rows = select_ev_threshold(scored, threshold_candidates=[-0.10, 0.0, 0.20], min_validation_bets=1)

        self.assertIn(threshold, [-0.10, 0.0, 0.20])
        self.assertEqual(len(rows), 3)
        self.assertTrue(all("roi" in row for row in rows))



    def test_rolling_validation_thresholds_returns_best_thresholds_by_outcome(self):
        candidates = build_candidate_bets(sample_market_features())
        thresholds, diagnostics = rolling_validation_thresholds(
            candidates,
            validation_seasons=("2018-19",),
            threshold_candidates=[0.0, 0.10],
            min_validation_bets_per_outcome=1,
            min_validation_folds=1,
        )

        self.assertTrue(set(thresholds).issubset({"H", "A"}))
        self.assertIn("aggregates", diagnostics)
        self.assertIn("raw", diagnostics)
        self.assertEqual(diagnostics["validation_seasons"], ["2018-19"])


    def test_apply_outcome_threshold_floors_keeps_stronger_cutoff(self):
        thresholds = apply_outcome_threshold_floors({"H": 0.14, "A": 0.23}, {"A": 0.45})

        self.assertEqual(thresholds, {"H": 0.14, "A": 0.45})

    def test_select_ev_threshold_by_outcome_returns_home_and_away_cutoffs(self):
        candidates = build_candidate_bets(sample_market_features())
        scored = predict_candidates(FakeMispricingModel([0.70, 0.10, 0.45, 0.50, 0.55, 0.20, 0.60, 0.15]), candidates)

        thresholds, diagnostics = select_ev_threshold_by_outcome(scored, min_validation_bets=1)
        selected = select_mispriced_bets_by_outcome(scored, thresholds)

        self.assertEqual(set(thresholds), {"H", "A"})
        self.assertEqual(set(diagnostics), {"H", "A"})
        self.assertTrue(set(selected["Outcome"]).issubset({"H", "A"}))
        self.assertIn("PredictedExpectedProfit", selected.columns)

    def test_select_mispriced_bets_uses_predicted_expected_profit(self):
        candidates = build_candidate_bets(sample_market_features()).head(2)
        scored = predict_candidates(FakeMispricingModel([0.70, 0.10]), candidates)
        selected = select_mispriced_bets(scored)

        self.assertEqual(selected[["RBallID", "Outcome"]].values.tolist(), [[1, "H"]])
        self.assertGreater(selected.loc[0, "PredictedExpectedProfit"], 0)
        self.assertTrue(selected.loc[0, "ValueBet"])

    def test_run_pipeline_writes_metrics_and_value_bets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_path = root / "mispricing.parquet"
            json_path = root / "mispricing.json"
            artifacts_dir = root / "artifacts"
            features = sample_market_features()
            candidates = build_candidate_bets(features)
            fake_model = FakeMispricingModel([0.70, 0.10, 0.55, 0.20, 0.60, 0.15])
            with patch("pipeline.mispricing_model.load_feature_data", return_value=features), \
                patch("pipeline.mispricing_model.add_market_features", return_value=(features, type("S", (), {"input_rows": 3, "output_rows": 3, "dropped_rows": 0})())), \
                patch("pipeline.mispricing_model.train_mispricing_classifier", return_value=fake_model):
                summary = run_pipeline(
                    artifacts_dir=artifacts_dir,
                    output_path=output_path,
                    dashboard_json_path=json_path,
                    forward_seasons=("2019-20", "2020-21"),
                )
            metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
            written = pd.read_parquet(output_path)
            output_exists = output_path.exists()
            json_exists = json_path.exists()

        self.assertEqual(summary.train_rows, 4)
        self.assertEqual(summary.test_rows, 4)
        self.assertTrue(output_exists)
        self.assertTrue(json_exists)
        self.assertIn("strategy_overall", metrics)
        self.assertEqual(len(written), summary.selected_bets)


if __name__ == "__main__":
    unittest.main()
