import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from pipeline.attack_defence_xg_benchmark import (
    split_train_holdout,
    STRENGTH_COLUMNS,
    build_attack_defence_odds_frame,
    build_holdout_predictions_frame,
    build_value_bet_roi,
    fit_league_baselines,
    fit_team_strengths,
    predict_lambdas_from_strengths,
    run_pipeline,
)
from pipeline.poisson_goal_model import OUTPUT_COLUMNS, lambdas_to_outcome_probabilities
from pipeline.stage3_train import BASE_FEATURE_COLUMNS


def sample_features() -> pd.DataFrame:
    rows = [
        ("2017-18", "2017-08-12", "Alpha", "Beta", 3, 0, "H", 0),
        ("2017-18", "2017-08-19", "Gamma", "Delta", 1, 1, "D", 1),
        ("2018-19", "2018-08-12", "Beta", "Gamma", 0, 2, "A", 2),
        ("2018-19", "2018-08-19", "Delta", "Alpha", 1, 3, "A", 2),
        ("2019-20", "2019-08-12", "Alpha", "Gamma", 2, 1, "H", 0),
        ("2019-20", "2019-08-19", "Promoted", "Beta", 0, 1, "A", 2),
    ]
    payload = []
    for idx, (season, date, home, away, home_goals, away_goals, result, result_code) in enumerate(rows, start=1):
        row = {
            "RBallID": idx,
            "League": "ENG",
            "Date": date,
            "Season": season,
            "HomeTeam": home,
            "AwayTeam": away,
            "HomeGoals": home_goals,
            "AwayGoals": away_goals,
            "Result": result,
            "ResultCode": result_code,
        }
        for offset, column in enumerate(BASE_FEATURE_COLUMNS):
            row[column] = float(idx + offset)
        payload.append(row)
    return pd.DataFrame(payload)


class AttackDefenceXgBenchmarkTests(unittest.TestCase):
    def test_split_uses_only_pre_holdout_rows_for_strengths(self):
        train, holdout = split_train_holdout(sample_features())
        strengths = fit_team_strengths(train, shrinkage_matches=0.0)

        self.assertEqual(train["Season"].unique().tolist(), ["2017-18", "2018-19"])
        self.assertEqual(holdout["Season"].unique().tolist(), ["2019-20"])
        self.assertNotIn("Promoted", strengths["Team"].tolist())
        self.assertEqual(strengths.columns.tolist(), STRENGTH_COLUMNS)

    def test_shrinkage_moves_low_sample_strengths_toward_default(self):
        train, _ = split_train_holdout(sample_features())
        unshrunk = fit_team_strengths(train, shrinkage_matches=0.0)
        shrunk = fit_team_strengths(train, shrinkage_matches=12.0)
        raw_alpha = float(unshrunk.loc[unshrunk["Team"] == "Alpha", "attack_strength_shrunk"].iloc[0])
        shrunk_alpha = float(shrunk.loc[shrunk["Team"] == "Alpha", "attack_strength_shrunk"].iloc[0])

        self.assertGreater(raw_alpha, 1.0)
        self.assertLess(abs(shrunk_alpha - 1.0), abs(raw_alpha - 1.0))

    def test_unseen_team_uses_default_strengths(self):
        train, holdout = split_train_holdout(sample_features())
        baselines = fit_league_baselines(train)
        strengths = fit_team_strengths(train)

        lambdas = predict_lambdas_from_strengths(holdout, baselines, strengths)
        promoted = lambdas.iloc[1]

        self.assertEqual(promoted["HomeAttackStrength"], 1.0)
        self.assertEqual(promoted["HomeDefenceStrength"], 1.0)
        self.assertTrue((lambdas[["Lambda_Home", "Lambda_Away"]] > 0.0).all().all())
        self.assertTrue(np.isfinite(lambdas.to_numpy(dtype=float)).all())

    def test_probability_and_odds_outputs_are_valid(self):
        holdout = sample_features().query("Season == '2019-20'").reset_index(drop=True)
        lambda_frame = pd.DataFrame(
            {
                "Lambda_Home": [1.6, 0.9],
                "Lambda_Away": [0.8, 1.2],
                "HomeAttackStrength": [1.1, 1.0],
                "AwayAttackStrength": [0.9, 1.2],
                "HomeDefenceStrength": [0.8, 1.0],
                "AwayDefenceStrength": [1.1, 0.9],
            }
        )
        proba = lambdas_to_outcome_probabilities(lambda_frame[["Lambda_Home", "Lambda_Away"]].to_numpy(), max_goals=8)

        odds = build_attack_defence_odds_frame(holdout, lambda_frame, proba)
        predictions = build_holdout_predictions_frame(holdout, lambda_frame, proba)

        self.assertEqual(odds.columns.tolist(), OUTPUT_COLUMNS)
        self.assertTrue(np.allclose(odds[["P_Home", "P_Draw", "P_Away"]].sum(axis=1), 1.0))
        self.assertTrue(np.allclose(odds["ModelOdds_Home"], 1.0 / odds["P_Home"]))
        self.assertTrue((odds[["ModelOdds_Home", "ModelOdds_Draw", "ModelOdds_Away"]] > 0.0).all().all())
        self.assertTrue({"League", "Lambda_Home", "HomeAttackStrength", "AwayDefenceStrength"}.issubset(predictions.columns))

    def test_value_bet_roi_uses_holdout_only_flat_stake_profit(self):
        value_bets = pd.DataFrame(
            [
                {"Season": "2019-20", "Outcome": "H", "Result": "H", "BestBookOdds": 2.50, "Edge": 0.20},
                {"Season": "2019-20", "Outcome": "A", "Result": "H", "BestBookOdds": 3.00, "Edge": 0.15},
                {"Season": "2018-19", "Outcome": "D", "Result": "D", "BestBookOdds": 4.00, "Edge": 0.25},
            ]
        )

        roi = build_value_bet_roi(value_bets)

        self.assertEqual(roi["total_bets"], 2)
        self.assertEqual(roi["wins"], 1)
        self.assertEqual(roi["losses"], 1)
        self.assertAlmostEqual(roi["flat_stake_profit"], 0.5)
        self.assertAlmostEqual(roi["flat_stake_roi"], 0.25)
        self.assertAlmostEqual(roi["average_edge"], 0.175)

    def test_run_pipeline_writes_benchmark_artifacts(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            features_dir = tmp_path / "features"
            artifacts_dir = tmp_path / "artifacts"
            output_path = tmp_path / "output" / "attack_defence_xg_model_odds.parquet"
            benchmark_path = tmp_path / "stage3" / "model_benchmarks.json"
            features_dir.mkdir(parents=True)
            benchmark_path.parent.mkdir(parents=True)
            sample_features().to_parquet(features_dir / "ENG_features.parquet", index=False)
            benchmark_path.write_text(
                json.dumps({"benchmarks": [{"model": "calibrated_xgboost", "log_loss": 0.9}]}),
                encoding="utf-8",
            )

            summary = run_pipeline(
                leagues=["ENG"],
                features_dir=features_dir,
                output_path=output_path,
                artifacts_dir=artifacts_dir,
                existing_benchmarks_path=benchmark_path,
                value_bets_path=tmp_path / "missing_value_bets.parquet",
                max_goals=8,
            )

            odds = pd.read_parquet(output_path)
            predictions = pd.read_parquet(artifacts_dir / "holdout_predictions.parquet")
            strengths = pd.read_parquet(artifacts_dir / "team_strengths.parquet")
            metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
            benchmarks = json.loads((artifacts_dir / "model_benchmarks.json").read_text(encoding="utf-8"))["benchmarks"]

        self.assertEqual(summary.train_rows, 4)
        self.assertEqual(summary.holdout_rows, 2)
        self.assertEqual(odds.columns.tolist(), OUTPUT_COLUMNS)
        self.assertTrue({"Lambda_Home", "Lambda_Away"}.issubset(predictions.columns))
        self.assertEqual(strengths.columns.tolist(), STRENGTH_COLUMNS)
        self.assertIn("league_baselines", metrics)
        self.assertEqual(benchmarks[-1]["model"], "attack_defence_xg")


if __name__ == "__main__":
    unittest.main()
