import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from pipeline.export_dashboard_data import (
    build_backtest,
    build_league_analytics,
    build_mlflow_runs,
    load_diagnostics,
    load_league_subset_policy,
    load_training_policy,
    build_project_summary,
    build_simulator,
    build_training_policy_strategy,
    build_strategy_comparison,
    filter_holdout_value_bets,
    run_pipeline,
)


def sample_features(league="ENG") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RBallID": f"{league}-m1",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Date": "2019-08-10",
                "Season": "2019-20",
                "League": league,
                "Result": "H",
                "HomeGoals": 2,
                "AwayGoals": 1,
                "HomeCorners": 5,
                "AwayCorners": 4,
                "HomeShotsOnTarget": 6,
                "AwayShotsOnTarget": 3,
            },
            {
                "RBallID": f"{league}-m2",
                "HomeTeam": "Chelsea",
                "AwayTeam": "Arsenal",
                "Date": "2019-08-17",
                "Season": "2019-20",
                "League": league,
                "Result": "D",
                "HomeGoals": 0,
                "AwayGoals": 0,
                "HomeCorners": 3,
                "AwayCorners": 7,
                "HomeShotsOnTarget": 2,
                "AwayShotsOnTarget": 5,
            },
        ]
    )


def sample_value_bets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RBallID": "ENG-m1",
                "HomeTeam": "Arsenal",
                "AwayTeam": "Chelsea",
                "Date": "2019-08-10",
                "Season": "2019-20",
                "League": "ENG",
                "Result": "H",
                "Outcome": "H",
                "ModelOdds": 1.90,
                "BestBookOdds": 2.20,
                "Edge": (2.20 / 1.90) - 1.0,
                "ValueBet": True,
                "BestBookmaker": "Pinnacle",
            },
            {
                "RBallID": "ENG-m2",
                "HomeTeam": "Chelsea",
                "AwayTeam": "Arsenal",
                "Date": "2019-08-17",
                "Season": "2019-20",
                "League": "ENG",
                "Result": "D",
                "Outcome": "A",
                "ModelOdds": 3.00,
                "BestBookOdds": 3.50,
                "Edge": (3.50 / 3.00) - 1.0,
                "ValueBet": True,
                "BestBookmaker": "Bet365",
            },
        ]
    )


class ExportDashboardDataTests(unittest.TestCase):
    def test_build_league_analytics_matches_stage6_contract(self):
        features = sample_features().assign(Date=lambda df: pd.to_datetime(df["Date"]))

        payload = build_league_analytics(features)

        self.assertEqual(payload["leagues"], ["ENG"])
        self.assertEqual(payload["seasons"], ["2019-20"])
        self.assertEqual(payload["summary"]["ENG"]["2019-20"]["avg_goals"], 1.5)
        self.assertEqual(payload["summary"]["ENG"]["2019-20"]["home_win_pct"], 0.5)
        self.assertEqual(payload["monthly_trends"]["ENG"][0]["month"], "2019-08")
        self.assertEqual(payload["monthly_trends"]["ENG"][0]["avg_corners"], 9.5)
        self.assertIn("avg_shots", payload["monthly_trends"]["ENG"][0])
        self.assertEqual(
            payload["team_standings"]["ENG"]["2019-20"][0],
            {"team": "Arsenal", "played": 2, "points": 4, "goals_for": 2, "goals_against": 1, "goal_diff": 1},
        )

    def test_build_simulator_matches_stage6_contract_and_uses_bookmaker_odds(self):
        payload = build_simulator(sample_value_bets(), stake=10.0)

        self.assertEqual(payload["bets"][0]["return"], 22.0)
        self.assertTrue(payload["bets"][0]["won"])
        self.assertFalse(payload["bets"][1]["won"])
        self.assertEqual(payload["summary"]["total_bets"], 2)
        self.assertEqual(payload["summary"]["wins"], 1)
        self.assertEqual(payload["summary"]["losses"], 1)
        self.assertEqual(payload["summary"]["starting_bankroll"], 20.0)
        self.assertEqual(payload["summary"]["ending_bankroll"], 22.0)
        self.assertEqual(payload["summary"]["total_profit"], 2.0)
        self.assertEqual(payload["summary"]["max_drawdown"], 10.0)




    def test_build_training_policy_strategy_replays_expanding_walk_forward_summary(self):
        training_policy = {
            "available": True,
            "aggregate": {"bets": 3, "wins": 2, "profit": 1.5, "roi": 0.5, "hit_rate": 2 / 3},
            "folds": [
                {
                    "test_seasons": ["2021-22"],
                    "value_bets": {"overall": {"bets": 3, "wins": 2, "profit": 1.5, "roi": 0.5, "hit_rate": 2 / 3}},
                }
            ],
        }

        strategy = build_training_policy_strategy(training_policy, stake=10.0)

        self.assertEqual(strategy["id"], "expanding_walk_forward_xgboost")
        self.assertEqual(strategy["summary"]["total_bets"], 3)
        self.assertEqual(strategy["summary"]["wins"], 2)
        self.assertEqual(strategy["summary"]["total_profit"], 15.0)
        self.assertEqual(strategy["summary"]["roi_pct"], 50.0)
        self.assertTrue(strategy["synthetic"])
        self.assertEqual(len(strategy["bets"]), 3)

    def test_build_strategy_comparison_ranks_available_strategy_simulators(self):
        strategies = {
            "xgboost_value": sample_value_bets(),
            "mispricing_model": sample_value_bets().iloc[[0]].copy(),
        }

        payload = build_strategy_comparison(strategies, primary_strategy_id="mispricing_model", stake=10.0)

        self.assertEqual(payload["primary_strategy_id"], "mispricing_model")
        self.assertEqual({row["id"] for row in payload["strategies"]}, {"xgboost_value", "mispricing_model"})
        self.assertTrue(all("summary" in row and "bets" in row for row in payload["strategies"]))
        self.assertGreaterEqual(payload["strategies"][0]["summary"]["roi_pct"], payload["strategies"][-1]["summary"]["roi_pct"])



    def test_build_training_policy_strategy_uses_real_value_bet_records_when_available(self):
        training_policy = {
            "available": True,
            "aggregate": {"bets": 2, "wins": 1, "profit": 1.2, "roi": 0.6, "hit_rate": 0.5},
            "folds": [
                {
                    "test_seasons": ["2021-22"],
                    "value_bets": {"overall": {"bets": 2, "wins": 1, "profit": 1.2, "roi": 0.6, "hit_rate": 0.5}},
                    "value_bet_records": [
                        {
                            "RBallID": "ENG-1",
                            "HomeTeam": "Arsenal",
                            "AwayTeam": "Chelsea",
                            "Date": "2021-08-13",
                            "Season": "2021-22",
                            "League": "ENG",
                            "Result": "H",
                            "Outcome": "H",
                            "ModelOdds": 1.8,
                            "BestBookOdds": 2.2,
                            "Edge": 0.22,
                            "BestBookmaker": "Bet365",
                        },
                        {
                            "RBallID": "SPA-1",
                            "HomeTeam": "Sevilla",
                            "AwayTeam": "Valencia",
                            "Date": "2021-08-14",
                            "Season": "2021-22",
                            "League": "SPA",
                            "Result": "A",
                            "Outcome": "H",
                            "ModelOdds": 2.0,
                            "BestBookOdds": 2.4,
                            "Edge": 0.2,
                            "BestBookmaker": "Pinnacle",
                        },
                    ],
                }
            ],
        }

        strategy = build_training_policy_strategy(training_policy, stake=10.0)

        self.assertFalse(strategy["synthetic"])
        self.assertEqual(strategy["bets"][0]["date"], "2021-08-13")
        self.assertEqual(strategy["bets"][0]["league"], "ENG")
        self.assertEqual(strategy["bets"][0]["home_team"], "Arsenal")
        self.assertEqual(strategy["bets"][0]["book_odds"], 2.2)
        self.assertEqual(strategy["summary"]["total_profit"], 2.0)

    def test_build_strategy_comparison_uses_training_policy_as_primary_when_available(self):
        strategies = {"xgboost_value": sample_value_bets()}
        training_policy = {
            "available": True,
            "aggregate": {"bets": 3, "wins": 2, "profit": 1.5, "roi": 0.5, "hit_rate": 2 / 3},
            "folds": [
                {
                    "test_seasons": ["2021-22"],
                    "value_bets": {"overall": {"bets": 3, "wins": 2, "profit": 1.5, "roi": 0.5, "hit_rate": 2 / 3}},
                }
            ],
        }

        payload = build_strategy_comparison(strategies, stake=10.0, training_policy=training_policy)

        self.assertEqual(payload["primary_strategy_id"], "expanding_walk_forward_xgboost")
        self.assertEqual(payload["strategies"][0]["id"], "expanding_walk_forward_xgboost")
        self.assertEqual(payload["strategies"][0]["summary"]["total_profit"], 15.0)

    def test_filter_holdout_value_bets_keeps_forward_test_seasons(self):
        value_bets = pd.concat(
            [
                sample_value_bets(),
                sample_value_bets().assign(Season="2020-21", RBallID=lambda df: df["RBallID"] + "-new"),
                sample_value_bets().assign(Season="2018-19", RBallID=lambda df: df["RBallID"] + "-old"),
            ],
            ignore_index=True,
        )

        holdout = filter_holdout_value_bets(value_bets)

        self.assertEqual(set(holdout["Season"]), {"2019-20", "2020-21"})
        self.assertEqual(len(holdout), 4)


    def test_build_backtest_loads_metrics_confusion_matrix_and_equity_curve(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics_path = tmp_path / "metrics.json"
            predictions_path = tmp_path / "holdout_predictions.parquet"
            metrics_path.write_text(
                json.dumps(
                    {
                        "holdout_log_loss": 0.94,
                        "holdout_brier_score": 0.22,
                        "holdout_accuracy": 0.56,
                        "holdout_f1_home": 0.61,
                        "holdout_f1_draw": 0.22,
                        "holdout_f1_away": 0.54,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {"Result": "H", "P_Home": 0.6, "P_Draw": 0.2, "P_Away": 0.2},
                    {"Result": "D", "P_Home": 0.4, "P_Draw": 0.3, "P_Away": 0.3},
                    {"Result": "A", "P_Home": 0.1, "P_Draw": 0.2, "P_Away": 0.7},
                ]
            ).to_parquet(predictions_path, index=False)

            mlruns_dir = tmp_path / "mlruns"
            run_dir = mlruns_dir / "123" / "run-a"
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "params").mkdir()
            (run_dir / "tags").mkdir()
            (run_dir / "meta.yaml").write_text("run_id: run-a\nrun_name: training-run\n", encoding="utf-8")
            (run_dir / "metrics" / "holdout_log_loss").write_text("1 0.91 0\n", encoding="utf-8")
            (run_dir / "metrics" / "holdout_accuracy").write_text("1 0.58 0\n", encoding="utf-8")
            (run_dir / "params" / "n_estimators").write_text("150", encoding="utf-8")

            payload = build_backtest(metrics_path, predictions_path, sample_value_bets(), mlruns_dir)

        self.assertEqual(payload["metrics"]["log_loss"], 0.94)
        self.assertEqual(payload["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [0, 0, 1]])
        self.assertEqual(payload["equity_curve"][0], {"date": None, "cumulative_pnl": 0.0, "value_bets_so_far": 0})
        self.assertEqual(payload["equity_curve"][-1]["cumulative_pnl"], 2.0)
        self.assertEqual(payload["mlflow_runs"][0]["run_id"], "run-a")
        self.assertEqual(payload["mlflow_runs"][0]["n_estimators"], 150)

    def test_build_mlflow_runs_returns_top_five_by_log_loss(self):
        with TemporaryDirectory() as tmp_dir:
            mlruns_dir = Path(tmp_dir) / "mlruns"
            for index, loss in enumerate([0.95, 0.88, 1.05, 0.91, 0.86, 0.99]):
                run_dir = mlruns_dir / "experiment" / f"run-{index}"
                (run_dir / "metrics").mkdir(parents=True)
                (run_dir / "params").mkdir()
                (run_dir / "meta.yaml").write_text(f"run_id: run-{index}\n", encoding="utf-8")
                (run_dir / "metrics" / "holdout_log_loss").write_text(f"1 {loss} 0\n", encoding="utf-8")
                (run_dir / "params" / "learning_rate").write_text("0.05", encoding="utf-8")

            runs = build_mlflow_runs(mlruns_dir)

        self.assertEqual([run["run_id"] for run in runs], ["run-4", "run-1", "run-3", "run-0", "run-5"])
        self.assertEqual(runs[0]["learning_rate"], 0.05)



    def test_build_project_summary_tells_final_model_story(self):
        strategy_comparison = {
            "strategies": [
                {"id": "xgboost_value", "label": "XGBoost value", "summary": {"total_bets": 10, "total_profit": 2.0, "roi_pct": 20.0, "hit_rate": 0.6}},
                {"id": "poisson_goal_model", "label": "Poisson goal model", "summary": {"total_bets": 10, "total_profit": -1.0, "roi_pct": -10.0, "hit_rate": 0.4}},
                {"id": "mispricing_model", "label": "Mispricing model", "summary": {"total_bets": 10, "total_profit": -2.0, "roi_pct": -20.0, "hit_rate": 0.3}},
            ]
        }
        training_policy = {
            "available": True,
            "aggregate": {"roi": 0.0547, "positive_roi_folds": 3, "folds": 5, "bets": 767},
            "folds": [{"test_seasons": ["2021-22"], "value_bets": {"overall": {"roi": 0.1}}}],
        }
        league_subset_policy = {
            "available": True,
            "subsets": [{"name": "all_five", "aggregate": {"roi": 0.0547, "profit": 41.96, "bets": 767, "positive_roi_folds": 3, "folds": 5}}],
        }

        payload = build_project_summary(strategy_comparison, training_policy, league_subset_policy)

        self.assertEqual(payload["status"], "Portfolio complete")
        self.assertEqual(payload["final_model"]["league_policy"], "all_five")
        self.assertEqual(payload["validation"]["expanding_walk_forward"]["roi"], 0.0547)
        self.assertIn("production", payload["final_model"]["production_semantics"].lower())
        self.assertEqual(payload["test_status"]["command"], ".venv/bin/python -m unittest discover tests")

    def test_load_diagnostics_returns_default_contract_when_missing(self):
        payload = load_diagnostics(Path("missing-diagnostics.json"))

        self.assertIn("value_bets_by_odds_range", payload)
        self.assertEqual(payload["value_bets_by_odds_range"], {"model_odds_ranges": [], "bookmaker_odds_ranges": []})

    def test_load_training_policy_marks_existing_artifact_available(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "policy.json"
            path.write_text(json.dumps({"model": "x", "aggregate": {"roi": 0.1}, "folds": []}), encoding="utf-8")

            payload = load_training_policy(path)

        self.assertTrue(payload["available"])
        self.assertEqual(payload["aggregate"]["roi"], 0.1)


    def test_load_league_subset_policy_marks_existing_artifact_available(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "league_policy.json"
            path.write_text(json.dumps({"model": "x", "subsets": [{"name": "all_five"}]}), encoding="utf-8")

            payload = load_league_subset_policy(path)

        self.assertTrue(payload["available"])
        self.assertEqual(payload["subsets"][0]["name"], "all_five")

    def test_run_pipeline_writes_all_dashboard_json_files(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            features_dir = tmp_path / "features"
            output_dir = tmp_path / "dashboard" / "data"
            features_dir.mkdir(parents=True)
            for league in ["ENG", "SPA", "FRA", "GER", "ITA"]:
                sample_features(league=league).to_parquet(features_dir / f"{league}_features.parquet", index=False)

            value_bets_path = tmp_path / "value_bets.parquet"
            sample_value_bets().to_parquet(value_bets_path, index=False)

            summary = run_pipeline(
                features_dir=features_dir,
                model_odds_path=tmp_path / "missing_model_odds.parquet",
                value_bets_path=value_bets_path,
                metrics_path=tmp_path / "missing_metrics.json",
                holdout_predictions_path=tmp_path / "missing_predictions.parquet",
                output_dir=output_dir,
                mlruns_dir=tmp_path / "missing_mlruns",
                training_policy_path=tmp_path / "missing_policy.json",
                league_subset_policy_path=tmp_path / "missing_league_policy.json",
                poisson_value_bets_path=tmp_path / "missing_poisson.parquet",
                market_aware_value_bets_path=tmp_path / "missing_market_aware.parquet",
                mispricing_value_bets_path=tmp_path / "missing_mispricing.parquet",
            )

            payloads = {}
            for filename in ["league_analytics.json", "backtest.json", "value_bets.json", "simulator.json", "strategy_comparison.json", "training_policy.json", "project_summary.json", "diagnostics.json"]:
                path = output_dir / filename
                self.assertTrue(path.exists(), filename)
                payloads[filename] = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(summary.files), 8)
        self.assertEqual(payloads["league_analytics.json"]["leagues"], ["ENG", "FRA", "GER", "ITA", "SPA"])
        self.assertEqual(payloads["value_bets.json"][0]["Date"], "2019-08-10")
        self.assertEqual(payloads["simulator.json"]["summary"]["roi_pct"], 10.0)
        self.assertIn("strategies", payloads["strategy_comparison.json"])
        self.assertEqual(payloads["strategy_comparison.json"]["primary_strategy_id"], "xgboost_value")
        self.assertFalse(payloads["training_policy.json"]["available"])
        self.assertEqual(payloads["project_summary.json"]["status"], "Portfolio complete")
        self.assertIn("value_bets_by_odds_range", payloads["diagnostics.json"])


if __name__ == "__main__":
    unittest.main()
