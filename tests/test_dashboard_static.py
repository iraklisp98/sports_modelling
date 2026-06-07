import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard"


class DashboardStaticTests(unittest.TestCase):
    def test_dashboard_entrypoint_references_local_assets_and_six_tabs(self):
        html = (DASHBOARD / "index.html").read_text(encoding="utf-8")

        self.assertIn('href="css/style.css"', html)
        self.assertIn('src="js/main.js"', html)
        self.assertEqual(len(re.findall(r'class="tab-btn', html)), 6)
        self.assertIn('data-tab="story"', html)
        self.assertIn("<th>Model odds</th>", html)
        self.assertIn("<th>Best book odds</th>", html)
        self.assertIn("<th>Bookmaker</th>", html)
        for tab_id in ["story", "analytics", "backtest", "odds", "calibration", "simulator"]:
            self.assertIn(f'id="{tab_id}"', html)

    def test_main_js_declares_json_contracts_and_renderers(self):
        script = (DASHBOARD / "js" / "main.js").read_text(encoding="utf-8")

        for filename in ["league_analytics.json", "backtest.json", "simulator.json", "strategy_comparison.json", "training_policy.json", "project_summary.json", "diagnostics.json", "value_bets.json"]:
            self.assertIn(f'loadJson("data/{filename}")' if filename != "value_bets.json" else 'loadJson("data/value_bets.json")', script)
        self.assertIn("REQUIRED_VALUE_BET_KEYS", script)
        self.assertIn("const escapeHtml", script)
        for function_name in ["renderProjectStory", "renderAnalytics", "renderBacktest", "renderCalibration", "renderSimulator", "renderStrategyComparison", "renderTrainingPolicy", "calculateSimulation"]:
            self.assertIn(f"function {function_name}", script)
        for key in ["RBallID", "HomeTeam", "AwayTeam", "Edge", "BestBookmaker"]:
            self.assertIn(f'"{key}"', script)

    def test_value_bets_json_matches_dashboard_contract_sample(self):
        records = json.loads((DASHBOARD / "data" / "value_bets.json").read_text(encoding="utf-8"))
        required = {
            "RBallID",
            "HomeTeam",
            "AwayTeam",
            "Date",
            "Season",
            "League",
            "Result",
            "Outcome",
            "ModelOdds",
            "BestBookOdds",
            "Edge",
            "ValueBet",
            "BestBookmaker",
        }

        self.assertGreater(len(records), 0)
        self.assertTrue(required.issubset(records[0]))
        self.assertIsInstance(records[0]["Edge"], (int, float))

    def test_all_dashboard_json_files_match_rendered_tab_contracts(self):
        analytics = json.loads((DASHBOARD / "data" / "league_analytics.json").read_text(encoding="utf-8"))
        backtest = json.loads((DASHBOARD / "data" / "backtest.json").read_text(encoding="utf-8"))
        simulator = json.loads((DASHBOARD / "data" / "simulator.json").read_text(encoding="utf-8"))
        strategy_comparison = json.loads((DASHBOARD / "data" / "strategy_comparison.json").read_text(encoding="utf-8"))
        training_policy = json.loads((DASHBOARD / "data" / "training_policy.json").read_text(encoding="utf-8"))
        project_summary = json.loads((DASHBOARD / "data" / "project_summary.json").read_text(encoding="utf-8"))
        diagnostics = json.loads((DASHBOARD / "data" / "diagnostics.json").read_text(encoding="utf-8"))

        self.assertTrue({"leagues", "league_labels", "seasons", "summary", "monthly_trends", "team_standings", "home_away_split"}.issubset(analytics))
        league = analytics["leagues"][0]
        season = analytics["seasons"][0]
        self.assertTrue({"matches", "avg_goals", "home_win_pct", "draw_pct", "away_win_pct"}.issubset(analytics["summary"][league][season]))
        self.assertTrue({"team", "played", "points", "goals_for", "goals_against", "goal_diff"}.issubset(analytics["team_standings"][league][season][0]))

        self.assertTrue({"metrics", "confusion_matrix", "equity_curve", "mlflow_runs"}.issubset(backtest))
        self.assertEqual(len(backtest["confusion_matrix"]), 3)
        self.assertEqual(backtest["equity_curve"][0]["value_bets_so_far"], 0)
        self.assertTrue({"run_id", "log_loss", "accuracy", "n_estimators"}.issubset(backtest["mlflow_runs"][0]))

        self.assertTrue({"value_bets_by_odds_range", "calibration_by_outcome_bucket"}.issubset(diagnostics))
        self.assertTrue({"model_odds_ranges", "bookmaker_odds_ranges"}.issubset(diagnostics["value_bets_by_odds_range"]))

        self.assertTrue({"default_stake", "bets", "summary"}.issubset(simulator))
        self.assertTrue({"date", "league", "home_team", "away_team", "outcome", "result", "model_odds", "book_odds", "best_bookmaker", "edge", "won"}.issubset(simulator["bets"][0]))
        self.assertTrue({"total_bets", "wins", "losses", "roi_pct", "max_drawdown"}.issubset(simulator["summary"]))
        self.assertTrue({"primary_strategy_id", "strategies", "default_stake"}.issubset(strategy_comparison))
        self.assertEqual(strategy_comparison["primary_strategy_id"], "expanding_walk_forward_xgboost")
        self.assertEqual(strategy_comparison["strategies"][0]["id"], "expanding_walk_forward_xgboost")
        self.assertTrue({"id", "label", "summary", "bets"}.issubset(strategy_comparison["strategies"][0]))
        self.assertTrue({"model_odds", "book_odds", "best_bookmaker"}.issubset(strategy_comparison["strategies"][0]["bets"][0]))
        self.assertTrue({"model", "aggregate", "folds", "available"}.issubset(training_policy))
        self.assertTrue({"status", "final_model", "validation", "model_decisions", "test_status"}.issubset(project_summary))
        self.assertTrue({"expanding_walk_forward", "league_subsets"}.issubset(project_summary["validation"]))


if __name__ == "__main__":
    unittest.main()
