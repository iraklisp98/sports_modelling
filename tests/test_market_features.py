import unittest

import pandas as pd

from pipeline.market_features import MARKET_FEATURE_COLUMNS, add_market_features_from_bookmaker_odds


def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RBallID": 1,
                "Date": "2019-08-10",
                "Season": "2019-20",
                "League": "ENG",
                "HomeTeam": "Arsenal FC",
                "AwayTeam": "Chelsea FC",
                "Result": "H",
            },
            {
                "RBallID": 2,
                "Date": "2019-08-17",
                "Season": "2019-20",
                "League": "ENG",
                "HomeTeam": "Chelsea",
                "AwayTeam": "Arsenal",
                "Result": "A",
            },
        ]
    )


def sample_bookmaker_odds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Date": "2019-08-10", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "B365_H": 2.0, "B365_D": 3.2, "B365_A": 3.8, "PS_H": 2.1, "PS_D": 3.1, "PS_A": 3.7},
            {"Date": "2019-08-17", "HomeTeam": "Chelsea", "AwayTeam": "Arsenal", "B365_H": 1.8, "B365_D": 3.6, "B365_A": 5.0},
        ]
    )


class MarketFeatureTests(unittest.TestCase):
    def test_add_market_features_normalizes_best_bookmaker_probabilities(self):
        featured, summary = add_market_features_from_bookmaker_odds(sample_features(), sample_bookmaker_odds())

        self.assertEqual(summary.input_rows, 2)
        self.assertEqual(summary.output_rows, 2)
        self.assertEqual(summary.dropped_rows, 0)
        self.assertTrue(set(MARKET_FEATURE_COLUMNS).issubset(featured.columns))
        self.assertTrue(all(abs(featured[["MarketProb_H", "MarketProb_D", "MarketProb_A"]].sum(axis=1) - 1.0) < 1e-9))
        self.assertAlmostEqual(featured.loc[0, "MarketBestOdds_H"], 2.1)
        self.assertGreater(featured.loc[0, "MarketBookmakerMargin"], 0.0)
        self.assertAlmostEqual(featured.loc[0, "MarketHomeAwayProbDiff"], featured.loc[0, "MarketProb_H"] - featured.loc[0, "MarketProb_A"])

    def test_add_market_features_drops_unmatched_rows(self):
        features = sample_features().assign(AwayTeam=["Chelsea", "Unknown"])

        featured, summary = add_market_features_from_bookmaker_odds(features, sample_bookmaker_odds())

        self.assertEqual(len(featured), 1)
        self.assertEqual(summary.dropped_rows, 1)


if __name__ == "__main__":
    unittest.main()
