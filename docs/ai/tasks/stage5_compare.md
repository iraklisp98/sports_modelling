# Stage 5 — Odds Comparison & Value Bet Flagging

**Status:** Complete  
**Script:** `pipeline/stage5_compare.py`  
**Input:** `data/output/model_odds.parquet` from Stage 4 + Football-Data.co.uk historical odds CSVs  
**Output:** `data/output/value_bets.parquet` + `dashboard/data/value_bets.json`

---

## What

Compare model-implied fair odds against historical bookmaker closing odds from Football-Data.co.uk. For each matched fixture, Stage 5 evaluates home-win and away-win outcomes only, takes the best available bookmaker price, computes the edge, and writes only bets that clear the 10% value threshold plus the odds sanity policy. Draw probabilities remain part of the model output, but draws are not surfaced as actionable value bets.

---

## Why This Approach

### Why Football-Data.co.uk instead of The Odds API?
The project is backtesting historical matches with `2019-20` kept as the first holdout season, followed by 2020-21 through 2025-26 for forward testing. A live odds API is useful for future fixtures, but it is the wrong source for historical validation unless the historical market is available for the exact period. Football-Data publishes season-level CSVs with match results, team names, dates, and historical bookmaker 1X2 odds, so the backtest can be reproduced without an API key.

### Why take the best bookmaker odds?
A bettor chooses the best available market price, not the average price. If Bet365 offers 2.10 and Pinnacle offers 2.25, the realistic comparison is against 2.25.

### Why is the edge formula `BestBookOdds / ModelOdds - 1`?
Stage 4 converts probability into fair decimal odds with `1 / p`. Lower model odds mean the model thinks the outcome is more likely. A bet has value when the bookmaker offers a higher price than the model's fair price:

```python
value_bet = best_bookmaker_odds >= 1.10 * model_odds
edge = (best_bookmaker_odds / model_odds) - 1
```

This fixes the common mistake of treating higher model odds as better. With decimal odds, higher model odds actually mean lower model probability.

### Why add odds sanity filters?
A 10% edge is necessary, but it is not enough. Very large bookmaker prices, tiny model probabilities, or extreme calculated edges are usually where model noise and historical data quirks show up. Stage 5 now uses a small risk policy before surfacing a value bet:

```text
ModelProbability >= 35%
1.20 <= BestBookOdds <= 8.00
10% <= Edge <= 30%
```

These bounds keep the dashboard focused on bets the model has enough confidence in and remove long-shot outliers that make the backtest look better than it really is. The values are exposed as CLI arguments so the policy is testable and tunable without changing code.

---

## Football-Data CSV Inputs

Stage 5 downloads missing files into `data/bookmaker_odds/football_data/` using this URL pattern:

```text
https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv
```

| League | Football-Data code | Seasons |
|---|---|---|
| Premier League | `E0` | `1011` through `2526` |
| La Liga | `SP1` | `1011` through `2526` |
| Ligue 1 | `F1` | `1011` through `2526` |
| Bundesliga | `D1` | `1011` through `2526` |
| Serie A | `I1` | `1011` through `2526` |

Supported bookmaker prefixes currently include `B365`, `PS`, `WH`, `BW`, `IW`, `LB`, `SB`, `SJ`, and `VC` where the matching `H`, `D`, and `A` columns exist.

---

## Build Steps

1. Load Stage 4 model odds from `data/output/model_odds.parquet`.
2. Ensure Football-Data CSVs exist locally; download the expected season/league files if missing.
3. Normalise Football-Data columns into date, home team, away team, season, league, and bookmaker outcome columns.
4. Match model rows to bookmaker rows by normalised `Date`, `HomeTeam`, and `AwayTeam`.
5. For home and away outcomes only, select the highest bookmaker odds and the bookmaker that offered it.
6. Compute `Edge = (BestBookOdds / ModelOdds) - 1`.
7. Keep rows where `Edge >= 0.10` and the odds sanity policy passes.
8. Write Parquet + dashboard JSON.

---

## Acceptance Criteria

- [x] Script exists: `pipeline/stage5_compare.py`
- [x] Football-Data URLs and local CSV cache path are defined
- [x] `data/output/value_bets.parquet` is written by `run_pipeline`
- [x] `dashboard/data/value_bets.json` is written by `run_pipeline`
- [x] Output columns are: `RBallID`, `HomeTeam`, `AwayTeam`, `Date`, `Season`, `League`, `Result`, `Outcome`, `ModelOdds`, `BestBookOdds`, `Edge`, `ValueBet`, `BestBookmaker`
- [x] Only home-win and away-win outcomes are eligible for value-bet output
- [x] All `Edge` values in the output are at least the configured threshold
- [x] Value bets must pass the configured probability, bookmaker-odds, and max-edge sanity policy
- [x] Unit tests cover best bookmaker selection, edge direction, draw exclusion, odds sanity filtering, CSV normalisation, and output writing

Narrow test command:

```bash
.venv/bin/python -m unittest tests.test_stage5_compare
```

Broader Stage 4/5 contract check:

```bash
.venv/bin/python -m unittest tests.test_stage4_odds_gen tests.test_stage5_compare
```

---

## Interview Q&A

**Q: Why did you use Football-Data.co.uk here instead of The Odds API?**  
A: "This is a historical backtest. I need bookmaker odds from the same seasons as the matches, not current or upcoming odds. Football-Data provides reproducible historical season CSVs, so anyone can rerun the comparison without an API key."

**Q: Walk me through the odds comparison logic.**  
A: "The model outputs probabilities, then Stage 4 converts them to fair decimal odds using `1 / probability`. Stage 5 compares those fair odds to the best historical bookmaker price. A bet is flagged only if the bookmaker price is at least 10% higher than my fair price and it passes sanity checks for model probability, bookmaker odds, and maximum edge."

**Q: Why is the edge formula `bookmaker_odds / model_odds - 1`?**  
A: "Because decimal model odds are fair odds. If my model says fair odds are 2.00 and the market offers 2.25, the market is paying 12.5% more than fair value. The opposite formula would incorrectly reward outcomes my model thinks are less likely."

**Q: How do you handle team name mismatches?**  
A: "The current implementation normalises whitespace and case, then matches on date, home team, and away team. I kept this deterministic first; if real CSV mismatches remain, the next production step would be an explicit mapping table rather than uncontrolled fuzzy matching."
