# Stage 5 — Odds Comparison & Value Bet Flagging

**Status:** Not started  
**Script:** `pipeline/stage5_compare.py`  
**Input:** `data/output/model_odds.parquet` from Stage 4  
**Output:** `data/output/value_bets.parquet` + `dashboard/data/value_bets.json`

---

## What

Fetch bookmaker odds from The Odds API for every match in the dataset. Match them to model odds by team name and date. For each match and each outcome, check whether the model's implied odds exceed the best bookmaker odds by at least 10%. Flag the ones that pass as value bets.

---

## Why This Approach

### Why The Odds API?
It's the cleanest, most reliable free-tier odds data source available. It aggregates multiple bookmakers in one call and returns decimal odds. The free tier allows enough historical requests for this dataset size.

### Why take the best bookmaker odds (maximum across bookmakers)?
You should always compare your model against the best available market price, not an average. If Bet365 offers 2.10 and Pinnacle offers 2.20, your model odds of 2.50 beat Pinnacle's 2.20 — that's the bet you'd place. Comparing against an average would understate the real edge.

### Why 10% and not 5% or 20%?
The model has uncertainty — log loss of ~0.94 means the probabilities aren't perfectly calibrated. A 5% edge could easily be noise. A 10% buffer absorbs calibration error while still being a meaningful signal. 20% would be too conservative — you'd almost never flag a bet.

### Why fuzzy match team names?
Team names differ between data sources. "Man United" in one source might be "Manchester United" or "Manchester Utd" in another. Fuzzy matching (via `rapidfuzz`) handles these variations automatically, falling back to a manual name mapping config for persistent mismatches.

---

## New Concepts to Learn Before Building

### The Odds API
Sign up for a free key at the-odds-api.com. Key endpoints:

```
GET /v4/sports
    Lists available sports and their keys

GET /v4/sports/{sport_key}/odds
    Returns odds for upcoming games
    ?apiKey=YOUR_KEY&regions=eu&markets=h2h&oddsFormat=decimal

GET /v4/historical/sports/{sport_key}/odds
    Returns odds as they were at a specific historical date
    ?apiKey=YOUR_KEY&date=2020-01-01T12:00:00Z&regions=eu&markets=h2h
```

For this project you'll use the **historical** endpoint to fetch the odds as they were before each match.

Sport keys for football:
- `soccer_england_league1` — Premier League (note: API naming)
- `soccer_france_ligue_1`
- `soccer_spain_la_liga`

### Rate limiting
The free tier has a request quota. Cache every API response locally so you only fetch each match once:

```python
import json, os, hashlib

def cache_key(url, params):
    return hashlib.md5((url + str(sorted(params.items()))).encode()).hexdigest()

def fetch_with_cache(url, params, cache_dir="data/odds_cache"):
    key = cache_key(url, params)
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return data
```

### Fuzzy matching
```python
from rapidfuzz import process, fuzz

def match_team_name(name, candidates, threshold=80):
    result = process.extractOne(name, candidates, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return result[0]
    return None
```

---

## How to Build It (Step by Step)

### Step 1 — Create the script and config files
- Create `pipeline/stage5_compare.py`
- Create `config/settings.yaml` with:

```yaml
odds_api:
  key: "YOUR_API_KEY_HERE"
  base_url: "https://api.the-odds-api.com/v4"
  bookmakers:
    - bet365
    - pinnaclesports
    - williamhill
  regions: "eu"
  markets: "h2h"
  odds_format: "decimal"

arbitrage:
  edge_threshold: 0.10   # 10%

leagues:
  ENG:
    sport_key: "soccer_england_premier_league"
  FRA:
    sport_key: "soccer_france_ligue_1"
  SPA:
    sport_key: "soccer_spain_la_liga"
```

### Step 2 — Load model odds
```python
import pandas as pd
import yaml

with open("config/settings.yaml") as f:
    config = yaml.safe_load(f)

df = pd.read_parquet("data/output/model_odds.parquet")
```

### Step 3 — Fetch bookmaker odds per match
For each match, call the historical odds endpoint. Group matches by date to minimise API calls (one call per date per league returns all matches that day).

```python
import requests
from datetime import datetime, timedelta

def fetch_odds_for_date(sport_key, date, config):
    date_str = date.strftime("%Y-%m-%dT12:00:00Z")
    url = f"{config['odds_api']['base_url']}/historical/sports/{sport_key}/odds"
    params = {
        "apiKey": config["odds_api"]["key"],
        "date": date_str,
        "regions": config["odds_api"]["regions"],
        "markets": config["odds_api"]["markets"],
        "oddsFormat": config["odds_api"]["odds_format"],
        "bookmakers": ",".join(config["odds_api"]["bookmakers"]),
    }
    return fetch_with_cache(url, params)
```

### Step 4 — Parse API response into a usable structure
The API returns a list of games. Each game has a list of bookmakers, each with a list of outcomes (Home / Draw / Away) and their odds.

```python
def parse_odds_response(data):
    records = []
    for game in data.get("data", []):
        home_team = game["home_team"]
        away_team = game["away_team"]
        commence = game["commence_time"][:10]  # date only

        for bookmaker in game.get("bookmakers", []):
            bk_name = bookmaker["key"]
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    records.append({
                        "HomeTeam_API": home_team,
                        "AwayTeam_API": away_team,
                        "Date_API": commence,
                        "Bookmaker": bk_name,
                        "Odds_Home": outcomes.get(home_team),
                        "Odds_Draw": outcomes.get("Draw"),
                        "Odds_Away": outcomes.get(away_team),
                    })
    return pd.DataFrame(records)
```

### Step 5 — Match API records to model odds rows
Use fuzzy matching on team names + exact date match:

```python
from rapidfuzz import process, fuzz

def find_best_bookmaker_odds(model_row, api_df, threshold=80):
    date_str = str(model_row["Date"])[:10]
    same_date = api_df[api_df["Date_API"] == date_str]

    if same_date.empty:
        return None

    api_home_names = same_date["HomeTeam_API"].unique().tolist()
    match = process.extractOne(model_row["HomeTeam"], api_home_names,
                               scorer=fuzz.token_sort_ratio)
    if not match or match[1] < threshold:
        return None

    matched_home = match[0]
    candidates = same_date[same_date["HomeTeam_API"] == matched_home]

    return {
        "BestOdds_Home": candidates["Odds_Home"].max(),
        "BestOdds_Draw": candidates["Odds_Draw"].max(),
        "BestOdds_Away": candidates["Odds_Away"].max(),
        "BestBookmaker": candidates.loc[candidates["Odds_Home"].idxmax(), "Bookmaker"],
    }
```

### Step 6 — Apply the arbitrage condition
```python
threshold = config["arbitrage"]["edge_threshold"]

def compute_value_bets(row):
    results = []
    for outcome, model_col, book_col in [
        ("H", "ModelOdds_Home", "BestOdds_Home"),
        ("D", "ModelOdds_Draw", "BestOdds_Draw"),
        ("A", "ModelOdds_Away", "BestOdds_Away"),
    ]:
        if pd.isna(row.get(book_col)):
            continue
        edge = (row[model_col] / row[book_col]) - 1
        if edge >= threshold:
            results.append({
                **row.to_dict(),
                "Outcome": outcome,
                "ModelOdds": row[model_col],
                "BestBookOdds": row[book_col],
                "Edge": edge,
                "ValueBet": True,
            })
    return results
```

### Step 7 — Write outputs
```python
import json, os

value_bets = []
for _, row in df.iterrows():
    value_bets.extend(compute_value_bets(row))

vb_df = pd.DataFrame(value_bets)

os.makedirs("data/output", exist_ok=True)
vb_df.to_parquet("data/output/value_bets.parquet", index=False)

os.makedirs("dashboard/data", exist_ok=True)
vb_df.to_json("dashboard/data/value_bets.json", orient="records", date_format="iso")
print(f"Found {len(vb_df)} value bets across {len(df)} matches.")
```

---

## Acceptance Criteria

- [ ] Script runs without errors: `python pipeline/stage5_compare.py`
- [ ] API responses cached locally in `data/odds_cache/` — script does not re-fetch on re-run
- [ ] `data/output/value_bets.parquet` created with columns: `HomeTeam`, `AwayTeam`, `Date`, `Season`, `Result`, `Outcome`, `ModelOdds`, `BestBookOdds`, `Edge`, `ValueBet`, `BestBookmaker`
- [ ] `dashboard/data/value_bets.json` created and valid JSON
- [ ] All `Edge` values are >= 0.10 (the threshold)
- [ ] Print statement reports a sensible number of value bets (expect 5–15% of all matches to be flagged)

---

## Interview Q&A

**Q: Walk me through your odds arbitrage logic.**  
A: "The model outputs win probabilities for each outcome. I convert those to implied decimal odds using `1/p`. I then fetch the best available bookmaker odds for the same match from The Odds API, taking the maximum across bookmakers to get the best market price. If my model odds are at least 10% higher than the best bookmaker price, that's a value bet — the market is underpricing the outcome relative to what my model believes."

**Q: How do you handle team name mismatches between your dataset and the API?**  
A: "I use fuzzy string matching with the `rapidfuzz` library. It computes a similarity score between the team names and returns the best match above a threshold. For names that consistently fail the threshold — like abbreviations or local name variants — I maintain a manual mapping in the config file."

**Q: How do you handle API rate limits?**  
A: "I cache every API response to disk using an MD5 hash of the URL and parameters as the filename. On subsequent runs, the script checks the cache first and only makes a network request if there's no cached file. For a dataset of 3 seasons this means you only ever hit the API once per match date."
