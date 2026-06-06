# Stage 2 — Feature Engineering

**Status:** Complete  
**Script:** `pipeline/stage2_features.py`  
**Input:** Parquet files from Stage 1 (`data/processed/ENG.parquet`, etc.)  
**Output:** Feature-enriched Parquet file per league in `data/features/`

---

## What

Take the clean match-level dataset and add the features the model needs: ELO ratings at kick-off, rolling form statistics, season win rates, and the match result label.

---

## Why This Approach

### Why separate feature engineering from ingestion?
Single responsibility. Stage 1 owns "make the raw data clean and structured." Stage 2 owns "add derived knowledge." If you want to add a new feature later, you edit one file without touching the cleaning logic. If the cleaning logic changes, features recompute correctly.

This is called a **modular pipeline** — each stage has a clear input contract and output contract. It's the pattern used in every production ML system.

### Why ELO?
ELO is a relative strength rating. It captures something rolling form can't: how strong a team is relative to everyone else in the league. Two teams with identical recent form but different ELO ratings have very different win probabilities. The model needs both signals.

### Why rolling 5-match windows?
5 matches is roughly one month of football — enough to capture recent form without including results that are no longer relevant. Too short (1–2 matches) is noisy. Too long (full season) loses the "current form" signal.

### Why compute features at match time, not after?
This is the most important constraint: **you cannot use information from the future when making a prediction.** Rolling stats, win rates, and ELO ratings must all be computed from matches that happened *before* the current match. If you accidentally include the current match in a rolling window, you have data leakage — your model will look great in training but fail completely in production.

Always sort by date before computing rolling features.

---

## New Concepts to Learn Before Building

### Data leakage
The single most common mistake in ML feature engineering. Leakage means your model has access to information it wouldn't have at prediction time. Example: if you compute `HomeGoals_Last5` including today's match, the model "cheated" — it saw the result before making the prediction.

**Rule:** Always compute rolling and cumulative features with `.shift(1)` or equivalent, so the current row is excluded from its own window.

### ELO rating system
ELO is a zero-sum rating. When a stronger team beats a weaker team, the stronger team gains a small number of points; when a weaker team wins, the weaker team gains a large number. The formula:

```
Expected_A = 1 / (1 + 10^((Rating_B - Rating_A) / 400))
K = 30  # how much ratings change per match

# After the match:
Rating_A_new = Rating_A + K * (Actual_A - Expected_A)
```

Where `Actual_A` is 1 for a win, 0.5 for a draw, 0 for a loss.

Home advantage: we add 80 points to the home team's rating when computing expected probabilities, but **not** when storing ratings.

---

## How to Build It (Step by Step)

### Step 1 — Create the script file
Create `pipeline/stage2_features.py`.

### Step 2 — Load the Stage 1 output
```python
import pandas as pd  # pandas is fine here; the dataset is now small and flat

df = pd.read_parquet("data/processed/ENG.parquet")
df = df.sort_values("Date").reset_index(drop=True)
```

Note: we switch to pandas for feature engineering because the operations are sequential and stateful (ELO updates depend on previous matches). PySpark's distributed model makes stateful iteration complex. pandas is the right tool here.

### Step 3 — Add the match result column
```python
def get_result(row):
    if row["HomeGoals"] > row["AwayGoals"]:
        return "H"
    elif row["HomeGoals"] < row["AwayGoals"]:
        return "A"
    else:
        return "D"

df["Result"] = df.apply(get_result, axis=1)
```

### Step 4 — Implement ELO ratings

Build the ELO system as a function that iterates through matches in date order and updates ratings after each match.

```python
def compute_elo(df, k=30, home_advantage=80, starting_elo=1500, regression_factor=0.2):
    ratings = {}      # team -> current ELO
    league_mean = starting_elo

    elo_home, elo_away = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        season = row["Season"]

        # Initialise new teams
        if home not in ratings:
            ratings[home] = league_mean - 100 if season != df["Season"].iloc[0] else starting_elo
        if away not in ratings:
            ratings[away] = league_mean - 100 if season != df["Season"].iloc[0] else starting_elo

        # Season regression: at the start of each new season, pull ratings toward mean
        # (handled separately — see Step 4b)

        r_home = ratings[home]
        r_away = ratings[away]

        elo_home.append(r_home)
        elo_away.append(r_away)

        # Compute expected scores (with home advantage)
        exp_home = 1 / (1 + 10 ** ((r_away - (r_home + home_advantage)) / 400))
        exp_away = 1 - exp_home

        # Actual scores
        if row["Result"] == "H":
            act_home, act_away = 1.0, 0.0
        elif row["Result"] == "A":
            act_home, act_away = 0.0, 1.0
        else:
            act_home, act_away = 0.5, 0.5

        # Update ratings
        ratings[home] = r_home + k * (act_home - exp_home)
        ratings[away] = r_away + k * (act_away - exp_away)

    df["HomeElo"] = elo_home
    df["AwayElo"] = elo_away
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    return df
```

**Why do we append before updating?** The ELO values stored in the dataframe must be the ratings *before* the match — not after. The model uses pre-match ratings to make predictions.

### Step 4b — Season regression
At the start of each season, pull all team ratings 20% toward the league mean. This reflects that promoted teams get stronger and champions often regress:

```python
def apply_season_regression(ratings, league_mean, factor=0.2):
    return {team: r + factor * (league_mean - r) for team, r in ratings.items()}
```

Call this once per season transition inside the main ELO loop.

### Step 5 — Rolling form features
For each team, compute the last 5 match values for goals, corners, and points from a unified team history. This matters because a team's recent form includes both home and away matches; splitting by `HomeTeam` and `AwayTeam` separately would throw away half of the history.

Implementation shape:

```python
# Build one row per team per match, then group by Team.
history = pd.concat([home_history, away_history], ignore_index=True)
history = history.sort_values(["Team", "Date", "RBallID"])

history["Goals_Last5"] = history.groupby("Team")["Goals"].transform(
    lambda values: values.shift(1).rolling(window=5, min_periods=1).mean()
)
```

The important detail is `shift(1)`: the current match is excluded before the rolling average is calculated. The first match in a team's history gets `0.0` after missing values are filled.

### Step 6 — Season win rate
For each team, compute win rate from the start of the current season up to, but not including, the current match. This uses the same unified team history as the rolling features, so a home team's season win rate includes wins it earned away from home earlier in the season.

Implementation shape:

```python
history["Win"] = (history["Points"] == 3.0).astype(float)
grouped = history.groupby(["Team", "Season"], sort=False)
previous_wins = grouped["Win"].cumsum() - history["Win"]
previous_matches = grouped.cumcount()
history["WinRate_Season"] = (previous_wins / previous_matches.replace(0, np.nan)).fillna(0.0)
```

Again, the feature is pre-match: current-match wins are subtracted before calculating the rate.

### Step 7 — Encode the target
XGBoost needs a numeric label:
```python
df["ResultCode"] = df["Result"].map({"H": 0, "D": 1, "A": 2})
```

### Step 8 — Write features output
```python
df.to_parquet("data/features/ENG_features.parquet", index=False)
```

---

## Acceptance Criteria

- [x] Script runs without errors: `python pipeline/stage2_features.py`
- [x] Output has columns: `HomeElo`, `AwayElo`, `EloDiff`, `HomeGoals_Last5`, `AwayGoals_Last5`, `HomeCorners_Last5`, `AwayCorners_Last5`, `HomePoints_Last5`, `AwayPoints_Last5`, `HomeWinRate_Season`, `AwayWinRate_Season`, `Result`, `ResultCode`
- [x] No data leakage: rolling features use `.shift(1)` — the first match of a team's history has `NaN` or `0` for rolling stats, not the result of that match
- [x] ELO values at row `i` reflect ratings before match `i`, not after
- [x] Five output files created: `data/features/ENG_features.parquet`, `FRA_features.parquet`, `GER_features.parquet`, `ITA_features.parquet`, `SPA_features.parquet`

---

## Interview Q&A

**Q: What is data leakage and how did you prevent it here?**  
A: "Data leakage is when a model has access at training time to information it wouldn't have at prediction time. In feature engineering it usually happens when you include the current observation in its own rolling window. I prevented it by always applying `.shift(1)` before computing rolling statistics — the current match is excluded from its own feature values."

**Q: Why did you use ELO instead of just raw win rates?**  
A: "Win rates don't account for strength of schedule. A team that beat 5 bottom-half sides looks identical to a team that drew with 5 top-half sides. ELO captures relative strength — a win against a high-rated opponent moves your rating much more than a win against a weak one. That makes it a much richer signal for predicting future performance."

**Q: Why switch from PySpark back to pandas for this stage?**  
A: "ELO computation is inherently sequential — each match's rating update depends on the previous one. Distributed processing works well for operations you can parallelize, but sequential state updates are easier and more correct in pandas. Using the right tool per stage is better engineering than using one tool everywhere."
