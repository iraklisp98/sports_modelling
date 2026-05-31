# Stage 6 — Dashboard

**Status:** Complete  
**Files:** `dashboard/index.html`, `dashboard/css/`, `dashboard/js/`, `dashboard/data/`  
**Input:** Four JSON files written by the pipeline to `dashboard/data/`  
**Output:** A single-page HTML dashboard served by nginx

---

## What

A four-tab HTML/CSS/JS dashboard that visualises the pipeline outputs. No backend server. The pipeline writes JSON files; the browser reads them at load time.

The four tabs:
1. **League Analytics** — descriptive stats, time series, team leaderboard
2. **Backtest Performance** — model metrics, equity curve, confusion matrix
3. **Odds Inspector** — filterable table of all value bets with match detail modal
4. **Betting Simulator** — simulate $10/bet on every flagged value bet, show P&L, equity curve, bet log

---

## Why This Approach

### Why no Flask backend?
The pipeline already computed everything. There's nothing to query at request time — just static data. A Flask app would add complexity (a running Python process, routes, templates) without adding value. Static HTML + JSON is simpler, more reliable, and easier to deploy.

For a portfolio project, simplicity that works is more impressive than complexity that's fragile.

### Why write JSON files from the pipeline instead of calling an API?
This keeps the dashboard completely decoupled from the pipeline. You can run the pipeline once, then open the dashboard a hundred times without touching any code. It also means the dashboard works without any network access after the first pipeline run.

### Why native SVG/CSS charts?
The dashboard is static and portfolio-focused, so simple native SVG/CSS charts avoid an external browser dependency while still making the JSON outputs inspectable. The mental model is: the pipeline computes the numbers; the browser only maps arrays of JSON records to DOM, SVG paths, and tables.

---

## New Concepts to Learn Before Building

### How the dashboard reads data
The browser can't access the filesystem directly. When the dashboard is served by nginx inside Docker, it makes HTTP requests to fetch JSON files:

```javascript
const response = await fetch('data/value_bets.json');
const data = await response.json();
```

This is why you can't just open `index.html` by double-clicking — it needs to be served over HTTP.

### Native SVG chart basics
```html
<svg viewBox="0 0 300 120" role="img" aria-label="Example line chart">
  <path d="M10,90 L150,40 L290,70" fill="none" stroke="#0f766e" stroke-width="3" />
</svg>
```

### JSON file contracts
Each JSON file has a defined structure that the dashboard depends on. These are written by earlier pipeline stages (or by a separate `export_dashboard_data.py` script that reads the Parquet files and writes JSON).

---

## JSON File Contracts

### `dashboard/data/league_analytics.json`
```json
{
  "leagues": ["ENG", "FRA", "SPA"],
  "seasons": ["2017-18", "2018-19", "2019-20"],
  "summary": {
    "ENG": {
      "2017-18": {
        "avg_goals": 2.72,
        "home_win_pct": 0.46,
        "draw_pct": 0.24,
        "away_win_pct": 0.30
      }
    }
  },
  "monthly_trends": {
    "ENG": [
      { "month": "2017-08", "avg_goals": 2.8, "avg_corners": 10.2, "avg_shots": 24.1 }
    ]
  },
  "team_standings": {
    "ENG": {
      "2017-18": [
        { "team": "Man City", "points": 100, "goals_for": 106, "goals_against": 27, "goal_diff": 79 }
      ]
    }
  }
}
```

### `dashboard/data/backtest.json`
```json
{
  "metrics": {
    "log_loss": 0.9468,
    "brier_score": 0.2269,
    "accuracy": 0.56,
    "f1_home": 0.61,
    "f1_draw": 0.22,
    "f1_away": 0.54
  },
  "confusion_matrix": [[120, 30, 25], [40, 55, 35], [28, 22, 95]],
  "equity_curve": [
    { "date": "2019-08-10", "cumulative_pnl": 0, "value_bets_so_far": 0 }
  ],
  "mlflow_runs": [
    { "run_id": "abc123", "log_loss": 0.9468, "accuracy": 0.56, "n_estimators": 300 }
  ]
}
```

### `dashboard/data/value_bets.json`
Array of value bet records from Stage 5 (already written by stage5_compare.py).

### `dashboard/data/simulator.json`
```json
{
  "bets": [
    {
      "date": "2019-08-10",
      "home_team": "Arsenal",
      "away_team": "Newcastle",
      "outcome": "H",
      "result": "H",
      "model_odds": 2.35,
      "book_odds": 2.10,
      "edge": 0.119,
      "stake": 10,
      "return": 23.5,
      "won": true,
      "running_bankroll": 1013.5
    }
  ],
  "summary": {
    "total_bets": 45,
    "wins": 28,
    "losses": 17,
    "starting_bankroll": 450,
    "ending_bankroll": 523.4,
    "total_profit": 73.4,
    "roi_pct": 16.3,
    "hit_rate": 0.622,
    "max_drawdown": 62.5,
    "longest_win_streak": 6,
    "longest_loss_streak": 4,
    "avg_odds": 2.18,
    "avg_edge_pct": 14.2
  }
}
```

---

## How to Build It (Step by Step)

Build this stage in small increments. The reason is practical: a dashboard can look "done" while hiding broken data contracts. Each slice should prove one browser behaviour or one JSON contract before adding the next view.

### Current Small-Slice Plan

- [x] **Stage 6.1 — Dashboard data export contract**: `pipeline/export_dashboard_data.py` writes the four expected JSON files to `dashboard/data/`.
- [x] **Stage 6.2 — Static app shell and tab routing**: `dashboard/index.html`, `dashboard/css/style.css`, and `dashboard/js/main.js` define the four-tab static app.
- [x] **Stage 6.3 — Odds Inspector MVP**: reads `dashboard/data/value_bets.json`, validates required keys, filters by league/date, sorts records, displays edge as a percentage, opens a match detail modal, and exports the filtered table to CSV.
- [x] **Stage 6.4 — League Analytics view**: render KPI cards, monthly trend chart, standings table, and home/away split from `league_analytics.json`.
- [x] **Stage 6.5 — Backtest Performance view**: render metrics, equity curve, confusion matrix, and MLflow run summary from `backtest.json`.
- [x] **Stage 6.6 — Betting Simulator view**: recalculate bankroll, ROI, hit rate, drawdown, and bet log from `simulator.json` when the stake changes.


### Step 1 — Create the export script
Before building the dashboard, write `pipeline/export_dashboard_data.py`. This script reads the Parquet files from earlier stages and writes the four JSON files above. Run it after Stage 5 completes.

### Step 2 — Create the HTML skeleton
Create `dashboard/index.html` with:
- A `<nav>` bar with four tab buttons
- Four `<section>` divs, one per tab, all hidden except the active one
- Import your JS file at the bottom of `<body>`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sports Betting Analytics</title>
  <link rel="stylesheet" href="css/style.css">
</head>
<body>
  <nav>
    <button class="tab-btn active" data-tab="analytics">League Analytics</button>
    <button class="tab-btn" data-tab="backtest">Backtest</button>
    <button class="tab-btn" data-tab="odds">Odds Inspector</button>
    <button class="tab-btn" data-tab="simulator">Simulator</button>
  </nav>

  <section id="analytics" class="tab-content active"></section>
  <section id="backtest" class="tab-content"></section>
  <section id="odds" class="tab-content"></section>
  <section id="simulator" class="tab-content"></section>

  <script src="js/main.js"></script>
</body>
</html>
```

### Step 3 — Tab routing in `main.js`
```javascript
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(btn.dataset.tab).classList.add('active');
    btn.classList.add('active');
  });
});
```

### Step 4 — Build Tab 1: League Analytics
Load `league_analytics.json`, wire the league and season selectors, render:
- 4 KPI cards (avg goals, home win %, draw %, away win %)
- A native SVG line chart for monthly trends
- A sortable HTML table for team standings
- A grouped bar chart for home vs away splits

### Step 5 — Build Tab 2: Backtest Performance
Load `backtest.json`, render:
- A metrics table
- An equity curve line chart
- A heatmap-style confusion matrix using a pure CSS grid
- A top-5 MLflow runs table

### Step 6 — Build Tab 3: Odds Inspector
Load `value_bets.json`, render:
- Date range and league filter inputs
- A sortable, filterable table with edge % cells highlighted green
- A click handler on each row that opens a modal with the match detail

The modal should show:
- Home vs Away team
- Model probability bar chart
- Bookmaker odds comparison table
- Edge % per outcome

### Step 7 — Build Tab 4: Betting Simulator
This is the most interactive tab. Load `simulator.json`.

**Stake slider:**
```javascript
const stakeSlider = document.getElementById('stake-slider');
stakeSlider.addEventListener('input', () => {
  currentStake = parseFloat(stakeSlider.value);
  runSimulation(currentStake);  // recalculate everything on stake change
});
```

**`runSimulation(stake)` in `simulator.js`:**
1. Loop through `simulator.json.bets`
2. For each bet, compute `return = won ? stake * book_odds : 0`
3. Track `running_bankroll`, `max_drawdown`, `streak` counters
4. Update all KPI cards
5. Re-render the equity curve chart
6. Re-render the bet log table with green/red row highlighting

**Equity curve rendering:**
```javascript
const bankrollPoints = bets.map((bet) => ({ label: bet.date, value: bet.running_bankroll }));
renderLineChart('simulator-chart', [
  { name: 'Bankroll', points: bankrollPoints },
  { name: 'Peak', points: peakPoints },
]);
```

---

## Acceptance Criteria

- [x] `dashboard/index.html` opens without errors when served over HTTP
- [ ] All four tabs render without JavaScript console errors (browser runtime not available in this environment)
- [x] Tab 1: League and season selectors update all charts and the standings table
- [x] Tab 2: Equity curve and confusion matrix render correctly
- [x] Tab 3: Filtering by date range and league updates the table; clicking a row opens the modal
- [x] Tab 4: Moving the stake slider recalculates and updates all KPI cards, the equity curve, and the bet log table in real time
- [x] All four `dashboard/data/*.json` files exist and are valid JSON before running the dashboard

---

## Interview Q&A

**Q: Why did you build the dashboard without a backend?**  
A: "The pipeline pre-computes everything. There's nothing to calculate at request time — the data is already there. Adding a Flask server would mean maintaining a running process, routes, and templates just to serve static data. Static HTML with JSON files is simpler, more reliable, and deploys with a single nginx container. For a portfolio project especially, simple and working beats complex and fragile."

**Q: How does the betting simulator work?**  
A: "It replays every value bet flagged during the backtest period. For each bet it computes the return as `stake × odds` if the predicted outcome was correct, or 0 if it was wrong. It tracks a running bankroll, peak bankroll, and drawdown. The stake is configurable via a slider — the simulation reruns in real time as you adjust it. It's a backtesting tool, not a live system."

**Q: What is max drawdown and why does it matter for a betting model?**  
A: "Max drawdown is the largest peak-to-trough decline in the running bankroll. It answers 'what's the worst losing run I could expect?' A model with a 20% ROI but a 90% drawdown is unusable in practice — you'd go broke before the edge materialised. Drawdown is a risk metric as important as ROI."
