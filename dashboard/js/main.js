const REQUIRED_VALUE_BET_KEYS = [
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
];

const state = {
  analytics: null,
  backtest: null,
  simulator: null,
  strategyComparison: null,
  activeStrategyId: null,
  diagnostics: null,
  valueBets: [],
  filteredBets: [],
  currentStake: 10,
};

const formatDate = (value) => value ? new Date(value).toISOString().slice(0, 10) : "Start";
const formatOdds = (value) => Number(value).toFixed(2);
const formatPct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const formatWholePct = (value) => `${Number(value).toFixed(1)}%`;
const formatCurrency = (value) => `$${Number(value).toFixed(2)}`;
const formatNumber = (value, digits = 2) => Number(value ?? 0).toFixed(digits);
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll("\"", "&quot;");

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((section) => section.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(button.dataset.tab).classList.add("active");
    });
  });
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not fetch ${path}: ${response.status}`);
  }
  return response.json();
}

function validateValueBets(records) {
  if (!Array.isArray(records)) {
    throw new Error("value_bets.json must be an array of records");
  }
  records.forEach((record, index) => {
    const missing = REQUIRED_VALUE_BET_KEYS.filter((key) => !(key in record));
    if (missing.length > 0) {
      throw new Error(`value_bets.json row ${index} is missing: ${missing.join(", ")}`);
    }
  });
}

function renderCards(targetId, cards) {
  document.getElementById(targetId).innerHTML = cards
    .map(([label, value]) => `<div class="summary-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function renderLineChart(targetId, series, options = {}) {
  const target = document.getElementById(targetId);
  const width = 720;
  const height = 280;
  const padding = { top: 22, right: 24, bottom: 34, left: 52 };
  const labels = series[0]?.points.map((point) => point.label) || [];
  const allValues = series.flatMap((item) => item.points.map((point) => Number(point.value))).filter(Number.isFinite);
  if (!labels.length || !allValues.length) {
    target.innerHTML = '<div class="empty-mini">No chart data available.</div>';
    return;
  }
  const dataMin = Math.min(...allValues);
  const dataMax = Math.max(...allValues);
  const min = options.min ?? Math.min(dataMin, 0);
  const max = Math.max(options.max ?? dataMax, min + 1);
  const xStep = labels.length > 1 ? (width - padding.left - padding.right) / (labels.length - 1) : 0;
  const yScale = (value) => height - padding.bottom - ((Number(value) - min) / (max - min)) * (height - padding.top - padding.bottom);
  const xScale = (index) => padding.left + index * xStep;
  const colors = ["#0f766e", "#2563eb", "#9333ea", "#dc2626"];
  const paths = series.map((item, seriesIndex) => {
    const d = item.points.map((point, index) => `${index === 0 ? "M" : "L"}${xScale(index).toFixed(1)},${yScale(point.value).toFixed(1)}`).join(" ");
    return `<path d="${d}" fill="none" stroke="${colors[seriesIndex % colors.length]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />`;
  }).join("");
  const legend = series.map((item, index) => `<span><i style="background:${colors[index % colors.length]}"></i>${escapeHtml(item.name)}</span>`).join("");
  const tickIndexes = labels.length > 8 ? [0, Math.floor(labels.length / 3), Math.floor((labels.length * 2) / 3), labels.length - 1] : labels.map((_, index) => index);
  const ticks = tickIndexes.map((index) => `<text x="${xScale(index)}" y="${height - 10}" text-anchor="middle">${escapeHtml(labels[index])}</text>`).join("");
  const yTicks = [min, min + (max - min) / 2, max].map((value) => `<text x="8" y="${yScale(value) + 4}" text-anchor="start">${formatNumber(value, 1)}</text><line x1="${padding.left}" x2="${width - padding.right}" y1="${yScale(value)}" y2="${yScale(value)}" />`).join("");

  target.innerHTML = `
    <div class="chart-legend">${legend}</div>
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(options.label || "Line chart")}">
      <g class="grid-lines">${yTicks}</g>
      <line x1="${padding.left}" x2="${width - padding.right}" y1="${height - padding.bottom}" y2="${height - padding.bottom}" class="axis-line" />
      ${paths}
      <g class="x-ticks">${ticks}</g>
    </svg>
  `;
}

function renderBarList(targetId, rows, keys) {
  const maxTotal = Math.max(1, ...rows.map((row) => keys.reduce((total, key) => total + Number(row[key.field] || 0), 0)));
  document.getElementById(targetId).innerHTML = rows.map((row) => {
    const bars = keys.map((key) => {
      const value = Number(row[key.field] || 0);
      return `<div class="split-bar ${key.className}" style="width:${(value / maxTotal) * 100}%"><span>${escapeHtml(key.label)} ${value}</span></div>`;
    }).join("");
    return `<div class="bar-row"><div class="bar-label">${escapeHtml(row.team)}</div><div class="bar-track">${bars}</div></div>`;
  }).join("");
}

function seasonMonthRange(season) {
  const startYear = Number(season.slice(0, 4));
  const endYear = startYear + 1;
  return { start: `${startYear}-08`, end: `${endYear}-07` };
}

function setupAnalytics() {
  const leagueSelect = document.getElementById("analytics-league");
  const seasonSelect = document.getElementById("analytics-season");
  const labels = state.analytics.league_labels || {};
  leagueSelect.innerHTML = state.analytics.leagues.map((league) => `<option value="${escapeHtml(league)}">${escapeHtml(labels[league] || league)}</option>`).join("");
  seasonSelect.innerHTML = state.analytics.seasons.map((season) => `<option value="${escapeHtml(season)}">${escapeHtml(season)}</option>`).join("");
  [leagueSelect, seasonSelect].forEach((control) => control.addEventListener("input", renderAnalytics));
  renderAnalytics();
}

function renderAnalytics() {
  const league = document.getElementById("analytics-league").value;
  const season = document.getElementById("analytics-season").value;
  const summary = state.analytics.summary?.[league]?.[season] || {};
  renderCards("analytics-kpis", [
    ["Matches", summary.matches ?? 0],
    ["Avg goals", formatNumber(summary.avg_goals)],
    ["Home win", formatPct(summary.home_win_pct ?? 0)],
    ["Draw / Away", `${formatPct(summary.draw_pct ?? 0)} / ${formatPct(summary.away_win_pct ?? 0)}`],
  ]);

  const range = seasonMonthRange(season);
  const monthly = (state.analytics.monthly_trends?.[league] || []).filter((row) => row.month >= range.start && row.month <= range.end);
  renderLineChart("monthly-chart", [
    { name: "Goals", points: monthly.map((row) => ({ label: row.month, value: row.avg_goals })) },
    { name: "Corners", points: monthly.map((row) => ({ label: row.month, value: row.avg_corners })) },
    { name: "Shots OT", points: monthly.map((row) => ({ label: row.month, value: row.avg_shots })) },
  ], { label: "Monthly league trends" });

  const standings = state.analytics.team_standings?.[league]?.[season] || [];
  document.getElementById("standings-body").innerHTML = standings.map((row) => `
    <tr>
      <td>${escapeHtml(row.team)}</td>
      <td>${row.played}</td>
      <td>${row.points}</td>
      <td>${row.goal_diff}</td>
      <td>${row.goals_for}</td>
      <td>${row.goals_against}</td>
    </tr>
  `).join("");

  const splitRows = (state.analytics.home_away_split?.[league]?.[season] || []).slice(0, 12);
  renderBarList("home-away-chart", splitRows, [
    { field: "home_points", label: "H", className: "home" },
    { field: "away_points", label: "A", className: "away" },
  ]);
}

function renderBacktest() {
  const metrics = state.backtest.metrics || {};
  renderCards("backtest-kpis", [
    ["Log loss", formatNumber(metrics.log_loss, 3)],
    ["Brier score", formatNumber(metrics.brier_score, 3)],
    ["Accuracy", formatPct(metrics.accuracy ?? 0)],
    ["F1 H/D/A", `${formatNumber(metrics.f1_home, 2)} / ${formatNumber(metrics.f1_draw, 2)} / ${formatNumber(metrics.f1_away, 2)}`],
  ]);

  const equity = state.backtest.equity_curve || [];
  renderLineChart("equity-chart", [
    { name: "Cumulative P&L", points: equity.map((row) => ({ label: row.date || "Start", value: row.cumulative_pnl })) },
  ], { label: "Backtest cumulative P&L" });

  const matrix = state.backtest.confusion_matrix || [];
  const maxCell = Math.max(1, ...matrix.flat());
  const labels = ["Home", "Draw", "Away"];
  document.getElementById("confusion-matrix").innerHTML = `
    <div></div>${labels.map((label) => `<strong>${label}</strong>`).join("")}
    ${matrix.map((row, rowIndex) => `
      <strong>${labels[rowIndex]}</strong>
      ${row.map((value) => `<div class="matrix-cell" style="--intensity:${value / maxCell}">${value}</div>`).join("")}
    `).join("")}
  `;

  document.getElementById("mlflow-runs-body").innerHTML = (state.backtest.mlflow_runs || []).map((run) => `
    <tr>
      <td><code>${escapeHtml(run.run_id)}</code></td>
      <td>${formatNumber(run.log_loss, 3)}</td>
      <td>${formatNumber(run.brier_score, 3)}</td>
      <td>${formatPct(run.accuracy ?? 0)}</td>
      <td>${escapeHtml(run.n_estimators ?? "")}</td>
      <td>${escapeHtml(run.max_depth ?? "")}</td>
      <td>${formatNumber(run.learning_rate, 4)}</td>
    </tr>
  `).join("");
}

function setupFilters(records) {
  const leagueFilter = document.getElementById("league-filter");
  const startDate = document.getElementById("start-date");
  const endDate = document.getElementById("end-date");
  const sortFilter = document.getElementById("sort-filter");

  const leagues = [...new Set(records.map((record) => record.League))].sort();
  leagues.forEach((league) => {
    const option = document.createElement("option");
    option.value = league;
    option.textContent = league;
    leagueFilter.appendChild(option);
  });

  const dates = records.map((record) => formatDate(record.Date)).sort();
  startDate.value = dates[0] || "";
  endDate.value = dates[dates.length - 1] || "";

  [leagueFilter, startDate, endDate, sortFilter].forEach((control) => {
    control.addEventListener("input", applyFilters);
  });
}

function applyFilters() {
  const league = document.getElementById("league-filter").value;
  const start = document.getElementById("start-date").value;
  const end = document.getElementById("end-date").value;
  const sort = document.getElementById("sort-filter").value;

  state.filteredBets = state.valueBets.filter((record) => {
    const date = formatDate(record.Date);
    const leagueMatch = league === "all" || record.League === league;
    return leagueMatch && (!start || date >= start) && (!end || date <= end);
  });

  state.filteredBets.sort((left, right) => {
    if (sort === "date-asc") return formatDate(left.Date).localeCompare(formatDate(right.Date));
    if (sort === "date-desc") return formatDate(right.Date).localeCompare(formatDate(left.Date));
    if (sort === "league-asc") return left.League.localeCompare(right.League) || formatDate(left.Date).localeCompare(formatDate(right.Date));
    return Number(right.Edge) - Number(left.Edge);
  });

  renderOddsSummary(state.filteredBets);
  renderOddsTable(state.filteredBets);
}

function renderOddsSummary(records) {
  const averageEdge = records.length
    ? records.reduce((total, record) => total + Number(record.Edge), 0) / records.length
    : 0;
  const uniqueMatches = new Set(records.map((record) => record.RBallID)).size;
  const bestEdge = records.length ? Math.max(...records.map((record) => Number(record.Edge))) : 0;
  const bookmakers = new Set(records.map((record) => record.BestBookmaker)).size;
  renderCards("odds-summary", [
    ["Value bets", records.length],
    ["Unique matches", uniqueMatches],
    ["Average edge", formatPct(averageEdge)],
    ["Best edge", `${formatPct(bestEdge)} across ${bookmakers} books`],
  ]);
}

function renderOddsTable(records) {
  const body = document.getElementById("value-bets-body");
  const visibleRecords = records.slice(0, 250);
  body.innerHTML = visibleRecords
    .map((record, index) => {
      const match = `${record.HomeTeam} vs ${record.AwayTeam}`;
      return `
        <tr data-index="${index}">
          <td>${formatDate(record.Date)}</td>
          <td>${escapeHtml(record.League)}</td>
          <td>${escapeHtml(match)}</td>
          <td>${escapeHtml(record.Outcome)}</td>
          <td>${formatOdds(record.ModelOdds)}</td>
          <td>${formatOdds(record.BestBookOdds)}</td>
          <td><span class="edge-badge">${formatPct(record.Edge)}</span></td>
          <td>${escapeHtml(record.BestBookmaker)}</td>
        </tr>
      `;
    })
    .join("");

  body.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", () => {
      openBetModal(visibleRecords[Number(row.dataset.index)]);
    });
  });
}

function openBetModal(record) {
  const modal = document.getElementById("bet-modal");
  const matchRecords = state.valueBets.filter((item) => String(item.RBallID) === String(record.RBallID));
  const flaggedByOutcome = Object.fromEntries(matchRecords.map((item) => [item.Outcome, item]));
  const outcomes = [
    ["H", "Home", record.P_Home, record.ModelOdds_Home],
    ["D", "Draw", record.P_Draw, record.ModelOdds_Draw],
    ["A", "Away", record.P_Away, record.ModelOdds_Away],
  ];
  const probabilityBars = outcomes.map(([code, label, probability]) => {
    const value = Number(probability || 0);
    return `<div class="prob-row"><span>${escapeHtml(label)}</span><div><i style="width:${value * 100}%"></i></div><strong>${formatPct(value)}</strong></div>`;
  }).join("");
  const oddsRows = outcomes.map(([code, label, probability, modelOdds]) => {
    const flagged = flaggedByOutcome[code];
    return `
      <tr>
        <td>${escapeHtml(label)}</td>
        <td>${formatOdds(modelOdds || (probability ? 1 / probability : 0))}</td>
        <td>${flagged ? formatOdds(flagged.BestBookOdds) : "-"}</td>
        <td>${flagged ? escapeHtml(flagged.BestBookmaker) : "-"}</td>
        <td>${flagged ? formatPct(flagged.Edge) : "-"}</td>
      </tr>
    `;
  }).join("");

  document.getElementById("modal-title").textContent = `${record.HomeTeam} vs ${record.AwayTeam}`;
  document.getElementById("modal-details").innerHTML = `
    <dt>Date</dt><dd>${escapeHtml(formatDate(record.Date))}</dd>
    <dt>League</dt><dd>${escapeHtml(record.League)}</dd>
    <dt>Season</dt><dd>${escapeHtml(record.Season)}</dd>
    <dt>Actual result</dt><dd>${escapeHtml(record.Result)}</dd>
    <dt>Model probabilities</dt><dd class="wide-detail"><div class="prob-bars">${probabilityBars}</div></dd>
    <dt>Available odds</dt><dd class="wide-detail">
      <div class="table-wrap modal-table">
        <table>
          <thead><tr><th>Outcome</th><th>Model odds</th><th>Best odds</th><th>Bookmaker</th><th>Edge</th></tr></thead>
          <tbody>${oddsRows}</tbody>
        </table>
      </div>
    </dd>
  `;
  modal.showModal();
}

function setupModal() {
  const modal = document.getElementById("bet-modal");
  document.getElementById("close-modal").addEventListener("click", () => modal.close());
}

function exportCsv() {
  const header = REQUIRED_VALUE_BET_KEYS.join(",");
  const rows = state.filteredBets.map((record) =>
    REQUIRED_VALUE_BET_KEYS.map((key) => `"${String(record[key]).replaceAll('"', '""')}"`).join(",")
  );
  const blob = new Blob([[header, ...rows].join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "value_bets.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function activeStrategy() {
  const strategies = state.strategyComparison?.strategies || [];
  return strategies.find((strategy) => strategy.id === state.activeStrategyId) || strategies[0] || state.simulator;
}

function calculateSimulation(stake) {
  const strategy = activeStrategy();
  const bets = strategy?.bets || [];
  const starting = stake * bets.length;
  let running = starting;
  let peak = starting;
  let maxDrawdown = 0;
  let wins = 0;
  let currentWin = 0;
  let currentLoss = 0;
  let longestWin = 0;
  let longestLoss = 0;
  const rows = bets.map((bet) => {
    const won = Boolean(bet.won);
    const payout = won ? stake * Number(bet.book_odds) : 0;
    const profit = payout - stake;
    running += profit;
    peak = Math.max(peak, running);
    maxDrawdown = Math.max(maxDrawdown, peak - running);
    if (won) {
      wins += 1;
      currentWin += 1;
      currentLoss = 0;
      longestWin = Math.max(longestWin, currentWin);
    } else {
      currentLoss += 1;
      currentWin = 0;
      longestLoss = Math.max(longestLoss, currentLoss);
    }
    return { ...bet, stake, return: payout, profit, running_bankroll: running };
  });
  const total = rows.length;
  const losses = total - wins;
  const ending = rows.length ? rows[rows.length - 1].running_bankroll : starting;
  const totalProfit = ending - starting;
  return {
    rows,
    summary: {
      total_bets: total,
      wins,
      losses,
      starting_bankroll: starting,
      ending_bankroll: ending,
      total_profit: totalProfit,
      roi_pct: starting ? (totalProfit / starting) * 100 : 0,
      hit_rate: total ? wins / total : 0,
      max_drawdown: maxDrawdown,
      longest_win_streak: longestWin,
      longest_loss_streak: longestLoss,
      avg_odds: total ? rows.reduce((sum, row) => sum + Number(row.book_odds), 0) / total : 0,
      avg_edge_pct: total ? rows.reduce((sum, row) => sum + Number(row.edge), 0) / total * 100 : 0,
    },
  };
}


function setupCalibration() {
  [document.getElementById("calibration-source"), document.getElementById("calibration-outcome")].forEach((control) => {
    control.addEventListener("input", renderCalibration);
  });
  renderCalibration();
}

function renderCalibration() {
  const source = document.getElementById("calibration-source").value;
  const outcome = document.getElementById("calibration-outcome").value;
  const ranges = state.diagnostics?.value_bets_by_odds_range?.[source] || [];
  const rows = ranges.filter((row) => outcome === "all" || row.outcome === outcome);
  const totalBets = rows.reduce((total, row) => total + Number(row.count || 0), 0);
  const totalWins = rows.reduce((total, row) => total + Number(row.wins || 0), 0);
  const weightedRoi = totalBets ? rows.reduce((total, row) => total + Number(row.flat_stake_roi || 0) * Number(row.count || 0), 0) / totalBets : 0;
  const weightedEdge = totalBets ? rows.reduce((total, row) => total + Number(row.avg_edge || 0) * Number(row.count || 0), 0) / totalBets : 0;
  renderCards("calibration-kpis", [
    ["Buckets", rows.length],
    ["Bets", totalBets],
    ["Hit rate", formatPct(totalBets ? totalWins / totalBets : 0)],
    ["ROI / Edge", `${formatPct(weightedRoi)} / ${formatPct(weightedEdge)}`],
  ]);

  document.getElementById("calibration-body").innerHTML = rows.map((row) => {
    const actual = row.actual_result_rates || {};
    return `
      <tr>
        <td>${escapeHtml(row.outcome)}</td>
        <td>${escapeHtml(row.bucket)}</td>
        <td>${row.count}</td>
        <td>${row.wins}</td>
        <td>${formatPct(row.hit_rate)}</td>
        <td>${formatPct(actual.H || 0)} / ${formatPct(actual.D || 0)} / ${formatPct(actual.A || 0)}</td>
        <td>${formatOdds(row.avg_model_odds)}</td>
        <td>${formatOdds(row.avg_book_odds)}</td>
        <td>${formatPct(row.avg_edge)}</td>
        <td>${formatPct(row.flat_stake_roi)}</td>
      </tr>
    `;
  }).join("");
}
function renderStrategyComparison() {
  const strategies = state.strategyComparison?.strategies || [];
  const body = document.getElementById("strategy-comparison-body");
  if (!strategies.length) {
    body.innerHTML = '<tr><td colspan="7">No strategy comparison data available.</td></tr>';
    return;
  }
  body.innerHTML = strategies.map((strategy) => {
    const summary = strategy.summary || {};
    const activeClass = strategy.id === state.activeStrategyId ? "active-strategy-row" : "";
    return `
      <tr class="${activeClass}" data-strategy-id="${escapeHtml(strategy.id)}">
        <td>${escapeHtml(strategy.label || strategy.id)}</td>
        <td>${summary.total_bets ?? 0}</td>
        <td>${summary.wins ?? 0}</td>
        <td>${formatPct(summary.hit_rate ?? 0)}</td>
        <td>${formatCurrency(summary.total_profit ?? 0)}</td>
        <td>${formatWholePct(summary.roi_pct ?? 0)}</td>
        <td>${formatCurrency(summary.max_drawdown ?? 0)}</td>
      </tr>
    `;
  }).join("");
  body.querySelectorAll("tr[data-strategy-id]").forEach((row) => {
    row.addEventListener("click", () => {
      state.activeStrategyId = row.dataset.strategyId;
      document.getElementById("strategy-select").value = state.activeStrategyId;
      renderSimulator();
    });
  });
}


function setupSimulator() {
  const slider = document.getElementById("stake-slider");
  const strategySelect = document.getElementById("strategy-select");
  const strategies = state.strategyComparison?.strategies || [];
  const defaultStake = Number(state.strategyComparison?.default_stake || state.simulator.default_stake || 10);
  strategySelect.innerHTML = strategies.map((strategy) => `<option value="${escapeHtml(strategy.id)}">${escapeHtml(strategy.label || strategy.id)}</option>`).join("");
  state.activeStrategyId = state.strategyComparison?.primary_strategy_id || strategies[0]?.id || null;
  if (state.activeStrategyId) {
    strategySelect.value = state.activeStrategyId;
  }
  slider.value = String(defaultStake);
  state.currentStake = defaultStake;
  strategySelect.addEventListener("input", () => {
    state.activeStrategyId = strategySelect.value;
    renderSimulator();
  });
  slider.addEventListener("input", () => {
    state.currentStake = Number(slider.value);
    renderSimulator();
  });
  renderSimulator();
}

function renderSimulator() {
  const stake = state.currentStake;
  document.getElementById("stake-value").textContent = formatCurrency(stake);
  const strategy = activeStrategy();
  const simulation = calculateSimulation(stake);
  renderStrategyComparison();
  const summary = simulation.summary;
  renderCards("simulator-kpis", [
    ["Starting bankroll", formatCurrency(summary.starting_bankroll)],
    ["Ending bankroll", formatCurrency(summary.ending_bankroll)],
    ["Profit / loss", formatCurrency(summary.total_profit)],
    ["ROI", formatWholePct(summary.roi_pct)],
    ["Hit rate", formatPct(summary.hit_rate)],
    ["Max drawdown", formatCurrency(summary.max_drawdown)],
  ]);

  let peak = summary.starting_bankroll;
  const bankrollPoints = [{ label: "Start", value: summary.starting_bankroll }];
  const peakPoints = [{ label: "Start", value: peak }];
  simulation.rows.forEach((row) => {
    peak = Math.max(peak, row.running_bankroll);
    bankrollPoints.push({ label: row.date, value: row.running_bankroll });
    peakPoints.push({ label: row.date, value: peak });
  });
  renderLineChart("simulator-chart", [
    { name: "Bankroll", points: bankrollPoints },
    { name: "Peak", points: peakPoints },
  ], { min: Math.min(summary.starting_bankroll, summary.ending_bankroll - summary.max_drawdown), label: "Simulator bankroll" });

  document.getElementById("simulator-stats").innerHTML = [
    ["Total bets", summary.total_bets],
    ["Wins / Losses", `${summary.wins} / ${summary.losses}`],
    ["Longest win streak", summary.longest_win_streak],
    ["Longest loss streak", summary.longest_loss_streak],
    ["Average odds", formatOdds(summary.avg_odds)],
    ["Average edge", formatWholePct(summary.avg_edge_pct)],
  ].map(([label, value]) => `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`).join("");

  document.getElementById("simulator-log-body").innerHTML = simulation.rows.slice(0, 250).map((row) => `
    <tr class="${row.won ? "win-row" : "loss-row"}">
      <td>${escapeHtml(row.date)}</td>
      <td>${escapeHtml(row.league)}</td>
      <td>${escapeHtml(`${row.home_team} vs ${row.away_team}`)}</td>
      <td>${escapeHtml(row.outcome)}</td>
      <td>${escapeHtml(row.result)}</td>
      <td>${formatOdds(row.book_odds)}</td>
      <td>${formatCurrency(row.stake)}</td>
      <td>${formatCurrency(row.return)}</td>
      <td>${formatCurrency(row.profit)}</td>
      <td>${formatCurrency(row.running_bankroll)}</td>
    </tr>
  `).join("");
}

async function init() {
  setupTabs();
  setupModal();
  document.getElementById("export-csv").addEventListener("click", exportCsv);

  const status = document.getElementById("data-status");
  try {
    const [analytics, backtest, simulator, strategyComparison, diagnostics, valueBets] = await Promise.all([
      loadJson("data/league_analytics.json"),
      loadJson("data/backtest.json"),
      loadJson("data/simulator.json"),
      loadJson("data/strategy_comparison.json"),
      loadJson("data/diagnostics.json"),
      loadJson("data/value_bets.json"),
    ]);
    validateValueBets(valueBets);
    state.analytics = analytics;
    state.backtest = backtest;
    state.simulator = simulator;
    state.strategyComparison = strategyComparison;
    state.diagnostics = diagnostics;
    state.valueBets = valueBets;
    setupAnalytics();
    renderBacktest();
    setupFilters(state.valueBets);
    setupCalibration();
    applyFilters();
    setupSimulator();
    status.textContent = `${state.valueBets.length} value bets loaded`;
  } catch (error) {
    status.textContent = "Data load failed";
    status.classList.add("error");
    document.getElementById("value-bets-body").innerHTML = `<tr><td colspan="8">${escapeHtml(error.message)}</td></tr>`;
  }
}

init();
