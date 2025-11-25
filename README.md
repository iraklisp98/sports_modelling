# Sports Betting Model: European Football Odds Prediction

A comprehensive machine learning pipeline for analyzing European football leagues and identifying profitable betting opportunities through advanced feature engineering and model ensembling.

## Project Overview

This project combines descriptive statistical analysis with machine learning to predict football match outcomes and identify discrepancies between model predictions and sportsbook odds. The goal is to systematically identify "safe bets"—matches where predicted win probabilities exceed offered odds by at least 10%—to build a sustainable passive income strategy.

## Project Architecture

### Phase 1: Descriptive Analysis
- **Data Collection**: Historical data from European football leagues over the last 10 years
- **Statistical Analysis**: League trends, team performance metrics, and seasonal patterns
- **Key Metrics**: Win rates, goal differentials, home/away performance, team form

### Phase 2: Feature Engineering & Predictive Modeling
- **Feature Engineering**: Extract meaningful in-game statistics and contextual features
- **Model Development**: Build and train multiple ML algorithms to predict match outcomes
- **Model Ensembling**: Compare different models and combine them to maximize predictive accuracy
- **Robustness Testing**: Cross-validation, stress testing, and performance benchmarking

### Phase 3: Odds Arbitrage & Betting Pipeline
- **Odds Comparison**: Compare model predictions against sportsbook odds
- **Safe Bet Identification**: Apply filter: `if (model_odds >= 1.10 * sportsbook_odds) then place_bet`
- **Betting Pipeline**: Automate bet placement and tracking for long-term profitability analysis

## Data Sources
- European Football Leagues: Premier League, La Liga, Serie A, Bundesliga, Ligue 1
- Historical Period: Last 10 years
- Match Statistics: Goals, shots, possession, fouls, injuries, team form

## Technologies & Libraries
- **Data Processing**: Pandas, NumPy
- **Feature Engineering**: Scikit-learn, custom pipelines
- **Modeling**: XGBoost, LightGBM, Random Forest, Logistic Regression, Neural Networks
- **Analysis**: Matplotlib, Seaborn, Plotly
- **Odds Integration**: API connections to betting sites
- **Orchestration**: Python scripting, task scheduling

## Installation

```bash
git clone https://github.com/iraklisp98/sports_modelling.git
cd sports_modelling
pip install -r requirements.txt
```

## Project Structure

```
sports_modelling/
├── data/
│   ├── raw/                 # Original historical data
│   └── processed/           # Cleaned and engineered features
├── analysis/
│   ├── descriptive/         # Statistical analysis notebooks
│   └── visualizations/      # Charts and insights
├── models/
│   ├── training/            # Model training scripts
│   ├── evaluation/          # Performance benchmarks
│   └── ensembles/           # Ensemble models
├── pipeline/
│   ├── feature_engineering/ # Feature creation
│   ├── odds_comparison/     # Sportsbook integration
│   └── betting/             # Automated betting logic
├── notebooks/               # Jupyter notebooks for exploration
└── requirements.txt         # Project dependencies
```

## Key Metrics & Success Criteria

- **Model Accuracy**: Target >60% prediction accuracy on holdout test set
- **Precision/Recall**: Optimized based on false positive cost (losing bets)
- **ROI**: Long-term return on investment from safe bets
- **Consistency**: Stable performance across different seasons and leagues

## Safe Bet Criteria

A bet is placed when:
```
model_implied_odds ≥ 1.10 × sportsbook_odds
```

This ensures a minimum 10% edge over offered odds, protecting against model uncertainty and minimizing variance.

## Usage

### 1. Run Descriptive Analysis
```bash
python analysis/descriptive/league_analysis.py
```

### 2. Train Models
```bash
python models/training/train_ensemble.py
```

### 3. Evaluate Performance
```bash
python models/evaluation/evaluate_models.py
```

### 4. Generate Predictions & Odds Comparison
```bash
python pipeline/odds_comparison/find_safe_bets.py
```

### 5. Execute Betting Pipeline
```bash
python pipeline/betting/place_bets.py
```

## Results & Performance

Results from model testing and backtesting will be documented in:
- `analysis/visualizations/model_comparison.html`
- `pipeline/betting/performance_report.csv`

## Risk Disclaimer

This project is for educational and research purposes. Sports betting carries financial risk. Past performance is not indicative of future results. Always bet responsibly and within your means.

## Contributing

Contributions are welcome! Please submit pull requests with:
- Feature engineering improvements
- New model architectures
- Better odds data sources
- Enhanced backtesting frameworks

## License

MIT License - See LICENSE file for details

## Author

iraklisp98

## Contact

For questions or collaboration inquiries, please open an issue on GitHub. 