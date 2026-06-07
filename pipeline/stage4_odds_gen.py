from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
try:
    from pipeline.model_features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, LEAGUE_FEATURE_COLUMNS, LEAGUES, MARKET_FEATURE_COLUMNS, add_league_indicator_features
    from pipeline.market_features import FOOTBALL_DATA_DIR, add_market_features
except ModuleNotFoundError:
    from model_features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, LEAGUE_FEATURE_COLUMNS, LEAGUES, MARKET_FEATURE_COLUMNS, add_league_indicator_features
    from market_features import FOOTBALL_DATA_DIR, add_market_features

FEATURES_DIR = Path('data/features')
OUTPUT_PATH = Path('data/output/model_odds.parquet')
MODEL_NAME = 'match_outcome_xgb'
MODEL_STAGE = 'Production'
MIN_ODDS_PROBABILITY = 1e-6
TRACKING_URI = 'file:mlruns'

REQUIRED_COLUMNS = [
    'RBallID',
    'HomeTeam',
    'AwayTeam',
    'Date',
    'Season',
    'League',
    *BASE_FEATURE_COLUMNS,
]

OUTPUT_COLUMNS = [
    'RBallID',
    'HomeTeam',
    'AwayTeam',
    'Date',
    'Season',
    'Result',
    'P_Home',
    'P_Draw',
    'P_Away',
    'ModelOdds_Home',
    'ModelOdds_Draw',
    'ModelOdds_Away',
]


@dataclass(frozen=True)
class OddsGenerationSummary:
    rows: int
    output_path: Path
    model_uri: str

    def line(self) -> str:
        return f'rows={self.rows}, output={self.output_path}, model={self.model_uri}'


def load_feature_data(
    leagues: Iterable[str] = LEAGUES,
    features_dir: Path = FEATURES_DIR,
) -> pd.DataFrame:
    frames = []
    for league in leagues:
        path = features_dir / f'{league}_features.parquet'
        if not path.exists():
            raise FileNotFoundError(f'Missing Stage 2 feature file: {path}')

        frame = pd.read_parquet(path).copy()
        frame['League'] = league
        frames.append(frame)

    if not frames:
        raise ValueError('At least one league is required for Stage 4')

    combined = pd.concat(frames, ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = add_league_indicator_features(combined, leagues=leagues)
    return combined.sort_values(['Date', 'RBallID'], kind='mergesort').reset_index(drop=True)


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f'Missing required Stage 4 columns: {missing}')


def load_production_model(
    model_name: str = MODEL_NAME,
    model_stage: str = MODEL_STAGE,
    tracking_uri: str | None = TRACKING_URI,
):
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError('MLflow is required to load the Production model') from exc

    mlflow.set_tracking_uri(tracking_uri or TRACKING_URI)
    model_uri = f'models:/{model_name}/{model_stage}'
    return mlflow.sklearn.load_model(model_uri)

def select_model_features(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    validate_feature_columns(df)
    featured = add_league_indicator_features(df)
    columns = list(feature_columns)
    missing = [column for column in columns if column not in featured.columns]
    if missing:
        raise ValueError(f'Missing required model feature columns: {missing}')
    return featured[columns].apply(pd.to_numeric, errors='raise').astype('float64')


def validate_model_class_order(model) -> None:
    classes = getattr(model, 'classes_', None)
    expected = np.array([0, 1, 2])
    if classes is None or not np.array_equal(np.asarray(classes), expected):
        raise ValueError(f'Model classes must be ordered as {expected.tolist()} for H/D/A outputs')


def validate_probability_matrix(proba: np.ndarray, tolerance: float = 0.001) -> np.ndarray:
    matrix = np.asarray(proba, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError('Expected a 2D probability matrix with three outcome columns')
    if not np.isfinite(matrix).all():
        raise ValueError('Probability matrix must contain only finite values')
    if (matrix < 0).any() or (matrix > 1).any():
        raise ValueError('Probability values must be between 0 and 1')

    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        max_delta = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f'Probability rows must sum to 1.0 ± {tolerance}; max delta={max_delta:.6f}')
    return matrix


def probabilities_to_odds(proba: np.ndarray, minimum_probability: float = MIN_ODDS_PROBABILITY) -> np.ndarray:
    if minimum_probability <= 0.0:
        raise ValueError('minimum_probability must be strictly positive')
    matrix = validate_probability_matrix(proba)
    if (matrix < 0).any():
        raise ValueError('Probabilities must be non-negative to convert to decimal odds')
    clipped = np.clip(matrix, minimum_probability, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return 1.0 / clipped


def build_model_odds_frame(df: pd.DataFrame, model) -> pd.DataFrame:
    validate_feature_columns(df)

    ordered = df.sort_values(['Date', 'RBallID'], kind='mergesort').reset_index(drop=True).copy()
    validate_model_class_order(model)
    expected_features = tuple(getattr(model, 'feature_names_in_', FEATURE_COLUMNS))
    features = select_model_features(ordered, feature_columns=expected_features)
    proba = validate_probability_matrix(model.predict_proba(features))
    odds = probabilities_to_odds(proba)

    output = ordered[['RBallID', 'HomeTeam', 'AwayTeam', 'Date', 'Season']].copy()
    if 'Result' in ordered.columns:
        output['Result'] = ordered['Result']
    else:
        output['Result'] = pd.NA
    output['P_Home'] = proba[:, 0]
    output['P_Draw'] = proba[:, 1]
    output['P_Away'] = proba[:, 2]
    output['ModelOdds_Home'] = odds[:, 0]
    output['ModelOdds_Draw'] = odds[:, 1]
    output['ModelOdds_Away'] = odds[:, 2]
    return output[OUTPUT_COLUMNS]


def run_pipeline(
    leagues: Iterable[str] = LEAGUES,
    features_dir: Path = FEATURES_DIR,
    output_path: Path = OUTPUT_PATH,
    model_name: str = MODEL_NAME,
    model_stage: str = MODEL_STAGE,
    tracking_uri: str | None = TRACKING_URI,
    football_data_dir: Path = FOOTBALL_DATA_DIR,
) -> OddsGenerationSummary:
    df = load_feature_data(leagues=leagues, features_dir=features_dir)
    model = load_production_model(model_name=model_name, model_stage=model_stage, tracking_uri=tracking_uri)
    expected_features = tuple(getattr(model, 'feature_names_in_', FEATURE_COLUMNS))
    if any(column in expected_features for column in MARKET_FEATURE_COLUMNS) and not set(MARKET_FEATURE_COLUMNS).issubset(df.columns):
        df, _ = add_market_features(df, football_data_dir=football_data_dir)
    odds_df = build_model_odds_frame(df, model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    odds_df.to_parquet(output_path, index=False)

    model_uri = f'models:/{model_name}/{model_stage}'
    return OddsGenerationSummary(rows=len(odds_df), output_path=output_path, model_uri=model_uri)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate model-implied odds from the Production Stage 3 model.')
    parser.add_argument('--features-dir', type=Path, default=FEATURES_DIR)
    parser.add_argument('--output-path', type=Path, default=OUTPUT_PATH)
    parser.add_argument('--model-name', default=MODEL_NAME)
    parser.add_argument('--model-stage', default=MODEL_STAGE)
    parser.add_argument('--tracking-uri', default=TRACKING_URI)
    parser.add_argument('--football-data-dir', type=Path, default=FOOTBALL_DATA_DIR)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    summary = run_pipeline(
        features_dir=args.features_dir,
        output_path=args.output_path,
        model_name=args.model_name,
        model_stage=args.model_stage,
        tracking_uri=args.tracking_uri,
        football_data_dir=args.football_data_dir,
    )
    print(summary.line())
