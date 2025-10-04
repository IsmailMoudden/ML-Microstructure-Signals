#!/usr/bin/env python3
"""
Synthetic Demo for ML Microstructure Signals

Basic workflow:
1. Generate synthetic order book data
2. Extract features
3. Generate labels
4. Train a model
5. Make predictions
6. Run backtest
7. Display results

Usage: python demo.py

Note: This is a simple demo - results might not be realistic.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ml_microstructure.data import SyntheticLOBGenerator, OrderBookProcessor
from ml_microstructure.features import FeaturePipeline
from ml_microstructure.models import create_model, ModelConfig
from ml_microstructure.utils.labeling import LabelGenerator
from ml_microstructure.backtest import BacktestRunner
from ml_microstructure.utils.io import save_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data():
    """Generate synthetic order book data."""
    logger.info("Generating synthetic order book data...")
    
    generator = SyntheticLOBGenerator(
        initial_price=100.0,
        tick_size=0.01,
        max_levels=10,
        arrival_rate=50.0,  # 50 orders per second
        duration_seconds=1800,  # 30 minutes
    )
    
    snapshots = generator.generate_data()
    logger.info(f"Generated {len(snapshots)} order book snapshots")
    
    return snapshots


def process_data(snapshots):
    """Process snapshots into DataFrame."""
    logger.info("Processing order book snapshots...")
    
    processor = OrderBookProcessor(max_levels=10)
    df = processor.process_snapshots(snapshots)
    
    logger.info(f"Processed {len(df)} data points")
    return df


def extract_features(df):
    """Extract features from data."""
    logger.info("Extracting features...")
    
    pipeline = FeaturePipeline()
    df_features = pipeline.extract_features(df)
    
    # Remove rows with NaN values
    df_features = df_features.dropna()
    
    logger.info(f"Extracted {len(df_features.columns)} features")
    return df_features


def generate_labels(df):
    """Generate labels for training."""
    logger.info("Generating labels...")
    
    label_generator = LabelGenerator(
        horizon=1,
        threshold=0.001,
        method="ternary"
    )
    
    labels = label_generator.generate_labels(df)
    df_labeled = df.copy()
    df_labeled['label'] = labels
    
    # Remove rows where labels couldn't be generated
    df_labeled = df_labeled.dropna(subset=['label'])
    
    logger.info(f"Generated labels for {len(df_labeled)} samples")
    logger.info(f"Label distribution: {df_labeled['label'].value_counts().to_dict()}")
    
    return df_labeled


def split_data(df):
    """Split data into train/test sets."""
    logger.info("Splitting data into train/test sets...")
    
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # 80/20 split
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    return train_df, test_df


def train_model(train_df):
    """Train a LightGBM model."""
    logger.info("Training LightGBM model...")
    
    # Prepare features
    feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'label']]
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    
    # Create model
    config = ModelConfig(
        model_type="lightgbm",
        random_state=42,
        class_weight="balanced",
        model_params={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        }
    )
    
    model = create_model(config)
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    return model, feature_cols


def make_predictions(model, test_df, feature_cols):
    """Make predictions on test data."""
    logger.info("Making predictions...")
    
    X_test = test_df[feature_cols]
    y_test = test_df['label']
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Add predictions to DataFrame
    df_pred = test_df.copy()
    df_pred['prediction'] = predictions
    df_pred['prob_up'] = probabilities[:, 2] if probabilities.shape[1] > 2 else probabilities[:, 1]
    df_pred['prob_down'] = probabilities[:, 0]
    df_pred['prob_flat'] = probabilities[:, 1] if probabilities.shape[1] > 2 else 1 - probabilities[:, 1]
    
    # Calculate accuracy
    accuracy = (predictions == y_test).mean()
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    return df_pred


def run_backtest(df_pred):
    """Run backtest on predictions."""
    logger.info("Running backtest...")
    
    # Create synthetic returns for backtesting
    np.random.seed(42)
    returns = np.random.normal(0, 0.001, len(df_pred))
    df_pred['return_forward_1'] = returns
    
    # Generate signals
    from ml_microstructure.backtest.signals import SignalConfig, SignalGenerator
    
    signal_config = SignalConfig(
        long_threshold=0.6,
        short_threshold=0.4,
        position_sizing="fixed",
        position_size=1.0
    )
    
    signal_generator = SignalGenerator(signal_config)
    df_signals = signal_generator.generate_signals(df_pred)
    
    # Execute trades
    from ml_microstructure.backtest.execution import ExecutionConfig, ExecutionEngine
    
    execution_config = ExecutionConfig(
        transaction_cost=0.001,
        slippage=0.0005,
        max_position=10.0
    )
    
    execution_engine = ExecutionEngine(execution_config)
    df_executed = execution_engine.execute_trades(df_signals)
    
    # Calculate metrics
    from ml_microstructure.backtest.metrics import MetricsConfig, BacktestMetrics
    
    metrics_config = MetricsConfig()
    metrics_calculator = BacktestMetrics(metrics_config)
    metrics = metrics_calculator.calculate_metrics(df_executed)
    
    logger.info("Backtest completed")
    return df_executed, metrics


def display_results(df_executed, metrics):
    """Display backtesting results."""
    print("\n" + "="*60)
    print("DEMO RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nData Overview:")
    print(f"  Total samples: {len(df_executed)}")
    print(f"  Time range: {df_executed['timestamp'].min()} to {df_executed['timestamp'].max()}")
    
    print(f"\nModel Performance:")
    print(f"  Total PnL: {metrics.get('total_pnl', 0):.2f}")
    print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    print(f"  Hit Rate: {metrics.get('hit_rate', 0):.2%}")
    
    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2f}")
    
    print(f"\nTrade Metrics:")
    print(f"  Average Win: {metrics.get('avg_win', 0):.2f}")
    print(f"  Average Loss: {metrics.get('avg_loss', 0):.2f}")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"  Turnover: {metrics.get('turnover', 0):.2f}")
    
    print("\n" + "="*60)
    
    # Save results
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    save_data(df_executed, output_dir / "backtest_results.parquet")
    save_data(metrics, output_dir / "metrics.json")
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main demo function."""
    print("🚀 ML Microstructure Signals - Synthetic Demo")
    print("=" * 50)
    
    try:
        # Step 1: Generate synthetic data
        snapshots = generate_synthetic_data()
        
        # Step 2: Process data
        df = process_data(snapshots)
        
        # Step 3: Extract features
        df_features = extract_features(df)
        
        # Step 4: Generate labels
        df_labeled = generate_labels(df_features)
        
        # Step 5: Split data
        train_df, test_df = split_data(df_labeled)
        
        # Step 6: Train model
        model, feature_cols = train_model(train_df)
        
        # Step 7: Make predictions
        df_pred = make_predictions(model, test_df, feature_cols)
        
        # Step 8: Run backtest
        df_executed, metrics = run_backtest(df_pred)
        
        # Step 9: Display results
        display_results(df_executed, metrics)
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("  1. Run the Streamlit dashboard: streamlit run ml_microstructure/dashboards/streamlit_app.py")
        print("  2. Explore the Jupyter notebooks in the notebooks/ directory")
        print("  3. Check the demo_output/ directory for saved results")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
