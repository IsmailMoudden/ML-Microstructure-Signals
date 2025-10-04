"""Main backtesting runner."""

import logging
from pathlib import Path
from typing import Dict, Optional

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ml_microstructure.backtest.signals import SignalGenerator, SignalConfig
from ml_microstructure.backtest.execution import ExecutionEngine, ExecutionConfig
from ml_microstructure.backtest.metrics import BacktestMetrics, MetricsConfig
from ml_microstructure.pipeline.predict import PredictionPipeline

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Main runner for backtesting pipeline."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize backtest runner.
        
        Args:
            config: Hydra configuration
        """
        self.config = config
        
        # Initialize components
        self.signal_config = SignalConfig(**config.signal)
        self.execution_config = ExecutionConfig(**config.execution)
        self.metrics_config = MetricsConfig(**config.metrics)
        
        self.signal_generator = SignalGenerator(self.signal_config)
        self.execution_engine = ExecutionEngine(self.execution_config)
        self.metrics_calculator = BacktestMetrics(self.metrics_config)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
    
    def load_predictions(self, predictions_path: str) -> pd.DataFrame:
        """Load predictions from file.
        
        Args:
            predictions_path: Path to predictions file
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading predictions from {predictions_path}")
        
        if predictions_path.endswith('.parquet'):
            df = pd.read_parquet(predictions_path)
        elif predictions_path.endswith('.csv'):
            df = pd.read_csv(predictions_path)
        else:
            raise ValueError(f"Unsupported file format: {predictions_path}")
        
        logger.info(f"Loaded {len(df)} predictions")
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from predictions.
        
        Args:
            df: DataFrame with predictions
            
        Returns:
            DataFrame with signals
        """
        logger.info("Generating trading signals")
        
        df_signals = self.signal_generator.generate_signals(df)
        
        logger.info(f"Generated signals for {len(df_signals)} samples")
        return df_signals
    
    def execute_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute trades based on signals.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with executed trades
        """
        logger.info("Executing trades")
        
        df_executed = self.execution_engine.execute_trades(df)
        
        logger.info(f"Executed trades for {len(df_executed)} samples")
        return df_executed
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate backtesting metrics.
        
        Args:
            df: DataFrame with executed trades
            
        Returns:
            Dictionary with calculated metrics
        """
        logger.info("Calculating backtesting metrics")
        
        metrics = self.metrics_calculator.calculate_metrics(df)
        
        logger.info("Metrics calculation completed")
        return metrics
    
    def run_walk_forward_analysis(self, df: pd.DataFrame, train_window: int, test_window: int) -> Dict:
        """Run walk-forward analysis.
        
        Args:
            df: DataFrame with predictions
            train_window: Training window size
            test_window: Test window size
            
        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"Running walk-forward analysis with train_window={train_window}, test_window={test_window}")
        
        results = []
        
        for i in range(0, len(df) - train_window - test_window, test_window):
            # Training period
            train_start = i
            train_end = i + train_window
            
            # Test period
            test_start = train_end
            test_end = train_end + test_window
            
            # Get data for this period
            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]
            
            # Generate signals for test period
            test_signals = self.signal_generator.generate_signals(test_data)
            
            # Execute trades
            test_executed = self.execution_engine.execute_trades(test_signals)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(test_executed)
            
            # Store results
            results.append({
                "period": i,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "metrics": metrics,
                "data": test_executed
            })
        
        logger.info(f"Walk-forward analysis completed with {len(results)} periods")
        return {"results": results}
    
    def run(self, predictions_path: Optional[str] = None, run_id: Optional[str] = None) -> Dict:
        """Run the complete backtesting pipeline.
        
        Args:
            predictions_path: Path to predictions file
            run_id: MLflow run ID for model predictions
            
        Returns:
            Dictionary with backtesting results
        """
        logger.info("Starting backtesting pipeline")
        
        # Load predictions
        if predictions_path:
            df = self.load_predictions(predictions_path)
        elif run_id:
            # Generate predictions using model
            from ml_microstructure.pipeline.predict import PredictionPipeline
            predict_config = OmegaConf.create({
                "data": self.config.data,
                "mlflow": self.config.mlflow,
                "run_id": run_id
            })
            predict_pipeline = PredictionPipeline(predict_config)
            df = predict_pipeline.run(run_id=run_id)
        else:
            raise ValueError("Either predictions_path or run_id must be provided")
        
        # Generate signals
        df_signals = self.generate_signals(df)
        
        # Execute trades
        df_executed = self.execute_trades(df_signals)
        
        # Calculate metrics
        metrics = self.calculate_metrics(df_executed)
        
        # Run walk-forward analysis if requested
        walk_forward_results = None
        if self.config.get("walk_forward", {}).get("enabled", False):
            walk_forward_results = self.run_walk_forward_analysis(
                df_signals,
                self.config.walk_forward.train_window,
                self.config.walk_forward.test_window
            )
        
        # Generate report
        report = self.metrics_calculator.generate_report(metrics)
        
        results = {
            "metrics": metrics,
            "data": df_executed,
            "signals": df_signals,
            "walk_forward": walk_forward_results,
            "report": report
        }
        
        logger.info("Backtesting pipeline completed successfully")
        return results
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save backtesting results.
        
        Args:
            results: Backtesting results
            output_path: Path to save results
        """
        logger.info(f"Saving backtesting results to {output_path}")
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        results["data"].to_parquet(f"{output_path}_data.parquet", index=False)
        
        # Save metrics
        import json
        with open(f"{output_path}_metrics.json", 'w') as f:
            json.dump(results["metrics"], f, indent=2, default=str)
        
        # Save report
        with open(f"{output_path}_report.txt", 'w') as f:
            f.write(results["report"])
        
        logger.info("Backtesting results saved successfully")


@hydra.main(version_base=None, config_path="../../configs", config_name="backtest")
def main(config: DictConfig) -> None:
    """Main backtesting function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run backtest
    runner = BacktestRunner(config)
    results = runner.run(
        predictions_path=config.get("predictions_path"),
        run_id=config.get("run_id")
    )
    
    # Save results if path provided
    if config.get("output_path"):
        runner.save_results(results, config.output_path)
    
    # Print report
    print(results["report"])


if __name__ == "__main__":
    main()



