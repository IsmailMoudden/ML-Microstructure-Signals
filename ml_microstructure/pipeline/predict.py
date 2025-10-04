"""Prediction pipeline for microstructure signal models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ml_microstructure.data import SyntheticLOBGenerator, OrderBookProcessor
from ml_microstructure.features import FeaturePipeline
from ml_microstructure.models import ModelFactory, ModelConfig

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Pipeline for making predictions with trained models."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize prediction pipeline.
        
        Args:
            config: Hydra configuration
        """
        self.config = config
        self.feature_pipeline = FeaturePipeline()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
    
    def load_model(self, run_id: Optional[str] = None) -> any:
        """Load trained model.
        
        Args:
            run_id: MLflow run ID. If None, loads latest model.
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from run_id: {run_id}")
        
        if run_id is None:
            # Get latest model
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            if not experiments:
                raise ValueError("No experiments found")
            
            experiment_id = experiments[0].experiment_id
            runs = client.search_runs(experiment_id, order_by=["start_time DESC"], max_results=1)
            if not runs:
                raise ValueError("No runs found")
            
            run_id = runs[0].info.run_id
        
        # Load model
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        
        logger.info(f"Model loaded successfully from run_id: {run_id}")
        return model
    
    def load_data(self) -> pd.DataFrame:
        """Load data for prediction.
        
        Returns:
            DataFrame with features
        """
        logger.info("Loading data for prediction")
        
        if self.config.data.type == "synthetic":
            # Generate synthetic data
            generator = SyntheticLOBGenerator(
                initial_price=self.config.data.synthetic.initial_price,
                tick_size=self.config.data.synthetic.tick_size,
                max_levels=self.config.data.synthetic.max_levels,
                arrival_rate=self.config.data.synthetic.arrival_rate,
                duration_seconds=self.config.data.synthetic.duration_seconds,
            )
            snapshots = generator.generate_data()
            
        elif self.config.data.type == "lobster":
            # Load LOBSTER data
            from ml_microstructure.data import LOBSTERLoader
            loader = LOBSTERLoader(
                symbol=self.config.data.lobster.symbol,
                date=self.config.data.lobster.date
            )
            snapshots = loader.load_data(self.config.data.lobster.file_path)
            
        elif self.config.data.type == "kaggle_crypto":
            # Load Kaggle crypto data
            from ml_microstructure.data import KaggleCryptoLoader
            loader = KaggleCryptoLoader(symbol=self.config.data.kaggle_crypto.symbol)
            snapshots = loader.load_data(self.config.data.kaggle_crypto.file_path)
            
        else:
            raise ValueError(f"Unknown data type: {self.config.data.type}")
        
        # Process snapshots into DataFrame
        processor = OrderBookProcessor(max_levels=self.config.data.max_levels)
        df = processor.process_snapshots(snapshots)
        
        logger.info(f"Loaded {len(df)} data points for prediction")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from data.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        logger.info("Extracting features")
        
        # Extract features
        df_features = self.feature_pipeline.extract_features(df)
        
        # Remove rows with NaN values
        df_features = df_features.dropna()
        
        logger.info(f"Extracted {len(df_features.columns)} features")
        return df_features
    
    def make_predictions(self, model: any, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on data.
        
        Args:
            model: Trained model
            df: DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Making predictions")
        
        # Select feature columns (exclude timestamp)
        feature_cols = [col for col in df.columns if col != 'timestamp']
        X = df[feature_cols]
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Add predictions to DataFrame
        df_pred = df.copy()
        df_pred['prediction'] = predictions
        df_pred['prob_up'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        df_pred['prob_down'] = probabilities[:, 2] if probabilities.shape[1] > 2 else probabilities[:, 1]
        df_pred['prob_flat'] = probabilities[:, 0] if probabilities.shape[1] > 2 else 1 - probabilities[:, 0]
        
        logger.info(f"Made predictions for {len(df_pred)} samples")
        return df_pred
    
    def save_predictions(self, df_pred: pd.DataFrame, output_path: str) -> None:
        """Save predictions to file.
        
        Args:
            df_pred: DataFrame with predictions
            output_path: Path to save predictions
        """
        logger.info(f"Saving predictions to {output_path}")
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet for efficiency
        df_pred.to_parquet(output_path, index=False)
        
        logger.info(f"Predictions saved successfully")
    
    def run(self, run_id: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
        """Run the complete prediction pipeline.
        
        Args:
            run_id: MLflow run ID. If None, uses latest model.
            output_path: Path to save predictions. If None, doesn't save.
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Starting prediction pipeline")
        
        # Load model
        model = self.load_model(run_id)
        
        # Load data
        df = self.load_data()
        
        # Extract features
        df_features = self.extract_features(df)
        
        # Make predictions
        df_pred = self.make_predictions(model, df_features)
        
        # Save predictions if path provided
        if output_path:
            self.save_predictions(df_pred, output_path)
        
        logger.info("Prediction pipeline completed successfully")
        return df_pred


@hydra.main(version_base=None, config_path="../../configs", config_name="predict")
def main(config: DictConfig) -> None:
    """Main prediction function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run pipeline
    pipeline = PredictionPipeline(config)
    predictions = pipeline.run(
        run_id=config.get("run_id"),
        output_path=config.get("output_path")
    )
    
    # Print summary
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"Total predictions: {len(predictions)}")
    print(f"Prediction distribution:")
    print(predictions['prediction'].value_counts().to_dict())
    print(f"Average probability (up): {predictions['prob_up'].mean():.4f}")
    print(f"Average probability (down): {predictions['prob_down'].mean():.4f}")
    print(f"Average probability (flat): {predictions['prob_flat'].mean():.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
