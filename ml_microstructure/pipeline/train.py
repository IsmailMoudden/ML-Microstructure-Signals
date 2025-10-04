"""Training pipeline for microstructure signal models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ml_microstructure.data import SyntheticLOBGenerator, OrderBookProcessor
from ml_microstructure.features import FeaturePipeline
from ml_microstructure.models import create_model, ModelConfig
from ml_microstructure.utils.labeling import LabelGenerator

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Pipeline for training microstructure signal models."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize training pipeline.
        
        Args:
            config: Hydra configuration
        """
        self.config = config
        self.feature_pipeline = FeaturePipeline()
        self.label_generator = LabelGenerator(
            horizon=self.config.labeling.horizon,
            threshold=self.config.labeling.threshold
        )
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
    
    def load_data(self) -> pd.DataFrame:
        """Load and process data.
        
        Returns:
            Processed DataFrame with features
        """
        logger.info("Loading data")
        
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
        
        logger.info(f"Loaded {len(df)} data points")
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
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate labels for training.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with labels
        """
        logger.info("Generating labels")
        
        # Generate labels
        labels = self.label_generator.generate_labels(df)
        
        # Add labels to DataFrame
        df_labeled = df.copy()
        df_labeled['label'] = labels
        
        # Remove rows where labels couldn't be generated
        df_labeled = df_labeled.dropna(subset=['label'])
        
        logger.info(f"Generated labels for {len(df_labeled)} samples")
        logger.info(f"Label distribution: {df_labeled['label'].value_counts().to_dict()}")
        
        return df_labeled
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets.
        
        Args:
            df: Labeled DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data")
        
        # Sort by timestamp to ensure proper time-based split
        df_sorted = df.sort_values('timestamp')
        
        # Calculate split indices
        n_samples = len(df_sorted)
        test_size = int(n_samples * self.config.data.test_size)
        val_size = int(n_samples * self.config.data.validation_size)
        
        # Time-based split (no leakage)
        train_end = n_samples - test_size - val_size
        val_end = n_samples - test_size
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training.
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        # Select feature columns (exclude timestamp and label)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
        
        X = df[feature_cols]
        y = df['label']
        
        return X, y
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train model with hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training model")
        
        # Create model configuration
        model_config = ModelConfig(
            model_type=self.config.model.type,
            random_state=self.config.model.random_state,
            class_weight=self.config.model.class_weight,
            model_params=self.config.model.params
        )
        
        # Create model
        model = create_model(model_config)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log configuration
            mlflow.log_params(OmegaConf.to_container(self.config, resolve=True))
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_metrics = model.evaluate(X_val, y_val)
            
            # Log metrics
            mlflow.log_metrics({
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["classification_report"]["macro avg"]["precision"],
                "val_recall": val_metrics["classification_report"]["macro avg"]["recall"],
                "val_f1": val_metrics["classification_report"]["macro avg"]["f1-score"],
            })
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Get feature importance if available
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance()
                mlflow.log_dict(feature_importance, "feature_importance.json")
            
            logger.info(f"Model trained successfully. Validation accuracy: {val_metrics['accuracy']:.4f}")
            
            return {
                "model": model,
                "metrics": val_metrics,
                "run_id": mlflow.active_run().info.run_id
            }
    
    def run(self) -> Dict:
        """Run the complete training pipeline.
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting training pipeline")
        
        # Load data
        df = self.load_data()
        
        # Extract features
        df_features = self.extract_features(df)
        
        # Generate labels
        df_labeled = self.generate_labels(df_features)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df_labeled)
        
        # Prepare features
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        X_test, y_test = self.prepare_features(test_df)
        
        # Train model
        results = self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        test_metrics = results["model"].evaluate(X_test, y_test)
        logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        
        # Save results
        results["test_metrics"] = test_metrics
        results["train_df"] = train_df
        results["val_df"] = val_df
        results["test_df"] = test_df
        
        logger.info("Training pipeline completed successfully")
        return results


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(config: DictConfig) -> None:
    """Main training function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run pipeline
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {config.model.type}")
    print(f"Validation Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"MLflow Run ID: {results['run_id']}")
    print("="*50)


if __name__ == "__main__":
    main()
