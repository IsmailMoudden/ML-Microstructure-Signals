"""Evaluation pipeline for microstructure signal models."""

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
from ml_microstructure.models import ModelFactory, ModelConfig
from ml_microstructure.utils.labeling import LabelGenerator

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Pipeline for evaluating trained models."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize evaluation pipeline.
        
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
    
    def load_model(self, run_id: str) -> any:
        """Load trained model.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from run_id: {run_id}")
        
        # Load model
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        
        logger.info(f"Model loaded successfully from run_id: {run_id}")
        return model
    
    def load_data(self) -> pd.DataFrame:
        """Load data for evaluation.
        
        Returns:
            DataFrame with features
        """
        logger.info("Loading data for evaluation")
        
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
        
        logger.info(f"Loaded {len(df)} data points for evaluation")
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
        """Generate labels for evaluation.
        
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
    
    def evaluate_model(self, model: any, df: pd.DataFrame) -> Dict:
        """Evaluate model performance.
        
        Args:
            model: Trained model
            df: DataFrame with features and labels
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model")
        
        # Select feature columns (exclude timestamp and label)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
        X = df[feature_cols]
        y = df['label']
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Calculate metrics
        accuracy = (predictions == y).mean()
        
        # Classification report
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(y, predictions, output_dict=True)
        cm = confusion_matrix(y, predictions)
        
        # Calculate additional metrics
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        
        # Calculate hit rate for each class
        hit_rates = {}
        for class_label in np.unique(y):
            mask = y == class_label
            if mask.sum() > 0:
                hit_rates[f'hit_rate_class_{class_label}'] = (predictions[mask] == class_label).mean()
        
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "hit_rates": hit_rates,
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "true_labels": y.tolist(),
        }
        
        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
        return results
    
    def analyze_predictions(self, results: Dict) -> Dict:
        """Analyze prediction patterns.
        
        Args:
            results: Evaluation results
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing predictions")
        
        predictions = np.array(results["predictions"])
        probabilities = np.array(results["probabilities"])
        true_labels = np.array(results["true_labels"])
        
        analysis = {}
        
        # Confidence analysis
        max_probs = np.max(probabilities, axis=1)
        analysis["avg_confidence"] = np.mean(max_probs)
        analysis["confidence_std"] = np.std(max_probs)
        
        # Confidence by accuracy
        correct_mask = predictions == true_labels
        analysis["avg_confidence_correct"] = np.mean(max_probs[correct_mask])
        analysis["avg_confidence_incorrect"] = np.mean(max_probs[~correct_mask])
        
        # Class-specific analysis
        for class_label in np.unique(true_labels):
            mask = true_labels == class_label
            if mask.sum() > 0:
                class_probs = probabilities[mask, class_label]
                analysis[f"avg_prob_class_{class_label}"] = np.mean(class_probs)
                analysis[f"std_prob_class_{class_label}"] = np.std(class_probs)
        
        # Prediction distribution
        pred_counts = np.bincount(predictions)
        analysis["prediction_distribution"] = pred_counts.tolist()
        
        logger.info("Prediction analysis completed")
        return analysis
    
    def save_results(self, results: Dict, analysis: Dict, output_path: str) -> None:
        """Save evaluation results.
        
        Args:
            results: Evaluation results
            analysis: Analysis results
            output_path: Path to save results
        """
        logger.info(f"Saving evaluation results to {output_path}")
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Combine results
        all_results = {**results, "analysis": analysis}
        
        # Save as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved successfully")
    
    def run(self, run_id: str, output_path: Optional[str] = None) -> Dict:
        """Run the complete evaluation pipeline.
        
        Args:
            run_id: MLflow run ID
            output_path: Path to save results. If None, doesn't save.
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting evaluation pipeline")
        
        # Load model
        model = self.load_model(run_id)
        
        # Load data
        df = self.load_data()
        
        # Extract features
        df_features = self.extract_features(df)
        
        # Generate labels
        df_labeled = self.generate_labels(df_features)
        
        # Evaluate model
        results = self.evaluate_model(model, df_labeled)
        
        # Analyze predictions
        analysis = self.analyze_predictions(results)
        
        # Combine results
        all_results = {**results, "analysis": analysis}
        
        # Save results if path provided
        if output_path:
            self.save_results(results, analysis, output_path)
        
        logger.info("Evaluation pipeline completed successfully")
        return all_results


@hydra.main(version_base=None, config_path="../../configs", config_name="evaluate")
def main(config: DictConfig) -> None:
    """Main evaluation function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run pipeline
    pipeline = EvaluationPipeline(config)
    results = pipeline.run(
        run_id=config.run_id,
        output_path=config.get("output_path")
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Average Confidence: {results['analysis']['avg_confidence']:.4f}")
    print(f"Confidence (Correct): {results['analysis']['avg_confidence_correct']:.4f}")
    print(f"Confidence (Incorrect): {results['analysis']['avg_confidence_incorrect']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
