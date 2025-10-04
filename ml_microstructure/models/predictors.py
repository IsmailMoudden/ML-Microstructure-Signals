"""Model predictors for microstructure signal prediction.

Different ML models for predicting price movements.
Some models work better than others - still experimenting.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from pydantic import BaseModel, Field, ConfigDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for model training - basic setup."""
    
    model_type: str = Field(description="Type of model to use")
    random_state: int = Field(default=42, description="Random state")
    test_size: float = Field(default=0.2, description="Test set size")
    validation_size: float = Field(default=0.2, description="Validation set size")
    class_weight: Optional[str] = Field(default="balanced", description="Class weight strategy")
    
    # Model-specific parameters
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")


class BasePredictor(ABC):
    """Base class for predictors - simple approach."""
    
    def __init__(self, config: ModelConfig) -> None:
        """Initialize predictor."""
        self.config = config
        self.model: Any = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BasePredictor":
        """Fit the model to training data.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        accuracy = (y_pred == y).mean()
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist(),
        }
    
    def save_model(self, filepath: str) -> None:
        """Save model to file.
        
        Args:
            filepath: Path to save model
        """
        raise NotImplementedError("Model saving not implemented")
    
    def load_model(self, filepath: str) -> None:
        """Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        raise NotImplementedError("Model loading not implemented")


class LogisticRegressionPredictor(BasePredictor):
    """Logistic Regression predictor."""
    
    def __init__(self, config: ModelConfig) -> None:
        """Initialize Logistic Regression predictor."""
        super().__init__(config)
        self.scaler = StandardScaler()
        
        # Default parameters
        default_params = {
            "random_state": config.random_state,
            "max_iter": 1000,
            "class_weight": config.class_weight,
        }
        default_params.update(config.model_params)
        
        self.model = LogisticRegression(**default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionPredictor":
        """Fit Logistic Regression model."""
        logger.info("Fitting Logistic Regression model")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info("Logistic Regression model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class RandomForestPredictor(BasePredictor):
    """Random Forest predictor."""
    
    def __init__(self, config: ModelConfig) -> None:
        """Initialize Random Forest predictor."""
        super().__init__(config)
        
        # Default parameters
        default_params = {
            "random_state": config.random_state,
            "class_weight": config.class_weight,
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        }
        default_params.update(config.model_params)
        
        self.model = RandomForestClassifier(**default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestPredictor":
        """Fit Random Forest model."""
        logger.info("Fitting Random Forest model")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info("Random Forest model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class LightGBMPredictor(BasePredictor):
    """LightGBM predictor."""
    
    def __init__(self, config: ModelConfig) -> None:
        """Initialize LightGBM predictor."""
        super().__init__(config)
        
        # Default parameters
        default_params = {
            "random_state": config.random_state,
            "class_weight": config.class_weight,
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        }
        default_params.update(config.model_params)
        
        self.model = LGBMClassifier(**default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMPredictor":
        """Fit LightGBM model."""
        logger.info("Fitting LightGBM model")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info("LightGBM model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class LSTMPredictor(BasePredictor):
    """LSTM predictor for sequence data."""
    
    def __init__(self, config: ModelConfig) -> None:
        """Initialize LSTM predictor."""
        super().__init__(config)
        
        # Default parameters
        default_params = {
            "sequence_length": 20,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
        }
        default_params.update(config.model_params)
        
        self.sequence_length = default_params["sequence_length"]
        self.hidden_size = default_params["hidden_size"]
        self.num_layers = default_params["num_layers"]
        self.dropout = default_params["dropout"]
        self.learning_rate = default_params["learning_rate"]
        self.batch_size = default_params["batch_size"]
        self.epochs = default_params["epochs"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        sequences_X = []
        sequences_y = []
        
        for i in range(self.sequence_length, len(X)):
            sequences_X.append(X[i-self.sequence_length:i])
            sequences_y.append(y[i])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMPredictor":
        """Fit LSTM model."""
        logger.info("Fitting LSTM model")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y.values)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=len(self.feature_names),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=len(np.unique(y))
        ).to(self.device)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)
        
        # Training loop - TODO: add early stopping later
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_fitted = True
        logger.info("LSTM model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()


class TransformerPredictor(BasePredictor):
    """Transformer predictor for sequence data."""
    
    def __init__(self, config: ModelConfig) -> None:
        """Initialize Transformer predictor."""
        super().__init__(config)
        
        # Default parameters
        default_params = {
            "sequence_length": 20,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
        }
        default_params.update(config.model_params)
        
        self.sequence_length = default_params["sequence_length"]
        self.d_model = default_params["d_model"]
        self.nhead = default_params["nhead"]
        self.num_layers = default_params["num_layers"]
        self.dropout = default_params["dropout"]
        self.learning_rate = default_params["learning_rate"]
        self.batch_size = default_params["batch_size"]
        self.epochs = default_params["epochs"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for Transformer."""
        sequences_X = []
        sequences_y = []
        
        for i in range(self.sequence_length, len(X)):
            sequences_X.append(X[i-self.sequence_length:i])
            sequences_y.append(y[i])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TransformerPredictor":
        """Fit Transformer model."""
        logger.info("Fitting Transformer model")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y.values)
        
        # Initialize model
        self.model = TransformerModel(
            input_size=len(self.feature_names),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=len(np.unique(y))
        ).to(self.device)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)
        
        # Training loop - TODO: add early stopping later
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_fitted = True
        logger.info("Transformer model fitted successfully")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()


class LSTMModel(nn.Module):
    """LSTM model for sequence prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int) -> None:
        """Initialize LSTM model."""
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        return output


class TransformerModel(nn.Module):
    """Transformer model for sequence prediction."""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, dropout: float, num_classes: int) -> None:
        """Initialize Transformer model."""
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        seq_len = x.size(1)
        
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer
        x = self.transformer(x)
        
        # Take the last output
        last_output = x[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        return output


# Simple model creation - no fancy factory pattern
def create_model(config: ModelConfig) -> BasePredictor:
    """Create model based on config. Simple approach."""
    model_type = config.model_type
    
    if model_type == "logistic_regression":
        return LogisticRegressionPredictor(config)
    elif model_type == "random_forest":
        return RandomForestPredictor(config)
    elif model_type == "lightgbm":
        return LightGBMPredictor(config)
    elif model_type == "lstm":
        return LSTMPredictor(config)
    elif model_type == "transformer":
        return TransformerPredictor(config)
    else:
        raise ValueError(f"Don't know this model: {model_type}")

def get_available_models():
    """What models we have"""
    return ["logistic_regression", "random_forest", "lightgbm", "lstm", "transformer"]



