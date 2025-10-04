"""Labeling utilities for microstructure signals."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class LabelConfig(BaseModel):
    """Configuration for label generation."""
    
    horizon: int = Field(default=1, description="Prediction horizon in time steps")
    threshold: float = Field(default=0.001, description="Threshold for flat labels (relative to mid price)")
    method: str = Field(default="ternary", description="Labeling method: 'ternary' or 'binary'")


class LabelGenerator:
    """Generate labels for microstructure signal prediction."""
    
    def __init__(self, horizon: int = 1, threshold: float = 0.001, method: str = "ternary") -> None:
        """Initialize label generator.
        
        Args:
            horizon: Prediction horizon in time steps
            threshold: Threshold for flat labels (relative to mid price)
            method: Labeling method ('ternary' or 'binary')
        """
        self.horizon = horizon
        self.threshold = threshold
        self.method = method
        
        if method not in ["ternary", "binary"]:
            raise ValueError(f"Unknown labeling method: {method}")
    
    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """Generate labels for the given data.
        
        Args:
            df: DataFrame with mid_price column
            
        Returns:
            Series with labels (0: down, 1: flat, 2: up for ternary)
        """
        logger.info(f"Generating {self.method} labels with horizon {self.horizon}")
        
        if "mid_price" not in df.columns:
            raise ValueError("DataFrame must contain 'mid_price' column")
        
        # Calculate future returns
        future_prices = df["mid_price"].shift(-self.horizon)
        current_prices = df["mid_price"]
        
        # Calculate relative returns
        returns = (future_prices - current_prices) / current_prices
        
        if self.method == "ternary":
            labels = self._generate_ternary_labels(returns)
        elif self.method == "binary":
            labels = self._generate_binary_labels(returns)
        else:
            raise ValueError(f"Unknown labeling method: {self.method}")
        
        logger.info(f"Generated {len(labels)} labels")
        logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
        
        return labels
    
    def _generate_ternary_labels(self, returns: pd.Series) -> pd.Series:
        """Generate ternary labels (up/flat/down).
        
        Args:
            returns: Series of returns
            
        Returns:
            Series with ternary labels (0: down, 1: flat, 2: up)
        """
        labels = pd.Series(index=returns.index, dtype=int)
        
        # Up labels
        labels[returns > self.threshold] = 2
        
        # Down labels  
        labels[returns < -self.threshold] = 0
        
        # Flat labels
        labels[(returns >= -self.threshold) & (returns <= self.threshold)] = 1
        
        return labels
    
    def _generate_binary_labels(self, returns: pd.Series) -> pd.Series:
        """Generate binary labels (up/down).
        
        Args:
            returns: Series of returns
            
        Returns:
            Series with binary labels (0: down, 1: up)
        """
        labels = pd.Series(index=returns.index, dtype=int)
        
        # Up labels
        labels[returns > 0] = 1
        
        # Down labels
        labels[returns <= 0] = 0
        
        return labels
    
    def get_label_weights(self, labels: pd.Series) -> dict:
        """Calculate class weights for imbalanced data.
        
        Args:
            labels: Series with labels
            
        Returns:
            Dictionary with class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get unique classes
        classes = np.unique(labels)
        
        # Calculate weights
        weights = compute_class_weight(
            "balanced",
            classes=classes,
            y=labels
        )
        
        # Create weight dictionary
        weight_dict = dict(zip(classes, weights))
        
        logger.info(f"Class weights: {weight_dict}")
        return weight_dict



