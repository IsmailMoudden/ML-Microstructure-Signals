"""Signal generation for backtesting."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class SignalConfig(BaseModel):
    """Configuration for signal generation."""
    
    long_threshold: float = Field(default=0.6, description="Threshold for long signals")
    short_threshold: float = Field(default=0.4, description="Threshold for short signals")
    position_sizing: str = Field(default="fixed", description="Position sizing method")
    position_size: float = Field(default=1.0, description="Fixed position size")
    kelly_fraction: float = Field(default=0.25, description="Kelly fraction for position sizing")
    max_position: float = Field(default=10.0, description="Maximum position size")


class SignalGenerator:
    """Generate trading signals from model predictions."""
    
    def __init__(self, config: SignalConfig) -> None:
        """Initialize signal generator.
        
        Args:
            config: Signal configuration
        """
        self.config = config
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from predictions."""
        logger.info("Generating trading signals")
        
        df_signals = df.copy()
        
        # Generate signals based on probabilities
        if "prob_up" in df.columns and "prob_down" in df.columns:
            # Ternary classification
            df_signals["signal"] = self._generate_ternary_signals(df)
        elif "prob_up" in df.columns:
            # Binary classification (up vs down)
            df_signals["signal"] = self._generate_binary_signals(df)
        else:
            raise ValueError("DataFrame must contain probability columns")
        
        # Calculate position sizes
        df_signals["position_size"] = self._calculate_position_sizes(df_signals)
        
        logger.info(f"Generated signals for {len(df_signals)} samples")
        logger.info(f"Signal distribution: {df_signals['signal'].value_counts().to_dict()}")
        
        return df_signals
    
    def _generate_ternary_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate ternary signals (long/short/flat).
        
        Args:
            df: DataFrame with probabilities
            
        Returns:
            Series with signals (1: long, -1: short, 0: flat)
        """
        signals = pd.Series(index=df.index, dtype=int)
        
        # Long signals
        signals[df["prob_up"] > self.config.long_threshold] = 1
        
        # Short signals
        signals[df["prob_down"] > self.config.short_threshold] = -1
        
        # Flat signals
        signals[
            (df["prob_up"] <= self.config.long_threshold) & 
            (df["prob_down"] <= self.config.short_threshold)
        ] = 0
        
        return signals
    
    def _generate_binary_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate binary signals (long/short).
        
        Args:
            df: DataFrame with probabilities
            
        Returns:
            Series with signals (1: long, -1: short)
        """
        signals = pd.Series(index=df.index, dtype=int)
        
        # Long signals
        signals[df["prob_up"] > self.config.long_threshold] = 1
        
        # Short signals
        signals[df["prob_up"] < self.config.short_threshold] = -1
        
        # Flat signals (middle range)
        signals[
            (df["prob_up"] >= self.config.short_threshold) & 
            (df["prob_up"] <= self.config.long_threshold)
        ] = 0
        
        return signals
    
    def _calculate_position_sizes(self, df: pd.DataFrame) -> pd.Series:
        """Calculate position sizes.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            Series with position sizes
        """
        if self.config.position_sizing == "fixed":
            return df["signal"].abs() * self.config.position_size
        
        elif self.config.position_sizing == "kelly":
            # Kelly position sizing based on win probability and odds
            win_prob = self._estimate_win_probability(df)
            odds = self._estimate_odds(df)
            
            kelly_fraction = win_prob - (1 - win_prob) / odds
            kelly_fraction = max(0, min(kelly_fraction, self.config.kelly_fraction))
            
            return df["signal"].abs() * kelly_fraction * self.config.position_size
        
        else:
            raise ValueError(f"Unknown position sizing method: {self.config.position_sizing}")
    
    def _estimate_win_probability(self, df: pd.DataFrame) -> float:
        """Estimate win probability from historical data.
        
        Args:
            df: DataFrame with signals and returns
            
        Returns:
            Estimated win probability
        """
        if "return_forward_1" in df.columns:
            # Use actual returns if available
            returns = df["return_forward_1"]
            signals = df["signal"]
            
            # Calculate win rate
            wins = ((signals > 0) & (returns > 0)) | ((signals < 0) & (returns < 0))
            win_prob = wins.mean()
        else:
            # Use probability-based estimation
            win_prob = 0.55  # Default assumption
        
        return win_prob
    
    def _estimate_odds(self, df: pd.DataFrame) -> float:
        """Estimate odds from historical data.
        
        Args:
            df: DataFrame with signals and returns
            
        Returns:
            Estimated odds
        """
        if "return_forward_1" in df.columns:
            # Use actual returns if available
            returns = df["return_forward_1"]
            signals = df["signal"]
            
            # Calculate average return magnitude
            avg_return = returns.abs().mean()
            odds = 1 / avg_return if avg_return > 0 else 1.0
        else:
            # Use probability-based estimation
            odds = 1.0  # Default assumption
        
        return odds
    
    def optimize_thresholds(self, df: pd.DataFrame, metric: str = "sharpe") -> Dict[str, float]:
        """Optimize signal thresholds for given metric.
        
        Args:
            df: DataFrame with predictions and returns
            metric: Optimization metric
            
        Returns:
            Dictionary with optimized thresholds
        """
        logger.info(f"Optimizing thresholds for metric: {metric}")
        
        if "return_forward_1" not in df.columns:
            raise ValueError("DataFrame must contain 'return_forward_1' column")
        
        best_score = -np.inf
        best_thresholds = {}
        
        # Grid search over thresholds
        long_thresholds = np.arange(0.5, 0.9, 0.05)
        short_thresholds = np.arange(0.1, 0.5, 0.05)
        
        for long_thresh in long_thresholds:
            for short_thresh in short_thresholds:
                if long_thresh <= short_thresh:
                    continue
                
                # Generate signals with these thresholds
                temp_config = SignalConfig(
                    long_threshold=long_thresh,
                    short_threshold=short_thresh,
                    position_sizing=self.config.position_sizing,
                    position_size=self.config.position_size
                )
                
                temp_generator = SignalGenerator(temp_config)
                temp_signals = temp_generator.generate_signals(df)
                
                # Calculate metric
                score = self._calculate_metric(temp_signals, metric)
                
                if score > best_score:
                    best_score = score
                    best_thresholds = {
                        "long_threshold": long_thresh,
                        "short_threshold": short_thresh,
                        "score": score
                    }
        
        logger.info(f"Optimized thresholds: {best_thresholds}")
        return best_thresholds
    
    def _calculate_metric(self, df: pd.DataFrame, metric: str) -> float:
        """Calculate specified metric.
        
        Args:
            df: DataFrame with signals and returns
            metric: Metric name
            
        Returns:
            Metric value
        """
        if metric == "sharpe":
            returns = df["signal"] * df["return_forward_1"]
            return returns.mean() / returns.std() if returns.std() > 0 else 0
        
        elif metric == "sortino":
            returns = df["signal"] * df["return_forward_1"]
            downside_returns = returns[returns < 0]
            return returns.mean() / downside_returns.std() if downside_returns.std() > 0 else 0
        
        elif metric == "hit_rate":
            returns = df["return_forward_1"]
            signals = df["signal"]
            wins = ((signals > 0) & (returns > 0)) | ((signals < 0) & (returns < 0))
            return wins.mean()
        
        else:
            raise ValueError(f"Unknown metric: {metric}")



