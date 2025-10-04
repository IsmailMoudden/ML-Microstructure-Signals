"""Feature extractors for microstructure signals.

Extracts various features from order book data.
Some extractors are experimental - results may vary.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Base class for feature extractors - simple approach."""
    
    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from data."""
        pass


class OrderFlowImbalanceExtractor(BaseExtractor):
    """Extract OFI features - not sure if this is the best approach."""
    
    def __init__(self, levels: int = 5, window: int = 10) -> None:
        """Initialize OFI extractor."""
        self.levels = levels
        self.window = window
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract OFI features - basic implementation."""
        logger.info("Extracting Order Flow Imbalance features")
        
        result_df = df.copy()
        
        # Calculate OFI for each level
        for level in range(1, self.levels + 1):
            bid_col = f"bid_size_{level}"
            ask_col = f"ask_size_{level}"
            
            if bid_col in df.columns and ask_col in df.columns:
                # OFI = bid_size - ask_size - simple formula
                ofi = df[bid_col] - df[ask_col]
                result_df[f"ofi_level_{level}"] = ofi
                
                # Rolling OFI
                result_df[f"ofi_level_{level}_rolling"] = ofi.rolling(self.window).mean()
                
                # OFI momentum
                result_df[f"ofi_level_{level}_momentum"] = ofi.diff()
        
        # Aggregate OFI across levels
        ofi_cols = [f"ofi_level_{i}" for i in range(1, self.levels + 1) if f"ofi_level_{i}" in result_df.columns]
        if ofi_cols:
            result_df["ofi_aggregate"] = result_df[ofi_cols].sum(axis=1)
            result_df["ofi_aggregate_rolling"] = result_df["ofi_aggregate"].rolling(self.window).mean()
        
        return result_df


class SpreadExtractor(BaseExtractor):
    """Extract spread-related features."""
    
    def __init__(self, window: int = 20) -> None:
        """Initialize spread extractor.
        
        Args:
            window: Rolling window for spread calculations
        """
        self.window = window
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract spread features.
        
        Args:
            df: DataFrame with order book data
            
        Returns:
            DataFrame with spread features
        """
        logger.info("Extracting spread features")
        
        result_df = df.copy()
        
        if "spread" in df.columns:
            # Basic spread features
            result_df["spread_log"] = np.log(df["spread"] + 1e-8)
            result_df["spread_rolling_mean"] = df["spread"].rolling(self.window).mean()
            result_df["spread_rolling_std"] = df["spread"].rolling(self.window).std()
            
            # Spread relative to mid price
            if "mid_price" in df.columns:
                result_df["spread_relative"] = df["spread"] / df["mid_price"]
                result_df["spread_relative_rolling"] = result_df["spread_relative"].rolling(self.window).mean()
            
            # Spread momentum
            result_df["spread_momentum"] = df["spread"].diff()
            result_df["spread_momentum_rolling"] = result_df["spread_momentum"].rolling(self.window).mean()
        
        return result_df


class DepthExtractor(BaseExtractor):
    """Extract depth-related features."""
    
    def __init__(self, levels: int = 5, window: int = 10) -> None:
        """Initialize depth extractor.
        
        Args:
            levels: Number of price levels to consider
            window: Rolling window for depth calculations
        """
        self.levels = levels
        self.window = window
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract depth features.
        
        Args:
            df: DataFrame with order book data
            
        Returns:
            DataFrame with depth features
        """
        logger.info("Extracting depth features")
        
        result_df = df.copy()
        
        # Aggregate depth features
        bid_size_cols = [f"bid_size_{i}" for i in range(1, self.levels + 1) if f"bid_size_{i}" in df.columns]
        ask_size_cols = [f"ask_size_{i}" for i in range(1, self.levels + 1) if f"ask_size_{i}" in df.columns]
        
        if bid_size_cols and ask_size_cols:
            # Total bid/ask depth
            result_df["total_bid_depth"] = df[bid_size_cols].sum(axis=1)
            result_df["total_ask_depth"] = df[ask_size_cols].sum(axis=1)
            result_df["total_depth"] = result_df["total_bid_depth"] + result_df["total_ask_depth"]
            
            # Depth imbalance
            result_df["depth_imbalance"] = (result_df["total_bid_depth"] - result_df["total_ask_depth"]) / result_df["total_depth"]
            
            # Rolling depth features
            result_df["total_depth_rolling"] = result_df["total_depth"].rolling(self.window).mean()
            result_df["depth_imbalance_rolling"] = result_df["depth_imbalance"].rolling(self.window).mean()
            
            # Depth momentum
            result_df["total_depth_momentum"] = result_df["total_depth"].diff()
            result_df["depth_imbalance_momentum"] = result_df["depth_imbalance"].diff()
        
        return result_df


class ImbalanceExtractor(BaseExtractor):
    """Extract order book imbalance features."""
    
    def __init__(self, levels: int = 5, window: int = 10) -> None:
        """Initialize imbalance extractor.
        
        Args:
            levels: Number of price levels to consider
            window: Rolling window for imbalance calculations
        """
        self.levels = levels
        self.window = window
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract imbalance features.
        
        Args:
            df: DataFrame with order book data
            
        Returns:
            DataFrame with imbalance features
        """
        logger.info("Extracting imbalance features")
        
        result_df = df.copy()
        
        # Level-specific imbalances
        for level in range(1, self.levels + 1):
            bid_col = f"bid_size_{level}"
            ask_col = f"ask_size_{level}"
            
            if bid_col in df.columns and ask_col in df.columns:
                total_size = df[bid_col] + df[ask_col]
                imbalance = (df[bid_col] - df[ask_col]) / (total_size + 1e-8)
                result_df[f"imbalance_level_{level}"] = imbalance
                
                # Rolling imbalance
                result_df[f"imbalance_level_{level}_rolling"] = imbalance.rolling(self.window).mean()
        
        # Weighted imbalance (closer to mid price gets higher weight)
        imbalance_cols = [f"imbalance_level_{i}" for i in range(1, self.levels + 1) if f"imbalance_level_{i}" in result_df.columns]
        if imbalance_cols:
            weights = np.array([1.0 / i for i in range(1, len(imbalance_cols) + 1)])
            weights = weights / weights.sum()
            
            weighted_imbalance = np.zeros(len(result_df))
            for i, col in enumerate(imbalance_cols):
                weighted_imbalance += weights[i] * result_df[col].fillna(0)
            
            result_df["weighted_imbalance"] = weighted_imbalance
            result_df["weighted_imbalance_rolling"] = result_df["weighted_imbalance"].rolling(self.window).mean()
        
        return result_df


class VWAPExtractor(BaseExtractor):
    """Extract Volume Weighted Average Price (VWAP) features."""
    
    def __init__(self, window: int = 20) -> None:
        """Initialize VWAP extractor.
        
        Args:
            window: Rolling window for VWAP calculation
        """
        self.window = window
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract VWAP features.
        
        Args:
            df: DataFrame with order book data
            
        Returns:
            DataFrame with VWAP features
        """
        logger.info("Extracting VWAP features")
        
        result_df = df.copy()
        
        if "mid_price" in df.columns:
            # Simple VWAP approximation using mid price and total depth
            if "total_depth" in df.columns:
                # VWAP = sum(price * volume) / sum(volume)
                result_df["vwap"] = df["mid_price"]  # Simplified for now
                result_df["vwap_rolling"] = result_df["vwap"].rolling(self.window).mean()
                
                # VWAP deviation
                result_df["vwap_deviation"] = df["mid_price"] - result_df["vwap_rolling"]
                result_df["vwap_deviation_abs"] = np.abs(result_df["vwap_deviation"])
                
                # VWAP momentum
                result_df["vwap_momentum"] = result_df["vwap"].diff()
        
        return result_df


class RollingReturnsExtractor(BaseExtractor):
    """Extract rolling returns and volatility features."""
    
    def __init__(self, windows: List[int] = None) -> None:
        """Initialize rolling returns extractor.
        
        Args:
            windows: List of rolling windows for returns calculation
        """
        self.windows = windows or [1, 5, 10, 20]
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract rolling returns features.
        
        Args:
            df: DataFrame with order book data
            
        Returns:
            DataFrame with rolling returns features
        """
        logger.info("Extracting rolling returns features")
        
        result_df = df.copy()
        
        if "mid_price" in df.columns:
            # Calculate returns for different horizons
            for window in self.windows:
                # Forward returns (for labeling)
                result_df[f"return_forward_{window}"] = df["mid_price"].shift(-window) / df["mid_price"] - 1
                
                # Backward returns (for features)
                result_df[f"return_backward_{window}"] = df["mid_price"] / df["mid_price"].shift(window) - 1
                
                # Rolling volatility
                result_df[f"volatility_{window}"] = df["mid_price"].pct_change().rolling(window).std()
                
                # Rolling skewness and kurtosis
                returns = df["mid_price"].pct_change()
                result_df[f"skewness_{window}"] = returns.rolling(window).skew()
                result_df[f"kurtosis_{window}"] = returns.rolling(window).kurt()
        
        return result_df


class MicropriceExtractor(BaseExtractor):
    """Extract microprice features."""
    
    def __init__(self, window: int = 10) -> None:
        """Initialize microprice extractor.
        
        Args:
            window: Rolling window for microprice calculations
        """
        self.window = window
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract microprice features.
        
        Args:
            df: DataFrame with order book data
            
        Returns:
            DataFrame with microprice features
        """
        logger.info("Extracting microprice features")
        
        result_df = df.copy()
        
        if "bid_size_1" in df.columns and "ask_size_1" in df.columns:
            # Microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
            if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
                bid_price = df["bid_price_1"]
                ask_price = df["ask_price_1"]
                bid_size = df["bid_size_1"]
                ask_size = df["ask_size_1"]
                
                total_size = bid_size + ask_size
                microprice = (bid_price * ask_size + ask_price * bid_size) / (total_size + 1e-8)
                
                result_df["microprice"] = microprice
                result_df["microprice_rolling"] = microprice.rolling(self.window).mean()
                
                # Microprice deviation from mid price
                if "mid_price" in df.columns:
                    result_df["microprice_deviation"] = microprice - df["mid_price"]
                    result_df["microprice_deviation_abs"] = np.abs(result_df["microprice_deviation"])
        
        return result_df


class FeaturePipeline:
    """Pipeline for extracting all features."""
    
    def __init__(self, extractors: List[BaseExtractor] = None) -> None:
        """Initialize feature pipeline.
        
        Args:
            extractors: List of feature extractors to use
        """
        self.extractors = extractors or self._get_default_extractors()
    
    def _get_default_extractors(self) -> List[BaseExtractor]:
        """Get default feature extractors."""
        return [
            OrderFlowImbalanceExtractor(),
            SpreadExtractor(),
            DepthExtractor(),
            ImbalanceExtractor(),
            VWAPExtractor(),
            RollingReturnsExtractor(),
            MicropriceExtractor(),
        ]
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features using the pipeline.
        
        Args:
            df: DataFrame with order book data
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info("Extracting features using pipeline")
        
        result_df = df.copy()
        
        for extractor in self.extractors:
            result_df = extractor.extract(result_df)
        
        # Remove any infinite or NaN values
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Extracted {len(result_df.columns)} features")
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be extracted.
        
        Returns:
            List of feature names
        """
        # This is a simplified version - in practice, you'd want to track
        # which features each extractor produces
        feature_names = []
        
        # Basic features
        feature_names.extend(["mid_price", "spread", "imbalance"])
        
        # Level-specific features
        for level in range(1, 11):
            feature_names.extend([
                f"bid_price_{level}", f"bid_size_{level}",
                f"ask_price_{level}", f"ask_size_{level}",
            ])
        
        # Derived features (simplified list)
        feature_names.extend([
            "ofi_level_1", "ofi_level_2", "ofi_level_3", "ofi_level_4", "ofi_level_5",
            "ofi_aggregate", "spread_log", "spread_relative", "total_depth",
            "depth_imbalance", "weighted_imbalance", "vwap", "microprice",
            "return_backward_1", "return_backward_5", "return_backward_10",
            "volatility_1", "volatility_5", "volatility_10",
        ])
        
        return feature_names
