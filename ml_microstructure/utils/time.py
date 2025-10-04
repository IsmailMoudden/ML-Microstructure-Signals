"""Time utilities for microstructure signals."""

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TimeUtils:
    """Utilities for time-based operations."""
    
    @staticmethod
    def resample_data(df: pd.DataFrame, freq: str = "1S") -> pd.DataFrame:
        """Resample data to specified frequency.
        
        Args:
            df: DataFrame with timestamp index
            freq: Resampling frequency
            
        Returns:
            Resampled DataFrame
        """
        logger.info(f"Resampling data to {freq} frequency")
        
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        # Set timestamp as index
        df_indexed = df.set_index("timestamp")
        
        # Resample
        df_resampled = df_indexed.resample(freq).last()
        
        # Reset index
        df_resampled = df_resampled.reset_index()
        
        logger.info(f"Resampled from {len(df)} to {len(df_resampled)} data points")
        return df_resampled
    
    @staticmethod
    def create_time_windows(df: pd.DataFrame, window_size: int, step_size: int = 1) -> List[pd.DataFrame]:
        """Create sliding time windows from data.
        
        Args:
            df: DataFrame with timestamp index
            window_size: Size of each window
            step_size: Step size between windows
            
        Returns:
            List of DataFrames for each window
        """
        logger.info(f"Creating time windows of size {window_size} with step {step_size}")
        
        windows = []
        
        for i in range(0, len(df) - window_size + 1, step_size):
            window = df.iloc[i:i + window_size].copy()
            windows.append(window)
        
        logger.info(f"Created {len(windows)} time windows")
        return windows
    
    @staticmethod
    def filter_by_time(df: pd.DataFrame, start_time: Optional[str] = None, 
                      end_time: Optional[str] = None) -> pd.DataFrame:
        """Filter data by time range.
        
        Args:
            df: DataFrame with timestamp column
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            
        Returns:
            Filtered DataFrame
        """
        logger.info(f"Filtering data from {start_time} to {end_time}")
        
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        df_filtered = df.copy()
        
        if start_time:
            df_filtered = df_filtered[df_filtered["timestamp"] >= pd.Timestamp(start_time)]
        
        if end_time:
            df_filtered = df_filtered[df_filtered["timestamp"] <= pd.Timestamp(end_time)]
        
        logger.info(f"Filtered from {len(df)} to {len(df_filtered)} data points")
        return df_filtered


# Create instance for easy access
time_utils = TimeUtils()



