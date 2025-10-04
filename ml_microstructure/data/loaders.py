"""Data loaders for order book data.

Handles loading and processing of order book data from different sources.
Some loaders might not work with all data formats - still testing.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class OrderBookSnapshot(BaseModel):
    """Order book snapshot - basic data structure."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timestamp: pd.Timestamp
    bid_prices: List[float] = Field(description="Bid prices")
    bid_sizes: List[float] = Field(description="Bid sizes")
    ask_prices: List[float] = Field(description="Ask prices")
    ask_sizes: List[float] = Field(description="Ask sizes")
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price simple average of best bid/ask."""
        return (self.bid_prices[0] + self.ask_prices[0]) / 2
    
    @property
    def spread(self) -> float:
        """Calculate spread  ask minus bid."""
        return self.ask_prices[0] - self.bid_prices[0]
    
    @property
    def imbalance(self) -> float:
        """Calculate imbalance - might need better formula later."""
        bid_volume = sum(self.bid_sizes)
        ask_volume = sum(self.ask_sizes)
        return (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.0


class OrderBookProcessor:
    """Processes order book snapshots."""
    
    def __init__(self, max_levels: int = 10) -> None:
        """Initialize processor."""
        self.max_levels = max_levels
    
    def process_snapshots(self, snapshots: List[OrderBookSnapshot]) -> pd.DataFrame:
        """Process snapshots into DataFrame."""
        data = []
        
        for snapshot in snapshots:
            row = {
                "timestamp": snapshot.timestamp,
                "mid_price": snapshot.mid_price,
                "spread": snapshot.spread,
                "imbalance": snapshot.imbalance,
            }
            
            # Add level-specific features
            for i in range(min(self.max_levels, len(snapshot.bid_prices))):
                row[f"bid_price_{i+1}"] = snapshot.bid_prices[i]
                row[f"bid_size_{i+1}"] = snapshot.bid_sizes[i]
                row[f"ask_price_{i+1}"] = snapshot.ask_prices[i]
                row[f"ask_size_{i+1}"] = snapshot.ask_sizes[i]
            
            data.append(row)
        
        return pd.DataFrame(data)


class BaseLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load_data(self, file_path: Union[str, Path]) -> List[OrderBookSnapshot]:
        """Load order book data from file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            List of order book snapshots
        """
        pass


class LOBSTERLoader(BaseLoader):
    """Loader for LOBSTER order book data."""
    
    def __init__(self, symbol: str = "AAPL", date: str = "2020-01-02") -> None:
        """Initialize LOBSTER loader.
        
        Args:
            symbol: Stock symbol
            date: Date in YYYY-MM-DD format
        """
        self.symbol = symbol
        self.date = date
    
    def load_data(self, file_path: Union[str, Path]) -> List[OrderBookSnapshot]:
        """Load LOBSTER order book data.
        
        Args:
            file_path: Path to LOBSTER order book file
            
        Returns:
            List of order book snapshots
        """
        logger.info(f"Loading LOBSTER data from {file_path}")
        
        # LOBSTER format: timestamp, bid_prices, bid_sizes, ask_prices, ask_sizes
        df = pd.read_csv(file_path, header=None)
        
        snapshots = []
        for _, row in df.iterrows():
            timestamp = pd.Timestamp(row[0], unit='ns')
            
            # Parse bid/ask data (LOBSTER format specific)
            bid_prices = [float(x) for x in row[1].split(',') if x]
            bid_sizes = [float(x) for x in row[2].split(',') if x]
            ask_prices = [float(x) for x in row[3].split(',') if x]
            ask_sizes = [float(x) for x in row[4].split(',') if x]
            
            snapshot = OrderBookSnapshot(
                timestamp=timestamp,
                bid_prices=bid_prices,
                bid_sizes=bid_sizes,
                ask_prices=ask_prices,
                ask_sizes=ask_sizes,
            )
            snapshots.append(snapshot)
        
        logger.info(f"Loaded {len(snapshots)} snapshots")
        return snapshots


class KaggleCryptoLoader(BaseLoader):
    """Loader for Kaggle cryptocurrency order book data."""
    
    def __init__(self, symbol: str = "BTC-USD") -> None:
        """Initialize Kaggle crypto loader.
        
        Args:
            symbol: Cryptocurrency symbol
        """
        self.symbol = symbol
    
    def load_data(self, file_path: Union[str, Path]) -> List[OrderBookSnapshot]:
        """Load Kaggle crypto order book data.
        
        Args:
            file_path: Path to crypto order book file
            
        Returns:
            List of order book snapshots
        """
        logger.info(f"Loading Kaggle crypto data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        snapshots = []
        for _, row in df.iterrows():
            timestamp = pd.Timestamp(row['timestamp'])
            
            # Kaggle crypto format: bid/ask prices and sizes as separate columns
            bid_prices = [row[f'bid_price_{i}'] for i in range(1, 11) if pd.notna(row[f'bid_price_{i}'])]
            bid_sizes = [row[f'bid_size_{i}'] for i in range(1, 11) if pd.notna(row[f'bid_size_{i}'])]
            ask_prices = [row[f'ask_price_{i}'] for i in range(1, 11) if pd.notna(row[f'ask_price_{i}'])]
            ask_sizes = [row[f'ask_size_{i}'] for i in range(1, 11) if pd.notna(row[f'ask_size_{i}'])]
            
            snapshot = OrderBookSnapshot(
                timestamp=timestamp,
                bid_prices=bid_prices,
                bid_sizes=bid_sizes,
                ask_prices=ask_prices,
                ask_sizes=ask_sizes,
            )
            snapshots.append(snapshot)
        
        logger.info(f"Loaded {len(snapshots)} snapshots")
        return snapshots


class SyntheticLOBGenerator:
    """Generates synthetic order book data using Poisson arrivals."""
    
    def __init__(
        self,
        initial_price: float = 100.0,
        tick_size: float = 0.01,
        max_levels: int = 10,
        arrival_rate: float = 100.0,  # orders per second
        duration_seconds: int = 3600,  # 1 hour
    ) -> None:
        """Initialize synthetic LOB generator.
        
        Args:
            initial_price: Starting mid price
            tick_size: Minimum price increment
            max_levels: Maximum number of price levels
            arrival_rate: Order arrival rate (Poisson parameter)
            duration_seconds: Duration of simulation in seconds
        """
        self.initial_price = initial_price
        self.tick_size = tick_size
        self.max_levels = max_levels
        self.arrival_rate = arrival_rate
        self.duration_seconds = duration_seconds
    
    def generate_data(self) -> List[OrderBookSnapshot]:
        """Generate synthetic order book data.
        
        Returns:
            List of synthetic order book snapshots
        """
        logger.info("Generating synthetic LOB data")
        
        # Initialize order book
        current_price = self.initial_price
        spread = self.tick_size * 2
        
        snapshots = []
        start_time = pd.Timestamp.now()
        
        # Generate Poisson arrivals
        np.random.seed(42)  # For reproducibility - TODO: make this configurable later
        inter_arrivals = np.random.exponential(1.0 / self.arrival_rate, self.duration_seconds * self.arrival_rate)
        timestamps = np.cumsum(inter_arrivals)
        
        for i, timestamp in enumerate(timestamps):
            if timestamp > self.duration_seconds:
                break
                
            current_time = start_time + pd.Timedelta(seconds=timestamp)
            
            # Generate synthetic bid/ask levels
            bid_prices = []
            bid_sizes = []
            ask_prices = []
            ask_sizes = []
            
            # Add some noise to mid price - this could be better but works for now
            price_noise = np.random.normal(0, self.tick_size * 0.1)
            current_price += price_noise
            
            for level in range(self.max_levels):
                # Bid side
                bid_price = current_price - spread/2 - level * self.tick_size
                bid_size = np.random.exponential(100)  # Exponential size distribution - not sure if realistic
                bid_prices.append(bid_price)
                bid_sizes.append(bid_size)
                
                # Ask side
                ask_price = current_price + spread/2 + level * self.tick_size
                ask_size = np.random.exponential(100)
                ask_prices.append(ask_price)
                ask_sizes.append(ask_size)
            
            snapshot = OrderBookSnapshot(
                timestamp=current_time,
                bid_prices=bid_prices,
                bid_sizes=bid_sizes,
                ask_prices=ask_prices,
                ask_sizes=ask_sizes,
            )
            snapshots.append(snapshot)
        
        logger.info(f"Generated {len(snapshots)} synthetic snapshots")
        return snapshots



