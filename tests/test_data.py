"""Tests for data module.

Basic tests for data loading and processing.
Some tests might fail with different data - still debugging.
"""

import pytest
import pandas as pd
import numpy as np

from ml_microstructure.data import OrderBookSnapshot, OrderBookProcessor, SyntheticLOBGenerator, LOBSTERLoader, KaggleCryptoLoader


class TestOrderBookSnapshot:
    """Test OrderBookSnapshot - basic tests."""
    
    def test_order_book_snapshot_creation(self):
        """Test snapshot creation - simple test."""
        snapshot = OrderBookSnapshot(
            timestamp=pd.Timestamp.now(),
            bid_prices=[100.0, 99.99, 99.98],
            bid_sizes=[100, 200, 300],
            ask_prices=[100.01, 100.02, 100.03],
            ask_sizes=[150, 250, 350]
        )
        
        assert snapshot.mid_price == 100.005
        assert abs(snapshot.spread - 0.01) < 1e-10
        assert abs(snapshot.imbalance + 0.111) < 0.01  # Approximate test
    
    def test_order_book_snapshot_empty(self):
        """Test OrderBookSnapshot with empty sizes."""
        snapshot = OrderBookSnapshot(
            timestamp=pd.Timestamp.now(),
            bid_prices=[100.0],
            bid_sizes=[0],
            ask_prices=[100.01],
            ask_sizes=[0]
        )
        
        assert snapshot.imbalance == 0.0


class TestOrderBookProcessor:
    """Test OrderBookProcessor class."""
    
    def test_process_snapshots(self):
        """Test processing snapshots."""
        snapshots = [
            OrderBookSnapshot(
                timestamp=pd.Timestamp.now(),
                bid_prices=[100.0, 99.99],
                bid_sizes=[100, 200],
                ask_prices=[100.01, 100.02],
                ask_sizes=[150, 250]
            ),
            OrderBookSnapshot(
                timestamp=pd.Timestamp.now() + pd.Timedelta(seconds=1),
                bid_prices=[100.0, 99.99],
                bid_sizes=[120, 220],
                ask_prices=[100.01, 100.02],
                ask_sizes=[170, 270]
            )
        ]
        
        processor = OrderBookProcessor(max_levels=2)
        df = processor.process_snapshots(snapshots)
        
        assert len(df) == 2
        assert "mid_price" in df.columns
        assert "spread" in df.columns
        assert "imbalance" in df.columns
        assert "bid_price_1" in df.columns
        assert "ask_price_1" in df.columns


class TestSyntheticLOBGenerator:
    """Test SyntheticLOBGenerator class."""
    
    def test_generate_data(self):
        """Test synthetic data generation."""
        generator = SyntheticLOBGenerator(
            initial_price=100.0,
            tick_size=0.01,
            max_levels=5,
            arrival_rate=10.0,
            duration_seconds=10
        )
        
        snapshots = generator.generate_data()
        
        assert len(snapshots) > 0
        assert all(isinstance(s, OrderBookSnapshot) for s in snapshots)
        assert all(len(s.bid_prices) == 5 for s in snapshots)
        assert all(len(s.ask_prices) == 5 for s in snapshots)
    
    def test_generate_data_reproducible(self):
        """Test that synthetic data generation is reproducible."""
        generator1 = SyntheticLOBGenerator(
            initial_price=100.0,
            tick_size=0.01,
            max_levels=5,
            arrival_rate=10.0,
            duration_seconds=10
        )
        
        generator2 = SyntheticLOBGenerator(
            initial_price=100.0,
            tick_size=0.01,
            max_levels=5,
            arrival_rate=10.0,
            duration_seconds=10
        )
        
        snapshots1 = generator1.generate_data()
        snapshots2 = generator2.generate_data()
        
        # Should be identical due to fixed random seed
        assert len(snapshots1) == len(snapshots2)
        assert snapshots1[0].bid_prices[0] == snapshots2[0].bid_prices[0]


class TestLOBSTERLoader:
    """Test LOBSTERLoader class."""
    
    def test_lobster_loader_creation(self):
        """Test LOBSTERLoader creation."""
        loader = LOBSTERLoader(symbol="AAPL", date="2020-01-02")
        
        assert loader.symbol == "AAPL"
        assert loader.date == "2020-01-02"
    
    def test_lobster_loader_file_not_found(self):
        """Test LOBSTERLoader with non-existent file."""
        loader = LOBSTERLoader(symbol="AAPL", date="2020-01-02")
        
        with pytest.raises(FileNotFoundError):
            loader.load_data("non_existent_file.csv")


class TestKaggleCryptoLoader:
    """Test KaggleCryptoLoader class."""
    
    def test_kaggle_crypto_loader_creation(self):
        """Test KaggleCryptoLoader creation."""
        loader = KaggleCryptoLoader(symbol="BTC-USD")
        
        assert loader.symbol == "BTC-USD"
    
    def test_kaggle_crypto_loader_file_not_found(self):
        """Test KaggleCryptoLoader with non-existent file."""
        loader = KaggleCryptoLoader(symbol="BTC-USD")
        
        with pytest.raises(FileNotFoundError):
            loader.load_data("non_existent_file.csv")



