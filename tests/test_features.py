"""Tests for features module."""

import pytest
import pandas as pd
import numpy as np

from ml_microstructure.features import (
    OrderFlowImbalanceExtractor,
    SpreadExtractor,
    DepthExtractor,
    ImbalanceExtractor,
    VWAPExtractor,
    RollingReturnsExtractor,
    MicropriceExtractor,
    FeaturePipeline,
)


class TestOrderFlowImbalanceExtractor:
    """Test OrderFlowImbalanceExtractor class."""
    
    def test_extract_ofi_features(self):
        """Test OFI feature extraction."""
        # Create test data
        df = pd.DataFrame({
            'bid_size_1': [100, 120, 110],
            'bid_size_2': [200, 220, 210],
            'ask_size_1': [150, 170, 160],
            'ask_size_2': [250, 270, 260],
        })
        
        extractor = OrderFlowImbalanceExtractor(levels=2, window=2)
        result = extractor.extract(df)
        
        assert 'ofi_level_1' in result.columns
        assert 'ofi_level_2' in result.columns
        assert 'ofi_aggregate' in result.columns
        
        # Check OFI calculation
        expected_ofi_1 = df['bid_size_1'] - df['ask_size_1']
        assert np.allclose(result['ofi_level_1'], expected_ofi_1)
    
    def test_extract_ofi_with_insufficient_levels(self):
        """Test OFI extraction with insufficient levels."""
        df = pd.DataFrame({
            'bid_size_1': [100, 120],
            'ask_size_1': [150, 170],
        })
        
        extractor = OrderFlowImbalanceExtractor(levels=5, window=2)
        result = extractor.extract(df)
        
        assert 'ofi_level_1' in result.columns
        assert 'ofi_aggregate' in result.columns


class TestSpreadExtractor:
    """Test SpreadExtractor class."""
    
    def test_extract_spread_features(self):
        """Test spread feature extraction."""
        df = pd.DataFrame({
            'spread': [0.01, 0.02, 0.015],
            'mid_price': [100.0, 101.0, 100.5],
        })
        
        extractor = SpreadExtractor(window=2)
        result = extractor.extract(df)
        
        assert 'spread_log' in result.columns
        assert 'spread_rolling_mean' in result.columns
        assert 'spread_relative' in result.columns
        
        # Check spread log calculation
        expected_spread_log = np.log(df['spread'] + 1e-8)
        assert np.allclose(result['spread_log'], expected_spread_log)
    
    def test_extract_spread_without_mid_price(self):
        """Test spread extraction without mid price."""
        df = pd.DataFrame({
            'spread': [0.01, 0.02, 0.015],
        })
        
        extractor = SpreadExtractor(window=2)
        result = extractor.extract(df)
        
        assert 'spread_log' in result.columns
        assert 'spread_rolling_mean' in result.columns
        assert 'spread_relative' not in result.columns


class TestDepthExtractor:
    """Test DepthExtractor class."""
    
    def test_extract_depth_features(self):
        """Test depth feature extraction."""
        df = pd.DataFrame({
            'bid_size_1': [100, 120, 110],
            'bid_size_2': [200, 220, 210],
            'ask_size_1': [150, 170, 160],
            'ask_size_2': [250, 270, 260],
        })
        
        extractor = DepthExtractor(levels=2, window=2)
        result = extractor.extract(df)
        
        assert 'total_bid_depth' in result.columns
        assert 'total_ask_depth' in result.columns
        assert 'total_depth' in result.columns
        assert 'depth_imbalance' in result.columns
        
        # Check depth calculation
        expected_bid_depth = df['bid_size_1'] + df['bid_size_2']
        expected_ask_depth = df['ask_size_1'] + df['ask_size_2']
        
        assert np.allclose(result['total_bid_depth'], expected_bid_depth)
        assert np.allclose(result['total_ask_depth'], expected_ask_depth)


class TestImbalanceExtractor:
    """Test ImbalanceExtractor class."""
    
    def test_extract_imbalance_features(self):
        """Test imbalance feature extraction."""
        df = pd.DataFrame({
            'bid_size_1': [100, 120, 110],
            'bid_size_2': [200, 220, 210],
            'ask_size_1': [150, 170, 160],
            'ask_size_2': [250, 270, 260],
        })
        
        extractor = ImbalanceExtractor(levels=2, window=2)
        result = extractor.extract(df)
        
        assert 'imbalance_level_1' in result.columns
        assert 'imbalance_level_2' in result.columns
        assert 'weighted_imbalance' in result.columns
        
        # Check imbalance calculation
        total_size_1 = df['bid_size_1'] + df['ask_size_1']
        expected_imbalance_1 = (df['bid_size_1'] - df['ask_size_1']) / (total_size_1 + 1e-8)
        
        assert np.allclose(result['imbalance_level_1'], expected_imbalance_1)


class TestVWAPExtractor:
    """Test VWAPExtractor class."""
    
    def test_extract_vwap_features(self):
        """Test VWAP feature extraction."""
        df = pd.DataFrame({
            'mid_price': [100.0, 101.0, 100.5],
            'total_depth': [1000, 1200, 1100],
        })
        
        extractor = VWAPExtractor(window=2)
        result = extractor.extract(df)
        
        assert 'vwap' in result.columns
        assert 'vwap_rolling' in result.columns
        assert 'vwap_deviation' in result.columns


class TestRollingReturnsExtractor:
    """Test RollingReturnsExtractor class."""
    
    def test_extract_rolling_returns_features(self):
        """Test rolling returns feature extraction."""
        df = pd.DataFrame({
            'mid_price': [100.0, 101.0, 100.5, 102.0, 101.5],
        })
        
        extractor = RollingReturnsExtractor(windows=[1, 2])
        result = extractor.extract(df)
        
        assert 'return_forward_1' in result.columns
        assert 'return_backward_1' in result.columns
        assert 'return_backward_2' in result.columns
        assert 'volatility_1' in result.columns
        assert 'volatility_2' in result.columns
        
        # Check return calculation
        expected_return_1 = df['mid_price'].pct_change()
        assert np.allclose(result['return_backward_1'], expected_return_1, equal_nan=True)


class TestMicropriceExtractor:
    """Test MicropriceExtractor class."""
    
    def test_extract_microprice_features(self):
        """Test microprice feature extraction."""
        df = pd.DataFrame({
            'bid_price_1': [100.0, 101.0, 100.5],
            'ask_price_1': [100.01, 101.01, 100.51],
            'bid_size_1': [100, 120, 110],
            'ask_size_1': [150, 170, 160],
            'mid_price': [100.005, 101.005, 100.505],
        })
        
        extractor = MicropriceExtractor(window=2)
        result = extractor.extract(df)
        
        assert 'microprice' in result.columns
        assert 'microprice_rolling' in result.columns
        assert 'microprice_deviation' in result.columns
        
        # Check microprice calculation
        total_size = df['bid_size_1'] + df['ask_size_1']
        expected_microprice = (
            df['bid_price_1'] * df['ask_size_1'] + 
            df['ask_price_1'] * df['bid_size_1']
        ) / (total_size + 1e-8)
        
        assert np.allclose(result['microprice'], expected_microprice)


class TestFeaturePipeline:
    """Test FeaturePipeline class."""
    
    def test_extract_features(self):
        """Test feature pipeline extraction."""
        # Create test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='1S'),
            'mid_price': np.random.uniform(100, 101, 10),
            'spread': np.random.uniform(0.01, 0.02, 10),
            'bid_price_1': np.random.uniform(99.9, 100, 10),
            'ask_price_1': np.random.uniform(100, 100.1, 10),
            'bid_size_1': np.random.uniform(100, 200, 10),
            'ask_size_1': np.random.uniform(100, 200, 10),
        })
        
        pipeline = FeaturePipeline()
        result = pipeline.extract_features(df)
        
        assert len(result) == len(df)
        assert len(result.columns) > len(df.columns)
        assert 'timestamp' in result.columns
        assert 'mid_price' in result.columns
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        pipeline = FeaturePipeline()
        feature_names = pipeline.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'mid_price' in feature_names
        assert 'spread' in feature_names



