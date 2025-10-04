"""Tests for backtest module."""

import pytest
import pandas as pd
import numpy as np

from ml_microstructure.backtest import (
    SignalConfig,
    SignalGenerator,
    ExecutionConfig,
    ExecutionEngine,
    MetricsConfig,
    BacktestMetrics,
)


class TestSignalConfig:
    """Test SignalConfig class."""
    
    def test_signal_config_creation(self):
        """Test SignalConfig creation."""
        config = SignalConfig(
            long_threshold=0.6,
            short_threshold=0.4,
            position_sizing="fixed",
            position_size=1.0
        )
        
        assert config.long_threshold == 0.6
        assert config.short_threshold == 0.4
        assert config.position_sizing == "fixed"
        assert config.position_size == 1.0
    
    def test_signal_config_defaults(self):
        """Test SignalConfig with defaults."""
        config = SignalConfig()
        
        assert config.long_threshold == 0.6
        assert config.short_threshold == 0.4
        assert config.position_sizing == "fixed"
        assert config.position_size == 1.0


class TestSignalGenerator:
    """Test SignalGenerator class."""
    
    def test_signal_generator_creation(self):
        """Test SignalGenerator creation."""
        config = SignalConfig()
        generator = SignalGenerator(config)
        
        assert generator.config == config
    
    def test_generate_ternary_signals(self):
        """Test ternary signal generation."""
        df = pd.DataFrame({
            'prob_up': [0.7, 0.3, 0.5, 0.8, 0.2],
            'prob_down': [0.2, 0.6, 0.3, 0.1, 0.7],
            'prob_flat': [0.1, 0.1, 0.2, 0.1, 0.1],
        })
        
        config = SignalConfig(long_threshold=0.6, short_threshold=0.4)
        generator = SignalGenerator(config)
        
        signals = generator.generate_signals(df)
        
        assert 'signal' in signals.columns
        assert 'position_size' in signals.columns
        
        # Check signal values
        assert signals['signal'].isin([-1, 0, 1]).all()
        
        # Check that high prob_up generates long signals
        assert signals.loc[signals['prob_up'] > 0.6, 'signal'].eq(1).all()
        
        # Check that high prob_down generates short signals
        assert signals.loc[signals['prob_down'] > 0.4, 'signal'].eq(-1).all()
    
    def test_generate_binary_signals(self):
        """Test binary signal generation."""
        df = pd.DataFrame({
            'prob_up': [0.7, 0.3, 0.5, 0.8, 0.2],
        })
        
        config = SignalConfig(long_threshold=0.6, short_threshold=0.4)
        generator = SignalGenerator(config)
        
        signals = generator.generate_signals(df)
        
        assert 'signal' in signals.columns
        assert 'position_size' in signals.columns
        
        # Check signal values
        assert signals['signal'].isin([-1, 0, 1]).all()
        
        # Check that high prob_up generates long signals
        assert signals.loc[signals['prob_up'] > 0.6, 'signal'].eq(1).all()
        
        # Check that low prob_up generates short signals
        assert signals.loc[signals['prob_up'] < 0.4, 'signal'].eq(-1).all()
    
    def test_calculate_position_sizes_fixed(self):
        """Test fixed position sizing."""
        df = pd.DataFrame({
            'signal': [1, -1, 0, 1, -1],
        })
        
        config = SignalConfig(position_sizing="fixed", position_size=2.0)
        generator = SignalGenerator(config)
        
        signals = generator.generate_signals(df)
        
        # Check position sizes
        assert signals.loc[signals['signal'] != 0, 'position_size'].eq(2.0).all()
        assert signals.loc[signals['signal'] == 0, 'position_size'].eq(0.0).all()
    
    def test_optimize_thresholds(self):
        """Test threshold optimization."""
        np.random.seed(42)
        df = pd.DataFrame({
            'prob_up': np.random.uniform(0, 1, 100),
            'prob_down': np.random.uniform(0, 1, 100),
            'return_forward_1': np.random.normal(0, 0.01, 100),
        })
        
        config = SignalConfig()
        generator = SignalGenerator(config)
        
        optimized = generator.optimize_thresholds(df, metric="sharpe")
        
        assert "long_threshold" in optimized
        assert "short_threshold" in optimized
        assert "score" in optimized
        assert optimized["long_threshold"] > optimized["short_threshold"]


class TestExecutionConfig:
    """Test ExecutionConfig class."""
    
    def test_execution_config_creation(self):
        """Test ExecutionConfig creation."""
        config = ExecutionConfig(
            transaction_cost=0.001,
            slippage=0.0005,
            max_position=10.0
        )
        
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005
        assert config.max_position == 10.0
    
    def test_execution_config_defaults(self):
        """Test ExecutionConfig with defaults."""
        config = ExecutionConfig()
        
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005
        assert config.max_position == 10.0


class TestExecutionEngine:
    """Test ExecutionEngine class."""
    
    def test_execution_engine_creation(self):
        """Test ExecutionEngine creation."""
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        assert engine.config == config
        assert engine.current_position == 0.0
        assert len(engine.trades) == 0
    
    def test_execute_trades(self):
        """Test trade execution."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=5, freq='1S'),
            'signal': [1, 0, -1, 1, 0],
            'position_size': [1.0, 0.0, 1.0, 1.0, 0.0],
            'mid_price': [100.0, 101.0, 102.0, 103.0, 104.0],
        })
        
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        result = engine.execute_trades(df)
        
        assert 'position' in result.columns
        assert 'trade_size' in result.columns
        assert 'transaction_cost' in result.columns
        assert 'slippage_cost' in result.columns
        assert 'total_cost' in result.columns
        assert 'pnl' in result.columns
        assert 'cumulative_pnl' in result.columns
        
        # Check that positions are updated
        assert result['position'].iloc[-1] != 0.0
    
    def test_apply_constraints(self):
        """Test position constraints."""
        config = ExecutionConfig(max_position=5.0)
        engine = ExecutionEngine(config)
        
        # Test max position constraint
        trade_size = engine._apply_constraints(10.0, 100.0)
        assert trade_size <= 5.0
        
        # Test min position constraint
        trade_size = engine._apply_constraints(-10.0, 100.0)
        assert trade_size >= -5.0
    
    def test_calculate_trade_cost(self):
        """Test trade cost calculation."""
        config = ExecutionConfig(transaction_cost=0.001, slippage=0.0005)
        engine = ExecutionEngine(config)
        
        cost = engine._calculate_trade_cost(1.0, 100.0)
        
        assert "transaction" in cost
        assert "slippage" in cost
        assert "total" in cost
        assert cost["total"] == cost["transaction"] + cost["slippage"]
    
    def test_get_trade_summary(self):
        """Test trade summary."""
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        # Execute some trades
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=3, freq='1S'),
            'signal': [1, -1, 1],
            'position_size': [1.0, 1.0, 1.0],
            'mid_price': [100.0, 101.0, 102.0],
        })
        
        engine.execute_trades(df)
        summary = engine.get_trade_summary()
        
        assert "total_trades" in summary
        assert "total_volume" in summary
        assert "total_cost" in summary
        assert summary["total_trades"] > 0
    
    def test_reset(self):
        """Test engine reset."""
        config = ExecutionConfig()
        engine = ExecutionEngine(config)
        
        # Execute some trades
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=2, freq='1S'),
            'signal': [1, -1],
            'position_size': [1.0, 1.0],
            'mid_price': [100.0, 101.0],
        })
        
        engine.execute_trades(df)
        
        # Reset
        engine.reset()
        
        assert engine.current_position == 0.0
        assert len(engine.trades) == 0


class TestMetricsConfig:
    """Test MetricsConfig class."""
    
    def test_metrics_config_creation(self):
        """Test MetricsConfig creation."""
        config = MetricsConfig(
            risk_free_rate=0.02,
            benchmark_return=0.0,
            trading_days=252
        )
        
        assert config.risk_free_rate == 0.02
        assert config.benchmark_return == 0.0
        assert config.trading_days == 252
    
    def test_metrics_config_defaults(self):
        """Test MetricsConfig with defaults."""
        config = MetricsConfig()
        
        assert config.risk_free_rate == 0.02
        assert config.benchmark_return == 0.0
        assert config.trading_days == 252


class TestBacktestMetrics:
    """Test BacktestMetrics class."""
    
    def test_backtest_metrics_creation(self):
        """Test BacktestMetrics creation."""
        config = MetricsConfig()
        metrics = BacktestMetrics(config)
        
        assert metrics.config == config
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        df = pd.DataFrame({
            'cumulative_pnl': [0, 10, 5, 15, 20],
            'pnl': [0, 10, -5, 10, 5],
            'trade_size': [0, 1, -1, 1, 0],
        })
        
        config = MetricsConfig()
        metrics = BacktestMetrics(config)
        
        result = metrics.calculate_metrics(df)
        
        assert "total_pnl" in result
        assert "total_trades" in result
        assert "volatility" in result
        assert "max_drawdown" in result
        assert "sharpe_ratio" in result
        assert "hit_rate" in result
        
        # Check total PnL
        assert result["total_pnl"] == 20.0
    
    def test_calculate_rolling_metrics(self):
        """Test rolling metrics calculation."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='1D'),
            'cumulative_pnl': np.cumsum(np.random.randn(10)),
            'pnl': np.random.randn(10),
        })
        
        config = MetricsConfig()
        metrics = BacktestMetrics(config)
        
        result = metrics.calculate_rolling_metrics(df, window=5)
        
        assert 'rolling_return' in result.columns
        assert 'rolling_volatility' in result.columns
        assert 'rolling_sharpe' in result.columns
        assert 'rolling_drawdown' in result.columns
    
    def test_generate_report(self):
        """Test report generation."""
        metrics = {
            "total_pnl": 100.0,
            "total_trades": 50,
            "sharpe_ratio": 1.5,
            "max_drawdown": -10.0,
            "hit_rate": 0.6,
        }
        
        config = MetricsConfig()
        metrics_calculator = BacktestMetrics(config)
        
        report = metrics_calculator.generate_report(metrics)
        
        assert isinstance(report, str)
        assert "BACKTESTING PERFORMANCE REPORT" in report
        assert "100.00" in report  # Total PnL
        assert "1.50" in report  # Sharpe ratio



