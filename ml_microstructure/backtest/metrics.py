"""Backtesting metrics calculation."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class MetricsConfig(BaseModel):
    """Configuration for metrics calculation."""
    
    risk_free_rate: float = Field(default=0.02, description="Risk-free rate (annual)")
    benchmark_return: float = Field(default=0.0, description="Benchmark return (annual)")
    trading_days: int = Field(default=252, description="Trading days per year")


class BacktestMetrics:
    """Calculate backtesting performance metrics."""
    
    def __init__(self, config: MetricsConfig) -> None:
        """Initialize metrics calculator.
        
        Args:
            config: Metrics configuration
        """
        self.config = config
    
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive backtesting metrics.
        
        Args:
            df: DataFrame with executed trades and PnL
            
        Returns:
            Dictionary with calculated metrics
        """
        logger.info("Calculating backtesting metrics")
        
        if "cumulative_pnl" not in df.columns:
            raise ValueError("DataFrame must contain 'cumulative_pnl' column")
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics(df))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(df))
        
        # Performance metrics
        metrics.update(self._calculate_performance_metrics(df))
        
        # Trade metrics
        metrics.update(self._calculate_trade_metrics(df))
        
        logger.info("Metrics calculation completed")
        return metrics
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic performance metrics.
        
        Args:
            df: DataFrame with PnL data
            
        Returns:
            Dictionary with basic metrics
        """
        total_pnl = df["cumulative_pnl"].iloc[-1]
        total_trades = len(df[df["trade_size"] != 0])
        
        return {
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "final_equity": total_pnl,
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics.
        
        Args:
            df: DataFrame with PnL data
            
        Returns:
            Dictionary with risk metrics
        """
        returns = df["pnl"].dropna()
        
        if len(returns) == 0:
            return {}
        
        # Volatility
        volatility = returns.std() * np.sqrt(self.config.trading_days)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.config.trading_days) if len(downside_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_pnl = df["cumulative_pnl"]
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        return {
            "volatility": volatility,
            "downside_deviation": downside_deviation,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "var_99": var_99,
        }
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            df: DataFrame with PnL data
            
        Returns:
            Dictionary with performance metrics
        """
        returns = df["pnl"].dropna()
        
        if len(returns) == 0:
            return {}
        
        # Annualized return
        total_return = df["cumulative_pnl"].iloc[-1]
        days = len(df)
        annualized_return = total_return * (self.config.trading_days / days)
        
        # Sharpe ratio
        excess_return = annualized_return - self.config.risk_free_rate
        volatility = returns.std() * np.sqrt(self.config.trading_days)
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.config.trading_days) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        max_drawdown = abs(self._calculate_risk_metrics(df).get("max_drawdown", 0))
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Information ratio
        benchmark_return = self.config.benchmark_return
        active_return = annualized_return - benchmark_return
        tracking_error = returns.std() * np.sqrt(self.config.trading_days)
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0
        
        return {
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "information_ratio": information_ratio,
        }
    
    def _calculate_trade_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-specific metrics.
        
        Args:
            df: DataFrame with trade data
            
        Returns:
            Dictionary with trade metrics
        """
        # Filter actual trades
        trades = df[df["trade_size"] != 0]
        
        if len(trades) == 0:
            return {}
        
        # Hit rate
        winning_trades = trades[trades["pnl"] > 0]
        hit_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        # Average win/loss
        avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
        losing_trades = trades[trades["pnl"] < 0]
        avg_loss = losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0
        
        # Win/loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Profit factor
        total_wins = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Turnover
        total_volume = trades["trade_size"].abs().sum()
        turnover = total_volume / len(trades) if len(trades) > 0 else 0
        
        return {
            "hit_rate": hit_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "profit_factor": profit_factor,
            "turnover": turnover,
        }
    
    def calculate_rolling_metrics(self, df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """Calculate rolling metrics.
        
        Args:
            df: DataFrame with PnL data
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        logger.info(f"Calculating rolling metrics with window {window}")
        
        df_rolling = df.copy()
        
        # Rolling returns
        df_rolling["rolling_return"] = df["pnl"].rolling(window).sum()
        
        # Rolling volatility
        df_rolling["rolling_volatility"] = df["pnl"].rolling(window).std() * np.sqrt(self.config.trading_days)
        
        # Rolling Sharpe ratio
        rolling_excess_return = df_rolling["rolling_return"] - self.config.risk_free_rate * (window / self.config.trading_days)
        df_rolling["rolling_sharpe"] = rolling_excess_return / df_rolling["rolling_volatility"]
        
        # Rolling drawdown
        rolling_max = df["cumulative_pnl"].rolling(window).max()
        df_rolling["rolling_drawdown"] = df["cumulative_pnl"] - rolling_max
        
        return df_rolling
    
    def generate_report(self, metrics: Dict[str, float]) -> str:
        """Generate formatted metrics report.
        
        Args:
            metrics: Dictionary with calculated metrics
            
        Returns:
            Formatted report string
        """
        report = "\n" + "="*60 + "\n"
        report += "BACKTESTING PERFORMANCE REPORT\n"
        report += "="*60 + "\n\n"
        
        # Basic metrics
        report += "BASIC METRICS\n"
        report += "-" * 20 + "\n"
        report += f"Total PnL: {metrics.get('total_pnl', 0):.2f}\n"
        report += f"Total Trades: {metrics.get('total_trades', 0)}\n"
        report += f"Final Equity: {metrics.get('final_equity', 0):.2f}\n\n"
        
        # Performance metrics
        report += "PERFORMANCE METRICS\n"
        report += "-" * 20 + "\n"
        report += f"Annualized Return: {metrics.get('annualized_return', 0):.2%}\n"
        report += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
        report += f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
        report += f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n"
        report += f"Information Ratio: {metrics.get('information_ratio', 0):.2f}\n\n"
        
        # Risk metrics
        report += "RISK METRICS\n"
        report += "-" * 20 + "\n"
        report += f"Volatility: {metrics.get('volatility', 0):.2%}\n"
        report += f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}\n"
        report += f"VaR (95%): {metrics.get('var_95', 0):.2f}\n"
        report += f"VaR (99%): {metrics.get('var_99', 0):.2f}\n\n"
        
        # Trade metrics
        report += "TRADE METRICS\n"
        report += "-" * 20 + "\n"
        report += f"Hit Rate: {metrics.get('hit_rate', 0):.2%}\n"
        report += f"Average Win: {metrics.get('avg_win', 0):.2f}\n"
        report += f"Average Loss: {metrics.get('avg_loss', 0):.2f}\n"
        report += f"Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}\n"
        report += f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
        report += f"Turnover: {metrics.get('turnover', 0):.2f}\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report



