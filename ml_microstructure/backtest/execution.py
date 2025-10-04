"""Execution engine for backtesting."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class ExecutionConfig(BaseModel):
    """Configuration for execution engine."""
    
    transaction_cost: float = Field(default=0.001, description="Transaction cost per trade")
    slippage: float = Field(default=0.0005, description="Slippage per trade")
    max_position: float = Field(default=10.0, description="Maximum position size")
    min_trade_size: float = Field(default=0.1, description="Minimum trade size")
    execution_delay: int = Field(default=1, description="Execution delay in time steps")


class ExecutionEngine:
    """Engine for executing trades with costs and slippage."""
    
    def __init__(self, config: ExecutionConfig) -> None:
        """Initialize execution engine.
        
        Args:
            config: Execution configuration
        """
        self.config = config
        self.current_position = 0.0
        self.trades = []
        self.equity_curve = []
    
    def execute_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute trades based on signals."""
        logger.info("Executing trades")
        
        df_executed = df.copy()
        
        # Initialize columns
        df_executed["position"] = 0.0
        df_executed["trade_size"] = 0.0
        df_executed["transaction_cost"] = 0.0
        df_executed["slippage_cost"] = 0.0
        df_executed["total_cost"] = 0.0
        df_executed["pnl"] = 0.0
        df_executed["cumulative_pnl"] = 0.0
        
        # Execute trades
        for i in range(len(df_executed)):
            signal = df_executed.iloc[i]["signal"]
            position_size = df_executed.iloc[i]["position_size"]
            mid_price = df_executed.iloc[i]["mid_price"]
            
            # Calculate trade size
            target_pos = signal * position_size  # shorter name
            trade_size = target_pos - self.current_position
            
            # Apply execution constraints
            trade_size = self._apply_constraints(trade_size, mid_price)
            
            if abs(trade_size) >= self.config.min_trade_size:
                # Execute trade
                costs = self._calculate_trade_cost(trade_size, mid_price)  # shorter name
                
                # Update position
                self.current_position += trade_size
                
                # Record trade
                df_executed.iloc[i, df_executed.columns.get_loc("position")] = self.current_position
                df_executed.iloc[i, df_executed.columns.get_loc("trade_size")] = trade_size
                df_executed.iloc[i, df_executed.columns.get_loc("transaction_cost")] = costs["transaction"]
                df_executed.iloc[i, df_executed.columns.get_loc("slippage_cost")] = costs["slippage"]
                df_executed.iloc[i, df_executed.columns.get_loc("total_cost")] = costs["total"]
                
                # Calculate PnL
                if i > 0:
                    price_change = mid_price - df_executed.iloc[i-1]["mid_price"]
                    position_pnl = self.current_position * price_change
                    df_executed.iloc[i, df_executed.columns.get_loc("pnl")] = position_pnl - costs["total"]
                
                # Update cumulative PnL
                if i > 0:
                    df_executed.iloc[i, df_executed.columns.get_loc("cumulative_pnl")] = (
                        df_executed.iloc[i-1]["cumulative_pnl"] + df_executed.iloc[i]["pnl"]
                    )
                else:
                    df_executed.iloc[i, df_executed.columns.get_loc("cumulative_pnl")] = df_executed.iloc[i]["pnl"]
                
                # Record trade details
                self.trades.append({
                    "timestamp": df_executed.iloc[i]["timestamp"],
                    "signal": signal,
                    "trade_size": trade_size,
                    "position": self.current_position,
                    "price": mid_price,
                    "cost": costs["total"],
                    "pnl": df_executed.iloc[i]["pnl"]
                })
        
        logger.info(f"Executed {len(self.trades)} trades")
        return df_executed
    
    def _apply_constraints(self, trade_size: float, price: float) -> float:
        """Apply execution constraints to trade size.
        
        Args:
            trade_size: Desired trade size
            price: Current price
            
        Returns:
            Constrained trade size
        """
        # Maximum position constraint
        max_trade = self.config.max_position - self.current_position
        min_trade = -self.config.max_position - self.current_position
        
        trade_size = max(min_trade, min(max_trade, trade_size))
        
        return trade_size
    
    def _calculate_trade_cost(self, trade_size: float, price: float) -> Dict[str, float]:
        """Calculate trade costs.
        
        Args:
            trade_size: Trade size
            price: Current price
            
        Returns:
            Dictionary with cost breakdown
        """
        trade_value = abs(trade_size) * price
        
        # Transaction cost
        transaction_cost = trade_value * self.config.transaction_cost
        
        # Slippage cost
        slippage_cost = trade_value * self.config.slippage
        
        # Total cost
        total_cost = transaction_cost + slippage_cost
        
        return {
            "transaction": transaction_cost,
            "slippage": slippage_cost,
            "total": total_cost
        }
    
    def get_trade_summary(self) -> Dict[str, float]:
        """Get summary of executed trades.
        
        Returns:
            Dictionary with trade summary
        """
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        summary = {
            "total_trades": len(self.trades),
            "total_volume": trades_df["trade_size"].abs().sum(),
            "total_cost": trades_df["cost"].sum(),
            "avg_trade_size": trades_df["trade_size"].abs().mean(),
            "avg_cost_per_trade": trades_df["cost"].mean(),
            "max_position": trades_df["position"].abs().max(),
            "total_pnl": trades_df["pnl"].sum(),
        }
        
        return summary
    
    def reset(self) -> None:
        """Reset execution engine state."""
        self.current_position = 0.0
        self.trades = []
        self.equity_curve = []



