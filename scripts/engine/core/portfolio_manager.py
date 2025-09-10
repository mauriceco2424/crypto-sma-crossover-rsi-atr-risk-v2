"""
Portfolio Manager

Handles position tracking, P&L calculation, and portfolio accounting.
Maintains the accounting identity: Equity_{t+1} = Equity_t + realizedPnL - fees
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from copy import deepcopy


class PortfolioManager:
    """
    Manages portfolio state, positions, and P&L calculations.
    
    Ensures proper accounting and provides real-time portfolio metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize portfolio manager with configuration.
        
        Args:
            config: Configuration dictionary from parameter_config.md
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initial portfolio state
        self.initial_capital = config['backtest']['initial_capital']
        self.cash = float(self.initial_capital)
        self.positions = {}  # symbol -> position info
        self.total_equity = float(self.initial_capital)
        
        # Performance tracking
        self.equity_history = []
        self.trade_history = []
        self.daily_returns = []
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_fees_paid = 0.0
        
        # Current prices for mark-to-market
        self.current_prices = {}
        
        # Performance metrics cache
        self._metrics_cache = {}
        self._cache_timestamp = None
        
        self.logger.info(f"Portfolio initialized with ${self.initial_capital:,.2f}")
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for mark-to-market valuation.
        
        Args:
            prices: Dictionary of symbol -> current price
        """
        self.current_prices.update(prices)
        self._update_unrealized_pnl()
        self._update_total_equity()
    
    def process_fill(self, fill_info: Dict[str, Any]) -> None:
        """
        Process a trade fill and update portfolio state.
        
        Args:
            fill_info: Fill information from order manager
        """
        symbol = fill_info['symbol']
        action = fill_info['action']
        quantity = fill_info['quantity']
        fill_price = fill_info['fill_price']
        fees = fill_info.get('fees', 0.0)
        timestamp = fill_info.get('timestamp')
        
        # Update cash for the trade
        trade_value = quantity * fill_price
        
        if action in ['buy', 'long']:
            self.cash -= trade_value + fees
            self._add_to_position(symbol, quantity, fill_price, 'long')
        elif action in ['sell', 'short']:
            self.cash += trade_value - fees
            self._add_to_position(symbol, -quantity, fill_price, 'short')
        elif action == 'close':
            realized_pnl = self._close_position(symbol, quantity, fill_price)
            if quantity > 0:  # Closing long position
                self.cash += trade_value - fees
            else:  # Closing short position
                self.cash -= trade_value + fees
        
        # Update P&L tracking
        self.total_fees_paid += fees
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': fill_price,
            'value': trade_value,
            'fees': fees,
            'cash_after': self.cash,
            'equity_after': self.total_equity
        }
        self.trade_history.append(trade_record)
        
        # Update total equity
        self._update_total_equity()
        
        # Log the trade
        self.logger.debug(f"Processed fill: {action} {quantity} {symbol} @ {fill_price:.4f} "
                         f"(fees: ${fees:.2f}, cash: ${self.cash:.2f})")
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position dictionary or None if no position
        """
        return self.positions.get(symbol)
    
    def get_position_size(self, symbol: str) -> float:
        """
        Get current position size for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position size (positive for long, negative for short)
        """
        position = self.positions.get(symbol)
        if position:
            return position['quantity']
        return 0.0
    
    def get_position_value(self, symbol: str) -> float:
        """
        Get current market value of position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current market value of position
        """
        position = self.positions.get(symbol)
        if not position:
            return 0.0
        
        current_price = self.current_prices.get(symbol, position['avg_price'])
        return position['quantity'] * current_price
    
    def get_available_cash(self) -> float:
        """Get available cash for new trades."""
        return self.cash
    
    def get_total_equity(self) -> float:
        """Get total portfolio equity."""
        return self.total_equity
    
    def get_positions_value(self) -> float:
        """Get total value of all positions."""
        total_value = 0.0
        for symbol, position in self.positions.items():
            current_price = self.current_prices.get(symbol, position['avg_price'])
            total_value += position['quantity'] * current_price
        return total_value
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get complete portfolio state.
        
        Returns:
            Dictionary with portfolio state information
        """
        positions_value = self.get_positions_value()
        
        # **ENHANCED VALIDATION**: Real-time accounting validation
        calculated_total_return = (self.total_equity / self.initial_capital - 1.0) * 100
        expected_equity = self.initial_capital + self.realized_pnl + self.unrealized_pnl - self.total_fees_paid
        equity_discrepancy = abs(self.total_equity - expected_equity)
        
        # Real-time validation flag for critical errors
        validation_warning = None
        if equity_discrepancy > (self.initial_capital * 0.01):  # >1% discrepancy
            validation_warning = f"ACCOUNTING_ERROR_{equity_discrepancy/self.initial_capital*100:.1f}pct_discrepancy"
        
        open_positions_count = len([p for p in self.positions.values() if p['quantity'] != 0])
        if open_positions_count > 5:
            validation_warning = f"HIGH_OPEN_POSITIONS_{open_positions_count}_trades"
        
        return {
            'cash': self.cash,
            'positions_value': positions_value,
            'total_equity': self.total_equity,
            'initial_capital': self.initial_capital,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_fees_paid': self.total_fees_paid,
            'positions': deepcopy(self.positions),
            'total_return': calculated_total_return,
            'open_positions_count': open_positions_count,
            'daily_return': self._calculate_daily_return() if self.equity_history else 0.0,
            # **ENHANCED VALIDATION FIELDS**
            'expected_equity': expected_equity,
            'equity_discrepancy': equity_discrepancy,
            'validation_warning': validation_warning,
            'accounting_verified': equity_discrepancy <= (self.initial_capital * 0.01)
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.equity_history:
            return {}
        
        # Use cached metrics if recent
        if (self._cache_timestamp and 
            self._cache_timestamp == len(self.equity_history) and 
            self._metrics_cache):
            return self._metrics_cache
        
        equity_series = pd.Series([entry['equity'] for entry in self.equity_history])
        returns = equity_series.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        metrics = {}
        
        try:
            # Basic metrics
            total_return = (self.total_equity / self.initial_capital - 1.0) * 100
            metrics['total_return_pct'] = total_return
            metrics['annualized_return_pct'] = self._annualize_return(total_return, len(returns))
            
            # Risk metrics
            metrics['volatility_pct'] = returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Sharpe ratio (assuming 0% risk-free rate)
            if metrics['volatility_pct'] > 0:
                metrics['sharpe_ratio'] = metrics['annualized_return_pct'] / metrics['volatility_pct']
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std() * np.sqrt(252) * 100
                metrics['sortino_ratio'] = metrics['annualized_return_pct'] / downside_std if downside_std > 0 else 0.0
            else:
                metrics['sortino_ratio'] = np.inf if total_return > 0 else 0.0
            
            # Drawdown metrics
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max * 100
            metrics['max_drawdown_pct'] = drawdown.min()
            
            # Win/loss metrics
            if self.trade_history:
                trade_pnls = []
                for trade in self.trade_history:
                    # This is simplified - would need proper P&L calculation
                    if 'realized_pnl' in trade:
                        trade_pnls.append(trade['realized_pnl'])
                
                if trade_pnls:
                    winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
                    losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
                    
                    metrics['win_rate_pct'] = len(winning_trades) / len(trade_pnls) * 100
                    metrics['avg_win'] = np.mean(winning_trades) if winning_trades else 0.0
                    metrics['avg_loss'] = np.mean(losing_trades) if losing_trades else 0.0
                    
                    if metrics['avg_loss'] != 0:
                        metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss'])
                    else:
                        metrics['profit_factor'] = np.inf if metrics['avg_win'] > 0 else 0.0
            
            # Trading activity
            metrics['total_trades'] = len(self.trade_history)
            metrics['total_fees_paid'] = self.total_fees_paid
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
        
        # Cache results
        self._metrics_cache = metrics
        self._cache_timestamp = len(self.equity_history)
        
        return metrics
    
    def record_daily_state(self, timestamp: datetime) -> None:
        """
        Record daily portfolio state for performance tracking.
        
        Args:
            timestamp: Current timestamp
        """
        equity_record = {
            'timestamp': timestamp,
            'equity': self.total_equity,
            'cash': self.cash,
            'positions_value': self.get_positions_value(),
            'unrealized_pnl': self.unrealized_pnl,
            'daily_return': self._calculate_daily_return()
        }
        
        self.equity_history.append(equity_record)
    
    def _add_to_position(self, symbol: str, quantity: float, price: float, side: str) -> None:
        """Add to or create a position."""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0.0,
                'avg_price': 0.0,
                'total_cost': 0.0,
                'side': side,
                'unrealized_pnl': 0.0
            }
        
        position = self.positions[symbol]
        
        # Update position
        new_total_cost = position['total_cost'] + (quantity * price)
        new_quantity = position['quantity'] + quantity
        
        if new_quantity != 0:
            position['avg_price'] = new_total_cost / new_quantity
        else:
            position['avg_price'] = 0.0
        
        position['quantity'] = new_quantity
        position['total_cost'] = new_total_cost
        
        # Remove position if quantity is zero
        if abs(position['quantity']) < 1e-8:
            del self.positions[symbol]
    
    def _close_position(self, symbol: str, quantity: float, price: float) -> float:
        """
        Close a position and calculate realized P&L.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to close (positive for closing long, negative for closing short)
            price: Closing price
            
        Returns:
            Realized P&L from the position closure
        """
        if symbol not in self.positions:
            self.logger.warning(f"Attempting to close non-existent position: {symbol}")
            return 0.0
        
        position = self.positions[symbol]
        
        # Calculate realized P&L
        if position['side'] == 'long':
            realized_pnl = quantity * (price - position['avg_price'])
        else:  # short
            realized_pnl = quantity * (position['avg_price'] - price)
        
        # Update position
        position['quantity'] -= quantity
        position['total_cost'] -= quantity * position['avg_price']
        
        # Remove position if fully closed
        if abs(position['quantity']) < 1e-8:
            del self.positions[symbol]
        
        # Update realized P&L
        self.realized_pnl += realized_pnl
        
        return realized_pnl
    
    def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L for all positions."""
        total_unrealized = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in self.current_prices:
                current_price = self.current_prices[symbol]
                
                if position['side'] == 'long':
                    unrealized = position['quantity'] * (current_price - position['avg_price'])
                else:  # short
                    unrealized = position['quantity'] * (position['avg_price'] - current_price)
                
                position['unrealized_pnl'] = unrealized
                total_unrealized += unrealized
        
        self.unrealized_pnl = total_unrealized
    
    def _update_total_equity(self) -> None:
        """Update total portfolio equity."""
        self.total_equity = self.cash + self.get_positions_value()
    
    def _calculate_daily_return(self) -> float:
        """Calculate daily return."""
        if len(self.equity_history) < 2:
            return 0.0
        
        previous_equity = self.equity_history[-2]['equity']
        current_equity = self.total_equity
        
        if previous_equity > 0:
            return (current_equity / previous_equity - 1.0) * 100
        return 0.0
    
    def _annualize_return(self, total_return_pct: float, num_periods: int) -> float:
        """
        Annualize a total return.
        
        Args:
            total_return_pct: Total return percentage
            num_periods: Number of periods (e.g., days)
            
        Returns:
            Annualized return percentage
        """
        if num_periods == 0:
            return 0.0
        
        # Assume daily periods, convert to annual
        periods_per_year = 252  # Trading days per year
        years = num_periods / periods_per_year
        
        if years == 0:
            return total_return_pct
        
        return ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100