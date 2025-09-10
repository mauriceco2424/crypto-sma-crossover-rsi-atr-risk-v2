"""
Slippage Model

Realistic slippage simulation for crypto markets.
"""

from typing import Dict, Any
import numpy as np


class SlippageModel:
    """
    Models realistic slippage for order execution.
    
    Supports multiple slippage models:
    - Fixed basis points
    - Volume-dependent
    - Volatility-adjusted
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize slippage model."""
        self.model_type = config.get('slippage_model', 'fixed_bps')
        self.slippage_bps = config.get('slippage_bps', 10)  # 0.10% default
        
    def calculate_slippage(self, 
                          order_size: float,
                          market_volume: float,
                          volatility: float = None) -> float:
        """Calculate slippage for an order."""
        if self.model_type == 'fixed_bps':
            return self.slippage_bps / 10000  # Convert bps to decimal
        elif self.model_type == 'volume_dependent':
            # Higher slippage for larger orders relative to volume
            impact = min(order_size / market_volume, 0.1)  # Cap at 10%
            return self.slippage_bps / 10000 * (1 + impact * 5)
        elif self.model_type == 'volatility_adjusted' and volatility:
            # Higher slippage in volatile markets
            vol_multiplier = 1 + (volatility - 0.02) * 10  # Baseline 2% vol
            return self.slippage_bps / 10000 * max(0.5, min(2.0, vol_multiplier))
        else:
            return self.slippage_bps / 10000