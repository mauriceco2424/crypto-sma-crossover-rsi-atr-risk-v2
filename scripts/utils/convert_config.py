#!/usr/bin/env python3
"""
Configuration Converter
Converts parameter_config.md to proper format for backtest engine
"""

import re
import yaml
import json
from pathlib import Path
from typing import Dict, Any


def parse_yaml_blocks(content: str) -> Dict[str, Any]:
    """Parse YAML blocks from markdown content."""
    config = {}
    
    # Find all YAML code blocks
    yaml_blocks = re.findall(r'```yaml\n(.*?)\n```', content, re.DOTALL)
    
    for block in yaml_blocks:
        # Parse each YAML block
        try:
            block_data = yaml.safe_load(block)
            if isinstance(block_data, dict):
                config.update(block_data)
        except yaml.YAMLError as e:
            print(f"Warning: Failed to parse YAML block: {e}")
            continue
    
    return config


def restructure_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Restructure config to match expected format."""
    
    # Create the main structure expected by the engine
    structured_config = {
        'backtest': {
            'start_date': config.get('start_date', '2023-01-01'),
            'end_date': config.get('end_date', '2024-01-01'),
            'initial_capital': config.get('initial_capital', 10000)
        },
        'universe': {
            'exchange': config.get('exchange', 'binance'),
            'market_type': config.get('market_type', 'spot'),
            'base_currency': config.get('base_currency', 'USDT'),
            'timeframe': config.get('timeframe', '1d'),
            'min_bars_required': config.get('MIN_BARS_REQUIRED', 365),
            'liquidity_threshold': config.get('LIQUIDITY_THRESHOLD', 250000),
            'exclude_stablecoins': config.get('exclude_stablecoins', True),
            'exclude_leveraged': config.get('exclude_leveraged', True)
        },
        'strategy_parameters': {
            # Trend and momentum
            'TREND_SMA_PERIOD': config.get('TREND_SMA_PERIOD', 200),
            'trend_filter': config.get('trend_filter', 'price_above_sma'),
            'RSI_PERIOD': config.get('RSI_PERIOD', 14),
            'RSI_THRESHOLD': config.get('RSI_THRESHOLD', 50),
            
            # Relative strength
            'RS_LOOKBACK': config.get('RS_LOOKBACK', 90),
            'RS_PERCENTILE': config.get('RS_PERCENTILE', 0.80),
            'RS_MIN_SYMBOLS': config.get('RS_MIN_SYMBOLS', 40),
            'tiebreak_1': config.get('tiebreak_1', 'roc_63'),
            'tiebreak_2': config.get('tiebreak_2', 'atr_pct'),
            
            # SMA crossover
            'FAST_SMA': config.get('FAST_SMA', 50),
            'SLOW_SMA': config.get('SLOW_SMA', 200),
            'VOLUME_MULTIPLIER': config.get('VOLUME_MULTIPLIER', 1.2),
            'VOLUME_LOOKBACK': config.get('VOLUME_LOOKBACK', 20),
            
            # Optional filters
            'USE_BREADTH_FILTER': config.get('USE_BREADTH_FILTER', False),
            'BREADTH_SYMBOL': config.get('BREADTH_SYMBOL', 'BTC/USDT'),
            'breadth_condition': config.get('breadth_condition', 'price_above_sma200'),
            
            # Exit signals
            'ATR_PERIOD': config.get('ATR_PERIOD', 14),
            'INITIAL_STOP_ATR_MULT': config.get('INITIAL_STOP_ATR_MULT', 2.0),
            'TRAILING_STOP_ATR_MULT': config.get('TRAILING_STOP_ATR_MULT', 3.0),
            'TAKE_PROFIT_R_MULT': config.get('TAKE_PROFIT_R_MULT', 3.0),
            'use_take_profit': config.get('use_take_profit', True),
            'MAX_HOLDING_DAYS': config.get('MAX_HOLDING_DAYS', 90),
            
            # Position management
            'RISK_PER_TRADE': config.get('RISK_PER_TRADE', 0.0075),
            'MAX_CONCURRENT_POSITIONS': config.get('MAX_CONCURRENT_POSITIONS', 30),
            'MAX_WEIGHT_PER_SYMBOL': config.get('MAX_WEIGHT_PER_SYMBOL', 0.10),
            'DAILY_DEPLOY_CAP': config.get('DAILY_DEPLOY_CAP', 0.20),
            'COOLDOWN_BARS': config.get('COOLDOWN_BARS', 7),
            'allow_pyramiding': config.get('allow_pyramiding', False),
            'accounting_mode': config.get('accounting_mode', 'mark_to_market'),
            'position_sizing_method': config.get('position_sizing_method', 'volatility_adjusted')
        },
        'execution': {
            'execution_delay': config.get('execution_delay', 'next_bar_open'),
            'slippage_model': config.get('slippage_model', 'fixed_bps'),
            'slippage_bps': config.get('slippage_bps', 10),
            'fee_model': config.get('fee_model', 'percentage'),
            'maker_fee_pct': config.get('maker_fee_pct', 0.10),
            'taker_fee_pct': config.get('taker_fee_pct', 0.10),
            'min_notional': config.get('min_notional', 10),
            'price_precision': config.get('price_precision', 8),
            'quantity_precision': config.get('quantity_precision', 8)
        },
        'optimization': {
            'enable_optimization': config.get('enable_optimization', True),
            'use_filter_gate_manager': config.get('use_filter_gate_manager', True),
            'use_data_processor_cache': config.get('use_data_processor_cache', True),
            'use_reference_engine': config.get('use_reference_engine', True),
            'cache_size_mb': config.get('cache_size_mb', 4800),
            'cache_dir': config.get('cache_dir', 'data/cache'),
            'num_workers': config.get('num_workers', 15),
            'chunk_size_mb': config.get('chunk_size_mb', 256)
        },
        'logging': {
            'log_level': config.get('log_level', 'INFO'),
            'log_to_file': config.get('log_to_file', True),
            'log_dir': config.get('log_dir', 'logs'),
            'save_trades': config.get('save_trades', True),
            'save_signals': config.get('save_signals', True),
            'save_metrics': config.get('save_metrics', True),
            'save_charts': config.get('save_charts', True),
            'output_dir': config.get('output_dir', 'data/runs'),
            'show_progress': config.get('show_progress', True),
            'progress_update_freq': config.get('progress_update_freq', 100)
        },
        'validation': {
            'validate_on_load': config.get('validate_on_load', True),
            'strict_validation': config.get('strict_validation', True),
            'check_lookahead': config.get('check_lookahead', True),
            'check_survivorship_bias': config.get('check_survivorship_bias', True),
            'check_min_samples': config.get('check_min_samples', True),
            'min_trades_required': config.get('min_trades_required', 30)
        }
    }
    
    return structured_config


def convert_config():
    """Convert parameter_config.md to proper format."""
    
    # Read parameter_config.md
    config_path = Path('parameter_config.md')
    if not config_path.exists():
        raise FileNotFoundError("parameter_config.md not found")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse YAML blocks
    config = parse_yaml_blocks(content)
    
    # Restructure to match expected format
    structured_config = restructure_config(config)
    
    # Write to temporary config file
    output_path = Path('temp_config.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structured_config, f, indent=2)
    
    print(f"Configuration converted and saved to: {output_path}")
    print("\nConfig summary:")
    print(f"  - Backtest period: {structured_config['backtest']['start_date']} to {structured_config['backtest']['end_date']}")
    print(f"  - Initial capital: ${structured_config['backtest']['initial_capital']:,}")
    print(f"  - Universe: {structured_config['universe']['exchange']} {structured_config['universe']['base_currency']} pairs")
    print(f"  - Timeframe: {structured_config['universe']['timeframe']}")
    print(f"  - Strategy parameters: {len(structured_config['strategy_parameters'])} items")
    
    return structured_config


if __name__ == "__main__":
    convert_config()