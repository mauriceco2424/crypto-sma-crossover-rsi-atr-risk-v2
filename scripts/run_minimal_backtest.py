#!/usr/bin/env python3
"""
Minimal Backtest Runner
Tests core backtest functionality with limited dataset
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import json
import uuid
from datetime import datetime
from scripts.engine.backtest import BacktestEngine
from scripts.engine.utils.logging_config import setup_logging

def run_minimal_backtest():
    """Run a minimal backtest with limited data."""
    logger = setup_logging(__name__)
    
    logger.info("Starting minimal backtest...")
    
    # Load and modify config for minimal test
    with open('temp_config.json', 'r') as f:
        config = json.load(f)
    
    # Reduce scope for testing
    config['backtest']['start_date'] = '2023-01-01'
    config['backtest']['end_date'] = '2023-01-15'  # Just 2 weeks
    config['universe']['symbols'] = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']  # Just 3 symbols
    config['strategy_parameters']['MAX_CONCURRENT_POSITIONS'] = 3
    
    # Save minimal config
    minimal_config_path = 'minimal_config.json'
    with open(minimal_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Minimal config: {config['backtest']['start_date']} to {config['backtest']['end_date']}")
    logger.info(f"Symbols: {config['universe']['symbols']}")
    
    try:
        # Create engine with minimal config
        engine = BacktestEngine(minimal_config_path)
        
        # Generate run ID
        run_id = f"minimal_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Running backtest with ID: {run_id}")
        
        # Execute backtest
        results = engine.run(run_id)
        
        logger.info("Backtest completed successfully!")
        logger.info(f"Results: {results}")
        
        return True, results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    print("=== Minimal Backtest Test ===")
    
    success, results = run_minimal_backtest()
    
    if success:
        print(f"\nBacktest completed successfully!")
        if results:
            print(f"Run ID: {results.get('run_id', 'unknown')}")
            print(f"Status: {results.get('status', 'unknown')}")
        exit(0)
    else:
        print(f"\nBacktest failed!")
        exit(1)