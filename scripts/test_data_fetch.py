#!/usr/bin/env python3
"""
Test Data Fetcher
Quick test to verify data fetching works
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import json
from scripts.engine.data.data_fetcher import DataFetcher
from scripts.engine.utils.logging_config import setup_logging

def test_data_fetch():
    """Test basic data fetching functionality."""
    logger = setup_logging(__name__)
    
    # Load config
    with open('temp_config.json', 'r') as f:
        config = json.load(f)
    
    # Create data fetcher
    data_fetcher = DataFetcher(config)
    
    # Test symbol discovery
    logger.info("Testing symbol discovery...")
    try:
        symbols = data_fetcher.get_available_symbols()
        logger.info(f"Found {len(symbols)} total symbols")
        
        # Filter for USDT pairs
        usdt_symbols = [s for s in symbols if s.endswith('USDT')]
        logger.info(f"Found {len(usdt_symbols)} USDT pairs")
        
        # Show first 10
        logger.info(f"First 10 USDT pairs: {usdt_symbols[:10]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Symbol discovery failed: {e}")
        return False

def test_small_data_fetch():
    """Test fetching data for a small set of symbols."""
    logger = setup_logging(__name__)
    
    # Load config
    with open('temp_config.json', 'r') as f:
        config = json.load(f)
    
    # Create data fetcher
    data_fetcher = DataFetcher(config)
    
    # Test small data fetch
    test_symbols = ['BTCUSDT', 'ETHUSDT']
    logger.info(f"Testing data fetch for: {test_symbols}")
    
    try:
        data = data_fetcher.fetch_historical_data(
            symbols=test_symbols,
            start_date='2023-01-01',
            end_date='2023-01-10',
            timeframe='1d'
        )
        
        for symbol, df in data.items():
            logger.info(f"{symbol}: {len(df)} rows")
            if len(df) > 0:
                logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
                logger.info(f"  Columns: {list(df.columns)}")
        
        return len(data) > 0
        
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Data Fetcher ===")
    
    print("\n1. Testing symbol discovery...")
    symbols_ok = test_data_fetch()
    
    print(f"\n2. Testing small data fetch...")
    data_ok = test_small_data_fetch()
    
    print(f"\n=== Results ===")
    print(f"Symbol discovery: {'OK' if symbols_ok else 'FAIL'}")
    print(f"Data fetch: {'OK' if data_ok else 'FAIL'}")
    
    if symbols_ok and data_ok:
        print("\nData fetcher working correctly!")
        exit(0)
    else:
        print("\nData fetcher has issues")
        exit(1)