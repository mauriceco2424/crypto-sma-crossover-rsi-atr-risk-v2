#!/usr/bin/env python3
"""
Run Registry Updater
Updates the run registry with new backtest run using lockfile protocol
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import json
import csv
import time
from datetime import datetime
from pathlib import Path
from scripts.engine.utils.logging_config import setup_logging

def update_run_registry(run_id: str) -> bool:
    """
    Update run registry with new backtest run using lockfile protocol.
    
    Args:
        run_id: The run ID to add to the registry
        
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logging(__name__)
    
    run_dir = Path(f"data/runs/{run_id}")
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return False
    
    # Load run data
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    
    if not manifest_path.exists() or not metrics_path.exists():
        logger.error("Required run files not found")
        return False
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Prepare registry entry
    registry_entry = {
        'run_id': manifest['run_id'],
        'timestamp': manifest.get('created_utc', datetime.utcnow().isoformat() + "Z"),
        'strategy': 'crypto_sma_crossover_rsi_atr_risk',
        'universe': manifest['universe_id'],
        'date_start': manifest['date_start'],
        'date_end': manifest['date_end'],
        'initial_capital': 10000,  # From config
        'status': manifest['status'],
        'total_return': metrics.get('total_return', 0.0),
        'cagr': metrics.get('CAGR', 0.0),
        'sharpe': metrics.get('Sharpe', 0.0),
        'sortino': metrics.get('Sortino', 0.0),
        'max_drawdown': metrics.get('MaxDD', 0.0),
        'n_trades': metrics.get('n_trades', 0),
        'win_rate': metrics.get('win_rate', 0.0),
        'config_hash': manifest.get('config_hash', ''),
        'engine_version': manifest.get('engine_version', '1.0.0')
    }
    
    # Registry and lockfile paths
    registry_dir = Path("docs/runs")
    registry_dir.mkdir(parents=True, exist_ok=True)
    
    registry_path = registry_dir / "run_registry.csv"
    lockfile_path = registry_dir / ".registry.lock"
    
    logger.info(f"Updating run registry for: {run_id}")
    
    # Lockfile protocol with timeout
    max_wait_time = 300  # 5 minutes
    wait_time = 0
    lock_acquired = False
    
    while wait_time < max_wait_time and not lock_acquired:
        try:
            if lockfile_path.exists():
                # Check if lockfile is stale (older than 5 minutes)
                lock_age = time.time() - lockfile_path.stat().st_mtime
                if lock_age > 300:
                    logger.warning(f"Stale lockfile detected (age: {lock_age:.1f}s), removing...")
                    lockfile_path.unlink()
                else:
                    logger.info("Registry locked by another process, waiting...")
                    time.sleep(5)
                    wait_time += 5
                    continue
            
            # Create lockfile
            with open(lockfile_path, 'w') as f:
                f.write(f"PID: {os.getpid()}\n")
                f.write(f"Timestamp: {datetime.utcnow().isoformat()}Z\n")
                f.write(f"Run ID: {run_id}\n")
            
            lock_acquired = True
            logger.info("Registry lock acquired")
            
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            time.sleep(1)
            wait_time += 1
    
    if not lock_acquired:
        logger.error(f"Could not acquire registry lock within {max_wait_time} seconds")
        return False
    
    try:
        # Check if registry exists, create header if not
        if not registry_path.exists():
            logger.info("Creating new run registry")
            with open(registry_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=registry_entry.keys())
                writer.writeheader()
        
        # Append new entry
        with open(registry_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=registry_entry.keys())
            writer.writerow(registry_entry)
        
        logger.info(f"Registry updated successfully for run: {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update registry: {e}")
        return False
        
    finally:
        # Always remove lockfile
        try:
            if lockfile_path.exists():
                lockfile_path.unlink()
                logger.info("Registry lock released")
        except Exception as e:
            logger.warning(f"Failed to remove lockfile: {e}")

def show_registry_summary():
    """Show summary of all runs in the registry."""
    registry_path = Path("docs/runs/run_registry.csv")
    
    if not registry_path.exists():
        print("No run registry found")
        return
    
    try:
        import pandas as pd
        df = pd.read_csv(registry_path)
        
        print(f"\n=== Run Registry Summary ===")
        print(f"Total runs: {len(df)}")
        
        if len(df) > 0:
            print(f"Date range: {df['date_start'].min()} to {df['date_end'].max()}")
            print(f"Strategies: {df['strategy'].nunique()}")
            print(f"Completed runs: {(df['status'] == 'completed').sum()}")
            print(f"Average CAGR: {df['cagr'].mean():.2%}")
            print(f"Average Sharpe: {df['sharpe'].mean():.2f}")
            print(f"Average Max DD: {df['max_drawdown'].mean():.2%}")
            
            print(f"\nRecent runs:")
            recent = df.sort_values('timestamp').tail(5)
            for _, row in recent.iterrows():
                print(f"  {row['run_id'][:20]:20s} | {row['status']:10s} | CAGR: {row['cagr']:7.2%} | Sharpe: {row['sharpe']:5.2f}")
    
    except Exception as e:
        print(f"Failed to load registry summary: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/update_run_registry.py <run_id>")
        sys.exit(1)
    
    run_id = sys.argv[1]
    print(f"=== Updating Run Registry ===")
    print(f"Run ID: {run_id}")
    
    success = update_run_registry(run_id)
    
    if success:
        print(f"\nRegistry updated successfully!")
        show_registry_summary()
        sys.exit(0)
    else:
        print(f"\nFailed to update registry!")
        sys.exit(1)