#!/usr/bin/env python3
"""
Run Data Validator
Validates the generated backtest run data for quality and consistency
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scripts.engine.utils.logging_config import setup_logging

def validate_run_data(run_id: str) -> dict:
    """
    Validate backtest run data for quality and consistency.
    
    Returns:
        Dict with validation results and any issues found
    """
    logger = setup_logging(__name__)
    run_dir = Path(f"data/runs/{run_id}")
    
    validation_results = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "status": "passed",
        "errors": [],
        "warnings": [],
        "checks": {}
    }
    
    logger.info(f"Validating run data for: {run_id}")
    
    # 1. Check required files exist
    required_files = ["manifest.json", "metrics.json", "trades.csv", "events.csv", "series.csv", "progress.json"]
    
    for file in required_files:
        file_path = run_dir / file
        if file_path.exists():
            validation_results["checks"][f"file_{file}"] = "exists"
        else:
            validation_results["errors"].append(f"Missing required file: {file}")
            validation_results["checks"][f"file_{file}"] = "missing"
    
    if validation_results["errors"]:
        validation_results["status"] = "failed"
        return validation_results
    
    # 2. Validate manifest.json
    logger.info("Validating manifest.json...")
    with open(run_dir / "manifest.json", 'r') as f:
        manifest = json.load(f)
    
    required_manifest_fields = ["run_id", "universe_id", "date_start", "date_end", "config_hash", "status"]
    for field in required_manifest_fields:
        if field in manifest:
            validation_results["checks"][f"manifest_{field}"] = "present"
        else:
            validation_results["errors"].append(f"Manifest missing field: {field}")
            validation_results["checks"][f"manifest_{field}"] = "missing"
    
    # 3. Validate metrics.json
    logger.info("Validating metrics.json...")
    with open(run_dir / "metrics.json", 'r') as f:
        metrics = json.load(f)
    
    required_metrics = ["CAGR", "Sortino", "Sharpe", "MaxDD", "n_trades", "win_rate"]
    for metric in required_metrics:
        if metric in metrics:
            validation_results["checks"][f"metric_{metric}"] = "present"
            # Sanity checks
            if metric == "win_rate" and not (0 <= metrics[metric] <= 1):
                validation_results["warnings"].append(f"Win rate outside expected range: {metrics[metric]}")
            elif metric == "MaxDD" and metrics[metric] > 0:
                validation_results["warnings"].append(f"Max drawdown should be negative: {metrics[metric]}")
        else:
            validation_results["errors"].append(f"Metrics missing field: {metric}")
            validation_results["checks"][f"metric_{metric}"] = "missing"
    
    # 4. Validate trades.csv
    logger.info("Validating trades.csv...")
    try:
        trades_df = pd.read_csv(run_dir / "trades.csv")
        validation_results["checks"]["trades_csv_readable"] = "yes"
        
        required_trade_columns = ["timestamp", "symbol", "side", "qty", "price", "fees", "realizedPnL"]
        for col in required_trade_columns:
            if col in trades_df.columns:
                validation_results["checks"][f"trades_{col}"] = "present"
            else:
                validation_results["errors"].append(f"Trades CSV missing column: {col}")
                validation_results["checks"][f"trades_{col}"] = "missing"
        
        # Check for valid timestamps (UTC format)
        if 'timestamp' in trades_df.columns:
            try:
                pd.to_datetime(trades_df['timestamp'])
                validation_results["checks"]["trades_timestamps_valid"] = "yes"
            except:
                validation_results["warnings"].append("Some trade timestamps may be invalid")
                validation_results["checks"]["trades_timestamps_valid"] = "partial"
        
        # Check for reasonable values
        if 'price' in trades_df.columns:
            if (trades_df['price'] <= 0).any():
                validation_results["errors"].append("Found non-positive prices in trades")
                validation_results["checks"]["trades_prices_positive"] = "no"
            else:
                validation_results["checks"]["trades_prices_positive"] = "yes"
        
        validation_results["checks"]["trade_count"] = len(trades_df)
        
    except Exception as e:
        validation_results["errors"].append(f"Failed to read trades.csv: {e}")
        validation_results["checks"]["trades_csv_readable"] = "no"
    
    # 5. Validate events.csv
    logger.info("Validating events.csv...")
    try:
        events_df = pd.read_csv(run_dir / "events.csv")
        validation_results["checks"]["events_csv_readable"] = "yes"
        
        required_event_columns = ["timestamp", "symbol", "event_type", "run_id"]
        for col in required_event_columns:
            if col in events_df.columns:
                validation_results["checks"][f"events_{col}"] = "present"
            else:
                validation_results["errors"].append(f"Events CSV missing column: {col}")
                validation_results["checks"][f"events_{col}"] = "missing"
        
        # Check event types
        if 'event_type' in events_df.columns:
            valid_events = {'filter_pass', 'buy_signal', 'tp_signal', 'tp_sell', 'sl_signal', 'sl_sell'}
            invalid_events = set(events_df['event_type']) - valid_events
            if invalid_events:
                validation_results["warnings"].append(f"Unknown event types: {invalid_events}")
                validation_results["checks"]["events_types_valid"] = "partial"
            else:
                validation_results["checks"]["events_types_valid"] = "yes"
        
        validation_results["checks"]["event_count"] = len(events_df)
        
    except Exception as e:
        validation_results["errors"].append(f"Failed to read events.csv: {e}")
        validation_results["checks"]["events_csv_readable"] = "no"
    
    # 6. Validate series.csv
    logger.info("Validating series.csv...")
    try:
        series_df = pd.read_csv(run_dir / "series.csv")
        validation_results["checks"]["series_csv_readable"] = "yes"
        
        required_series_columns = ["date", "equity", "monitored_count", "open_trades_count"]
        for col in required_series_columns:
            if col in series_df.columns:
                validation_results["checks"][f"series_{col}"] = "present"
            else:
                validation_results["errors"].append(f"Series CSV missing column: {col}")
                validation_results["checks"][f"series_{col}"] = "missing"
        
        # Check equity progression
        if 'equity' in series_df.columns:
            if (series_df['equity'] <= 0).any():
                validation_results["errors"].append("Found non-positive equity values")
                validation_results["checks"]["series_equity_positive"] = "no"
            else:
                validation_results["checks"]["series_equity_positive"] = "yes"
        
        # Check for monotonic dates
        if 'date' in series_df.columns:
            try:
                dates = pd.to_datetime(series_df['date'])
                if dates.is_monotonic_increasing:
                    validation_results["checks"]["series_dates_monotonic"] = "yes"
                else:
                    validation_results["warnings"].append("Series dates are not monotonic")
                    validation_results["checks"]["series_dates_monotonic"] = "no"
            except:
                validation_results["warnings"].append("Could not validate series date monotonicity")
                validation_results["checks"]["series_dates_monotonic"] = "unknown"
        
        validation_results["checks"]["series_rows"] = len(series_df)
        
    except Exception as e:
        validation_results["errors"].append(f"Failed to read series.csv: {e}")
        validation_results["checks"]["series_csv_readable"] = "no"
    
    # 7. Check figures directory
    figs_dir = run_dir / "figs"
    if figs_dir.exists():
        fig_count = len(list(figs_dir.glob("*")))
        validation_results["checks"]["figure_count"] = fig_count
        validation_results["checks"]["figs_directory"] = "exists"
    else:
        validation_results["warnings"].append("Figures directory not found")
        validation_results["checks"]["figs_directory"] = "missing"
        validation_results["checks"]["figure_count"] = 0
    
    # Final status determination
    if validation_results["errors"]:
        validation_results["status"] = "failed"
    elif validation_results["warnings"]:
        validation_results["status"] = "passed_with_warnings"
    else:
        validation_results["status"] = "passed"
    
    logger.info(f"Validation completed: {validation_results['status']}")
    logger.info(f"Errors: {len(validation_results['errors'])}, Warnings: {len(validation_results['warnings'])}")
    
    return validation_results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_run_data.py <run_id>")
        sys.exit(1)
    
    run_id = sys.argv[1]
    print(f"=== Validating Run Data: {run_id} ===")
    
    results = validate_run_data(run_id)
    
    print(f"\nValidation Status: {results['status'].upper()}")
    
    if results['errors']:
        print(f"\nERRORS ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['warnings']:
        print(f"\nWARNINGS ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    print(f"\nCHECKS PERFORMED:")
    for check, status in results['checks'].items():
        status_symbol = "✓" if status in ["present", "exists", "yes"] else "⚠" if status in ["partial", "unknown"] else "✗"
        try:
            print(f"  {status_symbol} {check}: {status}")
        except UnicodeEncodeError:
            symbol = "OK" if status in ["present", "exists", "yes"] else "WARN" if status in ["partial", "unknown"] else "FAIL"
            print(f"  {symbol} {check}: {status}")
    
    if results['status'] == 'passed':
        print(f"\n✓ All validations passed!")
        sys.exit(0)
    elif results['status'] == 'passed_with_warnings':
        print(f"\n⚠ Validation passed with warnings")
        sys.exit(0)
    else:
        print(f"\n✗ Validation failed")
        sys.exit(1)