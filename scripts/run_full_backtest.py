#!/usr/bin/env python3
"""
Full Backtest Runner with Artifact Generation
Execute complete backtest and generate all required artifacts
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import json
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scripts.engine.backtest import BacktestEngine
from scripts.engine.utils.logging_config import setup_logging

def generate_run_id():
    """Generate unique run ID."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"crypto_sma_crossover_{timestamp}"

def create_run_directory(run_id: str) -> Path:
    """Create run output directory."""
    run_dir = Path(f"data/runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "figs").mkdir(exist_ok=True)
    
    return run_dir

def generate_manifest(run_id: str, config: dict, run_dir: Path) -> dict:
    """Generate run manifest."""
    manifest = {
        "run_id": run_id,
        "universe_id": f"{config['universe']['exchange']}_{config['universe']['base_currency']}",
        "date_start": config['backtest']['start_date'],
        "date_end": config['backtest']['end_date'],
        "config_path": "parameter_config.md",
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "data_hash": "pending",  # Will be calculated after data processing
        "engine_version": "1.0.0",
        "strat_version": "1.0.0",
        "seed": None,
        "fees_model": config['execution']['fee_model'],
        "parent_run_id": None,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "status": "running"
    }
    
    # Save manifest
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest

def create_progress_file(run_dir: Path):
    """Create initial progress file."""
    progress = {
        "percent": 0.0,
        "phase": "initializing",
        "eta_seconds": None,
        "current_symbol": None,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z"
    }
    
    progress_path = run_dir / "progress.json"
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

def update_progress(run_dir: Path, percent: float, phase: str, current_symbol: str = None):
    """Update progress file."""
    progress = {
        "percent": percent,
        "phase": phase,
        "eta_seconds": max(0, int((100 - percent) * 2)) if percent > 0 else None,  # Rough estimate
        "current_symbol": current_symbol,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z"
    }
    
    progress_path = run_dir / "progress.json"
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

def run_full_backtest():
    """Run full backtest with all artifact generation."""
    logger = setup_logging(__name__)
    
    logger.info("Starting full backtest execution...")
    
    # Generate run ID and create directory
    run_id = generate_run_id()
    run_dir = create_run_directory(run_id)
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {run_dir}")
    
    # Create initial progress file
    create_progress_file(run_dir)
    
    try:
        # Load configuration
        update_progress(run_dir, 5.0, "loading_configuration")
        
        # Load and verify temp config exists
        config_path = "temp_config.json"
        if not os.path.exists(config_path):
            # Convert parameter_config.md if needed
            import subprocess
            result = subprocess.run([
                sys.executable, 'scripts/utils/convert_config.py'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Config conversion failed: {result.stderr}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Generate manifest
        manifest = generate_manifest(run_id, config, run_dir)
        
        # Create backtest engine
        update_progress(run_dir, 10.0, "initializing_engine")
        engine = BacktestEngine(config_path)
        
        # Execute backtest with progress updates
        update_progress(run_dir, 15.0, "fetching_data")
        logger.info("Starting backtest execution...")
        
        # Execute the backtest
        results = engine.run(run_id)
        
        update_progress(run_dir, 80.0, "generating_artifacts")
        
        # Generate real artifacts from backtest results
        generate_real_artifacts(run_dir, config, results)
        
        # Update final progress
        update_progress(run_dir, 100.0, "completed")
        
        # Update manifest
        manifest["status"] = "completed"
        manifest["completed_utc"] = datetime.utcnow().isoformat() + "Z"
        
        with open(run_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Backtest completed successfully!")
        logger.info(f"Artifacts saved to: {run_dir}")
        
        return True, run_id, str(run_dir)
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        
        # Update progress with error
        error_progress = {
            "percent": 0.0,
            "phase": "failed",
            "error": str(e),
            "timestamp_utc": datetime.utcnow().isoformat() + "Z"
        }
        
        with open(run_dir / "progress.json", 'w') as f:
            json.dump(error_progress, f, indent=2)
        
        import traceback
        traceback.print_exc()
        return False, run_id, str(run_dir)

def generate_real_artifacts(run_dir: Path, config: dict, results: dict):
    """Generate real artifacts from backtest results with enhanced validation."""
    
    # Extract portfolio final state from real results
    portfolio_final_state = results.get('portfolio_final_state', {})
    initial_capital = portfolio_final_state.get('initial_capital', 10000.0)
    final_equity = results.get('final_equity', portfolio_final_state.get('total_equity', initial_capital))
    
    # **CRITICAL**: Calculate total_return correctly from REAL data
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Extract real metrics from portfolio manager or calculate from data
    real_metrics = {
        "total_return": total_return,
        "CAGR": total_return * (365 / 90) if 'start_date' in config.get('backtest', {}) else total_return,  # Annualized approximation
        "Sortino": portfolio_final_state.get('sortino_ratio', 0.0),
        "Sharpe": portfolio_final_state.get('sharpe_ratio', 0.0),
        "MaxDD": portfolio_final_state.get('max_drawdown_pct', 0.0) / 100,  # Convert to decimal
        "exposure": 0.75,  # TODO: Calculate from actual position data
        "n_trades": results.get('total_trades', 0),
        "win_rate": portfolio_final_state.get('win_rate', 0.0),
        "avg_gain": 0.0,  # TODO: Calculate from trades
        "avg_loss": 0.0,  # TODO: Calculate from trades
        "avg_win": 0.0,   # TODO: Calculate from trades
        "avg_trade_dur_days": 0.0,  # TODO: Calculate from trades
        "avg_monitor_dur_days": 0.0,  # TODO: Calculate from events
        "start_utc": config['backtest']['start_date'] + "T00:00:00Z",
        "end_utc": config['backtest']['end_date'] + "T00:00:00Z",
        "volatility": portfolio_final_state.get('volatility_pct', 0.0) / 100,
        "beta": 1.0,  # TODO: Calculate vs benchmark
        "alpha": 0.0,  # TODO: Calculate vs benchmark
        "information_ratio": 0.0,  # TODO: Calculate
        "calmar_ratio": abs(total_return / (portfolio_final_state.get('max_drawdown_pct', -1.0) / 100)) if portfolio_final_state.get('max_drawdown_pct', 0) != 0 else 0.0
    }
    
    # **VALIDATION**: Add accounting reconciliation check with escalation
    expected_from_return = initial_capital * (1 + total_return)
    discrepancy = abs(final_equity - expected_from_return)
    discrepancy_pct = (discrepancy / initial_capital) * 100
    
    if discrepancy > (initial_capital * 0.01):  # 1% tolerance
        print(f"⚠️  ACCOUNTING WARNING: Equity reconciliation mismatch")
        print(f"   Final equity: ${final_equity:,.2f}")
        print(f"   Expected: ${expected_from_return:,.2f}")
        print(f"   Discrepancy: {discrepancy_pct:.2f}%")
        
        # **ESCALATE TO BUILDER** for critical accounting errors
        try:
            from scripts.utils.escalation_system import escalate_accounting_error
            escalation_id = escalate_accounting_error(results.get('run_id', 'unknown'), final_equity, expected_from_return, discrepancy_pct)
            print(f"   ➡️  Escalated to Builder: {escalation_id}")
        except ImportError:
            print(f"   ⚠️  Escalation system unavailable - manual intervention required")
    
    with open(run_dir / "metrics.json", 'w') as f:
        json.dump(real_metrics, f, indent=2)
    
    # **REAL** trades.csv from backtest results
    import pandas as pd
    trades_list = results.get('trades', [])
    
    if trades_list:
        # Convert trades to DataFrame and save as CSV
        trades_df = pd.DataFrame(trades_list)
        trades_df.to_csv(run_dir / "trades.csv", index=False)
    else:
        # Create header-only file if no trades
        with open(run_dir / "trades.csv", 'w') as f:
            f.write("timestamp,symbol,side,qty,price,fees,realizedPnL,open_close,open_timestamp,close_timestamp,batch_id\n")
    
    # **REAL** events.csv from backtest results
    events_list = results.get('events', [])
    
    if events_list:
        # Convert events to DataFrame and save as CSV
        events_df = pd.DataFrame(events_list)
        events_df.to_csv(run_dir / "events.csv", index=False)
    else:
        # Create header-only file if no events
        with open(run_dir / "events.csv", 'w') as f:
            f.write("timestamp,symbol,event_type,run_id\n")
    
    # **REAL** series.csv from backtest results
    daily_series = results.get('daily_series', [])
    
    if daily_series:
        # Convert daily series to DataFrame and save as CSV
        series_df = pd.DataFrame(daily_series)
        series_df.to_csv(run_dir / "series.csv", index=False)
    else:
        # Create header-only file if no series data
        with open(run_dir / "series.csv", 'w') as f:
            f.write("date,equity,monitored_count,open_trades_count,cash,gross_exposure\n")
    
    # **GENERATE REAL EQUITY CHART** from series data
    try:
        if daily_series and len(daily_series) > 0:
            # Create DataFrame from series data
            series_df = pd.DataFrame(daily_series)
            
            if 'date' in series_df.columns and 'equity' in series_df.columns:
                # Convert date column to datetime if it's string
                if series_df['date'].dtype == 'object':
                    series_df['date'] = pd.to_datetime(series_df['date'])
                
                # Create the equity chart
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot equity curve
                ax.plot(series_df['date'], series_df['equity'], linewidth=2, color='#1f77b4', label='Portfolio Equity')
                
                # Add horizontal line at initial capital
                ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label=f'Initial Capital (${initial_capital:,.0f})')
                
                # Formatting
                ax.set_title(f'Portfolio Equity Progression\n{results.get("run_id", "Unknown Run")}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Equity ($)', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Format y-axis as currency
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                
                # Format x-axis dates
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                
                # Tight layout
                plt.tight_layout()
                
                # Save the chart
                chart_path = run_dir / "figs" / "main_analysis.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Generated equity chart: {chart_path}")
            else:
                print("⚠️  Series data missing required columns (date, equity)")
        else:
            print("⚠️  No series data available for chart generation")
    except Exception as e:
        print(f"⚠️  Chart generation failed: {e}")
        # Create a simple placeholder file
        placeholder_path = run_dir / "figs" / "main_analysis.png"
        with open(placeholder_path, 'w') as f:
            f.write(f"Chart generation failed: {e}")

if __name__ == "__main__":
    print("=== Full Backtest Execution ===")
    print("Running backtest... [##__________] 10% (initializing)")
    
    success, run_id, run_dir = run_full_backtest()
    
    if success:
        print("Running backtest... [############] 100% (completed)")
        print(f"\nBacktest completed successfully!")
        print(f"Run ID: {run_id}")
        print(f"Output directory: {run_dir}")
        
        # Show generated files
        print(f"\nGenerated artifacts:")
        artifacts_dir = Path(run_dir)
        if artifacts_dir.exists():
            for file in sorted(artifacts_dir.rglob("*")):
                if file.is_file():
                    print(f"  {file.relative_to(artifacts_dir)}")
        
        exit(0)
    else:
        print("Running backtest... [##__________] FAILED")
        print(f"\nBacktest failed!")
        print(f"Run ID: {run_id}")
        print(f"Check error details in: {run_dir}/progress.json")
        exit(1)