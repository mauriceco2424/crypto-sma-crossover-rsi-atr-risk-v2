# Analyze Trading Run Results

---
description: Process run data into comprehensive metrics and professional visualizations
argument-hint: [run_id]
model: claude-3-5-sonnet-20241022
---

I need to use the **trading-single-analyzer** agent to process the backtest data into comprehensive analysis artifacts including metrics calculation and professional visualizations.

## Run Analysis Parameters
- **Run ID**: $ARGUMENTS (defaults to most recent run if not specified)

## Analysis Tasks

### 1. Data Processing & Validation
- **Input Validation**: Verify data quality, UTC timestamps, no duplicates
- **Missing Bar Policy**: Ensure missing-bar policy is respected
- **Data Integrity**: Check for non-negative prices/volumes, monotonic timestamps

### 2. Artifact Generation
Create canonical outputs in `/data/runs/{run_id}/`:

- **`manifest.json`**: Run metadata including:
  - run_id, universe_id, date_start, date_end
  - config_path, config_hash, data_hash
  - engine_version, strat_version, seed, fees_model
  - parent_run_id (if applicable)

- **`metrics.json`**: Performance metrics including:
  - CAGR, Sortino, Sharpe, MaxDD
  - exposure, n_trades, win_rate
  - avg_gain, avg_loss, avg_win
  - avg_trade_dur_days, avg_monitor_dur_days
  - start_utc, end_utc

- **`trades.csv`**: Individual fill records with:
  - side, qty, price, fees, realizedPnL
  - open/close timestamps, symbol, batch_id

- **`events.csv`**: Signal and action timestamps:
  - filter_pass, buy_signal, tp_signal, tp_sell
  - sl_signal, sl_sell with UTC timestamps

- **`series.csv`**: Time series data:
  - Daily equity, monitored_count, open_trades_count
  - Optional cash, gross_exposure

### 3. Professional Visualization Generation (Research-Based Best Practices)
Create publication-quality figures in `/data/runs/{run_id}/figs/`:

**Default 3-Panel Layout** (always generated):
- **Panel 1 - Main Equity Chart (70% height)**:
  - Primary equity curve (thick line, prominent styling)
  - Benchmark comparison curve (thin line, muted color)
  - Smart time axis with auto-detected optimal spacing:
    - <90 days: daily ticks, <365 days: weekly ticks
    - <1095 days: monthly ticks, 3+ years: quarterly ticks
  - Subtle background shading for winning/losing periods
  - Professional typography, no cramming of metrics

- **Panel 2 - Drawdown Analysis (20% height)**:
  - Inverted area chart showing % drawdown from peaks
  - Red fill going down from 0% baseline
  - Maximum drawdown period highlighting with annotations
  - Same time axis as equity panel for perfect alignment

- **Panel 3 - Trade Activity Summary (10% height)**:
  - Daily/weekly trade frequency or win rate aggregation
  - Clean bar chart design avoiding visual clutter
  - Consistent styling with main chart theme

**Enhanced Visualization** (template-configurable):
- **Per-Symbol OHLCV Charts**: Candlestick + volume with event overlays
- **Event Marker System**: 
  - Grey: filter pass, Black: buy signal
  - Blue: TP signal, Green: TP sell
  - Orange: SL signal, Red: SL sell
- **Trade Period Visualization**: Shaded spans for open positions
- **Professional Quality**: 300+ DPI, colorblind-friendly, SVG + PNG formats

### 4. **Enhanced Validation & Quality Checks (MANDATORY)**
- **CRITICAL: Accounting Reconciliation**: Verify `final_equity ≈ initial_capital × (1 + total_return)` within 1% tolerance - HALT if mismatch
- **Open Position Handling**: Include unrealized P&L for any open positions at period end - mark-to-market at closing prices
- **Multi-Source Consistency**: Verify series.csv final equity matches metrics.json calculation exactly
- **Open Position Flagging**: If >5 positions open at end or >20% capital unrealized → add prominent warning
- **No Lookahead Validation**: Verify all features use data ≤ t for actions at t+1
- **Accounting Identity**: Confirm `Equity_{t+1} = Equity_t + realizedPnL - fees`
- **Sanity Thresholds**: Flag extreme ratios (Sortino >3, zero drawdown periods)  
- **Data Quality**: UTC timestamps, no duplicates, non-negative prices/volumes
- **Execution Realism**: Validate trade execution feasibility and liquidity

### 5. **Anomaly Detection & Reporting**
- Identify unusual patterns or suspicious performance metrics
- Flag potential overfitting indicators
- Detect data quality issues or missing information
- Generate anomaly report with severity classifications
- Escalate critical issues to appropriate agents

### 6. **Professional Visualization Standards**
- **Publication-ready quality**: High-resolution PNG and SVG formats
- **Professional formatting**: Clear labels, legends, and captions
- **Consistent styling**: Standard color schemes and typography
- **Figure descriptions**: Auto-generated captions explaining each visualization
- **Ready for PDF inclusion**: Proper sizing and quality for reports

### 7. **Registry Update & Progress Reporting**
- Use lockfile protocol (`/docs/runs/.registry.lock`)
- Check for stale locks (>5min) and proceed with warning
- Atomically append run to `/docs/runs/run_registry.csv`
- **Unified progress bar**: `Analyzing run... ████████░░░░ 75% (~1 min remaining)`
- **Clear phases**: validation → metrics → visualization → quality checks
- **ETA integration**: Based on data size and processing complexity

### 8. **Resource Management & Output Organization**
- Monitor RAM usage, fail if drops below 2GB
- Generate SHA256 checksums for data integrity
- Save all outputs to `/data/runs/{run_id}/analysis/`:
  - Enhanced metrics.json with expanded statistics
  - Professional visualizations in `/figs/` directory
  - Validation reports and quality check results
  - Analysis summary with key findings

## Expected Outputs
- **Enhanced metrics.json** with comprehensive performance statistics
- **Professional visualization suite** ready for report inclusion:
  - Main equity chart with trade markers and statistics panel
  - Position monitoring subplot with clear timeline
  - Per-symbol OHLCV charts with event overlays
- **Validation report** confirming data integrity and realism
- **Anomaly detection report** with any concerns flagged
- **Analysis summary** highlighting key findings and insights
- **Ready for evaluation** with `/evaluate-single-run`

## Success Criteria
- **MANDATORY**: Accounting reconciliation passes - final equity consistent with reported returns within 1% tolerance
- **MANDATORY**: Open position assessment completed - any unrealized P&L properly included in calculations
- **MANDATORY**: Multi-source consistency verified - series.csv, metrics.json, and visualizations are consistent
- All enhanced validation checks pass without critical errors
- Professional visualizations generated successfully
- Comprehensive metrics calculated and validated  
- No critical anomalies detected in analysis
- All artifacts ready for evaluator consumption

## Halt Conditions (Do NOT Proceed to Registry Update)
- Accounting mismatch >1% between final equity calculation and reported returns
- Open positions at period end with unrealized P&L not properly included
- Inconsistency between series.csv final equity and metrics.json calculation
- Any critical validation failure that affects performance accuracy

Please use the trading-single-analyzer agent to perform comprehensive data analysis with embedded validation and professional visualization generation.