# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Framework to **build, evaluate, and iteratively optimize trading strategies** (backtest engine now, bot engine later). Goals: **realistic, live-executable strats**, **profitable in backtests**, and **optimizable**.

## Development Commands

**Core commands:**
- `python scripts/quiet_mode.py [on|off|status]` - Control agent output verbosity (default: shows progress)
- `pip install -r requirements.txt` - Install Python dependencies (pandas, numpy, matplotlib, scipy, etc.)
- `python scripts/initialization/initialize_strategy.py` - Transform skeleton to specific strategy project
- `pytest` - Run test suite (if tests exist)

**Key dependencies:** pandas, numpy, matplotlib, seaborn, plotly, scipy, scikit-learn, ccxt (crypto), tqdm (progress), numba (optimization)

## Abbreviations

* **EMR**: Engine Master Report (engine spec)
* **SMR**: Strategy Master Report (strategy spec)  
* **ECL/SCL**: Engine/Strategy Changelogs (append-only)
* **ECN/SER/SDCN**: Engine Change Notice / Strategy Evaluation Report / Strategy Definition Change Notice

## Always do this

* Start in **PLAN mode**; keep `cloud/tasks/<task>.md` with **goals, owners, deps, gates, milestones**
* Prefer **slash commands**: `/initialize`, `/validate-setup`, `/validate-strategy`, `/plan-strategy`, `/build-engine`, `/run`, `/analyze-single-run`, `/evaluate-single-run`, `/run-optimization`, `/evaluate-optimization`
* `/docs/**` is authoritative; changelogs are append-only
* **Progress bar requirement**: All Python scripts MUST implement unified progress reporting with ETA
* **Universal Speed Optimization**: All agents MUST leverage FilterGateManager, DataProcessor optimization, and ReferenceEngine for maximum speed (see .claude/docs/optimization-patterns.md)
* **Quiet Mode**: Use `python scripts/quiet_mode.py on` to suppress progress output and prevent screen flickering during long operations

## Workflow (dual-path: single-run + optimization)

**Common Setup & Planning (4 commands):**
1. **Setup & Validation**: `/validate-setup` → `/validate-strategy` → `/plan-strategy`
2. **Engine Building**: `/build-engine` (auto-generates parameter_config.md)

**Single-Run Path:**
3. **Individual Steps**: `/run` → `/analyze-single-run` → `/evaluate-single-run`

**Optimization Path:**
4. **Individual Steps**: `/run-optimization` → `/evaluate-optimization`

## Roles

**Common Agents:**
* **Orchestrator**: **ALWAYS ACTIVE** - coordinate ALL commands; plan/route; keep **EMR/SMR** authoritative; **ECL/SCL** append-only; block runs until docs fresh; **handle handoffs from ALL specialist agents for documentation updates**
* **Builder**: implement/optimize engine; hardware-aware session initialization; unit/smoke/perf tests; emit **ECN** (+ benchmarks with hardware profile); **MUST include FilterGateManager, optimized DataProcessor, and ReferenceEngine in generated engines for automatic speed optimization**

**Single-Run Agents:**
* **Single-Analyzer**: execute single backtests AND process run data; read parameter_config.md; emit **manifest, metrics, trades, events, series, figs**; sanity-check and flag anomalies  
* **Single-Evaluator**: **evaluate single-run performance** (assess metrics quality, compare to benchmarks); **strategic interpretation** (understand WHY strategy works/fails); **generate LaTeX PDF reports**; emit **SER**; if spec changed, **SDCN**

**Optimization Agents:**
* **Optimizer**: execute parameter sweeps AND process optimization data; read optimization_config.json; coordinate multiple backtests with walk-forward validation; create parameter performance matrices, robustness heatmaps, statistical validation; **MUST use OptimizationEngine with FilterGateManager, ReferenceEngine, and feature caching for 10-50x speedup**
* **Optimization-Evaluator**: **evaluate parameter optimization** (assess parameter significance, detect overfitting); **strategic parameter interpretation** (understand WHY parameters work); **generate optimization study PDF reports**

## Handoffs & gates

**Command Orchestration Pattern:**
ALL slash commands follow: **User → Orchestrator → Specialist Agent → Orchestrator → Documentation Update**

**Common Handoffs:**
* **`/build-engine`**: Orchestrator → Builder → **ECN + benchmarks** → Orchestrator updates **EMR** & appends **ECL**
* **`/validate-setup`**: Orchestrator → validation logic → documentation updates as needed
* **`/validate-strategy`**: Orchestrator → validation logic → SMR updates if needed

**Single-Run Handoffs:**
* **`/run`**: Orchestrator → Single-Analyzer → processed artifacts + run_registry row → Orchestrator
* **`/analyze-single-run`**: Orchestrator → Single-Analyzer → analysis artifacts → Single-Evaluator handoff → Orchestrator  
* **`/evaluate-single-run`**: Orchestrator → Single-Evaluator → **SER** (and **SDCN** if spec changed) → Orchestrator updates **SMR** & appends **SCL**

**Optimization Handoffs:**
* **`/run-optimization`**: Orchestrator → Optimizer → optimization data → Orchestrator
* **`/evaluate-optimization`**: Orchestrator → Optimization-Evaluator → **Optimization Report** → Orchestrator updates documentation

**Gates:**
* **Documentation Gate**: Orchestrator blocks ALL commands until EMR/SMR in sync with latest changelogs
* **Single-Run Gate**: `/analyze-single-run` must complete before `/evaluate-single-run`
* **Optimization Gate**: `/run-optimization` must complete before `/evaluate-optimization`
* **Handoff Gate**: ALL agents must complete handoff to Orchestrator for documentation updates

## Data & ingestion

Source & cache declared in EMR; OHLCV in UTC; define missing-bar policy; fees/slippage model versioned.

## Testing & failure handling

* Golden-set parity; unit tests for rule semantics; determinism (seeded) test; small-universe integration run
* On any failed hook/validator: **stop**, log, notify, do not proceed

## File layout

**Clean Script vs Data Separation:**

```
/scripts/                    # Execution scripts (organized by agent)
├── engine/                  # /build-engine generates here
│   ├── backtest.py         # Main backtest engine (with optimization components)
│   ├── strategy_engine.py  # Generated strategy code
│   ├── core/              # Portfolio, orders, risk management, FilterGateManager
│   ├── data/              # Data fetching and processing (with optimization)
│   ├── execution/         # Fill simulation, fees, slippage
│   ├── optimization/      # ReferenceEngine and optimization infrastructure
│   └── utils/             # Config parsing, progress tracking
├── analyzer/              # /run execution scripts
├── single_analysis/       # /analyze-single-run scripts
├── single_evaluation/     # /evaluate-single-run scripts
├── optimization/          # /run-optimization scripts (OptimizationEngine)
└── opt_evaluation/        # /evaluate-optimization scripts

/data/                      # Generated outputs (separate from scripts)
├── runs/{run_id}/         # Single backtest outputs
│   ├── manifest.json
│   ├── metrics.json
│   ├── trades.csv, events.csv, series.csv
│   └── figures/
├── optimization/{study_id}/  # Parameter optimization outputs
├── reports/               # Generated PDF reports
└── cache/                 # Data fetching cache

/docs/                     # Documentation and registries
├── EMR.md SMR.md ECL.md SCL.md
├── runs/run_registry.csv
├── notices/{ECN|SER|SDCN}/
└── research/
```

## Guardrails

* No lookahead (features ≤ t; actions next bar open)
* Fees/slippage + minNotional rounding; flag impossible fills
* Accounting: `Equity_{t+1} = Equity_t + realizedPnL − fees`
* Determinism; sanity flags (e.g., extreme Sortino, zero DD)

## Enhanced Validation Framework (MANDATORY)

**Universal Accounting Identity Gate (All Agents):**
* **VERIFY**: `Equity_final = Equity_initial + Sum(realized_PnL) + Sum(unrealized_PnL) - Sum(fees)`
* **CROSS-CHECK**: Visual equity curve direction matches reported return sign
* **RECONCILE**: `final_equity ≈ initial_capital × (1 + total_return)` within 1% tolerance

**Critical Red Flags (Immediate Escalation):**
* Positive metrics with declining equity chart → HALT, escalate to Builder
* Large unrealized losses not reflected in total return → FLAG for review  
* Final equity calculation doesn't reconcile with reported performance → STOP
* Open positions at period end without proper mark-to-market → WARN

**Mandatory Quality Gates:**
* **Visual-Numerical Consistency**: Equity visualization must match metrics direction
* **Open Position Disclosure**: >5 open trades or >20% unrealized capital requires explicit flagging
* **Multi-Source Validation**: series.csv, metrics.json, and visualizations must be consistent
* **Accounting Reconciliation**: All data sources must reconcile within 1% tolerance

## Progress Bar Standards

* **Python Scripts**: All must use unified progress libraries (tqdm, rich, progressbar2) with ETA
* **Format**: `Task description... ████████░░░░ 75% (~2 min remaining)`
* **One progress bar per major task** - no micro-task progress spam
* **Integration**: Python scripts report to shared progress tracking system
* **Claude Level**: Display umbrella progress for command coordination phases

## Visualization Standards

* **Research-Based Design**: No dual-axis charts, prioritize visual clarity over metric cramming
* **3-Panel Default**: Equity curve (70%) + Drawdown (20%) + Activity (10%) stacked layout
* **Professional Quality**: PDF vector graphics for LaTeX, PNG 300+ DPI for HTML
* **Smart Time Axis**: Auto-detect optimal spacing (<90d: daily, <365d: weekly, <1095d: monthly, 3y+: quarterly)
* **Template Configuration**: Basic/Enhanced/Advanced levels with benchmark and per-symbol options
* **File Formats**: Generate both PDF and PNG, use PDF for LaTeX reports, PNG for HTML fallback
* **Color Standards**: Colorblind-friendly palettes, professional typography, publication-ready quality

## Configuration Systems

**Single-Run Configuration:**
* **Strategy Template**: Defines parameter schema (types, ranges, descriptions)
* **Auto-generation**: `/build-engine` creates `parameter_config.md` template
* **Single Execution**: `/run` reads all settings from `parameter_config.md` (no CLI arguments)
* **Version Control**: All parameter configs are version-controlled for reproducibility

**Optimization Configuration:**
* **Optimization Template**: Defines parameter sweep ranges, search method, validation approach
* **Manual Creation**: User creates `optimization_config.json` with parameter ranges and search strategy
* **Search Methods**: Grid search, random search, Bayesian optimization
* **Walk-Forward Setup**: Training/validation windows, rolling periods, robustness testing
* **Optimization Execution**: `/run-optimization` reads `optimization_config.json` for parameter sweep specification

## Registry & Git

**Single-Run Registry:**
* Single-Analyzer appends `/docs/runs/run_registry.csv` per individual run
* Each row tracks single backtest execution with performance metrics
* **Enhanced validation tracking**: `accounting_verified`, `open_positions_end`, `unrealized_pnl_impact`, `validation_warnings`
* **Quality assurance**: Registry entry only created if all validation gates pass

**Optimization Registry:**
* Optimizer appends `/docs/optimization/optimization_registry.csv` per parameter study
* Each row tracks optimization study with parameter sweep metadata and results

**Git Integration:**
* On any `/docs/**` change (reports, changelogs, notices, registries) **auto-commit & push**
* Do **not** commit `/data/**` by default (individual runs and optimization studies)