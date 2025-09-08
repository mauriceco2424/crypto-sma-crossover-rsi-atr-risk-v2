# Trading Strategy Framework Skeleton

A production-ready framework for building, backtesting, and optimizing trading strategies using Claude Code agents and a sophisticated pipeline architecture.

## 🚀 Quick Start

### **Prerequisites**
- **Git Bash** (required for automated GitHub integration)
- **GitHub CLI** installed (`gh auth login` completed)

### **New Strategy Project Initialization**

1. **Create New Project Directory**:
   ```bash
   mkdir new_strat
   cd new_strat
   ```

2. **Clone Skeleton Content** (contents only, not the folder):
   ```bash
   git clone https://github.com/mauriceco2424/trading_bot_skeleton.git temp_skeleton
   mv temp_skeleton/* .
   mv temp_skeleton/.* . 2>/dev/null || true
   rm -rf temp_skeleton
   ```

3. **Define Your Strategy**:
   Edit `docs/SMR.md` following the `docs/guides/STRAT_TEMPLATE.md` format.
   **Key**: Update the `**Name**: <Strategy Name>` field with your actual strategy name.

4. **Initialize Your Strategy Project**:
   ```bash
   /initialize
   ```
   This **automatically**:
   - Reads strategy name from `docs/SMR.md` (e.g., "RSI Momentum Strategy")
   - Creates GitHub repository: `rsi-momentum-strategy`
   - Renames folder `new_strat` → `rsi-momentum-strategy`  
   - Updates workspace file: `rsi-momentum-strategy.code-workspace`
   - Updates all files with your strategy name
   - Sets up clean git repository with remote origin
   - Makes initial commit and pushes to GitHub

   **No manual GitHub repo creation needed!**

5. **Setup Dependencies and Validation**:
   ```bash
   /validate-setup
   ```
   (This automatically runs `pip install -r requirements.txt` if dependencies are missing)

6. **Build and Test Your Strategy**:
   ```bash
   /validate-strategy && /plan-strategy && /build-engine
   /run && /analyze-single-run && /evaluate-single-run
   ```

## 🌿 Git Branching Workflow

### **Professional Development Workflow**

The framework supports professional Git branching for safe strategy development:

#### **Branch Strategy**
- **`main`**: Production-ready, tested strategy versions
- **`develop`**: Integration branch for combining features
- **`feature/[strategy-change]`**: Individual improvements or experiments

#### **Typical Development Cycle**

1. **Start New Feature/Experiment**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/improve-rsi-thresholds
   ```

2. **Develop and Test Locally**:
   ```bash
   # Make changes to strategy parameters or logic
   /run && /analyze-single-run && /evaluate-single-run
   
   # Commit improvements
   git add .
   git commit -m "Optimize RSI thresholds for better Sortino ratio"
   ```

3. **Push Feature Branch**:
   ```bash
   git push -u origin feature/improve-rsi-thresholds
   ```

4. **Create Pull Request** (when ready to merge):
   ```bash
   gh pr create --base develop --title "Optimize RSI thresholds" --body "
   ## Changes
   - Improved RSI oversold threshold from 30 to 25
   - Added volume confirmation filter
   
   ## Performance
   - Sortino ratio: 2.1 → 2.4
   - Max drawdown: 15% → 12%
   
   ## Testing
   - [x] Backtested on 3-year dataset
   - [x] Walk-forward validation passed
   - [x] All hooks and validators pass
   "
   ```

5. **Merge to Production** (after testing):
   ```bash
   # Switch to main and merge develop
   git checkout main
   git pull origin main
   git merge develop
   git push origin main
   
   # Tag stable versions
   git tag v1.2.0 -m "RSI threshold optimization release"
   git push --tags
   ```

#### **Branch Protection Setup**
Configure branch protection in your GitHub repository:
- Require pull request reviews before merging to `main`
- Require status checks (backtests) to pass
- Dismiss stale reviews when new commits are pushed

### **Strategy Development Best Practices**

#### **Feature Branch Naming**
- `feature/optimize-entry-logic`
- `feature/add-volume-filter`
- `experiment/test-momentum-signals`
- `bugfix/fix-position-sizing`

#### **Commit Messages**
- Clear, descriptive commits
- Include performance impact when relevant
- Reference backtest results in commit descriptions

#### **When to Merge**
✅ **Merge when**:
- Backtest performance improves key metrics
- All validation hooks pass
- Walk-forward analysis confirms robustness
- Code review completed (if working in team)

❌ **Don't merge when**:
- Untested changes
- Performance degrades without clear reason
- Validation failures or overfitting detected


## 📋 What This Skeleton Provides

### 🤖 **6 Specialized Agents**

**Common Agents:**
- **trading-orchestrator**: Coordinates pipeline, manages quality gates, handles documentation
- **trading-builder**: Implements backtest engines, optimizes performance, writes ECNs

**Single-Run Agents:**
- **trading-single-analyzer**: Executes backtests AND processes single run data, generates artifacts and visualizations
- **trading-single-evaluator**: Evaluates single-run performance and generates PDF reports

**Optimization Agents:**
- **trading-optimizer**: Executes parameter sweeps AND processes optimization studies into parameter performance matrices with walk-forward analysis
- **trading-optimization-evaluator**: Evaluates parameter optimization results and generates optimization reports

### ⚡ **Streamlined Command System**

**Setup & Planning (4 commands):**
| Command | Purpose |
|---------|---------|
| `/validate-setup` | Validate framework setup and dependencies |
| `/validate-strategy` | Validate strategy specification |
| `/plan-strategy` | Plan strategy development approach |
| `/build-engine` | Build trading engine and generate parameter template |

**Single-Run Path (3 commands):**
| Command | Purpose |
|---------|---------|
| `/run` | Execute single backtest with parameter_config.md |
| `/analyze-single-run` | Process single run data into metrics and visualizations |
| `/evaluate-single-run` | Evaluate single-run performance and generate PDF report |

**Optimization Path (2 commands):**
| Command | Purpose |
|---------|---------|
| `/run-optimization` | Execute parameter sweep AND process optimization study into parameter performance matrices |
| `/evaluate-optimization` | Evaluate parameter optimization and generate optimization report |

### 🔧 **Production Hook System**
- **6 core hooks** with P0/P1/P2 priorities
- Resource validation, artifact integrity, accounting checks
- Configurable timeouts and error handling
- Safety hooks for lookahead and accounting validation

### 📚 **Documentation Framework**
- **EMR/SMR**: Engine and Strategy Master Reports with versioning
- **ECL/SCL**: Append-only changelogs
- **ECN/SER/SDCN**: Change notices and evaluation reports
- JSON schemas for all data structures

## 🏗️ Architecture

### **Directory Structure**
```
├── .claude/                    # Claude Code configuration
│   ├── agents/                # 6 specialized agents
│   └── commands/              # 9 streamlined commands
├── docs/                      # Authoritative documentation
│   ├── runs/                  # Run registry and results
│   └── schemas/               # JSON schemas
├── configs/                   # Strategy configurations
├── tools/hooks/               # Hook system
│   ├── core/                  # Essential hooks
│   ├── lib/                   # Hook infrastructure
│   └── config/                # Hook configuration
├── cloud/                     # State management
│   ├── tasks/                 # Task planning
│   └── state/                 # Runtime state
└── data/                      # Run data (not committed)
    ├── runs/                  # Backtest results
    └── sandbox/               # Development data
```

### **Dual Workflow Paths**

**Single-Run Workflow:**
1. **Setup** → `/validate-setup` → `/validate-strategy` → `/plan-strategy` → `/build-engine`
2. **Execute** → `/run` → `/analyze-single-run` → `/evaluate-single-run`

**Parameter Optimization Workflow:**
1. **Setup** → Same as single-run setup (+ create optimization_config.json)
2. **Execute** → `/run-optimization` → `/evaluate-optimization`

### **Quality Gates**
- **Docs Fresh Gate**: EMR/SMR in sync with latest changes
- **Pre-run Gates**: Tests pass, no conflicting runs
- **Post-run Gates**: Artifacts complete, anomalies flagged

## 🎯 Usage Examples

### **Single-Run Strategy Development**

```bash
/validate-setup && /validate-strategy && /plan-strategy && /build-engine
/run && /analyze-single-run && /evaluate-single-run
```

### **Parameter Optimization Study**

```bash
/validate-setup && /validate-strategy && /plan-strategy && /build-engine
# Create optimization_config.json with parameter ranges
/run-optimization && /evaluate-optimization
```

## 🔒 Safety & Validation

- **No-lookahead enforcement**: Features use data ≤ t for actions at t+1
- **Accounting integrity**: Rigorous P&L tracking with fees/slippage
- **Deterministic execution**: Seeded operations for reproducibility
- **Statistical validation**: Multiple-testing corrections, overfitting detection
- **Realism checks**: Liquidity, slippage, trade density validation

## 📊 Output Artifacts

### **Clean Script vs Data Separation**

**Scripts** (`scripts/` folder - organized by agent):
- `scripts/engine/` - Complete backtest engine with optimization components (generated by `/build-engine`)
- `scripts/analyzer/` - Run execution coordination 
- `scripts/single_analysis/` - Performance analysis
- `scripts/single_evaluation/` - Strategy evaluation and reports
- `scripts/optimization/` - High-performance parameter optimization with speed optimizations
- `scripts/opt_evaluation/` - Optimization evaluation

**Data** (`data/` folder - generated outputs only):
- `data/runs/{run_id}/` - Individual backtest outputs
  - `manifest.json`: Run metadata and hashes
  - `metrics.json`: Performance statistics
  - `trades.csv`, `events.csv`, `series.csv`: Detailed data
  - `figures/`: Professional visualizations
- `data/optimization/{study_id}/` - Parameter optimization studies
- `data/reports/` - Generated PDF reports
- `data/cache/` - Data fetching cache

## 🛠️ Customization

### **Adding New Strategies**
1. Create strategy specification (.md)
2. Use `/validate-setup` → `/validate-strategy` → `/plan-strategy` to plan implementation
3. Framework builds everything automatically with `/build-engine`

### **Extending Hooks**
Add custom hooks in `tools/hooks/extended/` with proper priority and error handling.

### **Custom Agents**
Extend agent capabilities by modifying `.claude/agents/` configurations.

## 📈 Performance

- **Universal Speed Optimization**: Significant speedup for parameter sweeps through FilterGateManager, feature caching, and reference run optimization
- **Strategy-Agnostic**: Speed optimizations work with ANY trading strategy automatically
- **Memory efficient**: Configurable caching with resource profiling
- **Hardware-aware**: Auto-configures based on system capabilities
- **Incremental computation**: Monotone gate shortcuts and universe reduction for faster iterations

## 📖 Documentation Guide

All user guides and documentation are organized in `docs/guides/`:

| Document | Purpose |
|----------|---------|
| 📋 **[STRAT_TEMPLATE.md](docs/guides/STRAT_TEMPLATE.md)** | Strategy specification template - **Use this to define your strategy** |
| 📖 **[User-Guide.md](docs/guides/User-Guide.md)** | Complete user manual with commands, optimization benefits, visualization system, and best practices |

**Quick Start**: Begin with `docs/guides/STRAT_TEMPLATE.md` to define your strategy, then follow the complete workflow in `docs/guides/User-Guide.md`.

## 🤝 Contributing

This is a skeleton framework. Customize for your specific needs:
- Add domain-specific features
- Extend hook system for your workflows  
- Modify agents for specialized strategies
- Add custom validation rules

## 🔧 Troubleshooting

### **Screen Flickering Issues**
If you experience screen flickering during agent operations (especially on Windows), set Git Bash as your default terminal in VS Code:

1. **In VS Code**: Open Command Palette (Ctrl+Shift+P)
2. Type: "Terminal: Select Default Profile"
3. Choose: "Git Bash" to set it as your default terminal
4. Restart your terminal

This resolves flickering issues caused by terminal output buffering and provides a smoother experience with Claude Code agents.

### **GitHub CLI Setup**
For automated repository creation, ensure GitHub CLI is properly configured:

1. **Install GitHub CLI**: Download from https://cli.github.com/ or use `winget install GitHub.cli`
2. **Authenticate**: Run `gh auth login` and follow the prompts
3. **Verify**: Test with `gh auth status` to confirm authentication
4. **Permissions**: Ensure your GitHub token has repository creation permissions

**Note**: The `/initialize` command requires Git Bash and authenticated GitHub CLI to automatically create and configure your strategy repository.

## 📝 License

MIT License - Use freely for your trading strategy development.

---

**Ready to build your next profitable strategy?** 

Run `python validate_setup.py` to verify your setup, then start with `/validate-setup` to begin your strategy development!