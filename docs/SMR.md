# SMR.md — "Crypto SMA Crossover + RSI + ATR Risk"

---

## 1) Strategy Overview
- **Name**: Crypto SMA Crossover + RSI + ATR Risk
- **Market/Universe**: Binance USDT spot pairs (exclude stablecoins & leveraged tokens)
- **Asset Selection**: Scan all eligible symbols daily; rank by 90-day Relative Strength (RS)
- **Timeframe**: 1D (daily candles, UTC)
- **Scope**: Long-only

---

## 1a) Strategy Description (Narrative)
Each day at the close, the strategy builds an eligible universe of Binance USDT pairs that meet history and liquidity requirements. It applies a trend filter (price above a long-term average) and a momentum screen (RSI above a neutral threshold), ranking survivors by 90-day Relative Strength to focus on the top tier. From this shortlist, the system monitors for a **Simple Moving Average (SMA) crossover** where the short-term SMA crosses above the long-term SMA, confirmed by sufficient volume. When a valid crossover occurs, it enters at the next day’s open. Once in a position, it uses an **ATR-based initial stop** sized to a fixed risk budget, a **trailing ATR stop** to lock in gains, and an optional **3R take-profit**. Exits are executed at the next day’s open after the hit day. Portfolio exposure is managed via volatility-adjusted sizing, symbol caps, and concurrency limits.

---

## 2) Entry Logic — *When do we BUY?*
- **Information / Markers Used**
  - Trend filter: CLOSE vs SMA200
  - Momentum: RSI(14)
  - Relative Strength (RS): 90-day return percentile across eligible symbols
  - Crossover: SMA(50) vs SMA(200)
  - Volume confirmation (moderate): recent quote volume vs 20-day average
  - (Optional breadth gate): BTC CLOSE > BTC SMA200

- **Parameters**
  - **Liquidity**: 30-day avg quote volume ≥ **250,000 USDT**
  - **Data requirement**: ≥ **365** daily bars
  - **Trend**: CLOSE > SMA200
  - **Momentum**: RSI(14) > **50**
  - **RS shortlist**: keep symbols with 90-day return in **top 20%** of eligible universe (min cap: top 40 symbols if universe is small).  
    - **Tie-breaks**: higher ROC(63), then lower ATR(14)%.
  - **Crossover**: SMA(50) crosses **above** SMA(200)
  - **Volume confirmation (soft)**: SumQuoteVolUSDT(3) ≥ **1.2×** SMA(QuoteVolUSDT,20)
  - **Breadth gate (optional, default OFF)**: BTC CLOSE > BTC SMA200

- **Mechanic / Condition**
  1. Symbol passes Liquidity + Data requirement.
  2. Passes Trend & Momentum at F (daily close).
  3. Is in RS shortlist at F.
  4. An **SMA(50) crossover above SMA(200)** occurs with soft volume confirmation → **BUY signal**.

- **Trigger Evaluation Time**: At daily close (F)  
- **Execution Rule**: **Buy at next day’s OPEN (F+1)** after signal day

---

## 3) Exit Logic — *When do we SELL?*
- **Information / Markers Used**
  - Initial stop: ATR(14)
  - Trailing stop: ATR(14) from highest close since entry
  - Take-profit: multiple of initial risk (R)
  - Time exit: max holding bars

- **Parameters**
  - **Initial Stop**: `SL = entry_price − 2.0 × ATR14_at_entry`
  - **Trailing Stop**: `Trail = highest_close_since_entry − 3.0 × ATR14_current`
  - **Take-Profit (optional)**: Exit at **+3R** (3 × initial risk)
  - **Time Exit (safety)**: **90** trading days

- **Mechanic / Condition**
  - After entry, compute initial risk `R = entry − SL`.
  - On each day H in the position, check intrabar hits in order:
    1) **Hard SL**, 2) **Take-Profit (if enabled)**, 3) **Trailing stop**, 4) **Time exit**.
  - If hit detected on H, mark exit reason and **execute SELL at H+1 OPEN**.

- **Collision Handling**: **SL precedence** → TP → Trailing → Time  
- **Execution Rule**: Exits execute at **next day’s OPEN** after the hit day

---

## 4) Position Management
> Picked modes are explicit; alter if needed.

### 4.1 Portfolio Accounting Mode
- **Mark-to-market** — Equity = cash + live value of open positions

### 4.2 Position Sizing Strategy
- **Volatility-Adjusted (Risk Targeting)**
  - **Risk budget per trade**: **0.75%** of portfolio equity
  - **Position size (units)**: `units = (risk_budget × equity) / (2.0 × ATR14_at_entry)`
  - Apply price-tick rounding, enforce exchange lot & minNotional; floor to nearest tradable lot; skip if below minNotional.

- **Caps / Constraints**
  - Max concurrent positions: **30**
  - Max weight per symbol: **10%** of equity
  - Daily new capital deployed cap: **20%** of equity
  - One open position per symbol (no pyramiding)

- **Re-entry / Cooldown / Scaling**
  - Cooldown **7 daily bars** after exit before a symbol can re-enter
  - No pyramiding; partial exits disabled (single-shot exit)

---

## 5) Filters & Eligibility
- **Data Requirements**: ≥ **365** daily bars available before any signal use
- **Tradability Filters**
  - 30-day average quote volume ≥ **250,000 USDT**
  - Exclude stablecoins, leveraged tokens, and obviously ill-quoted pairs
- **Run Boundaries**
  - Backtest defaults: START = `2019-01-01`; END = configurable

---

## 6) Conflict Handling
- **Buy vs Sell same bar**: **Sell takes precedence** (risk first)  
- **Exit Collisions**: **SL → TP → Trailing → Time** (strict order)

---

## 7) Visualization Configuration (Optional)
*(leave empty to use professional defaults)*

### 7.1 Visualization Level
- **Enhanced** — equity curve + drawdown + per-symbol charts with event overlays

### 7.2 Display Options
- **Benchmark Symbol**: BTC
- **Per-Symbol Analysis**: yes
- **Trade Markers**: all (mark entries, SL/TP/Trail exits, and SMA crossovers)
- **Time Period Shading**: yes (shade trending periods)

### 7.3 Custom Analysis Preferences
- Report RS percentile at entry, RSI value at entry, and SMA crossover frequency.

### 7.4 Other Strategy Notes
- Breadth gate via BTC SMA200 is supported but default OFF to avoid over-filtering.
- Consider enabling breadth gate during bearish crypto market conditions.

---

## Checklist (must be ticked before running)
- [ ] Market/Universe defined  
- [ ] Asset Selection method specified  
- [ ] Timeframe specified  
- [ ] Entry logic (markers, parameters, condition, trigger time, execution) defined  
- [ ] Exit logic (markers, parameters, condition, collisions, execution) defined  
- [ ] **Portfolio Accounting Mode chosen**  
- [ ] **Position Sizing Strategy chosen**  
- [ ] Data Requirements specified  
- [ ] Filters & Eligibility defined  
- [ ] Conflict Handling defined  
- [ ] (Optional) Visualization/Analysis preferences considered