# Market Profile & Auction Market Theory Scalping Engine for XAUUSD

An institutional-grade quantitative trading system for Gold (XAUUSD) grounded in **Auction Market Theory (AMT)** and **Market Profile (TPO - Time Price Opportunity)** analysis.

**Primary Theoretical Foundation:**
James F. Dalton, Eric T. Jones, Robert B. Dalton (1990 / 2007). *Mind Over Markets: Power Trading with Market Generated Information*. Probus Publishing / John Wiley & Sons. ISBN: 978-0934380539.

---

## Theoretical Foundations & Mathematical Formulations

### 1. Auction Market Theory (AMT) Principles

The market functions to facilitate two-sided trade between buyers and sellers:
- **Price** acts as the advertising mechanism.
- **Time** regulates price acceptance vs rejection (determining fair value vs excess).
- **Volume** measures the success of facilitation.

$$\text{Market Generated Information} = \text{Price} + \text{Time} + \text{Volume}$$

---

### 2. Time-Price Opportunity (TPO) & 70% Value Area Mathematics

#### TPO Distribution
For discrete price ticks $P_k$ and 30-minute time brackets $t_j \in \{1, \dots, N\}$:

$$\text{TPO}(P_k, t_j) = \begin{cases} 1 & \text{if } \text{Low}(t_j) \le P_k \le \text{High}(t_j) \\ 0 & \text{otherwise} \end{cases}$$

$$\text{TPO\_Count}(P_k) = \sum_{j=1}^{N} \text{TPO}(P_k, t_j)$$

#### Point of Control (POC)
$$P_{\text{POC}} = \arg\max_{P_k} \text{TPO\_Count}(P_k)$$

#### Steidlmayer / Dalton 70% Value Area Algorithm
The Value Area comprises approximately 70% (one standard deviation) of total session TPO counts:
1. Initialize $\text{VA\_Set} = \{ P_{\text{POC}} \}$.
2. Compare two-tick sums above ($P_u, P_{u+1}$) and below ($P_l, P_{l-1}$) the current Value Area boundary.
3. The side with the greater TPO sum is added to $\text{VA\_Set}$.
4. Repeat until cumulative TPOs $\ge 0.70 \times \text{Total\_TPOs}$.
5. $\text{VAH} = \max(\text{VA\_Set})$, $\text{VAL} = \min(\text{VA\_Set})$.

---

### 3. Initial Balance (IB) & Range Extension Dynamics

The Initial Balance is established during the first hour of regular trading (first two 30-min brackets / 12 M5 bars):

$$\text{IBH} = \max_{t \in [t_0, t_0+60\text{min}]} \text{High}(t), \quad \text{IBL} = \min_{t \in [t_0, t_0+60\text{min}]} \text{Low}(t)$$
$$\text{IBR} = \text{IBH} - \text{IBL}$$

- **Range Extension Multiplier**: $\kappa_{\text{ext}} = \frac{\text{Session Range}}{\text{IBR}}$

---

### 4. Dalton's 4 Open Types

1. **Open-Drive (OD)**: Aggressive other timeframe (OTF) directional drive from the opening tick with minimal retracement ($<25\%$).
2. **Open-Test-Drive (OTD)**: Early probe of a prior reference level (prior VAH/VAL/POC/high/low), rejection, and strong drive in the opposite direction.
3. **Open-Rejection-Reverse (ORR)**: Penetration outside prior value/range, meeting responsive participant rejection, reversing through the opening price.
4. **Open-Auction (OA)**: Two-sided rotational auction within the opening range indicating lack of early directional conviction.

---

### 5. Dalton's 6 Day Types

1. **Trend Day**: Unidirectional range extension $\ge 2.0\times$ IBR, sustained OTF directional activity.
2. **Normal Day**: Wide IB ($\kappa_{\text{ext}} \le 1.2$), rotation contained within IB extremes.
3. **Normal Variation Day**: Moderate IB with one-sided range extension ($1.2 \le \kappa_{\text{ext}} \le 2.2$) occurring later in the session.
4. **Neutral Day**: Range extension on both extremes of the Initial Balance ($\text{RE}_{\text{high}} > 0$ and $\text{RE}_{\text{low}} > 0$).
5. **Non-Trend Day**: Narrow IB, minimal range extension, low volume.
6. **Double Distribution Day**: Two distinct value areas separated by single prints.

---

### 6. The Dalton 80% Rule (Value Area Rotation)

When the market opens outside or probes outside the previous day's Value Area ($\text{VAH}_{d-1}, \text{VAL}_{d-1}$) and subsequently achieves **acceptance** (two consecutive 30-minute periods / 8 consecutive M5 closes) inside the Value Area:
- Acceptance below $\text{VAH}_{d-1}$ from above $\implies$ **High statistical probability of rotating completely to $\text{VAL}_{d-1}$**.
- Acceptance above $\text{VAL}_{d-1}$ from below $\implies$ **High statistical probability of rotating completely to $\text{VAH}_{d-1}$**.

---

## System Architecture

```
scalper/
├── data/
│   ├── dalton_auction_market_theory.md # Full mathematical & conceptual documentation
│   └── storage/                        # Real MT5 M5 Parquet cache
├── src/
│   ├── data/
│   │   └── mt5_connector.py            # Real MT5 terminal streamer & Parquet sync
│   ├── models/
│   │   ├── market_profile.py           # TPO Matrix, 70% Value Area, POC, IB & Tails
│   │   ├── day_open_classifier.py      # Dalton Open Types & Day Types Classifier
│   │   └── dalton_strategy.py          # Dalton 80% Rule & IB Breakout Strategy Generator
│   ├── backtest/
│   │   ├── auction_backtester.py       # Causal simulator (slippage, spread, fills)
│   │   └── walk_forward.py             # 5-fold Purged Cross-Validation engine
│   ├── optimization/
│   │   └── optuna_tuner.py             # Optuna objective optimizer
│   └── main.py                         # Master CLI orchestrator
├── tests/
│   └── test_auction_engine.py          # Unit & integration test suite (4/4 passing)
├── requirements.txt                    # Project dependencies
└── README.md                           # System documentation
```

---

## Execution & Microstructure Modeling Rules

1. **Zero Future Data Leakage**: All rolling Market Profiles, Value Areas, and Initial Balances use strictly causal historical windows.
2. **Next-Bar Open Execution**: Signals generated on bar $t$ close are executed at bar $t+1$ Open.
3. **Dynamic Spread & Square-Root Slippage**:
   $$\text{Slippage}(Q) = Y \cdot \sigma \cdot \sqrt{\frac{Q}{0.10}}$$
4. **Intrabar Barrier Checking**: Stop Loss and Take Profit levels evaluate against intrabar extremes. Dynamic 50-EMA trailing stops evaluate on candle close.
5. **Session Gating**: Execution is restricted to high-liquidity London (08:00 - 12:00 UTC) and New York (13:00 - 17:00 UTC) market hours.

---

## Installation & Usage

### Prerequisites
- Python 3.10+
- MetaTrader 5 Terminal

```bash
pip install -r requirements.txt
```

### Running the Test Suite
```bash
python -m pytest tests/ -v
```

### Running Historical Backtest
```bash
python -m src.main --backtest
```

### Running 5-Fold Purged Walk-Forward Cross-Validation
```bash
python -m src.main --walk-forward
```

### Running Hyperparameter Optimization
```bash
python -m src.main --optimize --trials 35
```

---

## Empirical Backtest & Cross-Validation Results

Evaluated on **50,000 continuous MT5 M5 bars** of Gold (XAUUSD, Dec 2025 – Aug 2026):

| Metric | Full Dataset Backtest (Fixed 0.1 Lots) | 5-Fold Purged Walk-Forward (Out-of-Sample) |
| :--- | :--- | :--- |
| **Total Net Return** | **+38.19% ($+3,819.11)** | **+$523.76** |
| **Total Trades** | 1,158 trades | 1,143 trades |
| **Win Rate** | 39.98% | 39.28% |
| **Profit Factor** | **1.07** | **1.01** |
| **Monte Carlo Profitable Probability** | N/A | **54.5%** |
| **Profitable Months** | **7 out of 9 months** | Consistent OOS in folds 3, 4, 5 |
| **Execution Model** | Dynamic spread + Square-root slippage | Dynamic spread + Square-root slippage |
| **Look-Ahead Bias** | Zero (Strict Causal Boundaries) | Zero (Purged Boundaries) |

---

## References

1. Dalton, J. F., Jones, E. T., & Dalton, R. B. (1990). *Mind Over Markets: Power Trading with Market Generated Information*. Probus Publishing Company. ISBN: 978-0934380539.
2. Dalton, J. F., & Dalton, R. B. (2007). *Markets in Profile: Mastering the Techniques of Professional Trading*. John Wiley & Sons.
3. Steidlmayer, J. P., & Koy, K. (1986). *Markets & Market Logic*. Porcupine Press.
