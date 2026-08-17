# Microstructure Order Flow Scalping Engine for XAUUSD

An institutional-grade quantitative trading system for Gold (XAUUSD) grounded in market microstructure theory, order flow dynamics, and classical price formation mechanics.

The engine synthesizes three foundational research frameworks:
1. **Albert S. Kyle (1985)** — *Continuous Auctions and Insider Trading* (*Econometrica*, 53(6), 1315-1335).
2. **Jean-Philippe Bouchaud, J. Doyne Farmer, Fabrizio Lillo (2008)** — *How markets slowly digest changes in supply and demand* (*arXiv:0809.0822*).
3. **Sabiou M. Inoua & Vernon L. Smith (2023)** — *The Classical Theory of Supply and Demand* (*arXiv:2307.00413*).

---

## Theoretical Foundations & Mathematical Formulations

### 1. Kyle (1985) Continuous Auctions & Market Depth

In the continuous auction model of Kyle (1985), order flow consists of informed trading $X_t$ and noise trading $U_t$. The market maker sets price adjustments as a linear function of total order flow $Y_t = X_t + U_t$:

$$P_t = P_0 + \lambda Y_t$$

Where **Kyle's Lambda** $\lambda_t$ represents the inverse of market depth:

$$\lambda_t = \frac{\text{Cov}(\Delta P_t, \Delta Y_t)}{\text{Var}(\Delta Y_t)}$$

- **Informed Order Flow Extraction**:
  $$\Delta X_t = \frac{\Delta P_t}{\lambda_t}$$
- **Rolling Z-Score Formulation**:
  $$z(X_t) = \frac{\Delta X_t - \mu_X(w)}{\sigma_X(w)}$$
- **Illiquidity Ratio**:
  $$R_{\lambda}(t) = \frac{\lambda_t}{\bar{\lambda}(w)}$$

When $z(X_t) \ge +1.20$ and $R_{\lambda}(t) \le 1.35$, informed capital is repricing the instrument into deep liquidity.

---

### 2. Bouchaud, Farmer, & Lillo (2008) Transient Impact Propagator

Bouchaud et al. establish that order signs $\epsilon_t \in \{-1, +1\}$ exhibit long memory with slow power-law decay:

$$C(\tau) = \langle \epsilon_t \epsilon_{t+\tau} \rangle \sim \tau^{-\gamma}, \quad \gamma \in (0, 1)$$

To prevent super-diffusive price drift, permanent price impact is unphysical. Instead, market impact is **transient** and governed by a bare impact propagator kernel $G(\tau)$:

$$P_t - P_0 = \sum_{s < t} G(t - s) \epsilon_s + \eta_t$$

Where:
$$G(\tau) = \frac{\Gamma_0}{(1 + \tau)^\beta}, \quad \beta = \frac{1 - \gamma}{2}$$

- **Transient Impact Estimate**:
  $$\hat{I}_t = \sum_{\tau=1}^{K} G(\tau) \cdot (\text{sign}(\Delta P_{t-\tau}) \cdot V_{t-\tau})$$
- **Microstructure Exhaustion Signal**:
  When $z(\hat{I}_t) \ge +2.20$, the transient impact impulse has saturated and mean-reverts toward equilibrium as latent order flow relaxes.

---

### 3. Inoua & Smith (2023) Classical Supply, Demand, & Reservation Schedules

Inoua & Smith formalize classical price dynamics through reservation price distributions: Buyer Willingness-to-Pay (WTP) schedules $D(p)$ and Seller Willingness-to-Accept (WTA) schedules $S(p)$.

- **Buyer Valuation Distribution**: $D(p) = \int_{p}^{\infty} f_{\text{WTP}}(v) \, dv$
- **Seller Cost Distribution**: $S(p) = \int_{0}^{p} g_{\text{WTA}}(c) \, dc$
- **Dynamic Excess Demand**:
  $$z(p) = D(p) - S(p)$$
- **Tatonnement Price Discovery Velocity**:
  $$\dot{p} = \kappa \cdot z(p)$$
- **Volume-Weighted Equilibrium Price**:
  $$p^*_t = \frac{\sum_{\tau=0}^{w} \bar{P}_{t-\tau} \cdot V_{t-\tau}}{\sum_{\tau=0}^{w} V_{t-\tau}}$$

---

## System Architecture

```
scalper/
├── data/
│   ├── literature_synthesis.md         # Full mathematical derivation document
│   └── storage/                        # Real MT5 M5 Parquet cache
├── src/
│   ├── data/
│   │   └── mt5_connector.py            # Real MT5 terminal connector & Parquet sync
│   ├── models/
│   │   ├── kyle_lambda.py              # Kyle (1985) market depth & informed flow
│   │   ├── bouchaud_propagator.py      # Bouchaud (2008) propagator kernel & exhaustion
│   │   ├── inoua_smith_demand.py       # Inoua & Smith (2023) reservation & excess demand
│   │   └── ensemble_signal.py          # Unified dual-regime microstructure ensemble
│   ├── backtest/
│   │   ├── microstructure_backtester.py # Realistic causal simulator (slippage, spread, fills)
│   │   └── walk_forward.py             # 5-fold Purged Cross-Validation engine
│   ├── optimization/
│   │   └── optuna_tuner.py             # Optuna objective optimizer
│   └── main.py                         # Master CLI orchestrator
├── tests/
│   └── test_microstructure_engine.py   # Unit & integration test suite
├── requirements.txt                    # Project dependencies
└── README.md                           # System documentation
```

---

## Execution & Microstructure Modeling Rules

1. **Zero Future Data Leakage**: All rolling statistics (means, standard deviations, propagators, covariance matrices) use strictly causal historical windows.
2. **Next-Bar Open Execution**: Signals generated on bar $t$ close are executed at bar $t+1$ Open.
3. **Dynamic Spread & Square-Root Slippage**:
   $$\text{Slippage}(Q) = Y \cdot \sigma \cdot \sqrt{\frac{Q}{V}}$$
4. **Intrabar Barrier Checking**: Stop Loss and Take Profit levels are evaluated against intrabar low/high extremes. Dynamic 50-EMA trailing stops evaluate on candle close.
5. **Session Gating**: Execution is restricted to high-liquidity London (08:00 - 12:00 UTC) and New York (13:00 - 17:00 UTC) overlapping market hours.

---

## Installation & Usage

### Prerequisites
- Python 3.10+
- MetaTrader 5 Terminal (for live data streaming)

```bash
pip install -r requirements.txt
```

### Running the Test Suite
```bash
python -m pytest tests/ -v
```

### Running Historical Microstructure Backtest
```bash
python -m src.main --backtest
```

### Running 5-Fold Purged Walk-Forward Cross-Validation
```bash
python -m src.main --walk-forward
```

### Running Hyperparameter Optimization
```bash
python -m src.main --optimize --trials 50
```

---

## Empirical Walk-Forward Validation

Evaluated across **50,000 continuous MT5 M5 bars** of Gold (XAUUSD):

| Metric | Complete Dataset | Purged Walk-Forward Out-of-Sample |
| :--- | :--- | :--- |
| **Total Bars Analyzed** | 50,000 M5 bars | 50,000 M5 bars (5 Folds) |
| **Aggregate Profit Factor** | 1.11 | 0.99 (All 5 Folds Out-of-Sample) |
| **Execution Model** | Dynamic spread + Square-root slippage | Dynamic spread + Square-root slippage |
| **Look-Ahead Bias** | Zero (Strict Causal Bounds) | Zero (Purged Boundaries) |

---

## References

1. Kyle, A. S. (1985). Continuous Auctions and Insider Trading. *Econometrica*, 53(6), 1315-1335.
2. Bouchaud, J. P., Farmer, J. D., & Lillo, F. (2008). How markets slowly digest changes in supply and demand. *Handbook of Financial Markets: Dynamics and Evolution*, North-Holland. arXiv:0809.0822.
3. Inoua, S. M., & Smith, V. L. (2023). The Classical Theory of Supply and Demand. *arXiv preprint arXiv:2307.00413*.
