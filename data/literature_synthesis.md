# Mathematical Synthesis: Microstructure, Order Flow Dynamics & Price Discovery Literature

This document synthesizes the foundational mathematical frameworks, theorems, and structural equations from the three seminal papers:
1. **Kyle (1985)** — *"Continuous Auctions and Insider Trading"* (*Econometrica*, 53(6), 1315-1335)
2. **Bouchaud, Farmer, & Lillo (2008)** — *"How markets slowly digest changes in supply and demand"* (*arXiv:0809.0822*)
3. **Inoua & Smith (2023)** — *"The Classical Theory of Supply and Demand"* (*arXiv:2307.00413*)

---

## 1. Kyle (1985): Continuous Auctions, Order Flow & Market Depth ($\lambda$)

### 1.1 The Theoretical Setup
The market operates over a time interval $t \in [0, T]$. There are three types of market participants:
1. **Informed Trader**: Possesses private information on the liquidation value $\tilde{v} \sim \mathcal{N}(p_0, \Sigma_0)$. Maximizes expected terminal profit $E[\int_0^T (\tilde{v} - P_t) dx_t \mid \tilde{v}]$.
2. **Noise Traders**: Submit aggregate random order flow $du_t = \sigma_u dW_t^u$, where $W_t^u$ is a standard Brownian motion with variance parameter $\sigma_u^2$.
3. **Competitive Market Makers**: Set price $P_t$ competitively given the aggregate order flow history $dY_t = dx_t + du_t$ under semi-strong market efficiency:
   $$P_t = E[\tilde{v} \mid Y_s, 0 \le s \le t]$$

### 1.2 Continuous-Time Equilibrium Equations
Kyle proves that a unique linear equilibrium exists:
- **Price Setting Rule (Market Impact $\lambda$)**:
  $$dP_t = \lambda(t) dY_t = \lambda(t) [dx_t + du_t]$$
  where $\lambda(t)$ is **Kyle's Lambda** (the illiquidity / price impact parameter):
  $$\lambda(t) = \frac{\sqrt{\Sigma_t}}{\sigma_u \sqrt{T - t}} = \frac{\sqrt{\Sigma_0}}{\sigma_u \sqrt{T}} = \text{constant} \quad (\lambda > 0)$$

- **Informed Trading Strategy ($\beta$)**:
  $$dx_t = \beta(t) (\tilde{v} - P_t) dt$$
  where the trading intensity is:
  $$\beta(t) = \frac{\sigma_u}{\sqrt{\Sigma_0} \sqrt{T}} = \text{constant}$$

- **Variance of Informed Private Information ($\Sigma_t$)**:
  $$\Sigma_t = \text{Var}(\tilde{v} \mid Y_s, 0 \le s \le t) = \Sigma_0 \left(1 - \frac{t}{T}\right)$$
  Private information is incorporated into price at a strictly constant linear rate:
  $$\frac{d\Sigma_t}{dt} = -\frac{\Sigma_0}{T}$$

### 1.3 Key Microstructure Insights for Algorithmic Execution
1. **Price Impact is Proportional to Order Flow Imbalance**: Price changes reflect the net order flow $dY_t$:
   $$\Delta P_t = \lambda \cdot (\text{Informed Flow}_t + \text{Noise Flow}_t)$$
2. **Order Flow Noise Camouflage**: The informed trader paces execution to camouflage informed flow inside background noise trading ($\sigma_u$).
3. **Information Decay**: The unrevealed information variance $\Sigma_t$ declines linearly over the trading horizon $T$.

---

## 2. Bouchaud, Farmer, & Lillo (2008): Long-Memory Order Flow & Propagator Models

### 2.1 The Long-Memory Property of Order Flow
Empirical transaction signs $\epsilon_t \in \{+1 \text{ (Buyer-Initiated)}, -1 \text{ (Seller-Initiated)}\}$ exhibit power-law autocorrelation with long memory:
$$C(\tau) = \langle \epsilon_t \epsilon_{t+\tau} \rangle \sim c_0 \tau^{-\gamma} \quad (0 < \gamma < 1)$$
Typically on major financial assets, $\gamma \approx 0.4 - 0.7$.

### 2.2 The Transient Impact (Propagator) Model
If price impact were permanent and linear, long-memory order flow would cause prices to be super-diffusive (predictable trends, violating no-arbitrage). Therefore, individual transaction impact must be **transient and decay over time**:
$$P_t = P_0 + \sum_{s < t} G(t - s) \epsilon_s v_s^\psi + \eta_t$$
where:
- $G(\tau)$ is the **Bare Impact Propagator function**:
  $$G(\tau) = \frac{\Gamma_0}{(1 + \tau)^\beta} \quad \text{with} \quad \beta = \frac{1 - \gamma}{2}$$
- $\psi \approx 0.2 - 0.3$ is the volume concavity exponent.
- $\eta_t$ is a martingale noise term.

### 2.3 The Master Square-Root Law of Market Impact
For a large metaorder of total volume $Q$ executed in a market with daily volume $V$ and daily volatility $\sigma_D$:
$$I(Q) = Y \cdot \sigma_D \sqrt{\frac{Q}{V}}$$
where $Y \approx 0.6 - 0.8$ is a universal dimensionless constant.

### 2.4 Asymmetric Liquidity & Spread Dynamics
- The bid-ask spread $S_t$ compensates market makers for adverse selection against informed order flow:
  $$S_t = 2 \lambda \sigma_u \sqrt{\Delta t} + C_{\text{inventory}} + C_{\text{fixed}}$$
- Following a temporary liquidity shock (e.g. order book sweep), the spread decays back to equilibrium following a power-law relaxation:
  $$S_t - S_0 \sim t^{-\theta}$$

---

## 3. Inoua & Smith (2023): Classical Reservation Price Theory & Supply-Demand Equilibrium

### 3.1 Reservation Price Distributions
Classical supply and demand are defined by observable monetary valuations rather than unobservable neoclassical utility functions:
- **Buyer Reservation Price (Willingness to Pay, WTP)**: $v \sim F_b(v)$ with density $f_b(v)$.
- **Seller Reservation Price (Willingness to Accept, WTA)**: $c \sim F_s(c)$ with density $f_s(c)$.

### 3.2 Market Demand & Supply Schedules
- **Market Demand**: $D(p) = N_b [1 - F_b(p)]$ (number of buyers with $v \ge p$).
- **Market Supply**: $S(p) = N_s F_s(p)$ (number of sellers with $c \le p$).

### 3.3 Excess Demand & Dynamic Price Discovery
- **Excess Demand Function**:
  $$z(p) = D(p) - S(p)$$
- **Decentralized Tatonnement Dynamics**:
  Price updates causally in the direction of order book excess demand:
  $$\frac{dp}{dt} = \kappa \cdot z(p) = \kappa [D(p) - S(p)]$$
  Equilibrium price $p^*$ occurs at the structural zero of excess demand: $z(p^*) = 0 \implies D(p^*) = S(p^*)$.

---

## 4. Unified Quantitative Architecture: Microstructure Order Flow Scalping Engine

Integrating the three papers creates a complete, production-grade microstructure trading engine:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Kyle (1985) Market Depth & Information Flow Layer                        │
│    • Estimates instantaneous Kyle's Lambda: λt = Cov(ΔPt, ΔYt) / Var(ΔYt)  │
│    • Tracks informed order flow vs background noise trading (σu)            │
│    • Paces execution intensity to minimize adverse price impact             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Bouchaud et al. (2008) Order Flow Memory & Propagator Layer             │
│    • Computes rolling order sign autocorrelation C(τ) and memory exponent γ │
│    • Evaluates transient price impact G(t-s) to identify exhausted runs     │
│    • Applies Square-Root Law I(Q) = Y · σ · sqrt(Q/V) for realistic slippage│
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Inoua & Smith (2023) Reservation Price & Excess Demand Layer             │
│    • Reconstructs buyer WTP and seller WTA order book distributions        │
│    • Computes real-time Excess Demand z(p) = D(p) - S(p)                    │
│    • Identifies structural liquidity imbalances and equilibrium reversion   │
└─────────────────────────────────────────────────────────────────────────────┘
```
