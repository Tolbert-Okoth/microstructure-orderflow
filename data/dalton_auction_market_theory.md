# Market Profile & Auction Market Theory: Theoretical Synthesis & Mathematical Framework

**Primary Reference:**
James F. Dalton, Eric T. Jones, Robert B. Dalton (1990 / 2007). *Mind Over Markets: Power Trading with Market Generated Information*. Probus Publishing / John Wiley & Sons. ISBN: 978-0934380539.

---

## 1. Core Principles of Auction Market Theory (AMT)

1. **The Purpose of the Market**: The market exists for one primary reason: to facilitate two-sided trade between buyers and sellers.
2. **Price vs. Value vs. Time**:
   - **Price** is the advertising mechanism (it moves up to find sellers and down to find buyers).
   - **Time** regulates price and determines acceptance vs. rejection (if price spends a lot of time at a level, that price is accepted as fair value; if price touches a level and immediately reverses, it represents rejection / excess).
   - **Volume** is the measure of facilitation (how successfully the market is conducting trade).
3. **Market Equation**:
   $$\text{Market Generated Information} = \text{Price} + \text{Time} + \text{Volume}$$

---

## 2. Time-Price Opportunity (TPO) & Value Area Mathematics

### 2.1 The TPO Distribution
A Time-Price Opportunity (TPO) represents a price level touched during a specific 30-minute time bracket (conventionally labeled A, B, C, ... Z).
Let $P_k$ be a discrete price tick (e.g. 0.10 for XAUUSD) and $t_j$ be the $j$-th 30-minute interval within the trading session $j \in \{1, \dots, N\}$.
The TPO indicator function is:
$$\text{TPO}(P_k, t_j) = \begin{cases} 1 & \text{if } \text{Low}(t_j) \le P_k \le \text{High}(t_j) \\ 0 & \text{otherwise} \end{cases}$$

The total TPO profile count at price $P_k$ is:
$$\text{TPO\_Count}(P_k) = \sum_{j=1}^{N} \text{TPO}(P_k, t_j)$$

### 2.2 Point of Control (POC)
The Point of Control $P_{\text{POC}}$ is the price level that generated the greatest number of TPOs (and/or greatest trading volume) during the session:
$$P_{\text{POC}} = \arg\max_{P_k} \text{TPO\_Count}(P_k)$$
In case of ties, the POC closest to the midpoint of the daily range is chosen.

### 2.3 Exact 70% Value Area Algorithm (Steidlmayer / Dalton)
The Value Area represents the price range that encompasses approximately **70%** (one standard deviation, $\sim 68.27\%$) of total session activity:
$$\text{Total\_TPOs} = \sum_{k} \text{TPO\_Count}(P_k)$$
$$\text{Target\_TPOs} = 0.70 \times \text{Total\_TPOs}$$

**Algorithm Execution**:
1. Initialize $\text{VA\_Set} = \{ P_{\text{POC}} \}$ with cumulative count $C = \text{TPO\_Count}(P_{\text{POC}})$.
2. Set upper pointer $u = \text{POC\_index} + 1$ and lower pointer $l = \text{POC\_index} - 1$.
3. At each step, calculate the two-tick sums:
   $$S_{\text{up}} = \text{TPO\_Count}(P_u) + \text{TPO\_Count}(P_{u+1})$$
   $$S_{\text{down}} = \text{TPO\_Count}(P_l) + \text{TPO\_Count}(P_{l-1})$$
4. If $S_{\text{up}} > S_{\text{down}}$:
   - Add $P_u, P_{u+1}$ to $\text{VA\_Set}$, $C \leftarrow C + S_{\text{up}}$, $u \leftarrow u + 2$.
5. If $S_{\text{down}} > S_{\text{up}}$:
   - Add $P_l, P_{l-1}$ to $\text{VA\_Set}$, $C \leftarrow C + S_{\text{down}}$, $l \leftarrow l - 2$.
6. If $S_{\text{up}} = S_{\text{down}}$:
   - Add both, $C \leftarrow C + S_{\text{up}} + S_{\text{down}}$, $u \leftarrow u + 2, l \leftarrow l - 2$.
7. Repeat until $C \ge \text{Target\_TPOs}$.
8. $\text{VAH} = \max(\text{VA\_Set})$, $\text{VAL} = \min(\text{VA\_Set})$.

---

## 3. Initial Balance (IB) & Session Anatomy

The **Initial Balance (IB)** represents the price range established during the first hour of trading (periods A and B, or first twelve 5-minute bars):
$$\text{IBH} = \max_{t \in [t_0, t_0 + 60\text{min}]} \text{High}(t)$$
$$\text{IBL} = \min_{t \in [t_0, t_0 + 60\text{min}]} \text{Low}(t)$$
$$\text{IBR} = \text{IBH} - \text{IBL}$$

### Range Extensions (RE):
- Upper Range Extension: $\text{RE}_{\text{high}} = \max(\text{High}_{\text{session}} - \text{IBH}, 0)$
- Lower Range Extension: $\text{RE}_{\text{low}} = \max(\text{IBL} - \text{Low}_{\text{session}}, 0)$
- IB Extension Multiplier: $\kappa_{\text{ext}} = \frac{\text{Session Range}}{\text{IBR}}$

---

## 4. Dalton's 4 Open Types (Early Session Hypothesis)

1. **Open-Drive (OD)**:
   - Price opens, immediately drives aggressively in one direction without any pullback beyond 1-2 ticks from the open.
   - Represents extreme Other Timeframe (OTF) directional conviction. High probability of a Trend Day.
2. **Open-Test-Drive (OTD)**:
   - Price opens, tests in one direction (testing prior day high/low, VAH/VAL, or POC), finds responsive rejection, and drives aggressively in the opposite direction.
3. **Open-Rejection-Reverse (ORR)**:
   - Price opens, penetrates outside prior value or range, meets intense responsive activity, reverses sharply back through the opening price, and drives across the entire value area.
4. **Open-Auction (OA)**:
   - Price opens and oscillates inside the opening range with no clear direction. Indicates balance, uncertainty, and two-sided rotational auction.

---

## 5. Dalton's 6 Day Types

1. **Trend Day (Double / Multi Distribution)**:
   - Wide range, narrow or moderate IB. Unidirectional range extensions throughout the session. Value Area continuously shifts. High directional volume.
2. **Normal Day**:
   - Wide Initial Balance created by early OTF entry. Little to no range extension during the remainder of the session ($\kappa_{\text{ext}} \le 1.2$). Rotational auction inside IB.
3. **Normal Variation Day**:
   - Moderate Initial Balance. Range extension occurs in one direction later in the day (periods C-E), extending range by $1.0\times$ to $2.0\times$ the IBR ($\kappa_{\text{ext}} \in [1.2, 2.2]$).
4. **Neutral Day**:
   - Range extension occurs on *both* sides of the Initial Balance ($\text{RE}_{\text{high}} > 0$ and $\text{RE}_{\text{low}} > 0$).
   - *Neutral-Center*: Close is near the POC (balanced auction).
   - *Neutral-Extreme*: Close is at an extreme of the day (one side overpowered the other late in the session).
5. **Non-Trend Day**:
   - Very narrow Initial Balance, no range extension ($\kappa_{\text{ext}} \approx 1.0$), low volume. The market is waiting for news or high-impact macroeconomic data.
6. **Running Trend Day**:
   - Most aggressive trend. Price never rotates; single prints form in every bracket; closes at the absolute high/low.

---

## 6. Structural Auction Formations & The Dalton 80% Rule

### 6.1 Excess & Tails vs. Poor Extremes
- **Buying Tail**: A single-print low consisting of at least 2 consecutive 30-minute brackets where price probed lower and was aggressively rejected by buyers. Represents true structural support.
- **Selling Tail**: A single-print high consisting of at least 2 consecutive brackets where price probed higher and was aggressively rejected by sellers.
- **Poor High / Poor Low**: High or low of the session consists of 2 or more TPOs side-by-side with no single-print excess. Indicates an **unfinished auction** that the market will revisit and auction through.

### 6.2 The Dalton 80% Rule (Value Area Rotation)
**Formulation**:
If the market opens outside or inside the previous day's Value Area ($\text{VAH}_{d-1}, \text{VAL}_{d-1}$), and subsequently **accepts** into the Value Area by printing two consecutive 30-minute bars (or six consecutive 5-minute bars) inside the Value Area boundaries:
- If price accepts *below* $\text{VAH}_{d-1}$ from above $\implies$ **80% statistical probability of rotating down to test $\text{VAL}_{d-1}$**.
- If price accepts *above* $\text{VAL}_{d-1}$ from below $\implies$ **80% statistical probability of rotating up to test $\text{VAH}_{d-1}$**.

### 6.3 Initiative vs. Responsive Auction Activity
- **Initiative Trade**: Trading outside of prior value in the direction of the breakout (e.g., buying above $\text{VAH}_{d-1}$ or selling below $\text{VAL}_{d-1}$). Requires high volume and high momentum.
- **Responsive Trade**: Trading against price extremes back toward fair value (e.g., buying below $\text{VAL}_{d-1}$ or selling above $\text{VAH}_{d-1}$).
