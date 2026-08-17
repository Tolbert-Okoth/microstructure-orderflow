"""
Dalton Auction Market Theory Strategy Generator.
Implements:
1. The Dalton 80% Rule (Value Area Acceptance & Full Rotation)
2. Initial Balance (IB) Initiative Breakout Expansion
3. Responsive Value Extremes & Poor High/Low Reversion
4. Institutional Session Gating & Multi-Scale Trend Alignment
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import pandas as pd

from src.models.market_profile import MarketProfileEngine, MarketProfileParams
from src.models.day_open_classifier import DayOpenClassifier, OpenTypeParams


@dataclass
class DaltonStrategyParams:
    """Parameters for Dalton AMT Execution Strategy."""
    # 80% Rule parameters
    acceptance_bars: int = 8            # Number of consecutive M5 bars closed inside Value Area to confirm acceptance (40 mins)
    va_buffer: float = 0.50             # Value Area boundary buffer in $ (e.g. $0.50 for XAUUSD)
    min_va_width: float = 5.0           # Minimum prior Value Area width ($) to justify rotation trade
    
    # IB Breakout parameters
    min_ibr: float = 5.0                # Minimum IB range ($) to avoid dead-session fakeouts
    max_ibr: float = 30.0               # Maximum IB range ($) to avoid exhausted openings
    ib_breakout_buffer: float = 0.40    # Buffer above/below IBH/IBL to confirm breakout
    
    # Session & Friction Filters
    max_spread_points: float = 45.0     # Max MT5 spread points
    allowed_sessions: List[Tuple[float, float]] = field(default_factory=lambda: [(8.0, 12.0), (13.0, 17.0)])
    
    # Multi-Scale Trend Alignment
    ema_fast_period: int = 20
    ema_slow_period: int = 50
    ema_macro_period: int = 288         # 24-hour macro trend anchor (288 M5 bars)


class DaltonAuctionStrategy:
    """
    Unified Dalton Auction Market Theory Signal Generator.
    Combines Market Profile Value Areas (VAH/VAL/POC), Initial Balance (IBH/IBL),
    and Auction Market Dynamics.
    """

    def __init__(self,
                 mp_params: MarketProfileParams = MarketProfileParams(),
                 strat_params: DaltonStrategyParams = DaltonStrategyParams()):
        self.mp_engine = MarketProfileEngine(mp_params)
        self.open_classifier = DayOpenClassifier()
        self.params = strat_params

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Computes causal Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def generate_auction_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes historical OHLCV data, builds causal daily Market Profiles,
        and generates causal trade signals.
        """
        data = df.copy().reset_index(drop=True)
        
        # 1. Compute Causal Market Profile Reference Levels (VAH, VAL, POC, IBH, IBL) if not precomputed
        if "prev_vah" not in data.columns:
            mp_df = self.mp_engine.compute_rolling_market_profiles(data)
            data = pd.concat([data, mp_df], axis=1)

        # 2. Compute Technical Anchors & ATR
        data["atr"] = self._compute_atr(data)
        data["ema_20"] = data["close"].ewm(span=self.params.ema_fast_period, adjust=False).mean()
        data["ema_50"] = data["close"].ewm(span=self.params.ema_slow_period, adjust=False).mean()
        data["ema_24h"] = data["close"].ewm(span=self.params.ema_macro_period, adjust=False).mean()

        macro_bull = data["close"] > data["ema_24h"]
        macro_bear = data["close"] < data["ema_24h"]

        # 3. Institutional Liquidity Window & Spread Filter
        h = data["time"].dt.hour
        m_min = data["time"].dt.minute
        time_decimal = h + m_min / 60.0

        in_session = np.zeros(len(data), dtype=bool)
        for start_t, end_t in self.params.allowed_sessions:
            in_session |= (time_decimal >= start_t) & (time_decimal <= end_t)

        spread_ok = (data["spread"] <= self.params.max_spread_points) if "spread" in data.columns else np.ones(len(data), dtype=bool)
        
        valid_context = in_session & spread_ok & ~data["prev_vah"].isna() & ~data["prev_val"].isna()

        # 4. Acceptance into Prior Value Area Tracker (Dalton 80% Rule)
        # Check consecutive closes inside [prev_val, prev_vah]
        inside_va = (data["close"] >= data["prev_val"]) & (data["close"] <= data["prev_vah"])
        
        # Rolling count of consecutive bars inside Value Area
        va_consec_inside = inside_va.groupby((~inside_va).cumsum()).cumsum()
        
        # Value Area acceptance achieved when consecutive bars inside >= acceptance_bars
        va_accepted = (va_consec_inside >= self.params.acceptance_bars)
        va_width = data["prev_vah"] - data["prev_val"]
        va_width_valid = (va_width >= self.params.min_va_width)

        # Prior day High / Low penetration before acceptance
        # Accepted from below (probed below VAL, now accepted inside -> Long rotation to VAH)
        entered_from_below = (data["low"].shift(self.params.acceptance_bars) < data["prev_val"].shift(self.params.acceptance_bars)) & va_accepted & va_width_valid
        
        # Accepted from above (probed above VAH, now accepted inside -> Short rotation to VAL)
        entered_from_above = (data["high"].shift(self.params.acceptance_bars) > data["prev_vah"].shift(self.params.acceptance_bars)) & va_accepted & va_width_valid

        # 5. Initial Balance (IB) Breakout Tracker
        # Post-IB execution (bars 12+ of the day)
        ib_active = data["is_ib_complete"] & (data["curr_ibr"] >= self.params.min_ibr) & (data["curr_ibr"] <= self.params.max_ibr)
        
        ib_breakout_high = ib_active & (data["close"] > data["curr_ibh"] + self.params.ib_breakout_buffer) & macro_bull
        ib_breakout_low = ib_active & (data["close"] < data["curr_ibl"] - self.params.ib_breakout_buffer) & macro_bear

        # 6. Poor Extreme Auction Incompletion (Repair Trades)
        # Price approaches prior poor high -> high probability of auction completion/breakout
        poor_high_target = data["prev_poor_high"] & (data["high"] >= data["prev_high"] - 1.0) & macro_bull
        poor_low_target = data["prev_poor_low"] & (data["low"] <= data["prev_low"] + 1.0) & macro_bear

        # 7. Generate Directional Signals
        trade_signals = np.zeros(len(data))
        strategy_types = ["NONE"] * len(data)
        custom_tp = np.full(len(data), np.nan)
        custom_sl = np.full(len(data), np.nan)

        # Strategy 1: Dalton 80% Rule Rotation Trades
        # Long: Accepted from below VAL -> Target VAH
        long_80_rule = valid_context & entered_from_below & (data["close"] < (data["prev_vah"] - 1.0))
        # Short: Accepted from above VAH -> Target VAL
        short_80_rule = valid_context & entered_from_above & (data["close"] > (data["prev_val"] + 1.0))

        trade_signals[long_80_rule] = 1.0
        trade_signals[short_80_rule] = -1.0

        # Strategy 2: Initial Balance Initiative Breakout Trades
        # Long: IB High Breakout
        long_ib_break = valid_context & ib_breakout_high & (trade_signals == 0)
        # Short: IB Low Breakout
        short_ib_break = valid_context & ib_breakout_low & (trade_signals == 0)

        trade_signals[long_ib_break] = 1.0
        trade_signals[short_ib_break] = -1.0

        data["trade_signal"] = trade_signals
        return data
