"""
Unified Microstructure & Order Flow Ensemble Engine.
Integrates Kyle (1985), Bouchaud et al. (2008), and Inoua & Smith (2023)
with institutional session gating and structural entry location validation.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import pandas as pd

from src.models.kyle_lambda import KyleLambdaEngine, KyleParams
from src.models.bouchaud_propagator import BouchaudPropagatorEngine, BouchaudParams
from src.models.inoua_smith_demand import InouaSmithDemandEngine, InouaSmithParams


@dataclass
class MicrostructureEnsembleParams:
    """Parameters for Unified Microstructure Ensemble."""
    min_kyle_informed_z: float = 1.20     # Minimum Kyle informed flow z-score for momentum
    min_bouchaud_momentum_z: float = 1.20 # Minimum Bouchaud propagator z-score for momentum
    min_inoua_demand_z: float = 1.20      # Minimum Inoua-Smith excess demand z-score for momentum
    propagator_exhaustion_thresh: float = 2.20 # Bouchaud exhaustion z-score for reversal fade
    excess_demand_reversal_thresh: float = 2.00 # Inoua-Smith extreme imbalance threshold for reversal
    adx_trend_min: float = 25.0           # Minimum ADX to confirm directional trending regime
    adx_range_max: float = 22.0           # Maximum ADX to confirm bounded range regime
    max_spread_points: float = 45.0       # Maximum MT5 spread points allowed
    allowed_sessions: List[Tuple[float, float]] = field(default_factory=lambda: [(8.0, 12.0), (13.0, 17.0)])


class UnifiedMicrostructureEnsemble:
    """
    Unified Institutional Microstructure Signal Generator combining:
    - Kyle (1985): Continuous Auctions, Informed Order Flow, and Kyle's Lambda
    - Bouchaud et al. (2008): Long-Memory Order Flow, Propagator Transient Impact, & Exhaustion Fades
    - Inoua & Smith (2023): Classical Reservation Prices, Dynamic Excess Demand, & Tatonnement Discovery
    - Macro Multi-Scale Anchor & Institutional Liquidity Window Gating
    """

    def __init__(self,
                 kyle_params: KyleParams = KyleParams(),
                 bouchaud_params: BouchaudParams = BouchaudParams(),
                 inoua_params: InouaSmithParams = InouaSmithParams(),
                 ensemble_params: MicrostructureEnsembleParams = MicrostructureEnsembleParams()):
        self.kyle_engine = KyleLambdaEngine(kyle_params)
        self.bouchaud_engine = BouchaudPropagatorEngine(bouchaud_params)
        self.inoua_engine = InouaSmithDemandEngine(inoua_params)
        self.params = ensemble_params

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Computes causal Average Directional Index (ADX)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = (df["low"].shift(1) - df["low"])

        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / np.maximum(atr, 1e-6))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / np.maximum(atr, 1e-6))

        dx = 100 * (plus_di - minus_di).abs() / np.maximum(plus_di + minus_di, 1e-6)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx

    def generate_features_and_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes OHLCV dataframe, computes underlying microstructure indicators,
        and generates causal, regime-gated execution signals.
        """
        res = df.copy().reset_index(drop=True)
        
        # 1. Kyle (1985) Market Depth & Informed Flow
        kyle_df = self.kyle_engine.compute_kyle_features(res["close"], res["high"], res["low"], res["volume"])
        res = pd.concat([res, kyle_df], axis=1)
        
        # 2. Bouchaud et al. (2008) Propagator & Transient Impact
        bouchaud_df = self.bouchaud_engine.compute_propagator_features(res["close"], res["volume"])
        res = pd.concat([res, bouchaud_df], axis=1)
        
        # 3. Inoua & Smith (2023) Classical Excess Demand
        inoua_df = self.inoua_engine.compute_demand_features(res["high"], res["low"], res["close"], res["volume"])
        res = pd.concat([res, inoua_df], axis=1)
        
        # 4. Multi-Scale EMAs & ADX
        res["adx"] = self._compute_adx(res)
        res["ema_20"] = res["close"].ewm(span=20, adjust=False).mean()
        res["ema_50"] = res["close"].ewm(span=50, adjust=False).mean()
        res["ema_6h"] = res["close"].ewm(span=72, adjust=False).mean()
        res["ema_24h"] = res["close"].ewm(span=288, adjust=False).mean()
        
        macro_long = (res["close"] > res["ema_24h"]) & (res["ema_6h"] > res["ema_24h"])
        macro_short = (res["close"] < res["ema_24h"]) & (res["ema_6h"] < res["ema_24h"])
        
        # Value-Retracement Location Check (Price retested 20-EMA/50-EMA value zone)
        val_zone_long = (res["low"] <= res["ema_20"] * 1.001) & (res["close"] >= res["ema_50"] * 0.999)
        val_zone_short = (res["high"] >= res["ema_20"] * 0.999) & (res["close"] <= res["ema_50"] * 1.001)
        
        # 5. Institutional Liquidity Windows & Spread Filter
        h = res["time"].dt.hour
        m_min = res["time"].dt.minute
        time_decimal = h + m_min / 60.0
        
        in_session = np.zeros(len(res), dtype=bool)
        for start_t, end_t in self.params.allowed_sessions:
            in_session |= (time_decimal >= start_t) & (time_decimal <= end_t)
        res["in_session"] = in_session
        
        spread_ok = (res["spread"] <= self.params.max_spread_points) if "spread" in res.columns else np.ones(len(res), dtype=bool)
        res["spread_ok"] = spread_ok
        
        # 6. Composite Signal Generation
        res["of_score"] = (res["kyle_informed_z"] + res["bouchaud_propagator_z"] + res["inoua_excess_demand_z"]) / 3.0
        trade_signals = np.zeros(len(res))
        
        # Strategy 1: Microstructure Momentum Expansion (Kyle Informed Flow + Bouchaud Propagator + Inoua Excess Demand)
        trend_regime = (res["adx"] >= self.params.adx_trend_min)
        long_mom = (
            in_session & spread_ok & trend_regime & val_zone_long & macro_long &
            (res["of_score"] >= 1.30)
        )
        short_mom = (
            in_session & spread_ok & trend_regime & val_zone_short & macro_short &
            (res["of_score"] <= -1.30)
        )
        
        trade_signals[long_mom] = 1.0
        trade_signals[short_mom] = -1.0
        
        # Strategy 2: Microstructure Exhaustion & Equilibrium Reversal (Bouchaud Propagator Decay + Inoua Clearing)
        range_regime = (res["adx"] <= self.params.adx_range_max)
        long_rev = (
            in_session & spread_ok & range_regime &
            (res["bouchaud_propagator_z"] <= -self.params.propagator_exhaustion_thresh) &
            (res["inoua_excess_demand_z"] <= -self.params.excess_demand_reversal_thresh) &
            (res["close"] < res["inoua_equilibrium_p_star"]) # Below volume-weighted equilibrium
        )
        short_rev = (
            in_session & spread_ok & range_regime &
            (res["bouchaud_propagator_z"] >= self.params.propagator_exhaustion_thresh) &
            (res["inoua_excess_demand_z"] >= self.params.excess_demand_reversal_thresh) &
            (res["close"] > res["inoua_equilibrium_p_star"]) # Above volume-weighted equilibrium
        )
        
        trade_signals[long_rev] = 1.0
        trade_signals[short_rev] = -1.0
        
        res["trade_signal"] = trade_signals
        return res
