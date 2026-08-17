"""
Albert S. Kyle (1985) - Continuous Auctions and Informed Trader Engine.
Estimates instantaneous Kyle's Lambda (price impact / illiquidity parameter)
and decomposes aggregate order flow into informed directional discovery vs noise trading.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class KyleParams:
    """Parameters for Kyle (1985) Market Depth & Information Engine."""
    rolling_window: int = 48        # 4-hour causal rolling estimation window (48 M5 bars)
    lambda_smooth_span: int = 12    # Smoothing span for Kyle's Lambda
    z_score_window: int = 96        # 8-hour normalization window for informed flow z-score
    min_volume_threshold: float = 1.0


class KyleLambdaEngine:
    """
    Implements the Kyle (1985) Microstructure Price Impact Model:
    1. Signed Order Flow Proxy: ΔY_t = sign(ΔP_t) · Volume_t
    2. Instantaneous Kyle's Lambda: λ_t = Cov(ΔP, ΔY) / Var(ΔY)
    3. Informed Flow: ΔX_t = ΔP_t / λ_t
    4. Noise Flow: ΔU_t = ΔY_t - ΔX_t
    5. Information Discovery Velocity: Rate of private information incorporation
    """

    def __init__(self, params: KyleParams = KyleParams()):
        self.params = params

    def compute_kyle_features(self, close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """
        Computes causal Kyle Lambda, informed flow, noise flow, and directional conviction z-scores.
        """
        # Price change
        delta_p = close.diff().fillna(0.0)
        
        # Microstructure signed order flow proxy (Lee-Ready / Tick rule direction · volume)
        # Using intra-bar price action: (Close - Open) or (Close - Close_{t-1})
        price_sign = np.sign(delta_p)
        signed_flow = price_sign * volume
        
        # Fast vectorized rolling covariance Cov(ΔP, ΔY) and variance Var(ΔY)
        # Cov(X, Y) = E[X·Y] - E[X]·E[Y], Var(Y) = E[Y²] - E[Y]²
        w = self.params.rolling_window
        min_p = max(5, w // 4)
        
        mean_p = delta_p.rolling(window=w, min_periods=min_p).mean()
        mean_y = signed_flow.rolling(window=w, min_periods=min_p).mean()
        mean_py = (delta_p * signed_flow).rolling(window=w, min_periods=min_p).mean()
        mean_y2 = (signed_flow ** 2).rolling(window=w, min_periods=min_p).mean()
        
        rolling_cov = mean_py - (mean_p * mean_y)
        rolling_var = mean_y2 - (mean_y ** 2)
        
        # Kyle's Lambda: λ_t = Cov(ΔP, ΔY) / Var(ΔY)
        # Bounded below by zero (positive price impact)
        kyle_lambda = rolling_cov / np.maximum(rolling_var, 1e-8)
        kyle_lambda = kyle_lambda.clip(lower=1e-6, upper=10.0).fillna(1e-4)
        
        # Smooth lambda to filter microstructure tick noise
        kyle_lambda_smooth = kyle_lambda.ewm(span=self.params.lambda_smooth_span, adjust=False).mean()
        
        # Informed Order Flow: ΔX_t = ΔP_t / λ_t
        informed_flow = delta_p / np.maximum(kyle_lambda_smooth, 1e-6)
        
        # Noise Flow: ΔU_t = ΔY_t - ΔX_t
        noise_flow = signed_flow - informed_flow
        
        # Cumulative Informed Flow over rolling horizon
        cum_informed = informed_flow.rolling(window=w, min_periods=w // 2).sum().fillna(0.0)
        
        # Standardized Informed Flow z-score
        z_w = self.params.z_score_window
        mean_inf = cum_informed.rolling(window=z_w, min_periods=z_w // 4).mean()
        std_inf = cum_informed.rolling(window=z_w, min_periods=z_w // 4).std()
        kyle_informed_z = (cum_informed - mean_inf) / np.maximum(std_inf, 1e-6)
        kyle_informed_z = kyle_informed_z.fillna(0.0).clip(-4.0, 4.0)
        
        # Kyle Liquidity Depth Ratio: Low Lambda = Deep liquid market; High Lambda = Thin market
        lambda_rolling_mean = kyle_lambda_smooth.rolling(window=z_w, min_periods=z_w // 4).mean().fillna(1e-4)
        kyle_illiquidity_ratio = kyle_lambda_smooth / np.maximum(lambda_rolling_mean, 1e-6)
        
        # Directional Signal (-1.0 to +1.0)
        kyle_signal = np.tanh(kyle_informed_z / 1.5)

        return pd.DataFrame({
            "kyle_lambda": kyle_lambda_smooth,
            "kyle_informed_flow": informed_flow,
            "kyle_noise_flow": noise_flow,
            "kyle_cum_informed": cum_informed,
            "kyle_informed_z": kyle_informed_z,
            "kyle_illiquidity_ratio": kyle_illiquidity_ratio,
            "kyle_signal": kyle_signal
        })
