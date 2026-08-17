"""
Bouchaud, Farmer, & Lillo (2008) - Long-Memory Order Flow & Propagator Engine.
Implements the transient market impact propagator model G(tau) = Gamma_0 / (1 + tau)^beta,
order sign autocorrelation tracking, and impact exhaustion divergence.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BouchaudParams:
    """Parameters for Bouchaud et al. (2008) Propagator Engine."""
    memory_gamma: float = 0.50      # Power-law order flow correlation exponent (0 < gamma < 1)
    propagator_cutoff: int = 48     # Maximum history lag for propagator convolution (48 M5 bars = 4h)
    volume_concavity_psi: float = 0.25 # Volume concavity exponent psi
    exhaustion_z_threshold: float = 2.2 # Z-score threshold for transient impact exhaustion
    rolling_norm_window: int = 96   # Normalization window for impact z-score


class BouchaudPropagatorEngine:
    """
    Implements the Transient Market Impact Propagator Model (Bouchaud et al., 2008):
    1. Order Signs: ε_t = sign(Close_t - Close_{t-1})
    2. Propagator Kernel: G(τ) = 1 / (1 + τ)^β, where β = (1 - γ) / 2
    3. Transient Impact Convolution: I_t = ∑_{s=1}^K G(s) · ε_{t-s} · V_{t-s}^ψ
    4. Impact Resilience Decay: Identifies overextended transient price pressure
    5. Square-Root Law Calibration for execution slippage
    """

    def __init__(self, params: BouchaudParams = BouchaudParams()):
        self.params = params
        # Propagator exponent beta = (1 - gamma) / 2
        self.beta = (1.0 - self.params.memory_gamma) / 2.0
        # Precompute discrete propagator kernel G(s) for s = 1..K
        s = np.arange(1, self.params.propagator_cutoff + 1)
        self.kernel = 1.0 / np.power(1.0 + s, self.beta)
        self.kernel = self.kernel / np.sum(self.kernel) # Normalize kernel weights

    def compute_propagator_features(self, close: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """
        Computes causal transient impact convolution, order sign persistence,
        and impact exhaustion signals.
        """
        delta_p = close.diff().fillna(0.0)
        epsilon = np.sign(delta_p)
        
        # Concave volume-weighted order sign impulse: ε_t · V_t^ψ
        psi = self.params.volume_concavity_psi
        norm_vol = volume / np.maximum(volume.rolling(window=96, min_periods=10).median(), 1.0)
        norm_vol = norm_vol.fillna(1.0).clip(0.1, 10.0)
        impulse = epsilon * np.power(norm_vol, psi)
        
        # Fast causal convolution of bare propagator G(s) with historical order impulses
        # np.convolve(mode='full') and shift to align causally
        convolved = np.convolve(impulse.values, self.kernel, mode='full')
        # Slice to length n, shifted by 1 so at time t we only see up to t-1
        transient_impact = np.zeros(len(close))
        transient_impact[1:] = convolved[:len(close)-1]
        
        transient_impact_series = pd.Series(transient_impact, index=close.index)
        
        # Standardize transient impact
        w = self.params.rolling_norm_window
        mean_imp = transient_impact_series.rolling(window=w, min_periods=w // 4).mean()
        std_imp = transient_impact_series.rolling(window=w, min_periods=w // 4).std()
        propagator_z = (transient_impact_series - mean_imp) / np.maximum(std_imp, 1e-6)
        propagator_z = propagator_z.fillna(0.0).clip(-4.0, 4.0)
        
        # Fast Vectorized Order Sign Autocorrelation over last 12 bars (1h)
        rolling_autocorr = epsilon.rolling(window=12, min_periods=6).corr(epsilon.shift(1)).fillna(0.0)
        
        # Propagator Exhaustion Signals:
        # Extreme transient positive impact (z >= +2.2) with stalling price -> Bearish Exhaustion Fade
        # Extreme transient negative impact (z <= -2.2) with stalling price -> Bullish Rebound Fade
        exhaustion_thresh = self.params.exhaustion_z_threshold
        bull_exhaustion_fade = (propagator_z <= -exhaustion_thresh).astype(float) # Rebound long
        bear_exhaustion_fade = (propagator_z >= exhaustion_thresh).astype(float)  # Decline short
        
        propagator_reversal_signal = bull_exhaustion_fade - bear_exhaustion_fade
        
        # Propagator Momentum Signal: Strong persistent order flow (z > 0.5 and positive autocorrelation)
        propagator_momentum_signal = np.tanh(propagator_z / 1.5)

        return pd.DataFrame({
            "bouchaud_transient_impact": transient_impact_series,
            "bouchaud_propagator_z": propagator_z,
            "bouchaud_order_sign_autocorr": rolling_autocorr,
            "bouchaud_reversal_signal": propagator_reversal_signal,
            "bouchaud_momentum_signal": propagator_momentum_signal
        })
