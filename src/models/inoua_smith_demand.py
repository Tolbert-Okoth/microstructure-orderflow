"""
Inoua & Smith (2023) - Classical Supply, Demand, and Reservation Price Engine.
Reconstructs cumulative Willingness-To-Pay (WTP) Demand D(p) and Willingness-To-Accept (WTA) Supply S(p),
computes Dynamic Excess Demand z(p) = D(p) - S(p), and evaluates classical price discovery velocity.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class InouaSmithParams:
    """Parameters for Inoua & Smith (2023) Classical Supply & Demand Engine."""
    formation_window: int = 48      # 4-hour rolling window on M5 to reconstruct reservation distributions
    z_score_window: int = 96        # 8-hour normalization window for excess demand z-score
    tatonnement_kappa: float = 1.0  # Price adjustment velocity multiplier
    extreme_imbalance_thresh: float = 2.0 # Multi-sigma imbalance threshold


class InouaSmithDemandEngine:
    """
    Implements the Classical Theory of Supply and Demand (Inoua & Smith, 2023):
    1. Buyer Valuation Distribution (WTP): D(p) = Volume traded with WTP >= p
    2. Seller Reservation Cost Distribution (WTA): S(p) = Volume traded with WTA <= p
    3. Dynamic Excess Demand Function: z(p) = D(p) - S(p)
    4. Tatonnement Price Discovery Velocity: dp/dt = κ · z(p)
    5. Equilibrium Reversion to Market Clearing Price p*
    """

    def __init__(self, params: InouaSmithParams = InouaSmithParams()):
        self.params = params

    def compute_demand_features(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.DataFrame:
        """
        Computes causal empirical demand D(p), supply S(p), excess demand z(p),
        and price discovery velocity.
        """
        w = self.params.formation_window
        n = len(close)
        
        # Intra-bar buyer vs seller volume allocation
        # Bullish bars allocate volume to buyer WTP; Bearish bars allocate to seller WTA
        bar_range = np.maximum(high - low, 1e-6)
        bull_ratio = (close - low) / bar_range
        bear_ratio = (high - close) / bar_range
        
        buyer_wtp_volume = volume * bull_ratio
        seller_wta_volume = volume * bear_ratio
        
        # Rolling cumulative demand D(p) and supply S(p)
        rolling_demand = buyer_wtp_volume.rolling(window=w, min_periods=w // 4).sum()
        rolling_supply = seller_wta_volume.rolling(window=w, min_periods=w // 4).sum()
        
        # Dynamic Excess Demand: z(p) = D(p) - S(p)
        excess_demand = rolling_demand - rolling_supply
        
        # Relative Excess Demand Ratio: z(p) / [D(p) + S(p)]
        total_market_liquidity = np.maximum(rolling_demand + rolling_supply, 1e-6)
        excess_demand_ratio = excess_demand / total_market_liquidity
        
        # Standardized Excess Demand Z-score over causal expanding/rolling baseline
        z_w = self.params.z_score_window
        mean_ed = excess_demand.rolling(window=z_w, min_periods=z_w // 4).mean()
        std_ed = excess_demand.rolling(window=z_w, min_periods=z_w // 4).std()
        excess_demand_z = (excess_demand - mean_ed) / np.maximum(std_ed, 1e-6)
        excess_demand_z = excess_demand_z.fillna(0.0).clip(-4.0, 4.0)
        
        # Classical Price Discovery Velocity: dp/dt = κ · z(p)
        tatonnement_velocity = self.params.tatonnement_kappa * excess_demand_ratio
        
        # Equilibrium Price Reconstruction p* (Volume-Weighted Price Equilibrium)
        typical_price = (high + low + close) / 3.0
        equilibrium_price_p_star = (typical_price * volume).rolling(window=w, min_periods=w // 4).sum() / total_market_liquidity
        
        # Price Deviation from Equilibrium: (P - p*) / p*
        price_equilibrium_gap = (close - equilibrium_price_p_star) / np.maximum(equilibrium_price_p_star, 1e-6)
        
        # Excess Demand Signal (-1.0 to +1.0)
        demand_signal = np.tanh(excess_demand_z / 1.5)

        return pd.DataFrame({
            "inoua_excess_demand": excess_demand,
            "inoua_excess_demand_ratio": excess_demand_ratio,
            "inoua_excess_demand_z": excess_demand_z,
            "inoua_tatonnement_velocity": tatonnement_velocity,
            "inoua_equilibrium_p_star": equilibrium_price_p_star,
            "inoua_price_equilibrium_gap": price_equilibrium_gap,
            "inoua_demand_signal": demand_signal
        })
