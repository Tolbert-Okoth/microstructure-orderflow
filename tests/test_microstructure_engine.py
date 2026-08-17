"""
Unit test suite for Quantitative Microstructure & Order Flow Engine.
Validates Kyle (1985), Bouchaud et al. (2008), Inoua & Smith (2023), and Backtest execution.
"""
import pytest
import numpy as np
import pandas as pd

from src.models.kyle_lambda import KyleLambdaEngine, KyleParams
from src.models.bouchaud_propagator import BouchaudPropagatorEngine, BouchaudParams
from src.models.inoua_smith_demand import InouaSmithDemandEngine, InouaSmithParams
from src.models.ensemble_signal import UnifiedMicrostructureEnsemble, MicrostructureEnsembleParams
from src.backtest.microstructure_backtester import MicrostructureBacktester, BacktestConfig


@pytest.fixture
def sample_ohlcv():
    """Generates synthetic 500-bar OHLCV test dataset with micro-trends."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2026-01-01 08:00", periods=n, freq="5min", tz="UTC")
    
    # Random walk with drift
    returns = np.random.normal(0.0001, 0.002, size=n)
    prices = 2000.0 * np.exp(np.cumsum(returns))
    
    highs = prices + np.random.uniform(0.5, 2.5, size=n)
    lows = prices - np.random.uniform(0.5, 2.5, size=n)
    opens = (highs + lows) / 2.0
    closes = prices
    volumes = np.random.randint(100, 1500, size=n)
    spreads = np.random.uniform(25.0, 45.0, size=n)
    
    return pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "spread": spreads
    })


def test_kyle_lambda_engine(sample_ohlcv):
    """Verifies Kyle (1985) market depth and informed flow decomposition."""
    engine = KyleLambdaEngine(KyleParams(rolling_window=24))
    res = engine.compute_kyle_features(
        sample_ohlcv["close"], sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["volume"]
    )
    
    assert "kyle_lambda" in res.columns
    assert "kyle_informed_flow" in res.columns
    assert "kyle_noise_flow" in res.columns
    assert "kyle_informed_z" in res.columns
    assert "kyle_signal" in res.columns
    
    # Kyle lambda must be strictly non-negative (positive price impact)
    assert (res["kyle_lambda"] > 0).all()
    # Informed + Noise flow = Total signed flow
    signed_flow = np.sign(sample_ohlcv["close"].diff().fillna(0.0)) * sample_ohlcv["volume"]
    diff = (res["kyle_informed_flow"] + res["kyle_noise_flow"]) - signed_flow
    np.testing.assert_allclose(diff.values, 0.0, atol=1e-5)


def test_bouchaud_propagator_engine(sample_ohlcv):
    """Verifies Bouchaud et al. (2008) long-memory propagator convolution."""
    engine = BouchaudPropagatorEngine(BouchaudParams(memory_gamma=0.5, propagator_cutoff=24))
    res = engine.compute_propagator_features(sample_ohlcv["close"], sample_ohlcv["volume"])
    
    assert "bouchaud_transient_impact" in res.columns
    assert "bouchaud_propagator_z" in res.columns
    assert "bouchaud_order_sign_autocorr" in res.columns
    assert "bouchaud_reversal_signal" in res.columns
    assert "bouchaud_momentum_signal" in res.columns
    
    # Propagator kernel weights sum to 1.0
    np.testing.assert_allclose(np.sum(engine.kernel), 1.0, atol=1e-5)
    # Z-scores are bounded
    assert (res["bouchaud_propagator_z"] >= -4.0).all() and (res["bouchaud_propagator_z"] <= 4.0).all()


def test_inoua_smith_demand_engine(sample_ohlcv):
    """Verifies Inoua & Smith (2023) classical supply, demand, and excess demand."""
    engine = InouaSmithDemandEngine(InouaSmithParams(formation_window=24))
    res = engine.compute_demand_features(
        sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], sample_ohlcv["volume"]
    )
    
    assert "inoua_excess_demand" in res.columns
    assert "inoua_excess_demand_z" in res.columns
    assert "inoua_tatonnement_velocity" in res.columns
    assert "inoua_equilibrium_p_star" in res.columns
    assert "inoua_demand_signal" in res.columns
    
    # Equilibrium price p* must be positive and within reasonable price bounds
    valid_p_star = res["inoua_equilibrium_p_star"].dropna()
    assert (valid_p_star > 1000.0).all() and (valid_p_star < 3000.0).all()


def test_unified_microstructure_ensemble(sample_ohlcv):
    """Verifies full ensemble feature generation, session gating, and trade signals."""
    ensemble = UnifiedMicrostructureEnsemble()
    res = ensemble.generate_features_and_signals(sample_ohlcv)
    
    assert "trade_signal" in res.columns
    assert "kyle_informed_z" in res.columns
    assert "bouchaud_propagator_z" in res.columns
    assert "inoua_excess_demand_z" in res.columns
    assert "in_session" in res.columns
    
    # Trade signal is discrete {-1.0, 0.0, 1.0}
    unique_sigs = np.unique(res["trade_signal"])
    for s in unique_sigs:
        assert s in [-1.0, 0.0, 1.0]


def test_microstructure_backtester(sample_ohlcv):
    """Verifies microstructure backtester accounting, fills, and spread cost."""
    ensemble = UnifiedMicrostructureEnsemble()
    data_with_signals = ensemble.generate_features_and_signals(sample_ohlcv)
    
    # Force test signals
    data_with_signals.loc[10, "trade_signal"] = 1.0
    data_with_signals.loc[50, "trade_signal"] = -1.0
    
    cfg = BacktestConfig(fixed_lots=0.10, max_holding_bars=10)
    backtester = MicrostructureBacktester(cfg)
    result = backtester.run_backtest(data_with_signals)
    
    assert "metrics" in result
    assert "trades" in result
    assert "equity_curve" in result
    
    m = result["metrics"]
    assert m["total_trades"] >= 2
    assert "profit_factor" in m
    assert "sharpe_ratio" in m
    assert "max_drawdown_pct" in m
