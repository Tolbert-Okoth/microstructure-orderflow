"""
Unit and Integration Tests for Auction Market Theory & Market Profile Engine.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.market_profile import MarketProfileEngine, MarketProfileParams
from src.models.day_open_classifier import DayOpenClassifier, OpenType, DayType
from src.models.dalton_strategy import DaltonAuctionStrategy, DaltonStrategyParams
from src.backtest.auction_backtester import AuctionBacktester, BacktestConfig


@pytest.fixture
def sample_session_df():
    """Generates synthetic 24-hour M5 session data (288 bars)."""
    np.random.seed(42)
    n_bars = 288
    times = pd.date_range("2026-08-17 00:00:00", periods=n_bars, freq="5min", tz="UTC")
    
    # Generate bell-shaped price distribution around 2350.0
    price_base = 2350.0
    walk = np.random.randn(n_bars).cumsum() * 0.5
    closes = price_base + walk
    highs = closes + np.random.uniform(0.2, 0.8, n_bars)
    lows = closes - np.random.uniform(0.2, 0.8, n_bars)
    opens = (highs + lows) / 2.0
    volumes = np.random.randint(50, 500, n_bars)
    spreads = np.full(n_bars, 30.0)

    return pd.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "spread": spreads
    })


def test_market_profile_engine(sample_session_df):
    engine = MarketProfileEngine(MarketProfileParams(tick_size=0.10, value_area_pct=0.70))
    prof = engine.calculate_session_profile(sample_session_df)
    
    assert prof is not None
    assert prof.vah_price >= prof.poc_price
    assert prof.poc_price >= prof.val_price
    assert prof.high >= prof.vah_price
    assert prof.val_price >= prof.low
    assert prof.tpo_count > 0
    assert prof.ibr_range > 0


def test_open_type_and_day_type_classification(sample_session_df):
    classifier = DayOpenClassifier()
    
    # Test open classification
    open_type = classifier.classify_open(
        sample_session_df.iloc[:12],
        prev_vah=2355.0,
        prev_val=2345.0,
        prev_poc=2350.0,
        prev_high=2360.0,
        prev_low=2340.0
    )
    assert isinstance(open_type, OpenType)

    # Test day type classification
    day_type = classifier.classify_day_type(
        ibh=2355.0,
        ibl=2345.0,
        session_high=2370.0,
        session_low=2344.0,
        session_close=2368.0
    )
    assert day_type == DayType.TREND_DAY_BULL


def test_dalton_strategy_signal_generation(sample_session_df):
    # Repeat for 3 days to test causal rolling profiles
    df_3days = pd.concat([
        sample_session_df,
        sample_session_df.assign(time=sample_session_df["time"] + pd.Timedelta(days=1)),
        sample_session_df.assign(time=sample_session_df["time"] + pd.Timedelta(days=2))
    ]).reset_index(drop=True)

    strategy = DaltonAuctionStrategy()
    data_with_signals = strategy.generate_auction_signals(df_3days)

    assert "trade_signal" in data_with_signals.columns
    assert "prev_vah" in data_with_signals.columns
    assert "prev_val" in data_with_signals.columns
    assert "curr_ibh" in data_with_signals.columns
    assert set(data_with_signals["trade_signal"].unique()).issubset({-1.0, 0.0, 1.0})


def test_auction_backtester_execution(sample_session_df):
    df_3days = pd.concat([
        sample_session_df,
        sample_session_df.assign(time=sample_session_df["time"] + pd.Timedelta(days=1)),
        sample_session_df.assign(time=sample_session_df["time"] + pd.Timedelta(days=2))
    ]).reset_index(drop=True)

    strategy = DaltonAuctionStrategy()
    data_with_signals = strategy.generate_auction_signals(df_3days)

    backtester = AuctionBacktester(BacktestConfig(fixed_lots=0.10))
    res = backtester.run_backtest(data_with_signals)

    assert "metrics" in res
    assert "trades" in res
    assert "equity_curve" in res
    assert res["metrics"]["initial_capital"] == 10000.0
