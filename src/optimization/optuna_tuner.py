"""
Optuna Parameter Optimizer for Auction Market Theory Strategy.
"""
import logging
from typing import Dict, Optional
import optuna
import pandas as pd

from src.models.dalton_strategy import DaltonAuctionStrategy, DaltonStrategyParams
from src.models.market_profile import MarketProfileEngine, MarketProfileParams
from src.backtest.auction_backtester import AuctionBacktester, BacktestConfig

logger = logging.getLogger("AuctionOptunaTuner")
optuna.logging.set_verbosity(optuna.logging.WARNING)


class AuctionOptunaTuner:
    """
    Finds optimal parameter combinations for Dalton 80% Rule, Value Area, and IB breakouts.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        # Precompute daily Market Profile levels once
        mp_engine = MarketProfileEngine()
        mp_df = mp_engine.compute_rolling_market_profiles(self.df)
        self.df = pd.concat([self.df, mp_df], axis=1)

    def objective(self, trial: optuna.Trial) -> float:
        # Suggest Strategy parameters
        acceptance_bars = trial.suggest_int("acceptance_bars", 2, 8, step=2)
        min_va_width = trial.suggest_float("min_va_width", 2.0, 6.0, step=1.0)
        min_ibr = trial.suggest_float("min_ibr", 2.0, 5.0, step=1.0)
        ib_breakout_buffer = trial.suggest_float("ib_breakout_buffer", 0.20, 0.80, step=0.20)
        
        # Suggest Execution barriers
        tp_atr_mult = trial.suggest_float("tp_atr_mult", 3.0, 6.0, step=0.5)
        sl_atr_mult = trial.suggest_float("sl_atr_mult", 1.4, 3.0, step=0.2)
        max_holding_bars = trial.suggest_int("max_holding_bars", 16, 40, step=4)

        strat_params = DaltonStrategyParams(
            acceptance_bars=acceptance_bars,
            min_va_width=min_va_width,
            min_ibr=min_ibr,
            ib_breakout_buffer=ib_breakout_buffer
        )

        bt_config = BacktestConfig(
            tp_atr_mult=tp_atr_mult,
            sl_atr_mult=sl_atr_mult,
            max_holding_bars=max_holding_bars,
            fixed_lots=0.10
        )

        strategy = DaltonAuctionStrategy(strat_params=strat_params)
        data_with_signals = strategy.generate_auction_signals(self.df)

        backtester = AuctionBacktester(bt_config)
        res = backtester.run_backtest(data_with_signals)
        m = res["metrics"]

        if m["total_trades"] < 15:
            return -100.0

        net_ret = m["total_return_pct"]
        pf = m["profit_factor"]
        dd = max(m["max_drawdown_pct"], 1.0)

        # Friction & Risk-adjusted Objective
        score = (net_ret * (pf if pf > 0 else 0.1)) / (dd ** 0.5)
        return score

    def run_optimization(self, n_trials: int = 35) -> Dict:
        logger.info(f"Starting Auction Market Theory Optimization ({n_trials} trials)...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self.objective, n_trials=n_trials)

        logger.info("Optimization complete.")
        logger.info(f"Best Score: {study.best_value:.4f}")
        logger.info(f"Best Parameters: {study.best_params}")
        return study.best_params


if __name__ == "__main__":
    from src.data.mt5_connector import MT5Connector
    connector = MT5Connector("XAUUSD")
    df = connector.load_cached_data()
    tuner = AuctionOptunaTuner(df.iloc[-25000:])
    best = tuner.run_optimization(35)
    print("\nOPTIMIZED PARAMETERS:")
    for k, v in best.items():
        print(f"  {k}: {v}")
