"""
Optuna Parameter Optimizer for Microstructure Order Flow Engine.
Grounded in Kyle (1985), Bouchaud et al. (2008), and Inoua & Smith (2023).
"""
import logging
from typing import Dict
import optuna
import pandas as pd
import numpy as np

from src.data.mt5_connector import MT5Connector
from src.models.kyle_lambda import KyleParams
from src.models.bouchaud_propagator import BouchaudParams
from src.models.inoua_smith_demand import InouaSmithParams
from src.models.ensemble_signal import UnifiedMicrostructureEnsemble, MicrostructureEnsembleParams
from src.backtest.microstructure_backtester import MicrostructureBacktester, BacktestConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OptunaOptimizer")


class MicrostructureOptunaTuner:
    """
    Optimizes the microstructure ensemble parameters across historical MT5 bars.
    """

    def __init__(self, df: pd.DataFrame, n_trials: int = 40):
        self.df = df
        self.n_trials = n_trials

    def objective(self, trial: optuna.Trial) -> float:
        """Evaluates trial parameters on broker microstructure backtest."""
        # Suggest Kyle parameters
        kyle_window = trial.suggest_int("kyle_window", 24, 72, step=12)
        min_kyle_z = trial.suggest_float("min_kyle_z", 1.0, 2.0, step=0.1)
        
        # Suggest Bouchaud parameters
        memory_gamma = trial.suggest_float("memory_gamma", 0.40, 0.70, step=0.05)
        min_bouchaud_z = trial.suggest_float("min_bouchaud_z", 1.0, 2.0, step=0.1)
        prop_exhaustion = trial.suggest_float("prop_exhaustion", 1.8, 3.0, step=0.2)
        
        # Suggest Inoua & Smith parameters
        inoua_window = trial.suggest_int("inoua_window", 24, 72, step=12)
        min_inoua_z = trial.suggest_float("min_inoua_z", 1.0, 2.0, step=0.1)
        demand_reversal_thresh = trial.suggest_float("demand_reversal_thresh", 1.8, 3.0, step=0.2)
        
        # Suggest Execution barriers
        tp_atr_mult = trial.suggest_float("tp_atr_mult", 3.0, 6.0, step=0.5)
        sl_atr_mult = trial.suggest_float("sl_atr_mult", 1.4, 3.0, step=0.2)
        max_holding_bars = trial.suggest_int("max_holding_bars", 16, 40, step=4)

        # Instantiate Ensemble
        ensemble = UnifiedMicrostructureEnsemble(
            kyle_params=KyleParams(rolling_window=kyle_window),
            bouchaud_params=BouchaudParams(memory_gamma=memory_gamma, exhaustion_z_threshold=prop_exhaustion),
            inoua_params=InouaSmithParams(formation_window=inoua_window, extreme_imbalance_thresh=demand_reversal_thresh),
            ensemble_params=MicrostructureEnsembleParams(
                min_kyle_informed_z=min_kyle_z,
                min_bouchaud_momentum_z=min_bouchaud_z,
                min_inoua_demand_z=min_inoua_z,
                propagator_exhaustion_thresh=prop_exhaustion,
                excess_demand_reversal_thresh=demand_reversal_thresh
            )
        )

        data_with_signals = ensemble.generate_features_and_signals(self.df)
        
        backtest_cfg = BacktestConfig(
            tp_atr_mult=tp_atr_mult,
            sl_atr_mult=sl_atr_mult,
            max_holding_bars=max_holding_bars,
            fixed_lots=0.10
        )
        
        backtester = MicrostructureBacktester(backtest_cfg)
        res = backtester.run_backtest(data_with_signals)
        m = res["metrics"]

        if m["total_trades"] < 15:
            return -100.0

        net_ret = m["total_return_pct"]
        pf = m["profit_factor"]
        
        score = net_ret * (pf if pf > 0 else 0.1)
        return score

    def run_optimization(self) -> Dict:
        """Executes Optuna study."""
        logger.info(f"Starting Microstructure Optimization ({self.n_trials} trials)...")
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        logger.info("Optimization complete.")
        logger.info(f"Best Score: {study.best_value:.4f}")
        logger.info(f"Best Parameters: {study.best_params}")

        return {
            "best_score": study.best_value,
            "best_params": study.best_params,
        }


if __name__ == "__main__":
    connector = MT5Connector("XAUUSD")
    df = connector.load_cached_data()
    sample_df = df.iloc[-25000:].copy().reset_index(drop=True)
    tuner = MicrostructureOptunaTuner(sample_df, n_trials=35)
    best = tuner.run_optimization()
    print("\nOPTIMIZED PARAMETERS:")
    for k, v in best["best_params"].items():
        print(f"  {k}: {v}")
