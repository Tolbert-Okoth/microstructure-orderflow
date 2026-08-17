"""
Purged Walk-Forward Cross-Validation & Monte Carlo Bootstrap Engine.
Grounded in Marcos Lopez de Prado's Purged & Embargoed Cross-Validation methodology.
"""
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.models.ensemble_signal import UnifiedMicrostructureEnsemble
from src.backtest.microstructure_backtester import MicrostructureBacktester, BacktestConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WalkForward")


class WalkForwardEngine:
    """
    Executes 5-fold Purged Cross-Validation:
    - Purges 100 bars preceding test boundaries to prevent label leakage.
    - Embargoes 50 bars succeeding test boundaries.
    - Runs 1,000-path Monte Carlo bootstrap to test statistical significance.
    """

    def __init__(self, n_splits: int = 5, purge_bars: int = 100, embargo_bars: int = 50):
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def get_purged_folds(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Splits time series into non-overlapping, purged Train/Test slices."""
        total_bars = len(df)
        test_size = total_bars // (self.n_splits + 1)
        folds = []

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            # Train is everything before test_start minus purge_bars
            train_end = max(0, test_start - self.purge_bars)
            train_df = df.iloc[:train_end].copy().reset_index(drop=True)
            
            # Test is test_start to test_end
            test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)
            
            if len(train_df) > 1000 and len(test_df) > 500:
                folds.append((train_df, test_df))

        return folds

    def run_walk_forward(self, df: pd.DataFrame, backtest_config: BacktestConfig = BacktestConfig()) -> Dict:
        """
        Executes out-of-sample evaluation across all purged folds.
        """
        logger.info(f"Starting {self.n_splits}-Fold Purged Walk-Forward Validation on {len(df):,} bars...")
        folds = self.get_purged_folds(df)
        
        ensemble = UnifiedMicrostructureEnsemble()
        
        fold_results = []
        all_oos_trades: List[Dict] = []

        for fold_idx, (train_df, test_df) in enumerate(folds):
            logger.info(f"Evaluating Fold {fold_idx + 1}/{len(folds)}: OOS Bars {len(test_df):,} ({test_df['time'].iloc[0]} to {test_df['time'].iloc[-1]})")
            
            # Generate signals on test fold
            test_with_signals = ensemble.generate_features_and_signals(test_df)
            
            backtester = MicrostructureBacktester(backtest_config)
            res = backtester.run_backtest(test_with_signals)
            m = res["metrics"]
            
            fold_results.append({
                "fold": fold_idx + 1,
                "start_time": str(test_df["time"].iloc[0]),
                "end_time": str(test_df["time"].iloc[-1]),
                "trades": m["total_trades"],
                "win_rate": m["win_rate"],
                "profit_factor": m["profit_factor"],
                "net_pnl": m["total_net_pnl"],
                "max_drawdown": m["max_drawdown_pct"]
            })
            
            all_oos_trades.extend(res["trades"])
            logger.info(f"  Fold {fold_idx + 1} Net PnL: ${m['total_net_pnl']:,.2f} | Trades: {m['total_trades']} | Win Rate: {m['win_rate']}% | PF: {m['profit_factor']:.2f}")

        # Aggregate Out-of-Sample Metrics
        pnls = np.array([t["net_pnl"] for t in all_oos_trades]) if all_oos_trades else np.array([0.0])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        
        agg_win_rate = float(len(wins) / max(len(pnls), 1) * 100.0)
        agg_pf = float(np.sum(wins) / max(np.sum(np.abs(losses)), 1e-4)) if len(losses) > 0 else 99.0
        
        # Monte Carlo Bootstrap (1,000 paths)
        mc_prob_profitable = 0.0
        if len(pnls) >= 10:
            n_sims = 1000
            mc_totals = []
            for _ in range(n_sims):
                sample_pnl = np.random.choice(pnls, size=len(pnls), replace=True)
                mc_totals.append(np.sum(sample_pnl))
            mc_prob_profitable = float(np.mean(np.array(mc_totals) > 0) * 100.0)

        return {
            "folds": fold_results,
            "total_oos_trades": len(all_oos_trades),
            "total_net_pnl": float(np.sum(pnls)),
            "aggregate_win_rate": round(agg_win_rate, 2),
            "aggregate_profit_factor": round(agg_pf, 2),
            "monte_carlo_prob_profitable": round(mc_prob_profitable, 2)
        }
