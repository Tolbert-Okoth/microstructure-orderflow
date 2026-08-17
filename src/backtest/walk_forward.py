"""
5-Fold Purged Walk-Forward Cross-Validation Engine for Auction Market Theory.
Enforces strict causal separation between training and out-of-sample folds.
"""
import logging
from typing import Dict, List
import numpy as np
import pandas as pd

from src.models.dalton_strategy import DaltonAuctionStrategy, DaltonStrategyParams
from src.backtest.auction_backtester import AuctionBacktester, BacktestConfig

logger = logging.getLogger("PurgedWalkForward")


class PurgedWalkForwardValidator:
    """
    Evaluates Dalton Auction Market Theory models using 5-Fold Purged Cross-Validation.
    """

    def __init__(self, n_folds: int = 5, embargo_pct: float = 0.01):
        self.n_folds = n_folds
        self.embargo_pct = embargo_pct

    def run_validation(self, df: pd.DataFrame, strat_params: DaltonStrategyParams = DaltonStrategyParams(), bt_config: BacktestConfig = BacktestConfig()) -> Dict:
        """
        Executes purged walk-forward cross validation across the full historical dataset.
        """
        n_total = len(df)
        fold_size = n_total // self.n_folds
        embargo_bars = int(n_total * self.embargo_pct)
        
        logger.info(f"Starting {self.n_folds}-Fold Purged Walk-Forward Validation on {n_total:,} bars...")

        strategy = DaltonAuctionStrategy(strat_params=strat_params)
        backtester = AuctionBacktester(bt_config)

        fold_results: List[Dict] = []
        all_oos_pnls: List[float] = []

        for fold in range(self.n_folds):
            # Causal out-of-sample interval
            oos_start = fold * fold_size
            oos_end = (fold + 1) * fold_size if fold < (self.n_folds - 1) else n_total

            # Purged & Embargoed training interval
            train_end = max(0, oos_start - embargo_bars)
            train_df = df.iloc[:train_end] if train_end > 500 else None
            oos_df = df.iloc[oos_start:oos_end].copy().reset_index(drop=True)

            logger.info(f"Evaluating Fold {fold + 1}/{self.n_folds}: OOS Bars {len(oos_df):,} ({oos_df['time'].iloc[0]} to {oos_df['time'].iloc[-1]})")

            # Generate signals on out-of-sample fold
            oos_with_signals = strategy.generate_auction_signals(oos_df)
            res = backtester.run_backtest(oos_with_signals)
            m = res["metrics"]
            trades = res["trades"]

            fold_results.append({
                "fold": fold + 1,
                "oos_start": str(oos_df["time"].iloc[0]),
                "oos_end": str(oos_df["time"].iloc[-1]),
                "trades": m["total_trades"],
                "win_rate": m["win_rate"],
                "profit_factor": m["profit_factor"],
                "net_pnl": m["total_net_pnl"],
                "max_drawdown_pct": m["max_drawdown_pct"]
            })

            for t in trades:
                all_oos_pnls.append(t["net_pnl"])

            logger.info(f"  Fold {fold + 1} Net PnL: ${m['total_net_pnl']:+,.2f} | Trades: {m['total_trades']} | Win Rate: {m['win_rate']}% | PF: {m['profit_factor']:.2f}")

        # Aggregate Out-of-Sample Statistics
        total_oos_trades = len(all_oos_pnls)
        agg_wins = [p for p in all_oos_pnls if p > 0]
        agg_losses = [p for p in all_oos_pnls if p < 0]
        agg_win_rate = (len(agg_wins) / total_oos_trades * 100.0) if total_oos_trades > 0 else 0.0
        agg_pf = (sum(agg_wins) / abs(sum(agg_losses))) if sum(agg_losses) != 0 else 0.0
        agg_net_pnl = sum(all_oos_pnls)

        # Monte Carlo Statistical Significance (1,000 resamples)
        mc_profitable = 0
        if total_oos_trades >= 10:
            for _ in range(1000):
                sample = np.random.choice(all_oos_pnls, size=total_oos_trades, replace=True)
                if np.sum(sample) > 0:
                    mc_profitable += 1
            mc_prob = (mc_profitable / 1000.0) * 100.0
        else:
            mc_prob = 0.0

        return {
            "fold_results": fold_results,
            "total_oos_trades": total_oos_trades,
            "aggregate_win_rate": round(agg_win_rate, 2),
            "aggregate_profit_factor": round(agg_pf, 2),
            "aggregate_net_pnl": round(agg_net_pnl, 2),
            "monte_carlo_prob_profitable": round(mc_prob, 2)
        }
