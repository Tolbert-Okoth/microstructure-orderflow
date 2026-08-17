"""
Quantitative Microstructure & Order Flow Scalping Engine - CLI Entry Point.
Grounded in Kyle (1985), Bouchaud et al. (2008), and Inoua & Smith (2023).
"""
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

from src.data.mt5_connector import MT5Connector
from src.models.ensemble_signal import UnifiedMicrostructureEnsemble, MicrostructureEnsembleParams
from src.backtest.microstructure_backtester import MicrostructureBacktester, BacktestConfig
from src.backtest.walk_forward import WalkForwardEngine
from src.optimization.optuna_tuner import MicrostructureOptunaTuner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Main")


def main():
    parser = argparse.ArgumentParser(description="Quantitative Microstructure & Order Flow Engine")
    parser.add_argument("--sync-data", action="store_true", help="Sync historical bars from MT5 terminal")
    parser.add_argument("--count", type=int, default=50000, help="Number of bars to fetch/load")
    parser.add_argument("--backtest", action="store_true", help="Run full-history microstructure backtest")
    parser.add_argument("--walk-forward", action="store_true", help="Run 5-fold Purged Walk-Forward cross-validation")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna parameter optimization study")
    parser.add_argument("--trials", type=int, default=35, help="Number of Optuna optimization trials")
    parser.add_argument("--kelly", action="store_true", help="Use Friction-Adjusted Kelly Position Sizing")
    parser.add_argument("--lots", type=float, default=0.10, help="Fixed lot size (used if --kelly is not set)")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Trading symbol")

    args = parser.parse_args()

    connector = MT5Connector(symbol=args.symbol)

    if args.sync_data:
        logger.info(f"Syncing {args.count:,} bars for {args.symbol} from MT5...")
        path = connector.sync_to_parquet(count=args.count)
        print(f"Data sync complete. Saved to: {path}")
        return

    # Load cached parquet data
    try:
        df = connector.load_cached_data()
        logger.info(f"Loaded {len(df):,} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    except Exception as e:
        logger.error(f"Failed to load cached data: {e}. Run with --sync-data first.")
        sys.exit(1)

    if args.optimize:
        logger.info(f"Running Optuna Optimization Study ({args.trials} trials)...")
        tuner = MicrostructureOptunaTuner(df.iloc[-25000:].copy().reset_index(drop=True), n_trials=args.trials)
        best = tuner.run_optimization()
        print("\n=======================================================")
        print("=== OPTUNA BEST HYPERPARAMETERS ===")
        print("=======================================================")
        for k, v in best["best_params"].items():
            print(f"  {k}: {v}")
        print("=======================================================")
        return

    if args.walk_forward:
        logger.info("Executing 5-Fold Purged Walk-Forward Validation...")
        wf_engine = WalkForwardEngine(n_splits=5)
        backtest_cfg = BacktestConfig(
            fixed_lots=None if args.kelly else args.lots,
            kelly_fraction=0.40
        )
        summary = wf_engine.run_walk_forward(df, backtest_config=backtest_cfg)
        print("\n=======================================================")
        print("=== PURGED WALK-FORWARD CROSS-VALIDATION REPORT ===")
        print("=======================================================")
        print(f"Total OOS Trades:             {summary['total_oos_trades']}")
        print(f"Total Net PnL:                ${summary['total_net_pnl']:,.2f}")
        print(f"Aggregate Win Rate:           {summary['aggregate_win_rate']}%")
        print(f"Aggregate Profit Factor:      {summary['aggregate_profit_factor']:.2f}")
        print(f"Monte Carlo Profitable Prob:  {summary['monte_carlo_prob_profitable']}%")
        print("=======================================================")
        return

    # Default: Run full-period microstructure backtest
    logger.info("Generating microstructure features and execution signals...")
    ensemble = UnifiedMicrostructureEnsemble()
    data_with_signals = ensemble.generate_features_and_signals(df)

    backtest_cfg = BacktestConfig(
        fixed_lots=None if args.kelly else args.lots,
        kelly_fraction=0.40
    )
    backtester = MicrostructureBacktester(backtest_cfg)
    result = backtester.run_backtest(data_with_signals)
    m = result["metrics"]

    mode_title = "FRICTION-ADJUSTED KELLY (0.40x)" if args.kelly else f"FIXED LOTS ({args.lots})"
    print(f"\n=======================================================")
    print(f"=== COMPLETE MICROSTRUCTURE BACKTEST: {mode_title} ===")
    print(f"=======================================================")
    print(f"Initial Capital:          ${m['initial_capital']:,.2f}")
    print(f"Final Capital:            ${m['final_capital']:,.2f}")
    print(f"Total Net Return:         {m['total_return_pct']:+.2f}% (${m['total_net_pnl']:+,.2f})")
    print(f"Total Trades:             {m['total_trades']} (Wins: {m['winning_trades']}, Losses: {m['losing_trades']})")
    print(f"Win Rate:                 {m['win_rate']}%")
    print(f"Profit Factor:            {m['profit_factor']:.2f}")
    print(f"Annualized Sharpe Ratio:  {m['sharpe_ratio']:.2f}")
    print(f"Downside Sortino Ratio:   {m['sortino_ratio']:.2f}")
    print(f"Max Drawdown:             {m['max_drawdown_pct']:.2f}%")
    print("\n--- MONTHLY PERFORMANCE BREAKDOWN ---")
    for month, info in m["monthly_breakdown"].items():
        print(f"  {month}: ${info['net_pnl']:+,.2f} | Trades: {info['trades']:2d} | Win Rate: {info['win_rate']:4.1f}% | PF: {info['profit_factor']:.2f}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
