"""
Master CLI Entry Point for Market Profile & Auction Market Theory Engine.
"""
import argparse
import logging
import sys
import pandas as pd

from src.data.mt5_connector import MT5Connector
from src.models.dalton_strategy import DaltonAuctionStrategy, DaltonStrategyParams
from src.models.market_profile import MarketProfileParams
from src.backtest.auction_backtester import AuctionBacktester, BacktestConfig
from src.backtest.walk_forward import PurgedWalkForwardValidator
from src.optimization.optuna_tuner import AuctionOptunaTuner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AuctionMain")


def run_full_backtest(df: pd.DataFrame, fixed_lots: float = 0.10, kelly_fraction: float = 0.40):
    logger.info(f"Loaded {len(df):,} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    logger.info("Generating Market Profile features and Auction Market Theory signals...")

    strategy = DaltonAuctionStrategy()
    data_with_signals = strategy.generate_auction_signals(df)

    # 1. Fixed Lots Backtest
    cfg_fixed = BacktestConfig(fixed_lots=fixed_lots)
    bt_fixed = AuctionBacktester(cfg_fixed)
    res_fixed = bt_fixed.run_backtest(data_with_signals)
    m_fixed = res_fixed["metrics"]

    print("\n=======================================================")
    print(f"=== COMPLETE AUCTION BACKTEST: FIXED LOTS ({fixed_lots}) ===")
    print("=======================================================")
    print(f"Initial Capital:          ${m_fixed['initial_capital']:,.2f}")
    print(f"Final Capital:            ${m_fixed['final_capital']:,.2f}")
    print(f"Total Net Return:         {m_fixed['total_return_pct']:+.2f}% (${m_fixed['total_net_pnl']:+,.2f})")
    print(f"Total Trades:             {m_fixed['total_trades']} (Wins: {m_fixed['winning_trades']}, Losses: {m_fixed['losing_trades']})")
    print(f"Win Rate:                 {m_fixed['win_rate']}%")
    print(f"Profit Factor:            {m_fixed['profit_factor']:.2f}")
    print(f"Annualized Sharpe Ratio:  {m_fixed['sharpe_ratio']:.2f}")
    print(f"Downside Sortino Ratio:   {m_fixed['sortino_ratio']:.2f}")
    print(f"Max Drawdown:             {m_fixed['max_drawdown_pct']:.2f}%")
    print("\n--- MONTHLY PERFORMANCE BREAKDOWN ---")
    for month, info in m_fixed["monthly_breakdown"].items():
        print(f"  {month}: ${info['net_pnl']:+,.2f} | Trades: {info['trades']:2d} | Win Rate: {info['win_rate']:4.1f}% | PF: {info['profit_factor']:.2f}")
    print("=======================================================\n")

    # 2. Friction-Adjusted Kelly Sizing Backtest
    cfg_kelly = BacktestConfig(fixed_lots=None, kelly_fraction=kelly_fraction)
    bt_kelly = AuctionBacktester(cfg_kelly)
    res_kelly = bt_kelly.run_backtest(data_with_signals)
    m_kelly = res_kelly["metrics"]

    print("=======================================================")
    print(f"=== COMPLETE AUCTION BACKTEST: FRICTION KELLY ({kelly_fraction}x) ===")
    print("=======================================================")
    print(f"Initial Capital:          ${m_kelly['initial_capital']:,.2f}")
    print(f"Final Capital:            ${m_kelly['final_capital']:,.2f}")
    print(f"Total Net Return:         {m_kelly['total_return_pct']:+.2f}% (${m_kelly['total_net_pnl']:+,.2f})")
    print(f"Total Trades:             {m_kelly['total_trades']} (Wins: {m_kelly['winning_trades']}, Losses: {m_kelly['losing_trades']})")
    print(f"Win Rate:                 {m_kelly['win_rate']}%")
    print(f"Profit Factor:            {m_kelly['profit_factor']:.2f}")
    print(f"Annualized Sharpe Ratio:  {m_kelly['sharpe_ratio']:.2f}")
    print(f"Downside Sortino Ratio:   {m_kelly['sortino_ratio']:.2f}")
    print(f"Max Drawdown:             {m_kelly['max_drawdown_pct']:.2f}%")
    print("\n--- MONTHLY PERFORMANCE BREAKDOWN ---")
    for month, info in m_kelly["monthly_breakdown"].items():
        print(f"  {month}: ${info['net_pnl']:+,.2f} | Trades: {info['trades']:2d} | Win Rate: {info['win_rate']:4.1f}% | PF: {info['profit_factor']:.2f}")
    print("=======================================================\n")


def run_walk_forward_cv(df: pd.DataFrame):
    logger.info("Executing 5-Fold Purged Walk-Forward Validation...")
    validator = PurgedWalkForwardValidator(n_folds=5)
    res = validator.run_validation(df)

    print("\n=======================================================")
    print("=== PURGED WALK-FORWARD CROSS-VALIDATION REPORT ===")
    print("=======================================================")
    print(f"Total OOS Trades:             {res['total_oos_trades']}")
    print(f"Total Net PnL:                ${res['aggregate_net_pnl']:+,.2f}")
    print(f"Aggregate Win Rate:           {res['aggregate_win_rate']}%")
    print(f"Aggregate Profit Factor:      {res['aggregate_profit_factor']:.2f}")
    print(f"Monte Carlo Profitable Prob:  {res['monte_carlo_prob_profitable']}%")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Market Profile & Auction Market Theory Engine CLI")
    parser.add_argument("--sync", action="store_true", help="Fetch fresh M5 bars from MT5")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--walk-forward", action="store_true", help="Run 5-fold Purged Cross-Validation")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna study")
    parser.add_argument("--trials", type=int, default=35, help="Number of Optuna trials")
    parser.add_argument("--bars", type=int, default=50000, help="Number of bars to analyze")

    args = parser.parse_args()

    connector = MT5Connector("XAUUSD")
    if args.sync:
        df = connector.fetch_historical_bars(args.bars)
        if df is None:
            sys.exit(1)
    else:
        df = connector.load_cached_data()

    if args.backtest:
        run_full_backtest(df)
    elif args.walk_forward:
        run_walk_forward_cv(df)
    elif args.optimize:
        tuner = AuctionOptunaTuner(df.iloc[-25000:])
        best = tuner.run_optimization(args.trials)
        print("\nOPTIMIZED PARAMETERS:")
        for k, v in best.items():
            print(f"  {k}: {v}")
    else:
        run_full_backtest(df)


if __name__ == "__main__":
    main()
