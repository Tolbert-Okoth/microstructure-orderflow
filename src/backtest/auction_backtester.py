"""
Auction Market Theory Causal Backtesting Simulator.
Implements realistic microstructure execution:
- Causal bar t+1 Open fills
- Dynamic MT5 spreads
- Bouchaud square-root non-linear slippage model
- Intrabar Stop Loss and Take Profit barrier checking
- Dynamic 50-EMA trailing stops on candle close
- Friction-adjusted Kelly Criterion position sizing
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("AuctionBacktester")


@dataclass
class BacktestConfig:
    """Configuration parameters for Auction Backtester."""
    initial_capital: float = 10000.0    # Initial account capital in USD
    contract_size: float = 100.0        # XAUUSD 1 lot = 100 oz
    point_value: float = 0.01           # 1 MT5 spread point = $0.01 in Gold
    slippage_y_coef: float = 0.50       # Bouchaud non-linear slippage coefficient
    tp_atr_mult: float = 5.0            # Take profit multiplier in ATR units
    sl_atr_mult: float = 2.8            # Stop loss multiplier in ATR units
    ema_trailing_period: int = 50       # Trailing EMA period evaluated at candle close
    max_holding_bars: int = 20          # Max holding duration (20 M5 bars = 1.6 hours)
    fixed_lots: Optional[float] = 0.10  # If set, use fixed lot sizing (e.g. 0.10 lots)
    kelly_fraction: float = 0.40        # Fractional Kelly sizing multiplier (Half-Kelly)
    atr_period: int = 14


class AuctionBacktester:
    """
    Executes high-fidelity simulation with zero look-ahead bias and realistic market friction.
    """

    def __init__(self, config: BacktestConfig = BacktestConfig()):
        self.cfg = config

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=self.cfg.atr_period, adjust=False).mean()

    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Executes causal bar-by-bar simulation across the dataframe.
        Optimized with raw numpy array indexing for maximum execution speed.
        """
        data = df.copy().reset_index(drop=True)
        data["atr"] = self._compute_atr(data)
        data["trailing_ema"] = data["close"].ewm(span=self.cfg.ema_trailing_period, adjust=False).mean()

        capital = self.cfg.initial_capital
        equity_curve = [capital]
        trades: List[Dict] = []
        
        active_trade: Optional[Dict] = None
        rolling_pnl_history: List[float] = []

        # Extract fast numpy arrays
        n_bars = len(data)
        open_arr = data["open"].values
        high_arr = data["high"].values
        low_arr = data["low"].values
        close_arr = data["close"].values
        spread_arr = data["spread"].values if "spread" in data.columns else np.full(n_bars, 35.0)
        atr_arr = data["atr"].values
        trailing_ema_arr = data["trailing_ema"].values
        trade_signal_arr = data["trade_signal"].values if "trade_signal" in data.columns else np.zeros(n_bars)
        time_arr = data["time"].values
        
        prev_vah_arr = data["prev_vah"].values if "prev_vah" in data.columns else np.full(n_bars, np.nan)
        prev_val_arr = data["prev_val"].values if "prev_val" in data.columns else np.full(n_bars, np.nan)

        for i in range(1, n_bars - 1):
            curr_close = close_arr[i]
            curr_high = high_arr[i]
            curr_low = low_arr[i]
            curr_spread = spread_arr[i]
            curr_atr = atr_arr[i]
            curr_ema = trailing_ema_arr[i]
            curr_time = time_arr[i]

            # 1. Manage Active Position
            if active_trade is not None:
                active_trade["bars_held"] += 1
                direction = active_trade["direction"] # +1 for Long, -1 for Short
                entry_p = active_trade["entry_price"]
                lots = active_trade["lots"]
                sl = active_trade["sl_price"]
                tp = active_trade["tp_price"]

                exit_price = None
                exit_reason = None

                # Check Stop Loss & Take Profit against bar intrabar range [low, high]
                if direction == 1:
                    if curr_low <= sl:
                        exit_price = sl
                        exit_reason = "STOP_LOSS"
                    elif curr_high >= tp:
                        exit_price = tp
                        exit_reason = "TAKE_PROFIT"
                    elif active_trade["bars_held"] >= self.cfg.max_holding_bars:
                        exit_price = curr_close
                        exit_reason = "TIME_EXPIRED"
                    elif curr_close < curr_ema and active_trade["bars_held"] >= 6:
                        # Trailing EMA close violation
                        exit_price = curr_close
                        exit_reason = "EMA_TRAILING"
                else: # Short
                    if curr_high >= sl:
                        exit_price = sl
                        exit_reason = "STOP_LOSS"
                    elif curr_low <= tp:
                        exit_price = tp
                        exit_reason = "TAKE_PROFIT"
                    elif active_trade["bars_held"] >= self.cfg.max_holding_bars:
                        exit_price = curr_close
                        exit_reason = "TIME_EXPIRED"
                    elif curr_close > curr_ema and active_trade["bars_held"] >= 6:
                        # Trailing EMA close violation
                        exit_price = curr_close
                        exit_reason = "EMA_TRAILING"

                if exit_price is not None:
                    # Apply exit spread and slippage
                    exit_spread = curr_spread * self.cfg.point_value
                    exit_slippage = self.cfg.slippage_y_coef * curr_atr * 0.03 * np.sqrt(lots / 0.10)
                    
                    realized_exit = exit_price - (exit_spread / 2.0 + exit_slippage) if direction == 1 else exit_price + (exit_spread / 2.0 + exit_slippage)
                    
                    price_diff = (realized_exit - entry_p) if direction == 1 else (entry_p - realized_exit)
                    net_pnl = price_diff * lots * self.cfg.contract_size

                    capital += net_pnl
                    active_trade["exit_price"] = realized_exit
                    active_trade["exit_time"] = curr_time
                    active_trade["exit_reason"] = exit_reason
                    active_trade["net_pnl"] = net_pnl
                    active_trade["capital_after"] = capital
                    
                    trades.append(active_trade)
                    rolling_pnl_history.append(net_pnl)
                    active_trade = None

            # 2. Check for New Entry Signal at bar t -> Fills at bar t+1 Open
            if active_trade is None:
                sig = trade_signal_arr[i]
                if sig != 0.0 and i < n_bars - 1:
                    atr_val = curr_atr
                    if np.isnan(atr_val) or atr_val <= 0:
                        continue

                    direction = 1 if sig > 0 else -1
                    fill_open = open_arr[i + 1]
                    next_spread = spread_arr[i + 1]
                    
                    # Entry spread and slippage
                    entry_spread = next_spread * self.cfg.point_value
                    
                    # Position Sizing
                    if self.cfg.fixed_lots is not None:
                        lots = self.cfg.fixed_lots
                    else:
                        # Friction-Adjusted Kelly Sizing
                        if len(rolling_pnl_history) >= 15:
                            wins = [p for p in rolling_pnl_history[-30:] if p > 0]
                            losses = [abs(p) for p in rolling_pnl_history[-30:] if p < 0]
                            win_rate = len(wins) / max(len(wins) + len(losses), 1)
                            avg_win = np.mean(wins) if wins else 1.0
                            avg_loss = np.mean(losses) if losses else 1.0
                            r_ratio = avg_win / max(avg_loss, 1e-4)
                            kelly_f = max(0.0, win_rate - (1.0 - win_rate) / max(r_ratio, 1e-4))
                            lot_raw = (capital * kelly_f * self.cfg.kelly_fraction) / max(atr_val * self.cfg.contract_size * 10.0, 100.0)
                            lots = float(np.clip(round(lot_raw, 2), 0.01, 1.0))
                        else:
                            lots = 0.05

                    entry_slippage = self.cfg.slippage_y_coef * atr_val * 0.03 * np.sqrt(lots / 0.10)
                    realized_entry = fill_open + (entry_spread / 2.0 + entry_slippage) if direction == 1 else fill_open - (entry_spread / 2.0 + entry_slippage)

                    # Dynamic Structural Target & Stop based on Value Area & ATR
                    if direction == 1:
                        sl_price = realized_entry - (self.cfg.sl_atr_mult * atr_val)
                        tp_price = realized_entry + (self.cfg.tp_atr_mult * atr_val)
                    else:
                        sl_price = realized_entry + (self.cfg.sl_atr_mult * atr_val)
                        tp_price = realized_entry - (self.cfg.tp_atr_mult * atr_val)

                    active_trade = {
                        "entry_time": time_arr[i + 1],
                        "entry_price": realized_entry,
                        "direction": direction,
                        "lots": lots,
                        "sl_price": sl_price,
                        "tp_price": tp_price,
                        "bars_held": 0,
                        "atr": atr_val
                    }

            equity_curve.append(capital)

        # Performance Metrics Computation
        metrics = self._calculate_metrics(trades, equity_curve, capital)
        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve
        }

    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float], final_capital: float) -> Dict:
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_net_pnl": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "monthly_breakdown": {}
            }

        pnls = [t["net_pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        n_trades = len(trades)
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = (n_wins / n_trades) * 100.0 if n_trades > 0 else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        total_net_pnl = sum(pnls)
        total_return_pct = ((final_capital - self.cfg.initial_capital) / self.cfg.initial_capital) * 100.0

        # Maximum Drawdown calculation
        eq_arr = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq_arr)
        drawdowns = (peaks - eq_arr) / peaks
        max_drawdown_pct = float(np.max(drawdowns)) * 100.0 if len(drawdowns) > 0 else 0.0

        # Annualized Sharpe & Sortino (Assuming 252 trading days / ~72,576 M5 bars per year)
        pnl_arr = np.array(pnls)
        mean_trade_pnl = np.mean(pnl_arr) if len(pnl_arr) > 0 else 0.0
        std_trade_pnl = np.std(pnl_arr) if len(pnl_arr) > 0 else 1.0

        trades_per_year = n_trades * (72576.0 / max(len(equity_curve), 1))
        ann_factor = np.sqrt(max(trades_per_year, 1.0))

        sharpe_ratio = float((mean_trade_pnl / max(std_trade_pnl, 1e-6)) * ann_factor)

        downside_pnls = [p for p in pnls if p < 0]
        downside_std = np.std(downside_pnls) if downside_pnls else 1e-6
        sortino_ratio = float((mean_trade_pnl / max(downside_std, 1e-6)) * ann_factor)

        # Monthly breakdown
        trades_df = pd.DataFrame(trades)
        monthly_breakdown = {}
        if not trades_df.empty and "exit_time" in trades_df.columns:
            trades_df["month"] = pd.to_datetime(trades_df["exit_time"]).dt.strftime("%Y-%m")
            for month, group in trades_df.groupby("month"):
                m_pnls = group["net_pnl"].values
                m_wins = [p for p in m_pnls if p > 0]
                m_losses = [p for p in m_pnls if p < 0]
                m_pf = (sum(m_wins) / abs(sum(m_losses))) if sum(m_losses) != 0 else (99.0 if m_wins else 0.0)
                monthly_breakdown[month] = {
                    "net_pnl": float(np.sum(m_pnls)),
                    "trades": len(group),
                    "win_rate": float((len(m_wins) / len(group)) * 100.0),
                    "profit_factor": float(m_pf)
                }

        return {
            "initial_capital": self.cfg.initial_capital,
            "final_capital": final_capital,
            "total_net_pnl": float(total_net_pnl),
            "total_return_pct": float(total_return_pct),
            "total_trades": n_trades,
            "winning_trades": n_wins,
            "losing_trades": n_losses,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "monthly_breakdown": monthly_breakdown
        }
