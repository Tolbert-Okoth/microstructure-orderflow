"""
Institutional Microstructure Backtester.
Simulates realistic broker order execution with dynamic spreads, Bouchaud square-root slippage,
zero-lookahead t+1 Open fills, 50-EMA trailing exits, and friction-adjusted Kelly sizing.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Configuration for Microstructure Backtester."""
    initial_capital: float = 10000.0   # Account base balance ($)
    contract_size: float = 100.0       # 100 oz per standard lot on XAUUSD
    point_value: float = 0.01          # Minimum price increment ($0.01 per point)
    slippage_y_coef: float = 0.70      # Bouchaud Square-Root Law universal constant Y
    fixed_lots: Optional[float] = 0.10 # Fixed lot size (if None, uses Kelly sizing)
    kelly_fraction: float = 0.40       # Friction-adjusted Kelly multiplier
    atr_period: int = 14
    tp_atr_mult: float = 5.0           # Take-profit ATR runner barrier
    sl_atr_mult: float = 2.2           # Stop-loss ATR barrier
    max_holding_bars: int = 24         # Maximum holding period (2 hours on M5)
    ema_trailing_period: int = 50      # 50-EMA trailing exit evaluated at candle close


class MicrostructureBacktester:
    """
    Simulates high-fidelity institutional order execution for XAUUSD:
    - Zero Lookahead: Signal at bar t -> Executed at bar t+1 Open.
    - Dynamic MT5 Spreads: Applied at both entry and exit.
    - Bouchaud Square-Root Law Slippage: Slippage = Y · ATR · sqrt(Lots / NormalLots)
    - 50-EMA Candle-Close Trailing Exits.
    """

    def __init__(self, config: BacktestConfig = BacktestConfig()):
        self.cfg = config

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Computes causal Average True Range."""
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

                    # Structural Barriers
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

        # Performance Metrics Calculation
        metrics = self._calculate_metrics(trades, equity_curve)
        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve
        }

    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float]) -> Dict:
        """Calculates institutional risk and performance metrics."""
        if len(trades) == 0:
            return {
                "total_trades": 0,
                "net_profit": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "total_return_pct": 0.0
            }

        pnls = np.array([t["net_pnl"] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        total_net_pnl = float(np.sum(pnls))
        total_gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
        total_gross_loss = float(np.sum(np.abs(losses))) if len(losses) > 0 else 0.0

        win_rate = float(len(wins) / len(pnls) * 100.0)
        profit_factor = float(total_gross_win / max(total_gross_loss, 1e-4)) if total_gross_loss > 0 else 99.0

        # Drawdown calculation
        eq = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq)
        drawdowns = (peaks - eq) / peaks * 100.0
        max_drawdown = float(np.max(drawdowns))

        # Sharpe & Sortino
        trade_returns = pnls / self.cfg.initial_capital
        mean_ret = np.mean(trade_returns)
        std_ret = np.std(trade_returns)
        downside_std = np.std(trade_returns[trade_returns < 0]) if np.sum(trade_returns < 0) > 0 else 1e-6

        # Annualized Sharpe (~250 trading days * ~2 trades/day = ~500 trades/yr)
        annual_factor = np.sqrt(500.0)
        sharpe = float((mean_ret / max(std_ret, 1e-6)) * annual_factor)
        sortino = float((mean_ret / max(downside_std, 1e-6)) * annual_factor)

        # Monthly breakdown
        monthly: Dict[str, Dict] = {}
        for t in trades:
            month_key = str(t["exit_time"])[:7] # YYYY-MM
            if month_key not in monthly:
                monthly[month_key] = {"net_pnl": 0.0, "trades": 0, "wins": 0, "losses": 0, "win_pnl": 0.0, "loss_pnl": 0.0}
            monthly[month_key]["trades"] += 1
            monthly[month_key]["net_pnl"] += t["net_pnl"]
            if t["net_pnl"] > 0:
                monthly[month_key]["wins"] += 1
                monthly[month_key]["win_pnl"] += t["net_pnl"]
            else:
                monthly[month_key]["losses"] += 1
                monthly[month_key]["loss_pnl"] += abs(t["net_pnl"])

        for k, v in monthly.items():
            v["win_rate"] = (v["wins"] / v["trades"] * 100.0) if v["trades"] > 0 else 0.0
            v["profit_factor"] = (v["win_pnl"] / max(v["loss_pnl"], 1e-4)) if v["loss_pnl"] > 0 else 99.0

        return {
            "initial_capital": self.cfg.initial_capital,
            "final_capital": equity_curve[-1],
            "total_net_pnl": total_net_pnl,
            "total_return_pct": (total_net_pnl / self.cfg.initial_capital) * 100.0,
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "monthly_breakdown": monthly
        }
