"""
MetaTrader 5 Connector & Parquet Storage Layer.
Streams real live tick and OHLCV rates directly from the local MT5 terminal.
"""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MT5Connector")

STORAGE_DIR = Path("data/storage")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class MT5Connector:
    """
    Handles robust local MT5 terminal connectivity, rate fetching,
    and parquet serialization for Gold (XAUUSD).
    """

    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "M5"):
        self.symbol = symbol
        self.timeframe_str = timeframe
        self.tf_map = {
            "M1": mt5.TIMEFRAME_M1 if mt5 else 1,
            "M5": mt5.TIMEFRAME_M5 if mt5 else 5,
            "M15": mt5.TIMEFRAME_M15 if mt5 else 15,
            "H1": mt5.TIMEFRAME_H1 if mt5 else 60,
            "H4": mt5.TIMEFRAME_H4 if mt5 else 240,
            "D1": mt5.TIMEFRAME_D1 if mt5 else 1440,
        }
        self.parquet_path = STORAGE_DIR / f"{self.symbol}_{self.timeframe_str}.parquet"

    def initialize(self) -> bool:
        """Initializes connection to MT5 terminal."""
        if mt5 is None:
            logger.error("MetaTrader5 python package is not installed.")
            return False
        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False
        
        # Ensure symbol is selected in MarketWatch
        if not mt5.symbol_select(self.symbol, True):
            logger.warning(f"Failed to select symbol {self.symbol} in MarketWatch.")
        logger.info(f"Connected to MT5 terminal. Symbol: {self.symbol}, Timeframe: {self.timeframe_str}")
        return True

    def shutdown(self):
        """Closes MT5 terminal connection."""
        if mt5:
            mt5.shutdown()
            logger.info("MT5 connection shut down.")

    def fetch_historical_bars(self, count: int = 50000) -> pd.DataFrame:
        """
        Fetches the latest `count` historical OHLCV bars directly from MT5.
        Includes real volume, tick volume, and dynamic bid-ask spread in points.
        """
        if not self.initialize():
            raise RuntimeError("Cannot fetch data: MT5 initialization failed.")
        
        tf = self.tf_map.get(self.timeframe_str, mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, count)
        self.shutdown()

        if rates is None or len(rates) == 0:
            raise ValueError(f"No rates returned for {self.symbol} from MT5: {mt5.last_error() if mt5 else ''}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        
        # Rename columns to standard lowercase
        df = df.rename(columns={
            "tick_volume": "volume",
            "real_volume": "real_volume"
        })
        
        # Fill zero spread with rolling median if missing
        if "spread" not in df.columns or df["spread"].iloc[-1] == 0:
            df["spread"] = 35.0  # Default ~35 points on XAUUSD
        else:
            df["spread"] = df["spread"].astype(float)
            df["spread"] = df["spread"].replace(0, np.nan).ffill().bfill()

        # Sort chronologically and drop duplicates
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        logger.info(f"Fetched {len(df):,} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        return df

    def sync_to_parquet(self, count: int = 50000) -> Path:
        """Fetches from MT5 and persists clean parquet file."""
        df = self.fetch_historical_bars(count=count)
        df.to_parquet(self.parquet_path, index=False)
        logger.info(f"Persisted clean historical dataset to {self.parquet_path}")
        return self.parquet_path

    def load_cached_data(self) -> pd.DataFrame:
        """Loads cached parquet data if exists, otherwise fetches and caches."""
        if self.parquet_path.exists():
            df = pd.read_parquet(self.parquet_path)
            logger.info(f"Loaded {len(df):,} cached bars from {self.parquet_path}")
            return df
        logger.info(f"No cache found at {self.parquet_path}. Fetching from MT5...")
        return self.fetch_historical_bars()
