"""
MT5 Data Connector for Real XAUUSD Bars.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import MetaTrader5 as mt5
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MT5Connector")


class MT5Connector:
    def __init__(self, symbol: str = "XAUUSD", timeframe: int = mt5.TIMEFRAME_M5, storage_dir: str = "data/storage"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.storage_dir / f"{self.symbol}_M5.parquet"

    def connect(self) -> bool:
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        logger.info(f"Connected to MetaTrader 5 Terminal v{mt5.version()}")
        return True

    def disconnect(self) -> None:
        mt5.shutdown()
        logger.info("MT5 connection closed.")

    def fetch_historical_bars(self, count: int = 50000) -> Optional[pd.DataFrame]:
        if not self.connect():
            return None
        
        try:
            logger.info(f"Fetching {count:,} M5 bars for {self.symbol}...")
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to fetch rates: {mt5.last_error()}")
                return None
            
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={"tick_volume": "volume"})
            df = df[["time", "open", "high", "low", "close", "volume", "spread", "real_volume"]]
            df = df.sort_values("time").reset_index(drop=True)
            
            # Save to Parquet
            df.to_parquet(self.parquet_path, engine="pyarrow", index=False)
            logger.info(f"Saved {len(df):,} bars to {self.parquet_path}")
            return df
        finally:
            self.disconnect()

    def load_cached_data(self) -> pd.DataFrame:
        if not self.parquet_path.exists():
            logger.info("Parquet cache missing, fetching live from MT5...")
            df = self.fetch_historical_bars()
            if df is None:
                raise RuntimeError("Could not fetch data from MT5 and no cache available.")
            return df
        
        df = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(df):,} cached bars from {self.parquet_path}")
        return df


if __name__ == "__main__":
    connector = MT5Connector("XAUUSD")
    df = connector.fetch_historical_bars(50000)
    if df is not None:
        print(f"Data successfully fetched: {len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
