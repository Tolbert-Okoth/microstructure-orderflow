"""
Dalton Open Type & Day Type Classifier.
Implements Dalton's 4 Open Types (Open-Drive, Open-Test-Drive, Open-Rejection-Reverse, Open-Auction)
and Auction Market Theory Day Types.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
import numpy as np
import pandas as pd


class OpenType(Enum):
    OPEN_DRIVE_BULL = "OPEN_DRIVE_BULL"
    OPEN_DRIVE_BEAR = "OPEN_DRIVE_BEAR"
    OPEN_TEST_DRIVE_BULL = "OPEN_TEST_DRIVE_BULL"
    OPEN_TEST_DRIVE_BEAR = "OPEN_TEST_DRIVE_BEAR"
    OPEN_REJECTION_REVERSE_BULL = "OPEN_REJECTION_REVERSE_BULL"
    OPEN_REJECTION_REVERSE_BEAR = "OPEN_REJECTION_REVERSE_BEAR"
    OPEN_AUCTION = "OPEN_AUCTION"
    UNKNOWN = "UNKNOWN"


class DayType(Enum):
    TREND_DAY_BULL = "TREND_DAY_BULL"
    TREND_DAY_BEAR = "TREND_DAY_BEAR"
    NORMAL_DAY = "NORMAL_DAY"
    NORMAL_VARIATION_BULL = "NORMAL_VARIATION_BULL"
    NORMAL_VARIATION_BEAR = "NORMAL_VARIATION_BEAR"
    NEUTRAL_DAY = "NEUTRAL_DAY"
    NON_TREND_DAY = "NON_TREND_DAY"


@dataclass
class OpenTypeParams:
    """Parameters for Open Type detection."""
    early_bars: int = 6               # First 30 minutes of session (6 M5 bars)
    drive_min_ticks: float = 1.50     # Min directional distance ($1.50) from open
    drive_max_retrace_pct: float = 0.25 # Max retracement allowed for Open-Drive


class DayOpenClassifier:
    """
    Classifies the market opening behavior relative to prior day's Value Area & Range.
    """

    def __init__(self, params: OpenTypeParams = OpenTypeParams()):
        self.params = params

    def classify_open(self,
                      session_first_bars: pd.DataFrame,
                      prev_vah: float,
                      prev_val: float,
                      prev_poc: float,
                      prev_high: float,
                      prev_low: float) -> OpenType:
        """
        Classifies the open based on the first 30-60 minutes of the session.
        """
        if len(session_first_bars) < self.params.early_bars:
            return OpenType.UNKNOWN

        open_p = float(session_first_bars["open"].iloc[0])
        highs = session_first_bars["high"].iloc[:self.params.early_bars].values
        lows = session_first_bars["low"].iloc[:self.params.early_bars].values
        closes = session_first_bars["close"].iloc[:self.params.early_bars].values

        max_high = np.max(highs)
        min_low = np.min(lows)
        end_close = closes[-1]

        bull_expansion = max_high - open_p
        bear_expansion = open_p - min_low

        # 1. Open-Drive (OD): Immediate unidirectional propulsion with minimal retracement
        if bull_expansion >= self.params.drive_min_ticks and bear_expansion <= (bull_expansion * self.params.drive_max_retrace_pct):
            if open_p >= prev_val:
                return OpenType.OPEN_DRIVE_BULL

        if bear_expansion >= self.params.drive_min_ticks and bull_expansion <= (bear_expansion * self.params.drive_max_retrace_pct):
            if open_p <= prev_vah:
                return OpenType.OPEN_DRIVE_BEAR

        # 2. Open-Test-Drive (OTD): Tested a reference level (prior high/low/VAH/VAL) and drove away
        # Tested low near prior VAL/Low then drove bull
        if abs(min_low - prev_val) <= 1.0 or abs(min_low - prev_low) <= 1.0:
            if end_close > open_p + 1.0:
                return OpenType.OPEN_TEST_DRIVE_BULL

        # Tested high near prior VAH/High then drove bear
        if abs(max_high - prev_vah) <= 1.0 or abs(max_high - prev_high) <= 1.0:
            if end_close < open_p - 1.0:
                return OpenType.OPEN_TEST_DRIVE_BEAR

        # 3. Open-Rejection-Reverse (ORR): Opened outside prior value, penetrated, rejected sharply back through open
        # Opened above VAH, probed, rejected back below open
        if open_p > prev_vah and min_low < open_p - 1.0 and end_close < prev_vah:
            return OpenType.OPEN_REJECTION_REVERSE_BEAR

        # Opened below VAL, probed, rejected back above open
        if open_p < prev_val and max_high > open_p + 1.0 and end_close > prev_val:
            return OpenType.OPEN_REJECTION_REVERSE_BULL

        # 4. Open-Auction (OA): Rotational two-sided trade inside opening range
        return OpenType.OPEN_AUCTION

    def classify_day_type(self,
                          ibh: float,
                          ibl: float,
                          session_high: float,
                          session_low: float,
                          session_close: float) -> DayType:
        """
        Classifies the completed session into Dalton's 6 Day Types.
        """
        ibr = max(ibh - ibl, 0.10)
        upper_extension = max(session_high - ibh, 0.0)
        lower_extension = max(ibl - session_low, 0.0)

        total_range = session_high - session_low
        extension_mult = total_range / ibr

        # Both sides extended -> Neutral Day
        if upper_extension >= (0.25 * ibr) and lower_extension >= (0.25 * ibr):
            return DayType.NEUTRAL_DAY

        # Unidirectional massive extension -> Trend Day
        if upper_extension >= (1.5 * ibr) and lower_extension < (0.25 * ibr):
            return DayType.TREND_DAY_BULL
        if lower_extension >= (1.5 * ibr) and upper_extension < (0.25 * ibr):
            return DayType.TREND_DAY_BEAR

        # Moderate one-sided extension -> Normal Variation Day
        if upper_extension >= (0.5 * ibr) and lower_extension < (0.25 * ibr):
            return DayType.NORMAL_VARIATION_BULL
        if lower_extension >= (0.5 * ibr) and upper_extension < (0.25 * ibr):
            return DayType.NORMAL_VARIATION_BEAR

        # Little extension -> Normal Day vs Non-Trend Day
        if extension_mult <= 1.15:
            if ibr < 3.0: # Narrow IB in Gold
                return DayType.NON_TREND_DAY
            return DayType.NORMAL_DAY

        return DayType.NORMAL_DAY
