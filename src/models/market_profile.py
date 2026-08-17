"""
Market Profile & Time-Price Opportunity (TPO) Engine.
Implements Steidlmayer & James Dalton (1990) 70% Value Area, Point of Control (POC),
Initial Balance (IB), Excess/Tails, and Poor High/Low unfinished auctions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class MarketProfileParams:
    """Parameters for Market Profile calculation."""
    tick_size: float = 0.10             # XAUUSD price tick increment ($0.10)
    value_area_pct: float = 0.70        # Value area target percentage (70% = 1 std dev)
    ib_bars: int = 12                   # First hour of session (12 M5 bars = two 30-min brackets)
    tail_min_ticks: int = 4             # Minimum single-print ticks to qualify as a structural tail
    poor_extreme_tolerance: float = 0.20 # Max distance ($) for poor high/low tie


@dataclass
class SessionProfile:
    """Encapsulates a single session's Market Profile structure."""
    date: pd.Timestamp
    high: float
    low: float
    open_price: float
    close_price: float
    volume: float
    poc_price: float
    vah_price: float
    val_price: float
    ibh_price: float
    ibl_price: float
    ibr_range: float
    tpo_count: int
    has_buying_tail: bool
    has_selling_tail: bool
    is_poor_high: bool
    is_poor_low: bool
    single_prints_upper: List[float] = field(default_factory=list)
    single_prints_lower: List[float] = field(default_factory=list)


class MarketProfileEngine:
    """
    Computes Time-Price Opportunities (TPO), Point of Control (POC),
    70% Value Area (VAH / VAL), and Initial Balance (IB) from intra-day bars.
    """

    def __init__(self, params: MarketProfileParams = MarketProfileParams()):
        self.params = params

    def _discretize_price(self, price: float) -> float:
        """Snaps price to discrete tick intervals."""
        return round(round(price / self.params.tick_size) * self.params.tick_size, 2)

    def calculate_session_profile(self, session_df: pd.DataFrame) -> Optional[SessionProfile]:
        """
        Builds complete Dalton Market Profile for a single trading session.
        Expects continuous M5 bars with columns: ['time', 'open', 'high', 'low', 'close', 'volume'].
        """
        if session_df.empty or len(session_df) < self.params.ib_bars:
            return None

        s_open = float(session_df["open"].iloc[0])
        s_close = float(session_df["close"].iloc[-1])
        s_high = float(session_df["high"].max())
        s_low = float(session_df["low"].min())
        s_vol = float(session_df["volume"].sum())
        s_date = pd.to_datetime(session_df["time"].iloc[0]).floor("D")

        # 1. Initial Balance (First hour / 12 M5 bars)
        ib_df = session_df.iloc[:self.params.ib_bars]
        ibh = float(ib_df["high"].max())
        ibl = float(ib_df["low"].min())
        ibr = max(ibh - ibl, self.params.tick_size)

        # 2. Build TPO Matrix across 30-minute brackets (each bracket = six 5-min bars)
        # Bracket index j in 0..N-1
        n_brackets = (len(session_df) + 5) // 6
        min_p = self._discretize_price(s_low)
        max_p = self._discretize_price(s_high)
        
        # Generate discrete price bins
        num_bins = int(round((max_p - min_p) / self.params.tick_size)) + 1
        if num_bins <= 0:
            return None

        price_bins = np.linspace(min_p, max_p, num_bins)
        price_bins = np.round(price_bins, 2)
        price_to_idx = {p: idx for idx, p in enumerate(price_bins)}

        tpo_matrix = np.zeros((num_bins, n_brackets), dtype=bool)

        for b_idx in range(n_brackets):
            b_df = session_df.iloc[b_idx * 6 : min((b_idx + 1) * 6, len(session_df))]
            if b_df.empty:
                continue
            b_high = float(b_df["high"].max())
            b_low = float(b_df["low"].min())
            
            # Mark touched prices
            start_p = self._discretize_price(b_low)
            end_p = self._discretize_price(b_high)
            
            for p_val in np.arange(start_p, end_p + self.params.tick_size / 2, self.params.tick_size):
                p_snap = self._discretize_price(p_val)
                if p_snap in price_to_idx:
                    tpo_matrix[price_to_idx[p_snap], b_idx] = True

        tpo_counts = np.sum(tpo_matrix, axis=1)
        total_tpos = int(np.sum(tpo_counts))
        if total_tpos == 0:
            return None

        # 3. Determine Point of Control (POC)
        # If tie, choose the price closest to the midpoint of the session range
        mid_price = (s_high + s_low) / 2.0
        max_tpo = np.max(tpo_counts)
        poc_candidates = np.where(tpo_counts == max_tpo)[0]
        
        best_poc_idx = poc_candidates[0]
        min_dist_to_mid = abs(price_bins[best_poc_idx] - mid_price)
        for cand in poc_candidates[1:]:
            dist = abs(price_bins[cand] - mid_price)
            if dist < min_dist_to_mid:
                min_dist_to_mid = dist
                best_poc_idx = cand
        
        poc_price = float(price_bins[best_poc_idx])

        # 4. Steidlmayer / Dalton 70% Value Area Algorithm
        target_tpos = self.params.value_area_pct * total_tpos
        cum_tpos = tpo_counts[best_poc_idx]
        va_indices = {best_poc_idx}

        up_ptr = best_poc_idx + 1
        dn_ptr = best_poc_idx - 1

        while cum_tpos < target_tpos and (up_ptr < num_bins or dn_ptr >= 0):
            # Sum next 2 ticks above
            up_sum = 0
            if up_ptr < num_bins:
                up_sum += tpo_counts[up_ptr]
                if up_ptr + 1 < num_bins:
                    up_sum += tpo_counts[up_ptr + 1]

            # Sum next 2 ticks below
            dn_sum = 0
            if dn_ptr >= 0:
                dn_sum += tpo_counts[dn_ptr]
                if dn_ptr - 1 >= 0:
                    dn_sum += tpo_counts[dn_ptr - 1]

            if up_sum == 0 and dn_sum == 0:
                break

            if up_sum > dn_sum:
                if up_ptr < num_bins:
                    va_indices.add(up_ptr)
                    cum_tpos += tpo_counts[up_ptr]
                if up_ptr + 1 < num_bins:
                    va_indices.add(up_ptr + 1)
                    cum_tpos += tpo_counts[up_ptr + 1]
                up_ptr += 2
            elif dn_sum > up_sum:
                if dn_ptr >= 0:
                    va_indices.add(dn_ptr)
                    cum_tpos += tpo_counts[dn_ptr]
                if dn_ptr - 1 >= 0:
                    va_indices.add(dn_ptr - 1)
                    cum_tpos += tpo_counts[dn_ptr - 1]
                dn_ptr -= 2
            else: # Equal
                if up_ptr < num_bins:
                    va_indices.add(up_ptr)
                    cum_tpos += tpo_counts[up_ptr]
                if up_ptr + 1 < num_bins:
                    va_indices.add(up_ptr + 1)
                    cum_tpos += tpo_counts[up_ptr + 1]
                if dn_ptr >= 0:
                    va_indices.add(dn_ptr)
                    cum_tpos += tpo_counts[dn_ptr]
                if dn_ptr - 1 >= 0:
                    va_indices.add(dn_ptr - 1)
                    cum_tpos += tpo_counts[dn_ptr - 1]
                up_ptr += 2
                dn_ptr -= 2

        va_prices = [price_bins[idx] for idx in va_indices]
        vah_price = float(np.max(va_prices))
        val_price = float(np.min(va_prices))

        # 5. Excess / Tails Detection vs. Poor Extremes
        # Buying Tail: Single print at low of day across at least 2 consecutive price ticks
        has_buying_tail = False
        if len(tpo_counts) >= 3:
            if tpo_counts[0] == 1 and tpo_counts[1] == 1:
                has_buying_tail = True

        # Selling Tail: Single print at high of day
        has_selling_tail = False
        if len(tpo_counts) >= 3:
            if tpo_counts[-1] == 1 and tpo_counts[-2] == 1:
                has_selling_tail = True

        # Poor High: Multiple TPOs at high without single-print excess
        is_poor_high = (tpo_counts[-1] >= 2)
        # Poor Low: Multiple TPOs at low without single-print excess
        is_poor_low = (tpo_counts[0] >= 2)

        return SessionProfile(
            date=s_date,
            high=s_high,
            low=s_low,
            open_price=s_open,
            close_price=s_close,
            volume=s_vol,
            poc_price=poc_price,
            vah_price=vah_price,
            val_price=val_price,
            ibh_price=ibh,
            ibl_price=ibl,
            ibr_range=ibr,
            tpo_count=total_tpos,
            has_buying_tail=has_buying_tail,
            has_selling_tail=has_selling_tail,
            is_poor_high=is_poor_high,
            is_poor_low=is_poor_low
        )

    def compute_rolling_market_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates over historical trading days causally, builds daily Market Profiles,
        and generates causal prior-day reference levels for every bar t.
        Guarantees 100% causal separation (bar t only sees completed day d-1 profile).
        """
        data = df.copy().reset_index(drop=True)
        data["date"] = data["time"].dt.floor("D")
        
        unique_dates = data["date"].unique()
        daily_profiles: Dict[pd.Timestamp, SessionProfile] = {}

        for d in unique_dates:
            day_df = data[data["date"] == d]
            prof = self.calculate_session_profile(day_df)
            if prof is not None:
                daily_profiles[d] = prof

        # Map prior-day profile metrics to current day's bars
        date_list = sorted(list(daily_profiles.keys()))
        date_to_prev = {}
        for idx in range(1, len(date_list)):
            curr_d = date_list[idx]
            prev_d = date_list[idx - 1]
            date_to_prev[curr_d] = daily_profiles[prev_d]

        # Initialize reference columns
        n_rows = len(data)
        prev_vah = np.full(n_rows, np.nan)
        prev_val = np.full(n_rows, np.nan)
        prev_poc = np.full(n_rows, np.nan)
        prev_high = np.full(n_rows, np.nan)
        prev_low = np.full(n_rows, np.nan)
        prev_ibh = np.full(n_rows, np.nan)
        prev_ibl = np.full(n_rows, np.nan)
        prev_ibr = np.full(n_rows, np.nan)
        prev_poor_high = np.zeros(n_rows, dtype=bool)
        prev_poor_low = np.zeros(n_rows, dtype=bool)

        curr_ibh_arr = np.full(n_rows, np.nan)
        curr_ibl_arr = np.full(n_rows, np.nan)
        curr_ibr_arr = np.full(n_rows, np.nan)
        is_ib_complete = np.zeros(n_rows, dtype=bool)

        # Vectorized population by date
        for d, day_indices in data.groupby("date").groups.items():
            idx_array = day_indices.values
            if d in date_to_prev:
                p_prof = date_to_prev[d]
                prev_vah[idx_array] = p_prof.vah_price
                prev_val[idx_array] = p_prof.val_price
                prev_poc[idx_array] = p_prof.poc_price
                prev_high[idx_array] = p_prof.high
                prev_low[idx_array] = p_prof.low
                prev_ibh[idx_array] = p_prof.ibh_price
                prev_ibl[idx_array] = p_prof.ibl_price
                prev_ibr[idx_array] = p_prof.ibr_range
                prev_poor_high[idx_array] = p_prof.is_poor_high
                prev_poor_low[idx_array] = p_prof.is_poor_low

            # Current day Initial Balance (Causal: Available starting from bar index 12 within the day)
            if len(idx_array) >= self.params.ib_bars:
                day_highs = data["high"].iloc[idx_array].values
                day_lows = data["low"].iloc[idx_array].values
                c_ibh = np.max(day_highs[:self.params.ib_bars])
                c_ibl = np.min(day_lows[:self.params.ib_bars])
                c_ibr = max(c_ibh - c_ibl, self.params.tick_size)

                # From bar 12 onwards, IB is fixed and known
                post_ib_indices = idx_array[self.params.ib_bars:]
                curr_ibh_arr[post_ib_indices] = c_ibh
                curr_ibl_arr[post_ib_indices] = c_ibl
                curr_ibr_arr[post_ib_indices] = c_ibr
                is_ib_complete[post_ib_indices] = True

        out = pd.DataFrame(index=data.index)
        out["prev_vah"] = prev_vah
        out["prev_val"] = prev_val
        out["prev_poc"] = prev_poc
        out["prev_high"] = prev_high
        out["prev_low"] = prev_low
        out["prev_ibh"] = prev_ibh
        out["prev_ibl"] = prev_ibl
        out["prev_ibr"] = prev_ibr
        out["prev_poor_high"] = prev_poor_high
        out["prev_poor_low"] = prev_poor_low
        out["curr_ibh"] = curr_ibh_arr
        out["curr_ibl"] = curr_ibl_arr
        out["curr_ibr"] = curr_ibr_arr
        out["is_ib_complete"] = is_ib_complete

        return out
