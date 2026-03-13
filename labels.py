import numpy as np
import pandas as pd

from config import HORIZON_BARS, SL_ATR_MULT, MIN_RR, USE_ADAPTIVE_RR


def build_labels_no_lookahead(df: pd.DataFrame, horizon: int = HORIZON_BARS,
                              sl_atr_mult: float = SL_ATR_MULT, min_rr: float = MIN_RR):
    labels = np.zeros(len(df), dtype=int)

    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    spread = df["spread"].values if "spread" in df.columns else np.zeros(len(df))
    atr_vals = df["M1_atr_14"].values

    # ── Adaptive RR: lower RR in low-volatility regimes ──
    atr_finite = atr_vals[np.isfinite(atr_vals) & (atr_vals > 0)]
    atr_median = np.median(atr_finite) if len(atr_finite) > 0 else 1.0

    for signal_idx in range(len(df) - horizon - 2):
        entry_idx = signal_idx + 1

        atr_now = atr_vals[signal_idx]
        if not np.isfinite(atr_now) or atr_now <= 0:
            continue

        spread_price = spread[entry_idx]
        mid_entry = open_[entry_idx]

        buy_entry = mid_entry + spread_price / 2.0
        sell_entry = mid_entry - spread_price / 2.0

        sl_dist = atr_now * sl_atr_mult

        # Adaptive RR: reduce target in low-vol regimes to get more balanced labels
        if USE_ADAPTIVE_RR and atr_now < atr_median:
            effective_rr = min_rr * 0.85
        else:
            effective_rr = min_rr

        tp_dist = sl_dist * effective_rr

        future_high = high[entry_idx:entry_idx + horizon]
        future_low = low[entry_idx:entry_idx + horizon]

        buy_tp = buy_entry + tp_dist
        buy_sl = buy_entry - sl_dist
        buy_tp_idx = np.where(future_high >= buy_tp)[0]
        buy_sl_idx = np.where(future_low <= buy_sl)[0]
        buy_ok = len(buy_tp_idx) > 0 and (len(buy_sl_idx) == 0 or buy_tp_idx[0] < buy_sl_idx[0])

        sell_tp = sell_entry - tp_dist
        sell_sl = sell_entry + sl_dist
        sell_tp_idx = np.where(future_low <= sell_tp)[0]
        sell_sl_idx = np.where(future_high >= sell_sl)[0]
        sell_ok = len(sell_tp_idx) > 0 and (len(sell_sl_idx) == 0 or sell_tp_idx[0] < sell_sl_idx[0])

        # Handle "both win" — pick the faster side
        if buy_ok and sell_ok:
            buy_speed = buy_tp_idx[0]
            sell_speed = sell_tp_idx[0]
            if buy_speed < sell_speed:
                labels[signal_idx] = 1
            elif sell_speed < buy_speed:
                labels[signal_idx] = 2
            # else: same bar → stays 0 (ambiguous)
        elif buy_ok:
            labels[signal_idx] = 1
        elif sell_ok:
            labels[signal_idx] = 2

    return pd.Series(labels, index=df.index, name="target")
