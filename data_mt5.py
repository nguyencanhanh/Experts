import os
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
import pytz

from config import DATA_CACHE_DIR, USE_LOCAL_DATA_CACHE


def mt5_init():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    print("MT5 initialized")


def mt5_shutdown():
    mt5.shutdown()
    print("MT5 shutdown")


def ensure_symbol(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Khong tim thay symbol: {symbol}")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Khong the select symbol: {symbol}")


def get_symbol_info(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Khong lay duoc symbol_info cho {symbol}")
    return info


def timeframe_to_name(timeframe: int) -> str:
    mapping = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_H1: "H1",
    }
    return mapping.get(timeframe, f"tf_{timeframe}")


def cache_path(symbol: str, timeframe: int) -> str:
    safe_symbol = "".join(ch if ch.isalnum() else "_" for ch in symbol)
    return os.path.join(DATA_CACHE_DIR, f"{safe_symbol}_{timeframe_to_name(timeframe)}.csv")


def normalize_rates_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()
    if "time" not in out.columns:
        raise ValueError("Rates data must include a time column")

    if pd.api.types.is_numeric_dtype(out["time"]):
        out["time"] = pd.to_datetime(out["time"], unit="s", utc=True)
    else:
        out["time"] = pd.to_datetime(out["time"], utc=True)

    return out.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)


def load_cached_rates(symbol: str, timeframe: int) -> pd.DataFrame:
    if not USE_LOCAL_DATA_CACHE:
        return pd.DataFrame()

    path = cache_path(symbol, timeframe)
    if not os.path.exists(path):
        return pd.DataFrame()

    cached = pd.read_csv(path)
    return normalize_rates_df(cached)


def merge_rates_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    valid = [normalize_rates_df(df) for df in frames if df is not None and len(df) > 0]
    if not valid:
        return pd.DataFrame()
    return normalize_rates_df(pd.concat(valid, ignore_index=True, sort=False))


def save_cached_rates(symbol: str, timeframe: int, df: pd.DataFrame):
    if not USE_LOCAL_DATA_CACHE or df is None or len(df) == 0:
        return

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    normalize_rates_df(df).to_csv(cache_path(symbol, timeframe), index=False)


def target_bars_for_years(timeframe: int, years_back: int) -> int:
    days = 365 * years_back
    if timeframe == mt5.TIMEFRAME_M1:
        return days * 24 * 60
    if timeframe == mt5.TIMEFRAME_M5:
        return days * 24 * 12
    if timeframe == mt5.TIMEFRAME_M15:
        return days * 24 * 4
    if timeframe == mt5.TIMEFRAME_H1:
        return days * 24
    return 50000


def fetch_rates_from_mt5(symbol: str, timeframe: int, target_bars: int) -> pd.DataFrame:
    chunk_size = 20000
    collected = []
    pos = 0
    total = 0

    for _ in range(100):
        remaining = target_bars - total
        if remaining <= 0:
            break

        count = min(chunk_size, remaining)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, pos, count)
        if rates is None or len(rates) == 0:
            break

        part = pd.DataFrame(rates)
        collected.append(part)
        got = len(part)
        total += got
        pos += got

        if got < count:
            break

    if len(collected) == 0:
        return pd.DataFrame()

    return normalize_rates_df(pd.concat(collected, ignore_index=True))


def get_rates(symbol: str, timeframe: int, years_back: int = 1, verbose: bool = False) -> pd.DataFrame:
    ensure_symbol(symbol)

    target_bars = target_bars_for_years(timeframe, years_back)
    cached_df = load_cached_rates(symbol, timeframe)
    live_df = fetch_rates_from_mt5(symbol, timeframe, target_bars)

    if len(live_df) == 0:
        utc = pytz.UTC
        end_time = datetime.now(tz=utc)
        start_time = end_time - timedelta(days=365 * years_back)
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        if rates is None or len(rates) == 0:
            if len(cached_df) == 0:
                raise RuntimeError(f"Khong lay duoc data {symbol} tf={timeframe}. err={mt5.last_error()}")
        else:
            live_df = normalize_rates_df(pd.DataFrame(rates))

    df = merge_rates_frames(cached_df, live_df)
    if len(df) == 0:
        raise RuntimeError(f"Du lieu rong sau khi lay data cho {symbol} tf={timeframe}")

    save_cached_rates(symbol, timeframe, df)

    cutoff = df["time"].max() - pd.Timedelta(days=365 * years_back + 3)
    df = df[df["time"] >= cutoff].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"Du lieu rong sau khi xu ly cho {symbol} tf={timeframe}")

    if verbose:
        tf_name = timeframe_to_name(timeframe)
        print(
            f"[DATA] {symbol} tf={tf_name} rows={len(df)} "
            f"from={df['time'].min()} to={df['time'].max()} "
            f"| cache_rows={len(cached_df)} live_rows={len(live_df)}"
        )

    return df


def get_recent_rates(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
    ensure_symbol(symbol)

    cached_df = load_cached_rates(symbol, timeframe)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        if len(cached_df) == 0:
            raise RuntimeError(f"Khong lay duoc recent data {symbol} tf={timeframe}. err={mt5.last_error()}")
        return cached_df.tail(count).reset_index(drop=True)

    recent_df = normalize_rates_df(pd.DataFrame(rates))
    merged_df = merge_rates_frames(cached_df, recent_df)
    save_cached_rates(symbol, timeframe, merged_df)
    return merged_df.tail(count).reset_index(drop=True)


def get_open_positions(symbol: str):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    return list(positions)


def get_supported_filling(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Khong lay duoc symbol_info cho {symbol}")

    filling = info.filling_mode
    if filling & mt5.SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    if filling & mt5.SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN
