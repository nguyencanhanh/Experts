import pytz
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta


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
        raise RuntimeError(f"Không tìm thấy symbol: {symbol}")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Không thể select symbol: {symbol}")


def get_symbol_info(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Không lấy được symbol_info cho {symbol}")
    return info


def get_rates(symbol: str, timeframe: int, years_back: int = 1, verbose: bool = False) -> pd.DataFrame:
    ensure_symbol(symbol)

    days = 365 * years_back
    if timeframe == mt5.TIMEFRAME_M1:
        target_bars = days * 24 * 60
    elif timeframe == mt5.TIMEFRAME_M5:
        target_bars = days * 24 * 12
    elif timeframe == mt5.TIMEFRAME_M15:
        target_bars = days * 24 * 4
    elif timeframe == mt5.TIMEFRAME_H1:
        target_bars = days * 24
    else:
        target_bars = 50000

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
        utc = pytz.UTC
        end_time = datetime.now(tz=utc)
        start_time = end_time - timedelta(days=365 * years_back)
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Không lấy được data {symbol} tf={timeframe}. err={mt5.last_error()}")
        df = pd.DataFrame(rates)
    else:
        df = pd.concat(collected, ignore_index=True)

    df = df.drop_duplicates(subset=["time"]).copy()
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    cutoff = df["time"].max() - pd.Timedelta(days=365 * years_back + 3)
    df = df[df["time"] >= cutoff].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"Dữ liệu rỗng sau khi xử lý cho {symbol} tf={timeframe}")

    if verbose:
        print(f"[DATA] {symbol} tf={timeframe} rows={len(df)} from={df['time'].min()} to={df['time'].max()}")

    return df


def get_recent_rates(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
    ensure_symbol(symbol)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Không lấy được recent data {symbol} tf={timeframe}. err={mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def get_open_positions(symbol: str):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    return list(positions)


def get_supported_filling(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Không lấy được symbol_info cho {symbol}")

    filling = info.filling_mode
    if filling & mt5.SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    if filling & mt5.SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN
