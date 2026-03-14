from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TIMEFRAME_NAMES = ("M1", "M5", "M15", "H1")


@dataclass
class OfflineSymbolInfo:
    symbol: str
    point: float
    digits: int
    trade_tick_size: float
    trade_tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float


def _normalized_column_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _rename_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias_groups = {
        "time": {"time", "datetime", "timestamp", "date", "timeutc", "datetimeutc"},
        "open": {"open", "o"},
        "high": {"high", "h"},
        "low": {"low", "l"},
        "close": {"close", "c"},
        "tick_volume": {"tickvolume", "volume", "tickvol", "realvolume"},
        "spread": {"spread"},
    }

    rename_map = {}
    for column in df.columns:
        normalized = _normalized_column_name(column)
        for target, aliases in alias_groups.items():
            if normalized in aliases:
                rename_map[column] = target
                break
    out = df.rename(columns=rename_map)

    for target in set(rename_map.values()):
        duplicate_columns = [column for column in out.columns if column == target]
        if len(duplicate_columns) <= 1:
            continue
        merged = out.loc[:, duplicate_columns].bfill(axis=1).iloc[:, 0]
        out = out.drop(columns=duplicate_columns)
        out[target] = merged

    return out


def normalize_rates_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = _rename_common_columns(df.copy())
    required = ["time", "open", "high", "low", "close"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if pd.api.types.is_numeric_dtype(out["time"]):
        unit = "ms" if float(out["time"].abs().max()) > 1e11 else "s"
        out["time"] = pd.to_datetime(out["time"], unit=unit, utc=True, errors="coerce")
    else:
        out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")

    for column in ["open", "high", "low", "close", "tick_volume", "spread"]:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["tick_volume"] = out["tick_volume"].fillna(0.0)
    out["spread"] = out["spread"].fillna(0.0)

    out = out.dropna(subset=required + ["time"]).copy()
    out = out.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def _candidate_paths(data_dir: Path, symbol: str, timeframe: str) -> list[Path]:
    symbol = symbol.strip()
    symbol_slug = "".join(ch for ch in symbol if ch.isalnum())
    timeframe = timeframe.upper()

    exact_candidates = [
        data_dir / f"{symbol}_{timeframe}.csv",
        data_dir / f"{symbol_slug}_{timeframe}.csv",
        data_dir / f"{symbol.lower()}_{timeframe.lower()}.csv",
        data_dir / f"{symbol_slug.lower()}_{timeframe.lower()}.csv",
        data_dir / f"{timeframe}.csv",
        data_dir / f"{timeframe.lower()}.csv",
    ]

    found = [path for path in exact_candidates if path.exists()]
    if found:
        return found

    matches = []
    for path in sorted(data_dir.rglob("*.csv")):
        stem = path.stem.lower()
        if timeframe.lower() not in stem:
            continue
        if symbol.lower() in stem or symbol_slug.lower() in stem or stem == timeframe.lower():
            matches.append(path)
    return matches


def resolve_rates_path(data_dir: str | Path, symbol: str, timeframe: str) -> Path:
    base_dir = Path(data_dir).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {base_dir}")

    matches = _candidate_paths(base_dir, symbol, timeframe)
    if not matches:
        raise FileNotFoundError(
            f"Could not find CSV for symbol={symbol} timeframe={timeframe} in {base_dir}. "
            f"Expected names like {symbol}_{timeframe}.csv or {timeframe}.csv"
        )
    return matches[0]


def load_rates_from_csv(data_dir: str | Path, symbol: str, timeframe: str, years_back: int = 1, verbose: bool = False) -> pd.DataFrame:
    path = resolve_rates_path(data_dir, symbol, timeframe)
    df = normalize_rates_df(pd.read_csv(path))
    if df.empty:
        raise RuntimeError(f"No valid rows found in {path}")

    if years_back > 0:
        cutoff = df["time"].max() - pd.Timedelta(days=365 * years_back + 3)
        df = df[df["time"] >= cutoff].reset_index(drop=True)

    if verbose:
        print(
            f"[OFFLINE DATA] {symbol} tf={timeframe} rows={len(df)} "
            f"from={df['time'].min()} to={df['time'].max()} source={path}"
        )
    return df


def _guess_price_digits(close_series: pd.Series, fallback: int = 2) -> int:
    sample = close_series.dropna().astype(float).head(500)
    if sample.empty:
        return fallback

    max_digits = fallback
    for value in sample:
        text = format(float(value), ".8f").rstrip("0").rstrip(".")
        if "." in text:
            max_digits = max(max_digits, len(text.split(".", 1)[1]))
    return max_digits


def _default_contract_size(symbol: str) -> float:
    symbol_upper = symbol.upper()
    if "XAU" in symbol_upper:
        return 100.0
    if "BTC" in symbol_upper:
        return 1.0
    return 1.0


def _default_volume_max(symbol: str) -> float:
    symbol_upper = symbol.upper()
    if "BTC" in symbol_upper:
        return 10.0
    return 100.0


def load_symbol_spec(path: str | Path | None) -> dict:
    if not path:
        return {}

    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Symbol spec file does not exist: {spec_path}")

    with spec_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_offline_symbol_info(symbol: str, close_series: pd.Series, spec: dict | None = None) -> OfflineSymbolInfo:
    spec = dict(spec or {})
    digits = int(spec.get("digits") or _guess_price_digits(close_series))
    point = float(spec.get("point") or (10 ** (-digits) if digits > 0 else 1.0))
    contract_size = float(spec.get("contract_size") or _default_contract_size(symbol))
    trade_tick_size = float(spec.get("trade_tick_size") or point)
    trade_tick_value = float(spec.get("trade_tick_value") or (contract_size * trade_tick_size))
    volume_min = float(spec.get("volume_min", 0.01))
    volume_max = float(spec.get("volume_max", _default_volume_max(symbol)))
    volume_step = float(spec.get("volume_step", 0.01))

    return OfflineSymbolInfo(
        symbol=symbol,
        point=point,
        digits=digits,
        trade_tick_size=trade_tick_size,
        trade_tick_value=trade_tick_value,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        contract_size=contract_size,
    )
