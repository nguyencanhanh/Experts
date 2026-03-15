import pandas as pd
from datetime import datetime

from config import (
    SESSION_FILTER, SESSION_WINDOWS_UTC,
    USE_NEWS_FILTER, NEWS_BLOCK_BEFORE_MIN, NEWS_BLOCK_AFTER_MIN, NEWS_IMPACT_ALLOW,
    RR_MAP, MIN_RR,
    MIN_ATR_PCT, MIN_M15_TREND_STRENGTH, MIN_H1_TREND_STRENGTH,
)


def empty_news_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["time_utc", "impact", "currency", "title"])


def parse_hhmm(s: str):
    hh, mm = s.split(":")
    return int(hh), int(mm)


def is_in_sessions(ts_utc: pd.Timestamp) -> bool:
    if not SESSION_FILTER:
        return True
    tod = ts_utc.time()
    for start_s, end_s in SESSION_WINDOWS_UTC:
        sh, sm = parse_hhmm(start_s)
        eh, em = parse_hhmm(end_s)
        start_t = datetime(2000, 1, 1, sh, sm).time()
        end_t = datetime(2000, 1, 1, eh, em).time()
        if start_t <= tod <= end_t:
            return True
    return False


def load_news_events(path: str) -> pd.DataFrame:
    if not USE_NEWS_FILTER:
        return empty_news_events_df()

    try:
        news = pd.read_csv(path)
    except FileNotFoundError:
        return empty_news_events_df()
    except pd.errors.EmptyDataError:
        return empty_news_events_df()

    if "time_utc" not in news.columns:
        raise ValueError("news_events.csv phải có cột time_utc")

    news["time_utc"] = pd.to_datetime(
        news["time_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    if news["time_utc"].isna().any():
        invalid = int(news["time_utc"].isna().sum())
        print(f"[NEWS] Dropping {invalid} rows with invalid time_utc from {path}")
        news = news.loc[~news["time_utc"].isna()].copy()
    news["impact"] = news.get("impact", "high").astype(str).str.lower()
    news["currency"] = news.get("currency", "").astype(str)
    news["title"] = news.get("title", "").astype(str)
    return news.sort_values("time_utc").reset_index(drop=True)


def is_in_news_window(ts_utc: pd.Timestamp, news_df: pd.DataFrame) -> bool:
    if news_df.empty or not USE_NEWS_FILTER:
        return False
    start = ts_utc - pd.Timedelta(minutes=NEWS_BLOCK_BEFORE_MIN)
    end = ts_utc + pd.Timedelta(minutes=NEWS_BLOCK_AFTER_MIN)
    subset = news_df[
        (news_df["time_utc"] >= start) &
        (news_df["time_utc"] <= end) &
        (news_df["impact"].isin(NEWS_IMPACT_ALLOW))
    ]
    return len(subset) > 0


def compute_rr_from_proba(p: float) -> float:
    for threshold, rr in RR_MAP:
        if p >= threshold:
            return rr
    return MIN_RR


def regime_filter(row: pd.Series) -> bool:
    return (
        row["M1_atr_pct"] > MIN_ATR_PCT and
        abs(row["M15_ema_spread_20_50"]) > MIN_M15_TREND_STRENGTH and
        abs(row["H1_ema_spread_20_50"]) > MIN_H1_TREND_STRENGTH
    )


def context_side_allowed(row: pd.Series, side: int) -> bool:
    if side == 1:
        return row["buy_context_score"] >= 1
    if side == 2:
        return row["sell_context_score"] >= 1
    return False


def confirm_entry(df_tail: pd.DataFrame, side: int) -> bool:
    """Scoring-based entry confirmation — need 3/5 conditions instead of all 5."""
    if len(df_tail) < 3:
        return False

    b0 = df_tail.iloc[-1]
    b1 = df_tail.iloc[-2]
    score = 0

    if side == 1:
        if b0["close"] > b0["open"]:
            score += 1
        if b0["close"] > b0["M1_ema_20"]:
            score += 1
        if b1["close"] >= b1["open"]:
            score += 1
        if b0["M15_trend_up_20_50"] == 1:
            score += 1
        if b0["H1_trend_up_20_50"] == 1:
            score += 1
        return score >= 3

    if side == 2:
        if b0["close"] < b0["open"]:
            score += 1
        if b0["close"] < b0["M1_ema_20"]:
            score += 1
        if b1["close"] <= b1["open"]:
            score += 1
        if b0["M15_trend_up_20_50"] == 0:
            score += 1
        if b0["H1_trend_up_20_50"] == 0:
            score += 1
        return score >= 3

    return False
