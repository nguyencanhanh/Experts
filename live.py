import os
import time
import json
import joblib
import torch
import pytz
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime

from config import (
    SYMBOL, MODEL_PATH, SCALER_PATH, FEATURES_PATH, METRICS_PATH,
    LIVE_BUY_THRESHOLD_OFFSET, LIVE_SELL_THRESHOLD_OFFSET,
    MAX_SPREAD_POINTS, COOLDOWN_BARS, REJECT_COOLDOWN_BARS,
    RISK_PER_TRADE, SL_ATR_MULT, USE_BREAK_EVEN,
    USE_TRAILING_STOP, USE_PARTIAL_TP, PARTIAL_TP_R,
    PARTIAL_CLOSE_RATIO, TRAILING_ATR_MULT,
    LIVE_LOG_PATH, TIMEFRAMES, SEQ_LEN, DEVICE,
    MAX_DAILY_LOSS_PCT, MAX_OPEN_POSITIONS, MAGIC,
    MT5_RECONNECT_WAIT_SEC, HEARTBEAT_INTERVAL_SEC,
    MAX_STALE_MINUTES, MAX_LATENCY_MS, MAX_POSITION_HOURS,
    EQUITY_DRAWDOWN_STOP_PCT,
)
from utils import append_csv_row, FeatureScaler
from data_mt5 import get_symbol_info, get_open_positions, get_recent_rates
from features import add_indicators, merge_timeframes, add_cross_features, get_base_features
from filters import (
    is_in_sessions, is_in_news_window, compute_rr_from_proba,
    regime_filter, context_side_allowed, confirm_entry, load_news_events,
)
from execution import place_market_order, modify_position_sl_tp, close_partial_position
from backtest import calc_lot_by_risk, round_volume
from model import CNNBiLSTMTransformer
from sequence_dataset import SequenceDataset


def build_live_sequence_frame(symbol: str):
    raw_m1 = get_recent_rates(symbol, TIMEFRAMES["M1"], 4000)
    raw_m5 = get_recent_rates(symbol, TIMEFRAMES["M5"], 1500)
    raw_m15 = get_recent_rates(symbol, TIMEFRAMES["M15"], 1000)
    raw_h1 = get_recent_rates(symbol, TIMEFRAMES["H1"], 500)

    base_m1 = raw_m1.copy()
    m1_feat = add_indicators(raw_m1, "M1")
    m5_feat = add_indicators(raw_m5, "M5")
    m15_feat = add_indicators(raw_m15, "M15")
    h1_feat = add_indicators(raw_h1, "H1")

    df = base_m1.merge(m1_feat, on="time", how="left")
    df = merge_timeframes(df, m5_feat, m15_feat, h1_feat)
    df = add_cross_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


def check_data_stale(df, max_stale_minutes: int = MAX_STALE_MINUTES) -> bool:
    """Check if data is stale (delayed more than max_stale_minutes)."""
    if len(df) == 0:
        return True
    last_time = df["time"].iloc[-1]
    if hasattr(last_time, "tz_localize"):
        now = datetime.now(pytz.UTC)
    else:
        now = np.datetime64("now")
    age_minutes = (now - last_time).total_seconds() / 60.0 if hasattr(last_time, "total_seconds") else 999
    try:
        age_minutes = (datetime.now(pytz.UTC) - last_time.to_pydatetime()).total_seconds() / 60.0
    except Exception:
        return False
    if age_minutes > max_stale_minutes:
        print(f"⚠️ Stale data: last bar is {age_minutes:.1f} min old")
        return True
    return False


def build_live_sequence_inputs(df_live, seq_len, feature_cols, scaler):
    """Build scaled 2D features and create SequenceDataset for live inference."""
    feat_df = get_base_features(df_live)
    feat_df = feat_df.dropna().reset_index(drop=True)

    df_aligned = df_live.iloc[-len(feat_df):].reset_index(drop=True).copy()
    arr = feat_df[feature_cols].values.astype("float32")

    # Scale the 2D features
    scaled = scaler.transform(arr)

    # Create dummy targets for the dataset
    n_seq = len(scaled) - seq_len + 1
    if n_seq <= 0:
        return None, None, None

    targets = np.zeros(n_seq, dtype=np.int64)
    ds = SequenceDataset(scaled, targets, seq_len, 0, n_seq)

    row_df = df_aligned.iloc[seq_len - 1:].reset_index(drop=True)
    return ds, row_df, scaled


def log_reject(signal_time, reason, p_buy, p_sell, spread_points, row):
    append_csv_row(LIVE_LOG_PATH, {
        "signal_time": str(signal_time),
        "event": reason,
        "p_buy": p_buy,
        "p_sell": p_sell,
        "spread_points": spread_points,
        "atr_pct": float(row["M1_atr_pct"]) if "M1_atr_pct" in row else None,
        "m15_trend_strength": float(row["M15_ema_spread_20_50"]) if "M15_ema_spread_20_50" in row else None,
        "h1_trend_strength": float(row["H1_ema_spread_20_50"]) if "H1_ema_spread_20_50" in row else None,
        "buy_context_score": float(row["buy_context_score"]) if "buy_context_score" in row else None,
        "sell_context_score": float(row["sell_context_score"]) if "sell_context_score" in row else None,
    })


def check_daily_loss_exceeded(symbol: str) -> bool:
    account = mt5.account_info()
    if account is None:
        return True

    utc = pytz.UTC
    today_start = datetime.now(utc).replace(hour=0, minute=0, second=0, microsecond=0)

    deals = mt5.history_deals_get(today_start, datetime.now(utc))
    if deals is None:
        deals = []

    daily_pnl = sum(
        d.profit + d.commission + d.swap
        for d in deals if d.symbol == symbol and d.magic == MAGIC
    )

    positions = get_open_positions(symbol)
    unrealized = sum(p.profit for p in positions)
    total_pnl = daily_pnl + unrealized

    loss_pct = abs(min(total_pnl, 0)) / max(account.balance, 1.0)
    if loss_pct >= MAX_DAILY_LOSS_PCT:
        print(f"⚠️ Daily loss limit: {loss_pct:.2%} >= {MAX_DAILY_LOSS_PCT:.2%}")
        return True
    return False


def check_equity_drawdown(symbol: str) -> bool:
    """Check equity drawdown from balance — additional safety layer."""
    account = mt5.account_info()
    if account is None:
        return True
    if account.balance <= 0:
        return True
    dd = (account.balance - account.equity) / account.balance
    if dd >= EQUITY_DRAWDOWN_STOP_PCT:
        print(f"⚠️ Equity drawdown: {dd:.2%} >= {EQUITY_DRAWDOWN_STOP_PCT:.2%}")
        return True
    return False


def ensure_mt5_connected() -> bool:
    try:
        info = mt5.terminal_info()
        if info is None:
            raise RuntimeError("No terminal info")
        return False
    except Exception:
        print("MT5 connection lost, reconnecting...")
        mt5.shutdown()
        time.sleep(MT5_RECONNECT_WAIT_SEC)
        if not mt5.initialize():
            raise RuntimeError(f"MT5 reconnect failed: {mt5.last_error()}")
        print("MT5 reconnected")
        return True


def close_expired_positions(symbol: str, max_hours: float = MAX_POSITION_HOURS):
    """Close positions that have been open too long without hitting TP/SL."""
    positions = get_open_positions(symbol)
    if not positions:
        return

    now = datetime.now(pytz.UTC)
    for pos in positions:
        try:
            open_time = datetime.fromtimestamp(pos.time, tz=pytz.UTC)
            hours_open = (now - open_time).total_seconds() / 3600.0
            if hours_open >= max_hours:
                print(f"⏰ Position {pos.ticket} open {hours_open:.1f}h — closing (timeout)")
                from execution import close_partial_position as _close
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    _close(symbol, pos, pos.volume)
        except Exception as e:
            print(f"Error closing expired position: {e}")


def manage_open_positions(symbol: str):
    positions = get_open_positions(symbol)
    if len(positions) == 0:
        return

    symbol_info = get_symbol_info(symbol)
    point = symbol_info.point

    df_live = build_live_sequence_frame(symbol)
    if len(df_live) < 5:
        return

    row = df_live.iloc[-2] if len(df_live) >= 2 else df_live.iloc[-1]
    atr_now = float(row["M1_atr_14"])
    if not (atr_now > 0):
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    for pos in positions:
        entry = pos.price_open
        sl = pos.sl
        tp = pos.tp
        volume = pos.volume

        if pos.type == mt5.POSITION_TYPE_BUY:
            current_price = tick.bid
            risk = max(entry - sl, point)
            current_profit_r = (current_price - entry) / risk
            candidate_trailing_sl = current_price - atr_now * TRAILING_ATR_MULT
            side = 1
        else:
            current_price = tick.ask
            risk = max(sl - entry, point) if sl > 0 else point
            current_profit_r = (entry - current_price) / risk
            candidate_trailing_sl = current_price + atr_now * TRAILING_ATR_MULT
            side = 2

        if USE_BREAK_EVEN and current_profit_r >= 1.0:
            if side == 1 and sl < entry:
                modify_position_sl_tp(symbol, pos.ticket, sl=round(entry, symbol_info.digits), tp=tp)
            elif side == 2 and (sl == 0 or sl > entry):
                modify_position_sl_tp(symbol, pos.ticket, sl=round(entry, symbol_info.digits), tp=tp)

        if USE_TRAILING_STOP:
            if side == 1 and candidate_trailing_sl > max(sl, entry if USE_BREAK_EVEN else -1e18):
                modify_position_sl_tp(symbol, pos.ticket, sl=round(candidate_trailing_sl, symbol_info.digits), tp=tp)
            elif side == 2 and (sl == 0 or candidate_trailing_sl < min(sl, entry if USE_BREAK_EVEN else 1e18)):
                modify_position_sl_tp(symbol, pos.ticket, sl=round(candidate_trailing_sl, symbol_info.digits), tp=tp)

        if USE_PARTIAL_TP and current_profit_r >= PARTIAL_TP_R and volume > symbol_info.volume_min:
            if volume >= round(symbol_info.volume_min * 1.9, 2):
                close_vol = round_volume(volume * PARTIAL_CLOSE_RATIO, symbol_info.volume_step, symbol_info.volume_min, volume)
                if close_vol < volume:
                    close_partial_position(symbol, pos, close_vol)


def predict_live(model, ds, batch_size=1024):
    """Predict probabilities for live data using SequenceDataset."""
    from trainer import predict_proba
    return predict_proba(model, ds, batch_size=batch_size)


def run_live():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH) and os.path.exists(METRICS_PATH)):
        raise RuntimeError("Thiếu model/scaler/features/metrics. Hãy train trước.")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    model = CNNBiLSTMTransformer(input_size=len(feature_cols)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    buy_threshold = min(float(metrics["best_buy_threshold"]) + LIVE_BUY_THRESHOLD_OFFSET, 0.75)
    sell_threshold = min(float(metrics["best_sell_threshold"]) + LIVE_SELL_THRESHOLD_OFFSET, 0.75)

    symbol_info = get_symbol_info(SYMBOL)
    news_df = load_news_events("news_events.csv")

    last_signal_bar_time = None
    last_trade_time = None
    last_reject_time = None
    last_heartbeat = time.time()

    print(f"Live V7 started | buy_th={buy_threshold} sell_th={sell_threshold}")
    print(f"Safety: max_daily_loss={MAX_DAILY_LOSS_PCT:.1%} max_positions={MAX_OPEN_POSITIONS}")
    print(f"        equity_dd_stop={EQUITY_DRAWDOWN_STOP_PCT:.1%} position_timeout={MAX_POSITION_HOURS}h")
    print(f"        max_latency={MAX_LATENCY_MS}ms max_stale={MAX_STALE_MINUTES}min")

    while True:
        try:
            ensure_mt5_connected()

            # Heartbeat
            if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL_SEC:
                account = mt5.account_info()
                bal = account.balance if account else 0
                eq = account.equity if account else 0
                print(f"♥ Heartbeat | balance={bal:.2f} equity={eq:.2f} | {datetime.now()}")
                last_heartbeat = time.time()

            # Safety checks
            if check_daily_loss_exceeded(SYMBOL):
                time.sleep(60)
                continue

            if check_equity_drawdown(SYMBOL):
                time.sleep(60)
                continue

            # Close expired positions
            close_expired_positions(SYMBOL)

            manage_open_positions(SYMBOL)

            df_live = build_live_sequence_frame(SYMBOL)

            # Stale data check
            if check_data_stale(df_live):
                time.sleep(10)
                continue

            ds, row_df, scaled = build_live_sequence_inputs(df_live, SEQ_LEN, feature_cols, scaler)

            if ds is None or len(ds) < 20:
                time.sleep(2)
                continue

            probs = predict_live(model, ds)

            signal_idx = len(row_df) - 2
            signal_row = row_df.iloc[signal_idx]
            signal_time = signal_row["time"]

            if last_signal_bar_time is not None and signal_time == last_signal_bar_time:
                time.sleep(2)
                continue
            last_signal_bar_time = signal_time

            if last_reject_time is not None:
                reject_gap = int((signal_time - last_reject_time).total_seconds() // 60)
                if reject_gap < REJECT_COOLDOWN_BARS:
                    time.sleep(2)
                    continue

            if not is_in_sessions(signal_time):
                time.sleep(2)
                continue

            if is_in_news_window(signal_time, news_df):
                time.sleep(2)
                continue

            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None:
                time.sleep(2)
                continue

            spread_points = (tick.ask - tick.bid) / symbol_info.point
            if spread_points > MAX_SPREAD_POINTS:
                log_reject(signal_time, "reject_spread", 0.0, 0.0, spread_points, signal_row)
                last_reject_time = signal_time
                time.sleep(2)
                continue

            # Max positions check
            positions = get_open_positions(SYMBOL)
            if len(positions) >= MAX_OPEN_POSITIONS:
                time.sleep(2)
                continue

            if last_trade_time is not None:
                bars_since = int((signal_time - last_trade_time).total_seconds() // 60)
                if bars_since < COOLDOWN_BARS:
                    time.sleep(2)
                    continue

            p_window = probs[max(0, signal_idx - 3 + 1):signal_idx + 1]
            p_buy = float(p_window[:, 1].mean())
            p_sell = float(p_window[:, 2].mean())

            side = 0
            pred_prob = 0.0
            if p_buy >= buy_threshold and p_buy > p_sell:
                side = 1
                pred_prob = p_buy
            elif p_sell >= sell_threshold and p_sell > p_buy:
                side = 2
                pred_prob = p_sell

            if side == 0:
                log_reject(signal_time, "reject_threshold", p_buy, p_sell, spread_points, signal_row)
                time.sleep(2)
                continue

            if not context_side_allowed(signal_row, side):
                log_reject(signal_time, "reject_context", p_buy, p_sell, spread_points, signal_row)
                last_reject_time = signal_time
                time.sleep(2)
                continue

            if not regime_filter(signal_row):
                log_reject(signal_time, "reject_regime", p_buy, p_sell, spread_points, signal_row)
                last_reject_time = signal_time
                time.sleep(2)
                continue

            confirm_window = row_df.iloc[max(0, signal_idx - 4):signal_idx + 1].copy()
            if not confirm_entry(confirm_window, side):
                log_reject(signal_time, "reject_confirm", p_buy, p_sell, spread_points, signal_row)
                last_reject_time = signal_time
                time.sleep(2)
                continue

            atr_now = float(signal_row["M1_atr_14"])
            if not (atr_now > 0):
                log_reject(signal_time, "reject_atr", p_buy, p_sell, spread_points, signal_row)
                time.sleep(2)
                continue

            # Re-check spread right before order (race condition mitigation)
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None:
                time.sleep(2)
                continue
            spread_points_now = (tick.ask - tick.bid) / symbol_info.point
            if spread_points_now > MAX_SPREAD_POINTS:
                log_reject(signal_time, "reject_spread_recheck", p_buy, p_sell, spread_points_now, signal_row)
                last_reject_time = signal_time
                time.sleep(2)
                continue

            rr = compute_rr_from_proba(pred_prob)
            sl_dist = atr_now * SL_ATR_MULT

            account = mt5.account_info()
            if account is None:
                time.sleep(2)
                continue

            volume = calc_lot_by_risk(symbol_info, account.balance, RISK_PER_TRADE, sl_dist)

            if side == 1:
                entry = tick.ask
                sl = entry - sl_dist
                tp = entry + sl_dist * rr
            else:
                entry = tick.bid
                sl = entry + sl_dist
                tp = entry - sl_dist * rr

            t_start = time.time()
            result = place_market_order(SYMBOL, side, volume, sl, tp)
            latency_ms = (time.time() - t_start) * 1000

            # Latency guard
            if latency_ms > MAX_LATENCY_MS:
                print(f"⚠️ High latency: {latency_ms:.0f}ms (max={MAX_LATENCY_MS}ms)")

            append_csv_row(LIVE_LOG_PATH, {
                "signal_time": str(signal_time),
                "event": "order_send",
                "side": "BUY" if side == 1 else "SELL",
                "p_buy": p_buy, "p_sell": p_sell,
                "pred_prob": pred_prob, "rr": rr,
                "volume": volume,
                "retcode": getattr(result, "retcode", None) if result is not None else None,
                "comment": getattr(result, "comment", "") if result is not None else "None",
                "order": getattr(result, "order", None) if result is not None else None,
                "deal": getattr(result, "deal", None) if result is not None else None,
                "spread_points": spread_points_now,
                "latency_ms": round(latency_ms, 1),
            })

            if result is not None and getattr(result, "retcode", None) == 10009:
                last_trade_time = signal_time

            time.sleep(2)

        except KeyboardInterrupt:
            print("Dừng live V7")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(5)
