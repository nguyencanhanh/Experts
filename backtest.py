import numpy as np
import pandas as pd

from config import (
    INITIAL_BALANCE, COOLDOWN_BARS, SIGNAL_SMOOTH_BARS, HORIZON_BARS,
    RISK_PER_TRADE, SL_ATR_MULT, MAX_SPREAD_POINTS,
    BACKTEST_ENTRY_SLIPPAGE_POINTS, BACKTEST_EXIT_SLIPPAGE_POINTS,
    USE_PARTIAL_TP, PARTIAL_CLOSE_RATIO, USE_BREAK_EVEN,
    USE_TRAILING_STOP, TRAILING_ATR_MULT,
    BUY_THRESHOLD_GRID, SELL_THRESHOLD_GRID,
    WF_TRAIN_BARS, WF_TEST_BARS, WF_STEP_BARS, SEQ_LEN,
)
from filters import (
    is_in_sessions, is_in_news_window, compute_rr_from_proba,
    regime_filter, context_side_allowed, confirm_entry,
)


def round_volume(volume: float, volume_step: float, volume_min: float, volume_max: float) -> float:
    volume = max(volume_min, min(volume_max, volume))
    steps = round(volume / volume_step)
    rounded = steps * volume_step
    digits = 2 if volume_step >= 0.01 else 3
    return round(rounded, digits)


def calc_lot_by_risk(symbol_info, balance: float, risk_per_trade: float, sl_distance_price: float) -> float:
    if sl_distance_price <= 0:
        return symbol_info.volume_min

    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    if tick_value is None or tick_value <= 0 or tick_size is None or tick_size <= 0:
        return symbol_info.volume_min

    money_per_1_price_move_per_lot = tick_value / tick_size
    risk_money = balance * risk_per_trade
    lot = risk_money / (sl_distance_price * money_per_1_price_move_per_lot)

    return round_volume(
        min(lot, symbol_info.volume_max),
        symbol_info.volume_step, symbol_info.volume_min, symbol_info.volume_max,
    )


def pnl_money(symbol_info, volume: float, entry: float, exit_price: float, side: int) -> float:
    direction = 1 if side == 1 else -1
    price_move = (exit_price - entry) * direction
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    if tick_value <= 0 or tick_size <= 0:
        return 0.0
    money_per_1_price_move_per_lot = tick_value / tick_size
    return price_move * money_per_1_price_move_per_lot * volume


def apply_slippage(price: float, point: float, side: int, is_entry: bool) -> float:
    slip = (BACKTEST_ENTRY_SLIPPAGE_POINTS if is_entry else BACKTEST_EXIT_SLIPPAGE_POINTS) * point
    if side == 1:
        return price + slip if is_entry else price - slip
    return price - slip if is_entry else price + slip


def simulate_trade(df, entry_idx, side, entry, sl, tp, volume, symbol_info, horizon, initial_risk):
    remaining_volume = volume
    realized_pnl = 0.0
    partial_done = False
    sl_current = sl
    tp_current = tp
    point = symbol_info.point

    end_idx = min(entry_idx + horizon - 1, len(df) - 1)

    for j in range(entry_idx, end_idx + 1):
        row = df.iloc[j]
        high_j = row["high"]
        low_j = row["low"]
        atr_j = row["M1_atr_14"]

        one_r_price = entry + initial_risk if side == 1 else entry - initial_risk

        if USE_PARTIAL_TP and not partial_done:
            hit_partial = (high_j >= one_r_price) if side == 1 else (low_j <= one_r_price)
            if hit_partial:
                close_vol = remaining_volume * PARTIAL_CLOSE_RATIO
                partial_exit = apply_slippage(one_r_price, point, side, False)
                realized_pnl += pnl_money(symbol_info, close_vol, entry, partial_exit, side)
                remaining_volume -= close_vol
                partial_done = True
                if USE_BREAK_EVEN:
                    sl_current = entry

        if USE_BREAK_EVEN and not partial_done:
            hit_be = (high_j >= one_r_price) if side == 1 else (low_j <= one_r_price)
            if hit_be:
                sl_current = entry

        if USE_TRAILING_STOP and np.isfinite(atr_j) and atr_j > 0:
            if side == 1:
                sl_current = max(sl_current, row["close"] - atr_j * TRAILING_ATR_MULT)
            else:
                sl_current = min(sl_current, row["close"] + atr_j * TRAILING_ATR_MULT)

        if side == 1:
            hit_tp = high_j >= tp_current
            hit_sl = low_j <= sl_current
        else:
            hit_tp = low_j <= tp_current
            hit_sl = high_j >= sl_current

        if hit_tp and hit_sl:
            exit_price = apply_slippage(sl_current, point, side, False)
            realized_pnl += pnl_money(symbol_info, remaining_volume, entry, exit_price, side)
            return j, exit_price, "SL", realized_pnl

        if hit_sl:
            exit_price = apply_slippage(sl_current, point, side, False)
            realized_pnl += pnl_money(symbol_info, remaining_volume, entry, exit_price, side)
            return j, exit_price, "SL", realized_pnl

        if hit_tp:
            exit_price = apply_slippage(tp_current, point, side, False)
            realized_pnl += pnl_money(symbol_info, remaining_volume, entry, exit_price, side)
            return j, exit_price, "TP", realized_pnl

    last_close = apply_slippage(df.iloc[end_idx]["close"], point, side, False)
    realized_pnl += pnl_money(symbol_info, remaining_volume, entry, last_close, side)
    return end_idx, last_close, "TIME", realized_pnl


def score_summary(summary: dict) -> float:
    """Regularized scoring — penalizes extreme thresholds to reduce overfit."""
    n = summary["trades"]
    if n == 0:
        return 0.0

    net_norm = summary["net_profit"] / INITIAL_BALANCE
    pf = max(summary["profit_factor"], 0.1)
    dd = max(1 - summary["max_drawdown"], 0.01)
    wr = max(summary["win_rate"], 0.01)
    trade_factor = min(n / 20.0, 1.0)

    # Penalize extreme thresholds (too high = too few trades, likely overfit)
    bt = summary.get("buy_threshold", 0.55)
    st = summary.get("sell_threshold", 0.55)
    threshold_penalty = 1.0 - 0.3 * max(0, bt - 0.60) - 0.3 * max(0, st - 0.60)
    threshold_penalty = max(threshold_penalty, 0.5)

    return net_norm * pf * dd * (wr ** 1.5) * trade_factor * threshold_penalty


def backtest_strategy(row_df: pd.DataFrame, probs: np.ndarray, symbol_info, news_df: pd.DataFrame,
                      buy_threshold: float, sell_threshold: float):
    df = row_df.copy().reset_index(drop=True)

    balance = INITIAL_BALANCE
    equity_peak = INITIAL_BALANCE
    max_drawdown = 0.0
    trades = []
    equity_curve = []

    skip_session = skip_news = skip_spread = skip_side = 0
    skip_context = skip_regime = skip_confirm = skip_atr = 0

    i = max(SIGNAL_SMOOTH_BARS + 5, 10)
    last_trade_entry_idx = -999999

    while i < len(df) - HORIZON_BARS - 2:
        signal_idx = i
        entry_idx = signal_idx + 1

        if entry_idx - last_trade_entry_idx < COOLDOWN_BARS:
            i += 1
            continue

        signal_row = df.iloc[signal_idx]
        ts = signal_row["time"]

        if not is_in_sessions(ts):
            skip_session += 1
            i += 1
            continue

        if is_in_news_window(ts, news_df):
            skip_news += 1
            i += 1
            continue

        entry_row = df.iloc[entry_idx]
        spread_points = float(entry_row.get("spread", 0))
        if spread_points > MAX_SPREAD_POINTS:
            skip_spread += 1
            i += 1
            continue

        p_window = probs[max(0, signal_idx - SIGNAL_SMOOTH_BARS + 1):signal_idx + 1]
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
            skip_side += 1
            i += 1
            continue

        if not context_side_allowed(signal_row, side):
            skip_context += 1
            i += 1
            continue

        if not regime_filter(signal_row):
            skip_regime += 1
            i += 1
            continue

        confirm_window = df.iloc[max(0, signal_idx - 4):signal_idx + 1].copy()
        if not confirm_entry(confirm_window, side):
            skip_confirm += 1
            i += 1
            continue

        atr_now = float(signal_row["M1_atr_14"])
        if not np.isfinite(atr_now) or atr_now <= 0:
            skip_atr += 1
            i += 1
            continue

        rr = compute_rr_from_proba(pred_prob)
        sl_dist = atr_now * SL_ATR_MULT
        point = symbol_info.point
        spread_price = spread_points * point
        entry_open = float(entry_row["open"])

        if side == 1:
            raw_entry = entry_open + spread_price / 2.0
            entry = apply_slippage(raw_entry, point, side, True)
            sl = entry - sl_dist
            tp = entry + sl_dist * rr
        else:
            raw_entry = entry_open - spread_price / 2.0
            entry = apply_slippage(raw_entry, point, side, True)
            sl = entry + sl_dist
            tp = entry - sl_dist * rr

        volume = calc_lot_by_risk(symbol_info, balance, RISK_PER_TRADE, sl_dist)

        exit_idx, exit_price, exit_reason, pnl = simulate_trade(
            df, entry_idx, side, entry, sl, tp, volume, symbol_info, HORIZON_BARS, sl_dist
        )

        balance += pnl
        equity_peak = max(equity_peak, balance)
        drawdown = (equity_peak - balance) / equity_peak if equity_peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)

        trades.append({
            "signal_time": str(ts),
            "entry_time": str(df.iloc[entry_idx]["time"]),
            "exit_time": str(df.iloc[exit_idx]["time"]),
            "side": "BUY" if side == 1 else "SELL",
            "p_buy": p_buy, "p_sell": p_sell, "pred_prob": pred_prob,
            "buy_threshold": buy_threshold, "sell_threshold": sell_threshold,
            "rr": rr, "entry": entry, "sl_initial": sl, "tp_initial": tp,
            "exit_price": exit_price, "exit_reason": exit_reason,
            "volume": volume, "pnl": pnl, "balance_after": balance,
            "spread_points": spread_points,
        })

        equity_curve.append({
            "time": str(df.iloc[exit_idx]["time"]),
            "balance": balance, "drawdown": drawdown,
        })

        last_trade_entry_idx = entry_idx
        i = exit_idx + 1

    print("DEBUG BACKTEST SKIPS:")
    print("skip_session =", skip_session)
    print("skip_news    =", skip_news)
    print("skip_spread  =", skip_spread)
    print("skip_side    =", skip_side)
    print("skip_context =", skip_context)
    print("skip_regime  =", skip_regime)
    print("skip_confirm =", skip_confirm)
    print("skip_atr     =", skip_atr)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    if len(trades_df) == 0:
        summary = {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 0.0,
            "net_profit": 0.0, "ending_balance": float(balance),
            "max_drawdown": float(max_drawdown), "avg_pnl": 0.0,
            "buy_threshold": buy_threshold, "sell_threshold": sell_threshold,
        }
        return trades_df, equity_df, summary

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    summary = {
        "trades": int(len(trades_df)),
        "wins": int((trades_df["pnl"] > 0).sum()),
        "losses": int((trades_df["pnl"] < 0).sum()),
        "win_rate": float((trades_df["pnl"] > 0).mean()),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": float(profit_factor),
        "net_profit": float(trades_df["pnl"].sum()),
        "ending_balance": float(balance),
        "max_drawdown": float(max_drawdown),
        "avg_pnl": float(trades_df["pnl"].mean()),
        "buy_threshold": float(buy_threshold),
        "sell_threshold": float(sell_threshold),
    }
    return trades_df, equity_df, summary


def optimize_thresholds(valid_row_df, valid_probs, symbol_info, news_df):
    results = []
    for bth in BUY_THRESHOLD_GRID:
        for sth in SELL_THRESHOLD_GRID:
            _, _, summary = backtest_strategy(valid_row_df, valid_probs, symbol_info, news_df, bth, sth)
            summary["score"] = float(score_summary(summary))
            results.append(summary)

    results_df = pd.DataFrame(results).sort_values(
        ["score", "profit_factor", "net_profit"], ascending=False,
    ).reset_index(drop=True)

    return results_df, results_df.iloc[0].to_dict()


def run_walkforward(bundle, trainer_predict_fn, trainer_train_fn, symbol_info, news_df):
    """Walk-forward analysis — updated for new SequenceBundle API."""
    from utils import FeatureScaler
    from sequence_dataset import SequenceDataset

    wf_results = []
    wf_trades = []
    wf_equity = []

    start = 0
    window_id = 0
    seq_len = bundle.seq_len

    while True:
        train_start = start
        train_end = train_start + WF_TRAIN_BARS
        test_start = train_end
        test_end = test_start + WF_TEST_BARS

        if test_end > bundle.n_sequences:
            break

        # Fit scaler on training features (2D)
        feat_start = train_start
        feat_end = train_end + seq_len - 1
        scaler = FeatureScaler()
        scaler.fit(bundle.features[feat_start:feat_end])

        # Scale all features
        scaled = scaler.transform(bundle.features)

        # Split train/valid
        valid_cut = int((train_end - train_start) * 0.85)
        actual_valid_start = train_start + valid_cut

        train_ds = SequenceDataset(scaled, bundle.targets, seq_len, train_start, valid_cut)
        valid_ds = SequenceDataset(scaled, bundle.targets, seq_len, actual_valid_start, train_end - actual_valid_start)
        test_ds = SequenceDataset(scaled, bundle.targets, seq_len, test_start, test_end - test_start)

        row_df_valid = bundle.row_df.iloc[actual_valid_start:train_end].reset_index(drop=True)
        row_df_test = bundle.row_df.iloc[test_start:test_end].reset_index(drop=True)

        model = trainer_train_fn(train_ds, valid_ds, input_size=bundle.features.shape[-1])

        valid_probs = trainer_predict_fn(model, valid_ds)
        _, best = optimize_thresholds(row_df_valid, valid_probs, symbol_info, news_df)

        buy_threshold = float(best["buy_threshold"])
        sell_threshold = float(best["sell_threshold"])

        test_probs = trainer_predict_fn(model, test_ds)
        trades_df, equity_df, summary = backtest_strategy(
            row_df_test, test_probs, symbol_info, news_df, buy_threshold, sell_threshold
        )

        summary["window_id"] = window_id
        summary["train_start_idx"] = train_start
        summary["train_end_idx"] = train_end
        summary["test_start_idx"] = test_start
        summary["test_end_idx"] = test_end
        wf_results.append(summary)

        if len(trades_df) > 0:
            trades_df["window_id"] = window_id
            wf_trades.append(trades_df)

        if len(equity_df) > 0:
            equity_df["window_id"] = window_id
            wf_equity.append(equity_df)

        print(f"WF window={window_id} trades={summary['trades']} pf={summary['profit_factor']:.2f} net={summary['net_profit']:.2f}")

        start += WF_STEP_BARS
        window_id += 1

    wf_results_df = pd.DataFrame(wf_results)
    wf_trades_df = pd.concat(wf_trades, ignore_index=True) if len(wf_trades) > 0 else pd.DataFrame()
    wf_equity_df = pd.concat(wf_equity, ignore_index=True) if len(wf_equity) > 0 else pd.DataFrame()

    if len(wf_results_df) == 0:
        return wf_results_df, wf_trades_df, wf_equity_df, {}

    agg = {
        "windows": int(len(wf_results_df)),
        "trades": int(wf_results_df["trades"].sum()),
        "avg_win_rate": float(wf_results_df["win_rate"].mean()),
        "avg_profit_factor": float(wf_results_df["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0).mean()),
        "total_net_profit": float(wf_results_df["net_profit"].sum()),
        "avg_max_drawdown": float(wf_results_df["max_drawdown"].mean()),
    }
    return wf_results_df, wf_trades_df, wf_equity_df, agg
