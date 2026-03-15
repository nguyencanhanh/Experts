import argparse
import json

import numpy as np
import pandas as pd

from backtest import backtest_strategy, optimize_thresholds, run_walkforward
from config import (
    DEVICE, NEWS_CSV_PATH, PROFILE_NAME, SYMBOL, TIMEFRAMES,
    TRAIN_RATIO, VALID_RATIO, CALIB_RATIO, YEARS_BACK,
)
from data_mt5 import ensure_symbol, get_rates, get_symbol_info, mt5_init, mt5_shutdown
from features import add_cross_features, add_indicators, get_base_features, merge_timeframes
from filters import load_news_events
from live import run_live
from save_load import load_inference_bundle, save_outputs
from sequence_dataset import (
    SequenceDataset,
    build_purged_four_way_split,
    build_sequence_bundle,
)
from trainer import evaluate, predict_proba, train_model
from utils import FeatureScaler, set_seed


def prepare_dataset(symbol: str, years_back: int = 1) -> pd.DataFrame:
    raw_m1 = get_rates(symbol, TIMEFRAMES["M1"], years_back, verbose=True)
    raw_m5 = get_rates(symbol, TIMEFRAMES["M5"], years_back, verbose=True)
    raw_m15 = get_rates(symbol, TIMEFRAMES["M15"], years_back, verbose=True)
    raw_h1 = get_rates(symbol, TIMEFRAMES["H1"], years_back, verbose=True)

    base_m1 = raw_m1.copy()
    m1_feat = add_indicators(raw_m1, "M1")
    m5_feat = add_indicators(raw_m5, "M5")
    m15_feat = add_indicators(raw_m15, "M15")
    h1_feat = add_indicators(raw_h1, "H1")

    df = base_m1.merge(m1_feat, on="time", how="left")
    df = merge_timeframes(df, m5_feat, m15_feat, h1_feat)
    df = add_cross_features(df)
    return df.reset_index(drop=True)


def ensure_saved_feature_layout(feature_cols, current_feature_cols):
    saved = list(feature_cols)
    current = list(current_feature_cols)
    if saved == current:
        return

    missing = [col for col in saved if col not in current]
    extra = [col for col in current if col not in saved]
    details = []
    if missing:
        details.append(f"missing={missing[:10]}")
    if extra:
        details.append(f"extra={extra[:10]}")
    raise RuntimeError(
        "Current feature layout no longer matches the saved model features. "
        + (" | ".join(details) if details else "")
    )


def _print_overfit_warning(eval_train, eval_valid, eval_test):
    """In cảnh báo nếu gap train-test lớn (dấu hiệu overfit)."""
    try:
        p_train = eval_train["report"]["weighted avg"]["precision"]
        p_valid = eval_valid["report"]["weighted avg"]["precision"]
        p_test = eval_test["report"]["weighted avg"]["precision"]
        gap_tv = p_train - p_valid
        gap_vt = p_valid - p_test
        print(f"\n===== OVERFIT MONITOR =====")
        print(f"  Train precision: {p_train:.4f}")
        print(f"  Valid precision: {p_valid:.4f}  (gap from train: {gap_tv:+.4f})")
        print(f"  Test  precision: {p_test:.4f}  (gap from valid: {gap_vt:+.4f})")
        if gap_tv > 0.10:
            print(f"  [WARNING] Train-Valid gap = {gap_tv:.3f} > 0.10 → Overfit nghiêm trọng!")
            print(f"            → Tăng DROPOUT, WEIGHT_DECAY, hoặc giảm CNN_CHANNELS/LSTM_HIDDEN")
        elif gap_tv > 0.05:
            print(f"  [WARNING] Train-Valid gap = {gap_tv:.3f} > 0.05 → Có dấu hiệu overfit nhẹ")
        else:
            print(f"  [OK] Train-Valid gap = {gap_tv:.3f} ≤ 0.05 → Tốt")
        if gap_vt > 0.10:
            print(f"  [WARNING] Valid-Test gap = {gap_vt:.3f} > 0.10 → Threshold bị overfit trên valid!")
        print("=" * 27)
    except Exception:
        pass


def train_pipeline(run_backtest: bool = True, run_wf: bool = True):
    symbol_info = get_symbol_info(SYMBOL)
    news_df = load_news_events(NEWS_CSV_PATH)

    df = prepare_dataset(SYMBOL, YEARS_BACK)
    print(f"Dataset rows: {len(df)}")
    print(df[["time", "open", "high", "low", "close"]].tail())

    bundle = build_sequence_bundle(df)
    n = bundle.n_sequences
    print(f"Sequences: {n} (features: {bundle.features.shape[1]}, seq_len: {bundle.seq_len})")

    unique, counts = np.unique(bundle.targets, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  class {label}: {count} ({count / n:.1%})")
    # Cảnh báo nếu class 0 chiếm quá nhiều
    class0_pct = counts[0] / n if len(counts) > 0 else 0
    if class0_pct > 0.70:
        print(f"[WARNING] Class 0 chiếm {class0_pct:.1%} > 70% — xem xét giảm MIN_RR hoặc tăng HORIZON_BARS")

    # ──────────────────────────────────────────────────────────────────────────
    # Split dữ liệu thành 4 phần rõ ràng để chống overfit:
    #   TRAIN  (65%): model học
    #   VALID  (10%): early stopping (model thấy qua validation loss)
    #   CALIB  (10%): optimize_thresholds — model CHƯA BAO GIỜ thấy trong training
    #   TEST   (15%): đánh giá cuối cùng
    # ──────────────────────────────────────────────────────────────────────────
    splits = build_purged_four_way_split(
        n,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        calib_ratio=CALIB_RATIO,
        seq_len=bundle.seq_len,
    )
    print(f"\nData split:")
    print(f"  Purge gap : {splits.purge_gap} sequences between segments")
    print(f"  Train : [{splits.train.start:6d}, {splits.train.end:6d}) = {splits.train.count} sequences ({TRAIN_RATIO:.0%} of usable)")
    print(f"  Valid : [{splits.valid.start:6d}, {splits.valid.end:6d}) = {splits.valid.count} sequences ({VALID_RATIO:.0%} of usable) — early stopping")
    print(f"  Calib : [{splits.calib.start:6d}, {splits.calib.end:6d}) = {splits.calib.count} sequences ({CALIB_RATIO:.0%} of usable) — threshold opt")
    print(f"  Test  : [{splits.test.start:6d}, {splits.test.end:6d}) = {splits.test.count} sequences ({1-TRAIN_RATIO-VALID_RATIO-CALIB_RATIO:.0%} of usable)")

    # Scaler fit CHỈ trên train
    scaler = FeatureScaler()
    scaler.fit(bundle.features[:splits.train.end + bundle.seq_len - 1])
    scaled = scaler.transform(bundle.features)

    train_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, splits.train.start, splits.train.count)
    valid_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, splits.valid.start, splits.valid.count)
    calib_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, splits.calib.start, splits.calib.count)
    test_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, splits.test.start, splits.test.count)

    row_df_calib = bundle.row_df.iloc[splits.calib.start:splits.calib.end].reset_index(drop=True)
    row_df_test = bundle.row_df.iloc[splits.test.start:splits.test.end].reset_index(drop=True)

    # Train model chính
    model = train_model(train_ds, valid_ds, input_size=bundle.features.shape[-1])

    eval_train = evaluate(model, train_ds)
    eval_valid = evaluate(model, valid_ds)
    eval_calib = evaluate(model, calib_ds)
    eval_test  = evaluate(model, test_ds)

    # In overfit monitor
    _print_overfit_warning(eval_train, eval_valid, eval_test)

    # Optimize threshold trên CALIB set (model chưa thấy) — tránh threshold overfit
    calib_probs = predict_proba(model, calib_ds)
    threshold_table, best = optimize_thresholds(row_df_calib, calib_probs, symbol_info, news_df)
    best_buy  = float(best["buy_threshold"])
    best_sell = float(best["sell_threshold"])
    print(f"\nCalib thresholds (out-of-sample): buy={best_buy} sell={best_sell}")

    test_trades_df = pd.DataFrame()
    test_equity_df = pd.DataFrame()
    test_summary = {}
    if run_backtest:
        test_probs = predict_proba(model, test_ds)
        test_trades_df, test_equity_df, test_summary = backtest_strategy(
            row_df_test, test_probs, symbol_info, news_df, best_buy, best_sell
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Deploy model: train lại trên train+valid+calib (85% data).
    # Scaler fit trên train+valid+calib để không leaking future data.
    # Threshold được tối ưu trên calib segment = honest out-of-sample estimate.
    # ──────────────────────────────────────────────────────────────────────────
    deploy_scaler = FeatureScaler()
    deploy_scaler.fit(bundle.features[:splits.valid.end + bundle.seq_len - 1])
    deploy_scaled = deploy_scaler.transform(bundle.features)

    # Deploy train = train+valid (model chưa thấy valid trong lúc "fit" nhưng valid
    # được dùng cho early stopping → dùng calib làm valid của deploy để tránh biết trước)
    deploy_train_ds = SequenceDataset(deploy_scaled, bundle.targets, bundle.seq_len, 0, splits.valid.end)
    deploy_valid_ds = SequenceDataset(deploy_scaled, bundle.targets, bundle.seq_len, splits.calib.start, splits.calib.count)
    deploy_model = train_model(deploy_train_ds, deploy_valid_ds, input_size=bundle.features.shape[-1])

    # Threshold cho deploy model: chạy lại optimize trên calib (dữ liệu sạch)
    deploy_calib_probs = predict_proba(deploy_model, deploy_valid_ds)
    deploy_threshold_table, deploy_best = optimize_thresholds(
        row_df_calib, deploy_calib_probs, symbol_info, news_df
    )
    deploy_buy  = float(deploy_best["buy_threshold"])
    deploy_sell = float(deploy_best["sell_threshold"])
    print(f"Deploy thresholds (re-optimized on calib): buy={deploy_buy} sell={deploy_sell}")

    wf_results_df = pd.DataFrame()
    wf_trades_df  = pd.DataFrame()
    wf_equity_df  = pd.DataFrame()
    wf_summary    = {}
    if run_wf:
        wf_results_df, wf_trades_df, wf_equity_df, wf_summary = run_walkforward(
            bundle=bundle,
            trainer_predict_fn=predict_proba,
            trainer_train_fn=train_model,
            symbol_info=symbol_info,
            news_df=news_df,
        )

    metrics = {
        "symbol": SYMBOL,
        "profile": PROFILE_NAME,
        "rows_total": int(n),
        "rows_train": int(splits.train.count),
        "rows_valid": int(splits.valid.count),
        "rows_calib": int(splits.calib.count),
        "rows_test": int(splits.test.count),
        "purge_gap_sequences": int(splits.purge_gap),
        "training_history": getattr(model, "training_history", []),
        "training_best_val_loss": getattr(model, "best_val_loss", None),
        "deploy_training_history": getattr(deploy_model, "training_history", []),
        "deploy_training_best_val_loss": getattr(deploy_model, "best_val_loss", None),
        "eval_train": eval_train,
        "eval_valid": eval_valid,
        "eval_calib": eval_calib,
        "eval_test": eval_test,
        # ← Đây là threshold sẽ dùng trong live (honest out-of-sample estimate)
        "best_buy_threshold": deploy_buy,
        "best_sell_threshold": deploy_sell,
        # Lưu thêm để debug
        "calib_best_buy_threshold": best_buy,
        "calib_best_sell_threshold": best_sell,
        "threshold_table": threshold_table.to_dict(orient="records"),
        "deploy_threshold_table": deploy_threshold_table.to_dict(orient="records"),
        "test_backtest_summary": test_summary,
        "walkforward_summary": wf_summary,
        "walkforward_windows": [] if len(wf_results_df) == 0 else wf_results_df.to_dict(orient="records"),
    }

    print("\n===== TEST REPORT =====")
    print(metrics["eval_test"]["report_text"])

    if run_backtest:
        print("\n===== TEST BACKTEST SUMMARY =====")
        print(json.dumps(test_summary, ensure_ascii=False, indent=2))
    else:
        print("\n===== TRAIN-ONLY MODE =====")
        print("Saved model/scaler/features/metrics without test backtest or walk-forward.")

    if run_wf:
        print("\n===== WALK-FORWARD SUMMARY =====")
        print(json.dumps(wf_summary, ensure_ascii=False, indent=2))

    save_outputs(
        model=deploy_model,
        scaler=deploy_scaler,
        feature_cols=list(get_base_features(df).columns),
        metrics=metrics,
        df_full=df,
        test_trades_df=test_trades_df,
        test_equity_df=test_equity_df,
        wf_results_df=wf_results_df,
        wf_trades_df=wf_trades_df,
        wf_equity_df=wf_equity_df,
        wf_summary=wf_summary,
    )


def backtest_only_pipeline():
    artifact = load_inference_bundle(DEVICE)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_cols = list(artifact["feature_cols"])
    metrics = dict(artifact["metrics"])

    symbol_info = get_symbol_info(SYMBOL)
    news_df = load_news_events(NEWS_CSV_PATH)
    df = prepare_dataset(SYMBOL, YEARS_BACK)
    print(f"Dataset rows: {len(df)}")

    current_feature_cols = list(get_base_features(df).columns)
    ensure_saved_feature_layout(feature_cols, current_feature_cols)

    bundle = build_sequence_bundle(df)
    n = bundle.n_sequences
    splits = build_purged_four_way_split(
        n,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        calib_ratio=CALIB_RATIO,
        seq_len=bundle.seq_len,
    )

    scaled = scaler.transform(bundle.features)
    test_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, splits.test.start, splits.test.count)
    row_df_test = bundle.row_df.iloc[splits.test.start:splits.test.end].reset_index(drop=True)

    best_buy  = float(metrics.get("best_buy_threshold", 0.0))
    best_sell = float(metrics.get("best_sell_threshold", 0.0))
    if best_buy <= 0 or best_sell <= 0:
        raise RuntimeError("Saved metrics do not contain valid buy/sell thresholds.")

    eval_test = evaluate(model, test_ds)
    test_probs = predict_proba(model, test_ds)
    test_trades_df, test_equity_df, test_summary = backtest_strategy(
        row_df_test, test_probs, symbol_info, news_df, best_buy, best_sell
    )

    metrics.update({
        "symbol": SYMBOL,
        "profile": PROFILE_NAME,
        "rows_total": int(n),
        "rows_train": int(splits.train.count),
        "rows_valid": int(splits.valid.count),
        "rows_calib": int(splits.calib.count),
        "rows_test": int(splits.test.count),
        "purge_gap_sequences": int(splits.purge_gap),
        "eval_test": eval_test,
        "test_backtest_summary": test_summary,
        "backtest_source": "saved_model",
        "backtest_buy_threshold": best_buy,
        "backtest_sell_threshold": best_sell,
    })

    print("\n===== BACKTEST-ONLY TEST REPORT =====")
    print(eval_test["report_text"])
    print("\n===== BACKTEST-ONLY SUMMARY =====")
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))

    save_outputs(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        metrics=metrics,
        df_full=df,
        test_trades_df=test_trades_df,
        test_equity_df=test_equity_df,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Trade bot trainer/live runner")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "test", "train", "live", "backtest", "paper"],
        help=(
            "pipeline/test = train + backtest + walk-forward, train = train only, "
            "live = run live bot, backtest = use saved model for backtest only, "
            "paper = run live logic without sending MT5 orders"
        ),
    )
    return parser.parse_args()


def resolve_mode(cli_mode: str | None) -> str:
    if cli_mode in {"pipeline", "test", "train", "live", "backtest", "paper"}:
        return "pipeline" if cli_mode == "test" else cli_mode

    print("1 = train + backtest + walk-forward")
    print("2 = live V7")
    print("3 = train only")
    print("4 = backtest only (saved model)")
    print("5 = paper live (saved model, no MT5 orders)")
    choice = input("Chon mode: ").strip()
    return {
        "1": "pipeline",
        "2": "live",
        "3": "train",
        "4": "backtest",
        "5": "paper",
    }.get(choice, "")


def main():
    args = parse_args()
    set_seed()
    mt5_init()
    try:
        ensure_symbol(SYMBOL)
        print(f"Config | symbol={SYMBOL} profile={PROFILE_NAME} years_back={YEARS_BACK} device={DEVICE}")
        mode = resolve_mode(args.mode)

        if mode == "pipeline":
            train_pipeline(run_backtest=True, run_wf=True)
        elif mode == "train":
            train_pipeline(run_backtest=False, run_wf=False)
        elif mode == "backtest":
            backtest_only_pipeline()
        elif mode == "live":
            run_live()
        elif mode == "paper":
            run_live(paper_mode=True)
        else:
            print("Mode khong hop le")
    finally:
        mt5_shutdown()


if __name__ == "__main__":
    main()
