import json
import pandas as pd
import numpy as np

from config import (
    SYMBOL, MODEL_PATH, SCALER_PATH, FEATURES_PATH, METRICS_PATH,
    TRAIN_RATIO, VALID_RATIO, TIMEFRAMES, YEARS_BACK, SEQ_LEN, DEVICE,
)
from utils import set_seed, FeatureScaler
from data_mt5 import mt5_init, mt5_shutdown, ensure_symbol, get_symbol_info, get_rates
from features import add_indicators, merge_timeframes, add_cross_features, get_base_features
from filters import load_news_events
from sequence_dataset import build_sequence_bundle, SequenceDataset
from trainer import train_model, predict_proba, evaluate
from backtest import backtest_strategy, optimize_thresholds, run_walkforward
from save_load import save_outputs
from live import run_live


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


def train_pipeline():
    symbol_info = get_symbol_info(SYMBOL)
    news_df = load_news_events("news_events.csv")

    df = prepare_dataset(SYMBOL, YEARS_BACK)
    print(f"Dataset rows: {len(df)}")
    print(df[["time", "open", "high", "low", "close"]].tail())

    bundle = build_sequence_bundle(df)
    n = bundle.n_sequences
    print(f"Sequences: {n} (features: {bundle.features.shape[1]}, seq_len: {bundle.seq_len})")

    # ── Label distribution ──
    unique, counts = np.unique(bundle.targets, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  class {u}: {c} ({c / n:.1%})")

    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    # ── Fit scaler on training features (2D — memory efficient) ──
    scaler = FeatureScaler()
    scaler.fit(bundle.features[:train_end + bundle.seq_len - 1])
    scaled = scaler.transform(bundle.features)

    # ── Create lazy datasets ──
    train_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, 0, train_end)
    valid_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, train_end, valid_end - train_end)
    test_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, valid_end, n - valid_end)

    row_df_valid = bundle.row_df.iloc[train_end:valid_end].reset_index(drop=True)
    row_df_test = bundle.row_df.iloc[valid_end:].reset_index(drop=True)

    # ── Train ──
    model = train_model(train_ds, valid_ds, input_size=bundle.features.shape[-1])

    # ── Evaluate ──
    eval_train = evaluate(model, train_ds)
    eval_valid = evaluate(model, valid_ds)
    eval_test = evaluate(model, test_ds)

    # ── Optimize thresholds on validation ──
    valid_probs = predict_proba(model, valid_ds)
    threshold_table, best = optimize_thresholds(row_df_valid, valid_probs, symbol_info, news_df)

    best_buy = float(best["buy_threshold"])
    best_sell = float(best["sell_threshold"])

    # ── Test backtest ──
    test_probs = predict_proba(model, test_ds)
    test_trades_df, test_equity_df, test_summary = backtest_strategy(
        row_df_test, test_probs, symbol_info, news_df, best_buy, best_sell
    )

    # ── FIX: Refit scaler on train+valid for deploy model ──
    deploy_scaler = FeatureScaler()
    deploy_scaler.fit(bundle.features[:valid_end + bundle.seq_len - 1])
    deploy_scaled = deploy_scaler.transform(bundle.features)

    deploy_train_ds = SequenceDataset(deploy_scaled, bundle.targets, bundle.seq_len, 0, valid_end)
    deploy_valid_ds = SequenceDataset(deploy_scaled, bundle.targets, bundle.seq_len, train_end, valid_end - train_end)
    deploy_model = train_model(deploy_train_ds, deploy_valid_ds, input_size=bundle.features.shape[-1])

    # ── Walk-forward ──
    wf_results_df, wf_trades_df, wf_equity_df, wf_summary = run_walkforward(
        bundle=bundle,
        trainer_predict_fn=predict_proba,
        trainer_train_fn=train_model,
        symbol_info=symbol_info,
        news_df=news_df,
    )

    metrics = {
        "rows_total": int(n),
        "rows_train": int(train_end),
        "rows_valid": int(valid_end - train_end),
        "rows_test": int(n - valid_end),
        "eval_train": eval_train,
        "eval_valid": eval_valid,
        "eval_test": eval_test,
        "best_buy_threshold": best_buy,
        "best_sell_threshold": best_sell,
        "threshold_table": threshold_table.to_dict(orient="records"),
        "test_backtest_summary": test_summary,
        "walkforward_summary": wf_summary,
        "walkforward_windows": [] if len(wf_results_df) == 0 else wf_results_df.to_dict(orient="records"),
    }

    print("\n===== TEST REPORT =====")
    print(metrics["eval_test"]["report_text"])

    print("\n===== TEST BACKTEST SUMMARY =====")
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))

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


def main():
    set_seed()
    mt5_init()
    try:
        ensure_symbol(SYMBOL)

        print("1 = train + backtest + walk-forward")
        print("2 = live V7")
        mode = input("Chọn mode: ").strip()

        if mode == "1":
            train_pipeline()
        elif mode == "2":
            run_live()
        else:
            print("Mode không hợp lệ")
    finally:
        mt5_shutdown()


if __name__ == "__main__":
    main()
