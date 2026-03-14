import argparse
import json

import numpy as np
import pandas as pd

from backtest import backtest_strategy, optimize_thresholds, run_walkforward
from config import DEVICE, NEWS_CSV_PATH, PROFILE_NAME, SYMBOL, TIMEFRAMES, TRAIN_RATIO, VALID_RATIO, YEARS_BACK
from data_mt5 import ensure_symbol, get_rates, get_symbol_info, mt5_init, mt5_shutdown
from features import add_cross_features, add_indicators, get_base_features, merge_timeframes
from filters import load_news_events
from live import run_live
from save_load import load_inference_bundle, save_outputs
from sequence_dataset import SequenceDataset, build_sequence_bundle
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

    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    scaler = FeatureScaler()
    scaler.fit(bundle.features[:train_end + bundle.seq_len - 1])
    scaled = scaler.transform(bundle.features)

    train_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, 0, train_end)
    valid_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, train_end, valid_end - train_end)
    test_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, valid_end, n - valid_end)

    row_df_valid = bundle.row_df.iloc[train_end:valid_end].reset_index(drop=True)
    row_df_test = bundle.row_df.iloc[valid_end:].reset_index(drop=True)

    model = train_model(train_ds, valid_ds, input_size=bundle.features.shape[-1])

    eval_train = evaluate(model, train_ds)
    eval_valid = evaluate(model, valid_ds)
    eval_test = evaluate(model, test_ds)

    valid_probs = predict_proba(model, valid_ds)
    threshold_table, best = optimize_thresholds(row_df_valid, valid_probs, symbol_info, news_df)

    best_buy = float(best["buy_threshold"])
    best_sell = float(best["sell_threshold"])

    test_trades_df = pd.DataFrame()
    test_equity_df = pd.DataFrame()
    test_summary = {}
    if run_backtest:
        test_probs = predict_proba(model, test_ds)
        test_trades_df, test_equity_df, test_summary = backtest_strategy(
            row_df_test, test_probs, symbol_info, news_df, best_buy, best_sell
        )

    deploy_scaler = FeatureScaler()
    deploy_scaler.fit(bundle.features[:valid_end + bundle.seq_len - 1])
    deploy_scaled = deploy_scaler.transform(bundle.features)

    deploy_train_ds = SequenceDataset(deploy_scaled, bundle.targets, bundle.seq_len, 0, valid_end)
    deploy_valid_ds = SequenceDataset(deploy_scaled, bundle.targets, bundle.seq_len, train_end, valid_end - train_end)
    deploy_model = train_model(deploy_train_ds, deploy_valid_ds, input_size=bundle.features.shape[-1])

    wf_results_df = pd.DataFrame()
    wf_trades_df = pd.DataFrame()
    wf_equity_df = pd.DataFrame()
    wf_summary = {}
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
        "rows_train": int(train_end),
        "rows_valid": int(valid_end - train_end),
        "rows_test": int(n - valid_end),
        "training_history": getattr(model, "training_history", []),
        "training_best_val_loss": getattr(model, "best_val_loss", None),
        "deploy_training_history": getattr(deploy_model, "training_history", []),
        "deploy_training_best_val_loss": getattr(deploy_model, "best_val_loss", None),
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
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)
    if valid_end >= n:
        raise RuntimeError("Not enough sequences to build a backtest split.")

    scaled = scaler.transform(bundle.features)
    test_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, valid_end, n - valid_end)
    row_df_test = bundle.row_df.iloc[valid_end:].reset_index(drop=True)

    best_buy = float(metrics.get("best_buy_threshold", 0.0))
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
        "rows_train": int(train_end),
        "rows_valid": int(valid_end - train_end),
        "rows_test": int(n - valid_end),
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
