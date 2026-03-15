from __future__ import annotations

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Offline/Colab trainer and backtester")
    parser.add_argument("--symbol", default="BTCUSD", help="Trading symbol, for example XAUUSD or BTCUSD")
    parser.add_argument("--profile", default="btc_active", help="Profile name, for example base, xau_active, btc_base, btc_active")
    parser.add_argument("--years-back", type=int, default=1, help="How much history to keep from the uploaded CSVs")
    parser.add_argument("--data-dir", default="data/cache/mt5", help="Directory containing offline CSVs")
    parser.add_argument("--news-csv", default="data/inputs/news_events.csv", help="Optional news CSV path")
    parser.add_argument("--symbol-spec", default="", help="Optional JSON file describing point/tick/volume settings")
    parser.add_argument(
        "--mode",
        choices=["train", "pipeline", "backtest"],
        default="pipeline",
        help="train = train only, pipeline = train + backtest + walk-forward, backtest = use saved model",
    )
    return parser.parse_args()


def configure_env(args):
    os.environ["TRADE_BOT_SYMBOL"] = args.symbol.strip().upper()
    os.environ["TRADE_BOT_PROFILE"] = args.profile.strip().lower() or "base"
    os.environ["TRADE_BOT_YEARS_BACK"] = str(max(int(args.years_back), 1))


def main():
    args = parse_args()
    configure_env(args)

    import numpy as np
    import pandas as pd

    from backtest import backtest_strategy, optimize_thresholds, run_walkforward
    from config import DEVICE, PROFILE_NAME, SYMBOL, TRAIN_RATIO, VALID_RATIO, CALIB_RATIO, YEARS_BACK
    from features import add_cross_features, add_indicators, get_base_features, merge_timeframes
    from filters import load_news_events
    from offline_data import build_offline_symbol_info, load_rates_from_csv, load_symbol_spec
    from save_load import load_inference_bundle, save_outputs
    from sequence_dataset import (
        SequenceDataset,
        build_purged_four_way_split,
        build_sequence_bundle,
    )
    from trainer import evaluate, predict_proba, train_model
    from utils import FeatureScaler, set_seed

    def prepare_dataset_offline(symbol: str, years_back: int, data_dir: str) -> pd.DataFrame:
        raw_m1 = load_rates_from_csv(data_dir, symbol, "M1", years_back, verbose=True)
        raw_m5 = load_rates_from_csv(data_dir, symbol, "M5", years_back, verbose=True)
        raw_m15 = load_rates_from_csv(data_dir, symbol, "M15", years_back, verbose=True)
        raw_h1 = load_rates_from_csv(data_dir, symbol, "H1", years_back, verbose=True)

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

    def build_symbol_info(df: pd.DataFrame):
        spec = load_symbol_spec(args.symbol_spec)
        return build_offline_symbol_info(SYMBOL, df["close"], spec=spec)

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
        news_df = load_news_events(args.news_csv)
        df = prepare_dataset_offline(SYMBOL, YEARS_BACK, args.data_dir)
        symbol_info = build_symbol_info(df)

        print(f"Config | symbol={SYMBOL} profile={PROFILE_NAME} years_back={YEARS_BACK} device={DEVICE}")
        print(f"Offline symbol spec | point={symbol_info.point} digits={symbol_info.digits} contract_size={symbol_info.contract_size}")
        print(f"Dataset rows: {len(df)}")
        print(df[["time", "open", "high", "low", "close"]].tail())

        bundle = build_sequence_bundle(df, point_size=symbol_info.point)
        n = bundle.n_sequences
        print(f"Sequences: {n} (features: {bundle.features.shape[1]}, seq_len: {bundle.seq_len})")

        unique, counts = np.unique(bundle.targets, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  class {label}: {count} ({count / n:.1%})")
        
        class0_pct = counts[0] / n if len(counts) > 0 else 0
        if class0_pct > 0.70:
            print(f"[WARNING] Class 0 chiếm {class0_pct:.1%} > 70% — xem xét giảm MIN_RR hoặc tăng HORIZON_BARS")

        # Split dữ liệu: Train / Valid / Calib / Test
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

        scaler = FeatureScaler()
        scaler.fit(bundle.features[:splits.train.end + bundle.seq_len - 1])
        scaled = scaler.transform(bundle.features)

        train_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, splits.train.start, splits.train.count)
        valid_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, splits.valid.start, splits.valid.count)
        calib_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, splits.calib.start, splits.calib.count)
        test_ds = SequenceDataset(scaled, bundle.targets, bundle.seq_len, splits.test.start, splits.test.count)

        row_df_calib = bundle.row_df.iloc[splits.calib.start:splits.calib.end].reset_index(drop=True)
        row_df_test = bundle.row_df.iloc[splits.test.start:splits.test.end].reset_index(drop=True)

        model = train_model(train_ds, valid_ds, input_size=bundle.features.shape[-1])

        eval_train = evaluate(model, train_ds)
        eval_valid = evaluate(model, valid_ds)
        eval_calib = evaluate(model, calib_ds)
        eval_test  = evaluate(model, test_ds)

        _print_overfit_warning(eval_train, eval_valid, eval_test)

        # Optimize threshold trên CALIB set
        calib_probs = predict_proba(model, calib_ds)
        threshold_table, best = optimize_thresholds(row_df_calib, calib_probs, symbol_info, news_df)

        best_buy = float(best["buy_threshold"])
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

        # Deploy model
        deploy_scaler = FeatureScaler()
        deploy_scaler.fit(bundle.features[:splits.valid.end + bundle.seq_len - 1])
        deploy_scaled = deploy_scaler.transform(bundle.features)

        deploy_train_ds = SequenceDataset(deploy_scaled, bundle.targets, bundle.seq_len, 0, splits.valid.end)
        deploy_valid_ds = SequenceDataset(deploy_scaled, bundle.targets, bundle.seq_len, splits.calib.start, splits.calib.count)
        deploy_model = train_model(deploy_train_ds, deploy_valid_ds, input_size=bundle.features.shape[-1])

        deploy_calib_probs = predict_proba(deploy_model, deploy_valid_ds)
        deploy_threshold_table, deploy_best = optimize_thresholds(
            row_df_calib, deploy_calib_probs, symbol_info, news_df
        )
        deploy_buy = float(deploy_best["buy_threshold"])
        deploy_sell = float(deploy_best["sell_threshold"])
        print(f"Deploy thresholds (re-optimized on calib): buy={deploy_buy} sell={deploy_sell}")

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
            "best_buy_threshold": deploy_buy,
            "best_sell_threshold": deploy_sell,
            "calib_best_buy_threshold": best_buy,
            "calib_best_sell_threshold": best_sell,
            "threshold_table": threshold_table.to_dict(orient="records"),
            "deploy_threshold_table": deploy_threshold_table.to_dict(orient="records"),
            "test_backtest_summary": test_summary,
            "walkforward_summary": wf_summary,
            "walkforward_windows": [] if len(wf_results_df) == 0 else wf_results_df.to_dict(orient="records"),
            "data_source": "offline_csv",
            "offline_data_dir": os.path.abspath(args.data_dir),
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

        news_df = load_news_events(args.news_csv)
        df = prepare_dataset_offline(SYMBOL, YEARS_BACK, args.data_dir)
        symbol_info = build_symbol_info(df)
        print(f"Config | symbol={SYMBOL} profile={PROFILE_NAME} years_back={YEARS_BACK} device={DEVICE}")
        print(f"Dataset rows: {len(df)}")

        current_feature_cols = list(get_base_features(df).columns)
        ensure_saved_feature_layout(feature_cols, current_feature_cols)

        bundle = build_sequence_bundle(df, point_size=symbol_info.point)
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
            "rows_train": int(splits.train.count),
            "rows_valid": int(splits.valid.count),
            "rows_calib": int(splits.calib.count),
            "rows_test": int(splits.test.count),
            "purge_gap_sequences": int(splits.purge_gap),
            "eval_test": eval_test,
            "test_backtest_summary": test_summary,
            "backtest_source": "saved_model_offline_csv",
            "backtest_buy_threshold": best_buy,
            "backtest_sell_threshold": best_sell,
            "data_source": "offline_csv",
            "offline_data_dir": os.path.abspath(args.data_dir),
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

    set_seed()
    if args.mode == "train":
        train_pipeline(run_backtest=False, run_wf=False)
    elif args.mode == "backtest":
        backtest_only_pipeline()
    else:
        train_pipeline(run_backtest=True, run_wf=True)


if __name__ == "__main__":
    main()
