import os
import json
import joblib
import torch
import pandas as pd

from config import (
    MODEL_PATH, SCALER_PATH, FEATURES_PATH, METRICS_PATH,
    DATASET_PATH, BT_TRADES_PATH, BT_SUMMARY_PATH, BT_EQUITY_PATH,
    WF_PATH, WF_TRADES_PATH, WF_EQUITY_PATH, SEQ_LEN,
)
from features import get_base_features
from model import CNNBiLSTMTransformer, infer_model_arch_from_state_dict
from utils import ensure_parent_dir


def save_outputs(model, scaler, feature_cols: list, metrics: dict, df_full: pd.DataFrame,
                 test_trades_df: pd.DataFrame = None,
                 test_equity_df: pd.DataFrame = None,
                 wf_results_df: pd.DataFrame = None,
                 wf_trades_df: pd.DataFrame = None,
                 wf_equity_df: pd.DataFrame = None,
                 wf_summary: dict = None):
    for path in [
        MODEL_PATH, SCALER_PATH, FEATURES_PATH, METRICS_PATH,
        DATASET_PATH, BT_TRADES_PATH, BT_SUMMARY_PATH, BT_EQUITY_PATH,
        WF_PATH, WF_TRADES_PATH, WF_EQUITY_PATH,
    ]:
        ensure_parent_dir(path)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(feature_cols),
            "model_arch": getattr(model, "arch_config", None),
        },
        MODEL_PATH,
    )

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save dataset (subset of rows for inspection)
    feature_df = get_base_features(df_full).dropna().reset_index(drop=True)
    base_cols = ["time", "open", "high", "low", "close"]
    base_cols = [c for c in base_cols if c in df_full.columns]
    base_df = df_full[base_cols].iloc[-len(feature_df):].reset_index(drop=True)
    save_df = pd.concat([base_df, feature_df], axis=1)
    save_df.to_csv(DATASET_PATH, index=False)

    if test_trades_df is not None:
        test_trades_df.to_csv(BT_TRADES_PATH, index=False)
    if test_equity_df is not None:
        test_equity_df.to_csv(BT_EQUITY_PATH, index=False)

    if metrics.get("test_backtest_summary") is not None:
        with open(BT_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics["test_backtest_summary"], f, ensure_ascii=False, indent=2)

    if wf_results_df is not None or wf_trades_df is not None or wf_summary is not None:
        payload = {
            "walkforward_windows": [] if wf_results_df is None else wf_results_df.to_dict(orient="records"),
            "walkforward_summary": {} if wf_summary is None else wf_summary,
        }
        with open(WF_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if wf_trades_df is not None and len(wf_trades_df) > 0:
            wf_trades_df.to_csv(WF_TRADES_PATH, index=False)
        if wf_equity_df is not None and len(wf_equity_df) > 0:
            wf_equity_df.to_csv(WF_EQUITY_PATH, index=False)


def load_inference_bundle(device: str):
    if not (all([
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH,
        METRICS_PATH,
    ])):
        raise RuntimeError("Artifact paths are not configured.")

    missing = [path for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH, METRICS_PATH] if not os.path.exists(path)]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Missing model artifacts. Run the training pipeline first (e.g. `python main.py --mode pipeline`). "
            f"Missing files: {missing_str}"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    state_dict = checkpoint["model_state_dict"]
    model_arch = checkpoint.get("model_arch")
    if model_arch is None:
        model_arch = infer_model_arch_from_state_dict(
            state_dict,
            input_size=checkpoint.get("input_size", len(feature_cols)),
        )

    checkpoint_input_size = int(model_arch.get("input_size", checkpoint.get("input_size", len(feature_cols))))
    if checkpoint_input_size != len(feature_cols):
        raise RuntimeError(
            f"Feature count mismatch: checkpoint expects {checkpoint_input_size}, "
            f"but features file has {len(feature_cols)} columns."
        )

    model = CNNBiLSTMTransformer(**model_arch).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "checkpoint": checkpoint,
        "model_arch": model_arch,
    }
