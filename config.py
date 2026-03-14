import os
import torch

try:
    import MetaTrader5 as mt5
except ImportError:
    class _MT5Stub:
        TIMEFRAME_M1 = "M1"
        TIMEFRAME_M5 = "M5"
        TIMEFRAME_M15 = "M15"
        TIMEFRAME_H1 = "H1"

    mt5 = _MT5Stub()

def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _slugify(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


SYMBOL = (os.getenv("TRADE_BOT_SYMBOL") or "XAUUSD").strip().upper()
YEARS_BACK = _env_int("TRADE_BOT_YEARS_BACK", 1)
ARTIFACT_VERSION = (os.getenv("TRADE_BOT_ARTIFACT_VERSION") or "v7").strip() or "v7"
PROFILE_NAME = (os.getenv("TRADE_BOT_PROFILE") or "base").strip().lower() or "base"
SYMBOL_SLUG = _slugify(SYMBOL)
PROFILE_SLUG = _slugify(PROFILE_NAME)
ARTIFACT_PREFIX = f"{SYMBOL_SLUG}_{ARTIFACT_VERSION}" + (f"_{PROFILE_SLUG}" if PROFILE_NAME != "base" else "")
IS_XAU_ACTIVE_PROFILE = PROFILE_NAME in {"xau_active", "active_xau", "gold_active"} and ("XAU" in SYMBOL)
IS_BTC_BASE_PROFILE = PROFILE_NAME in {"btc", "btc_base", "crypto_base"} and ("BTC" in SYMBOL)

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}

# ── Model / Training ──
SEQ_LEN = 64
BATCH_SIZE = 256
MAX_EPOCHS = 60
LR = 5e-4
MAX_LR = 1.5e-3

CNN_CHANNELS = 192
LSTM_HIDDEN = 192
LSTM_LAYERS = 2
DROPOUT = 0.30
DROP_PATH_RATE = 0.1

TRANSFORMER_DIM = 192
TRANSFORMER_HEADS = 6
TRANSFORMER_LAYERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

EARLY_STOPPING_PATIENCE = 12
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
USE_AMP = torch.cuda.is_available()
USE_EMA_WEIGHTS = True
EMA_DECAY = 0.997

# Scheduler
USE_ONECYCLE = False
USE_COSINE_ANNEALING = True
WARMUP_EPOCHS = 3

# Gradient accumulation
GRAD_ACCUM_STEPS = 2

# Mixup augmentation
USE_MIXUP = True
MIXUP_ALPHA = 0.2

# Multi-seed ensemble
USE_ENSEMBLE = False
ENSEMBLE_SEEDS = [42, 123, 456]

# ── Labeling ──
HORIZON_BARS = 30
SL_ATR_MULT = 1.4
MIN_RR = 1.3
USE_ADAPTIVE_RR = True

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15

# ── Walk-forward ──
WF_TRAIN_BARS = 60000
WF_TEST_BARS = 10000
WF_STEP_BARS = 10000

# ── Session / News ──
SESSION_FILTER = True
SESSION_WINDOWS_UTC = [
    ("07:00", "16:30"),
    ("12:00", "21:00"),
]

MAX_SPREAD_POINTS = 280

USE_NEWS_FILTER = True
BASE_DATA_DIR = "data"
INPUT_DATA_DIR = os.path.join(BASE_DATA_DIR, "inputs")
CACHE_DATA_DIR = os.path.join(BASE_DATA_DIR, "cache")

ARTIFACTS_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")
DATASET_DIR = os.path.join(ARTIFACTS_DIR, "datasets")
BACKTEST_DIR = os.path.join(ARTIFACTS_DIR, "backtest")
WALKFORWARD_DIR = os.path.join(ARTIFACTS_DIR, "walkforward")
LOG_DIR = os.path.join(ARTIFACTS_DIR, "logs")

NEWS_CSV_PATH = os.path.join(INPUT_DATA_DIR, "news_events.csv")
NEWS_BLOCK_BEFORE_MIN = 20
NEWS_BLOCK_AFTER_MIN = 20
NEWS_IMPACT_ALLOW = {"high"}

# ── Backtest ──
BACKTEST_ENTRY_SLIPPAGE_POINTS = 25
BACKTEST_EXIT_SLIPPAGE_POINTS = 25

SIGNAL_SMOOTH_BARS = 3
COOLDOWN_BARS = 10
REJECT_COOLDOWN_BARS = 2

BUY_THRESHOLD_GRID = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]
SELL_THRESHOLD_GRID = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]

LIVE_BUY_THRESHOLD_OFFSET = 0.02
LIVE_SELL_THRESHOLD_OFFSET = 0.02

MIN_ATR_PCT = 0.00014
MIN_M15_TREND_STRENGTH = 0.00004
MIN_H1_TREND_STRENGTH = 0.00004

if IS_XAU_ACTIVE_PROFILE:
    # Slightly more active XAU variant: easier labels and gentler live filters.
    HORIZON_BARS = 24
    SL_ATR_MULT = 1.30
    MIN_RR = 1.20
    SIGNAL_SMOOTH_BARS = 2
    COOLDOWN_BARS = 6
    REJECT_COOLDOWN_BARS = 1
    BUY_THRESHOLD_GRID = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    SELL_THRESHOLD_GRID = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    LIVE_BUY_THRESHOLD_OFFSET = 0.01
    LIVE_SELL_THRESHOLD_OFFSET = 0.01
    MIN_ATR_PCT = 0.00012
    MIN_M15_TREND_STRENGTH = 0.00003
    MIN_H1_TREND_STRENGTH = 0.00003

if IS_BTC_BASE_PROFILE:
    # BTC trades continuously, so the XAU session/news gates are usually too restrictive.
    SESSION_FILTER = False
    USE_NEWS_FILTER = False

RR_MAP = [
    (0.82, 2.00),
    (0.76, 1.80),
    (0.70, 1.60),
    (0.64, 1.45),
    (0.00, 1.30),
]

# ── Asset-specific ──
ROUND_LEVELS = [10, 50, 100]
if IS_BTC_BASE_PROFILE:
    ROUND_LEVELS = [100, 500, 1000]

# Backward-compatible alias used by older code/comments.
GOLD_ROUND_LEVELS = ROUND_LEVELS

# ── Account / Risk ──
INITIAL_BALANCE = 10000.0
RISK_PER_TRADE = 0.002
MAX_LOT = 0.10
MAGIC = 26032026
DEVIATION = 20

USE_BREAK_EVEN = True
BE_TRIGGER_R = 1.0

USE_PARTIAL_TP = True
PARTIAL_TP_R = 1.0
PARTIAL_CLOSE_RATIO = 0.50

USE_TRAILING_STOP = True
TRAILING_ATR_MULT = 1.2

# ── Paths ──
MODEL_PATH = os.path.join(MODEL_DIR, f"{ARTIFACT_PREFIX}_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, f"{ARTIFACT_PREFIX}_scaler.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, f"{ARTIFACT_PREFIX}_features.joblib")
METRICS_PATH = os.path.join(REPORTS_DIR, f"{ARTIFACT_PREFIX}_metrics.json")
USE_LOCAL_DATA_CACHE = True
DATA_CACHE_DIR = os.path.join(CACHE_DATA_DIR, "mt5")

DATASET_PATH = os.path.join(DATASET_DIR, f"{ARTIFACT_PREFIX}_dataset.csv")
BT_TRADES_PATH = os.path.join(BACKTEST_DIR, f"{ARTIFACT_PREFIX}_backtest_trades.csv")
BT_SUMMARY_PATH = os.path.join(BACKTEST_DIR, f"{ARTIFACT_PREFIX}_backtest_summary.json")
BT_EQUITY_PATH = os.path.join(BACKTEST_DIR, f"{ARTIFACT_PREFIX}_backtest_equity.csv")
WF_PATH = os.path.join(WALKFORWARD_DIR, f"{ARTIFACT_PREFIX}_walkforward.json")
WF_TRADES_PATH = os.path.join(WALKFORWARD_DIR, f"{ARTIFACT_PREFIX}_walkforward_trades.csv")
WF_EQUITY_PATH = os.path.join(WALKFORWARD_DIR, f"{ARTIFACT_PREFIX}_walkforward_equity.csv")
LIVE_LOG_PATH = os.path.join(LOG_DIR, f"{ARTIFACT_PREFIX}_live_log.csv")
PAPER_LIVE_LOG_PATH = os.path.join(LOG_DIR, f"{ARTIFACT_PREFIX}_paper_live_log.csv")

# ── Live safety ──
MAX_DAILY_LOSS_PCT = 0.03
MAX_OPEN_POSITIONS = 1
MT5_RECONNECT_WAIT_SEC = 10
HEARTBEAT_INTERVAL_SEC = 300
MAX_STALE_MINUTES = 5
MAX_LATENCY_MS = 500
MAX_POSITION_HOURS = 4
EQUITY_DRAWDOWN_STOP_PCT = 0.05
