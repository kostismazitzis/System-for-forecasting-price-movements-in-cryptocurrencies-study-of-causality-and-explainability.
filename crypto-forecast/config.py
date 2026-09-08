from pathlib import Path


# ============================================================
# ΒΑΣΙΚΕΣ ΔΙΑΔΡΟΜΕΣ ΤΟΥ PROJECT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ΡΥΘΜΙΣΕΙΣ BITCOIN
# ============================================================

BINANCE_BASE_URL = "https://data-api.binance.vision"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1d"

DEFAULT_LIMIT = 365


# ============================================================
# ΡΥΘΜΙΣΕΙΣ GOOGLE TRENDS
# ============================================================

GOOGLE_TRENDS_KEYWORD = "Bitcoin"

# Κενή τιμή σημαίνει παγκόσμια δεδομένα.
GOOGLE_TRENDS_GEO = ""

GOOGLE_TRENDS_TIMEFRAME = "today 12-m"

GOOGLE_TRENDS_LANGUAGE = "en-US"

GOOGLE_TRENDS_TIMEZONE = 0


# ============================================================
# ΡΥΘΜΙΣΕΙΣ FEATURE ENGINEERING
# ============================================================

PRICE_LAGS = [1, 2, 3, 5, 7]
TREND_LAGS = [1, 2, 3, 5, 7]

MOVING_AVERAGE_WINDOWS = [3, 7, 14]

VOLATILITY_WINDOWS = [7, 14]

TARGET_COLUMN = "target"

RANDOM_STATE = 42


# ============================================================
# ΡΥΘΜΙΣΕΙΣ ΑΞΙΟΛΟΓΗΣΗΣ
# ============================================================

DEFAULT_MIN_TRAIN_SIZE = 60

DEFAULT_TEST_SIZE = 1


# ============================================================
# ΑΡΧΕΙΑ ΕΞΟΔΟΥ
# ============================================================

BITCOIN_DATA_FILE = DATA_DIR / "bitcoin_prices.csv"

TRENDS_DATA_FILE = DATA_DIR / "google_trends.csv"

DATASET_FILE = DATA_DIR / "bitcoin_trends_dataset.csv"

MODEL_FILE = MODEL_DIR / "bitcoin_direction_model.joblib"

METRICS_FILE = DATA_DIR / "evaluation_metrics.json"


# ============================================================
# ΡΥΘΜΙΣΕΙΣ FLASK
# ============================================================

HOST = "127.0.0.1"

PORT = 5000

DEBUG = True

DATASET_FILE = DATA_DIR / "bitcoin_trends_dataset.csv"
MODEL_FILE = MODEL_DIR / "bitcoin_direction_model.joblib"
METRICS_FILE = DATA_DIR / "evaluation_metrics.json"