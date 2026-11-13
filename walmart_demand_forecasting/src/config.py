from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

KAGGLE_COMPETITION = "m5-forecasting-accuracy"
SALES_FILE = DATA_DIR / "sales_train_evaluation.csv"
CALENDAR_FILE = DATA_DIR / "calendar.csv"
PRICES_FILE = DATA_DIR / "sell_prices.csv"
MODEL_FILE = MODEL_DIR / "model.joblib"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
VALIDATION_FILE = OUTPUT_DIR / "validation_daily.csv"
FORECAST_FILE = OUTPUT_DIR / "forecast_daily.csv"
TOP_ITEMS_FILE = OUTPUT_DIR / "forecast_top_items.csv"

STORE_ID = "CA_1"
CATEGORY_ID = "FOODS"
STATE_ID = STORE_ID.split("_")[0]
HORIZON = 28

LAGS = [28, 35, 42]
ROLLING_WINDOWS = [7, 28]

CATEGORICAL_FEATURES = ["item_id", "dept_id", "event_name_1", "event_type_1"]
NUMERIC_FEATURES = [
    "wday", "month", "snap",
    "sell_price", "price_change", "price_vs_item_mean",
    *[f"lag_{lag}" for lag in LAGS],
    *[f"rolling_mean_{window}" for window in ROLLING_WINDOWS],
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "sales"
RANDOM_STATE = 42
