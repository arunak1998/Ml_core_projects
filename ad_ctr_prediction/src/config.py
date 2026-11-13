from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

KAGGLE_DATASET = "pavansanagapati/ad-displayclick-data-on-taobaocom"
IMPRESSIONS_FILE = DATA_DIR / "raw_sample.csv"
ADS_FILE = DATA_DIR / "ad_feature.csv"
USERS_FILE = DATA_DIR / "user_profile.csv"
MODEL_FILE = MODEL_DIR / "model.joblib"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"

USER_SAMPLE_FRACTION = 0.2
TEST_DATE = "2017-05-13"
MIN_CATEGORY_COUNT = 50
DEMO_ROWS = 15
RANDOM_STATE = 42

COLUMN_NAMES = {
    "user": "user_id", "userid": "user_id", "adgroup_id": "ad_id", "pid": "ad_slot", "clk": "clicked",
    "cate_id": "category_id", "brand": "brand_id", "customer": "advertiser_id",
    "final_gender_code": "gender", "pvalue_level": "spending_level", "shopping_level": "shopping_depth",
    "occupation": "is_student", "new_user_class_level": "city_level", "cms_group_id": "user_group",
}
GENDER_NAMES = {1: "male", 2: "female"}

CATEGORICAL_FEATURES = ["ad_slot", "gender", "user_group", "category_id", "brand_id"]
NUMERIC_FEATURES = [
    "hour",
    "age_level", "spending_level", "shopping_depth", "is_student", "city_level",
    "price", "price_vs_category",
    "user_past_views", "user_past_clicks", "ad_past_views", "ad_past_clicks",
    "user_category_past_views", "user_category_past_clicks",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "clicked"
