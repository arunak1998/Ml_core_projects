from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

KAGGLE_DATASET = "netflix-inc/netflix-prize-data"
RATINGS_FILE = DATA_DIR / "combined_data_1.txt"
TITLES_FILE = DATA_DIR / "movie_titles.csv"
MODEL_FILE = MODEL_DIR / "model.joblib"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
RECOMMENDATIONS_FILE = OUTPUT_DIR / "recommendations.csv"
SIMILAR_MOVIES_FILE = OUTPUT_DIR / "similar_movies.csv"

MAX_ROWS = 10_000_000
MIN_MOVIE_RATINGS = 500
MIN_USER_RATINGS = 50

N_FACTORS = 24
TOP_K = 10
EVAL_USERS = 2000
EVAL_NEGATIVES = 99
DEMO_USERS = 3
DEMO_MOVIES = 3
RANDOM_STATE = 42
