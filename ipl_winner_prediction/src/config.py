from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
DATA_FILE = DATA_DIR / "ipl_csv2.zip"
BALLS_FILE = "all_matches.csv"
MATCH_INFO_FIELDS = ["date", "winner", "method", "target_runs", "target_overs"]
MODEL_FILE = MODEL_DIR / "model.joblib"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.csv"

TEST_SEASONS = [2025, 2026]

TEAM_RENAMES = {
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}
INACTIVE_TEAMS = [
    "Gujarat Lions",
    "Kochi Tuskers Kerala",
    "Pune Warriors",
    "Rising Pune Supergiant",
    "Rising Pune Supergiants",
]

CATEGORICAL_FEATURES = ["batting_team", "bowling_team"]
NUMERIC_FEATURES = [
    "target",
    "runs_left",
    "balls_left",
    "wickets_left",
    "current_run_rate",
    "required_run_rate",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "batting_team_won"
