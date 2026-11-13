import joblib
import pandas as pd

from src import config
from src.features import add_chase_features

SAMPLE_SITUATIONS = pd.DataFrame([
    {"batting_team": "Chennai Super Kings", "bowling_team": "Mumbai Indians", "target": 180, "runs_scored": 135, "overs": 16.0, "wickets_out": 4},
    {"batting_team": "Royal Challengers Bengaluru", "bowling_team": "Kolkata Knight Riders", "target": 200, "runs_scored": 60, "overs": 8.0, "wickets_out": 4},
    {"batting_team": "Mumbai Indians", "bowling_team": "Gujarat Titans", "target": 165, "runs_scored": 150, "overs": 18.3, "wickets_out": 8},
    {"batting_team": "Sunrisers Hyderabad", "bowling_team": "Rajasthan Royals", "target": 190, "runs_scored": 100, "overs": 10.0, "wickets_out": 2},
    {"batting_team": "Delhi Capitals", "bowling_team": "Punjab Kings", "target": 150, "runs_scored": 30, "overs": 6.0, "wickets_out": 3},
])


def overs_to_balls(overs):
    completed_overs = overs.astype(int)
    extra_balls = ((overs - completed_overs) * 10).round().astype(int)
    return completed_overs * 6 + extra_balls


def load_model():
    return joblib.load(config.MODEL_FILE)


def predict_win_probability(model, situations):
    df = situations.copy()
    df["balls_bowled"] = overs_to_balls(df["overs"])
    features = add_chase_features(df)[config.FEATURES]
    win_probability = model.predict_proba(features)[:, 1].astype(float)
    df["batting_win_%"] = (win_probability * 100).round(1)
    df["bowling_win_%"] = (100 - df["batting_win_%"]).round(1)
    return df.drop(columns="balls_bowled")


def save_predictions(predictions):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    predictions.to_csv(config.PREDICTIONS_FILE, index=False)
