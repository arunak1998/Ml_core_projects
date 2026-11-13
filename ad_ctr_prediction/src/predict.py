import joblib

from src import config

DISPLAY_COLUMNS = ["gender", "age_level", "spending_level", "city_level", "price", "ad_slot", "hour", "user_past_clicks"]


def load_model():
    return joblib.load(config.MODEL_FILE)


def predict_click_probability(model, X):
    return model.predict_proba(X)[:, 1]


def demo_predictions(model, X_test, y_test):
    sample = X_test.sample(config.DEMO_ROWS, random_state=config.RANDOM_STATE)
    predictions = sample[DISPLAY_COLUMNS].copy()
    predictions["predicted_ctr_%"] = (predict_click_probability(model, sample) * 100).round(2)
    predictions["clicked"] = y_test.loc[sample.index]
    return predictions.sort_values("predicted_ctr_%", ascending=False)


def save_predictions(predictions):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    predictions.to_csv(config.PREDICTIONS_FILE, index=False)
