import joblib

from src import config


def load_model():
    return joblib.load(config.MODEL_FILE)


def predict_sales(model, df):
    df = df.copy()
    df["forecast"] = model.predict(df[config.FEATURES]).clip(min=0)
    return df


def daily_totals(df):
    columns = ["sales", "forecast"] if df["sales"].notna().any() else ["forecast"]
    totals = df.groupby("date")[columns].sum().round(0).reset_index()
    return totals.rename(columns={"sales": "actual_units", "forecast": "forecast_units"})


def top_items(df, n=10):
    totals = df.groupby("item_id", observed=True)["forecast"].sum().round(0)
    return totals.nlargest(n).reset_index().rename(columns={"forecast": "forecast_units_next_28_days"})


def save_predictions(validation_daily, forecast_daily, forecast_items):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    validation_daily.to_csv(config.VALIDATION_FILE, index=False)
    forecast_daily.to_csv(config.FORECAST_FILE, index=False)
    forecast_items.to_csv(config.TOP_ITEMS_FILE, index=False)
