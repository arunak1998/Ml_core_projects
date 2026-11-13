import json

import joblib
import numpy as np
from lightgbm import LGBMRegressor

from src import config


def split_data(df):
    known = df[df[config.TARGET].notna()]
    validation_start = known["day_num"].max() - config.HORIZON + 1
    train = known[(known["day_num"] < validation_start) & known["rolling_mean_28"].notna()]
    validation = known[known["day_num"] >= validation_start]
    future = df[df[config.TARGET].isna()]
    return train, validation, future


def train_model(train):
    model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.1,
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )
    model.fit(train[config.FEATURES], train[config.TARGET])
    return model


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(actual, predicted):
    return float(np.abs(actual - predicted).sum() / actual.sum() * 100)


def store_daily_wape(validation, predicted):
    daily = validation.assign(predicted=predicted).groupby("date")[[config.TARGET, "predicted"]].sum()
    return wape(daily[config.TARGET], daily["predicted"])


def evaluate_model(model, validation):
    actual = validation[config.TARGET]
    predicted = model.predict(validation[config.FEATURES]).clip(min=0)
    baseline = validation["rolling_mean_28"]
    return {
        "rmse": round(rmse(actual, predicted), 4),
        "baseline_rmse": round(rmse(actual, baseline), 4),
        "item_wape_%": round(wape(actual, predicted), 2),
        "baseline_item_wape_%": round(wape(actual, baseline), 2),
        "store_daily_wape_%": round(store_daily_wape(validation, predicted), 2),
        "baseline_store_daily_wape_%": round(store_daily_wape(validation, baseline.to_numpy()), 2),
        "items": validation["id"].nunique(),
        "validation_days": config.HORIZON,
    }


def save_model(model):
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_FILE)


def save_metrics(metrics):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2))
