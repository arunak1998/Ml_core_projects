import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score

from src import config


def split_data(df):
    is_test = df["date"] == config.TEST_DATE
    train, test = df[~is_test], df[is_test]
    return train[config.FEATURES], test[config.FEATURES], train[config.TARGET], test[config.TARGET]


def train_model(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        min_data_per_group=200,
        cat_smooth=50,
        cat_l2=10,
        max_cat_threshold=32,
        reg_lambda=5,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, train_ctr):
    click_probability = model.predict_proba(X_test)[:, 1]
    model_loss = log_loss(y_test, click_probability)
    baseline_loss = log_loss(y_test, np.full(len(y_test), train_ctr))
    return {
        "roc_auc": round(roc_auc_score(y_test, click_probability), 4),
        "log_loss": round(model_loss, 4),
        "baseline_log_loss": round(baseline_loss, 4),
        "log_loss_improvement_%": round((1 - model_loss / baseline_loss) * 100, 2),
        "actual_ctr_%": round(float(y_test.mean()) * 100, 2),
        "test_impressions": len(y_test),
    }


def feature_importance(model):
    importance = pd.DataFrame({
        "feature": model.feature_name_,
        "importance_gain": model.booster_.feature_importance(importance_type="gain").round(0),
    })
    return importance.sort_values("importance_gain", ascending=False)


def save_model(model):
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_FILE)


def save_metrics(metrics, importance):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    importance.to_csv(config.IMPORTANCE_FILE, index=False)
