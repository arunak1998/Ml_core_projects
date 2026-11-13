import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src import config


def split_data(df):
    is_test = df["season"].isin(config.TEST_SEASONS)
    train, test = df[~is_test], df[is_test]
    return train[config.FEATURES], test[config.FEATURES], train[config.TARGET], test[config.TARGET]


def build_model():
    encoder = ColumnTransformer(
        [("teams", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    return Pipeline([("encoder", encoder), ("classifier", classifier)])


def train_model(X_train, y_train):
    model = build_model()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    win_probability = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, win_probability > 0.5), 4),
        "roc_auc": round(roc_auc_score(y_test, win_probability), 4),
        "log_loss": round(log_loss(y_test, win_probability), 4),
        "test_balls": len(y_test),
    }


def save_model(model):
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_FILE)


def save_metrics(metrics):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2))
