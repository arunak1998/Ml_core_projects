from src.data_loader import load_data
from src.features import build_features
from src.predict import demo_predictions, load_model, save_predictions
from src.preprocess import clean_data
from src.train import evaluate_model, feature_importance, save_metrics, save_model, split_data, train_model


def main():
    print("[1/6] Loading Taobao impressions, ads and users")
    impressions, ads, users = load_data()
    print(f"      {len(impressions):,} impressions from {impressions['user_id'].nunique():,} users")

    print("[2/6] Cleaning: joining ads + users, readable names, time, rare categories")
    df = clean_data(impressions, ads, users)

    print("[3/6] Building features: user profile, product, context, past activity")
    df = build_features(df)

    print("[4/6] Training LightGBM")
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model)

    print("[5/6] Evaluating on the last day")
    metrics = evaluate_model(model, X_test, y_test, train_ctr=y_train.mean())
    importance = feature_importance(model)
    save_metrics(metrics, importance)
    print(f"      {metrics}")
    print(importance.to_string(index=False))

    print("[6/6] Predicting click probability")
    predictions = demo_predictions(load_model(), X_test, y_test)
    save_predictions(predictions)
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
