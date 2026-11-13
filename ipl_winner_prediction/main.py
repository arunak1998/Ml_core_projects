from src.data_loader import load_data
from src.features import build_features
from src.predict import SAMPLE_SITUATIONS, load_model, predict_win_probability, save_predictions
from src.preprocess import clean_data
from src.train import evaluate_model, save_metrics, save_model, split_data, train_model


def main():
    print("[1/6] Loading data")
    deliveries, matches = load_data()

    print("[2/6] Cleaning and standardizing")
    chase = clean_data(deliveries, matches)

    print("[3/6] Building features")
    df = build_features(chase)
    print(f"      {len(df):,} ball-by-ball chase situations")

    print("[4/6] Training XGBoost")
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model)

    print("[5/6] Evaluating on unseen seasons")
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics)
    print(f"      {metrics}")

    print("[6/6] Predicting live match situations")
    predictions = predict_win_probability(load_model(), SAMPLE_SITUATIONS)
    save_predictions(predictions)
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
