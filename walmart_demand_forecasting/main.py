from src import config
from src.data_loader import load_data
from src.features import build_features
from src.predict import daily_totals, load_model, predict_sales, save_predictions, top_items
from src.preprocess import clean_data
from src.train import evaluate_model, save_metrics, save_model, split_data, train_model


def main():
    print("[1/6] Loading Walmart sales, calendar and prices")
    sales, calendar, prices = load_data()

    print(f"[2/6] Cleaning {config.CATEGORY_ID} in store {config.STORE_ID}")
    df = clean_data(sales, calendar, prices)
    print(f"      {df['id'].nunique():,} items, {len(df):,} item-days")

    print("[3/6] Building lag, rolling, price and calendar features")
    df = build_features(df)

    print("[4/6] Training LightGBM (tweedie)")
    train, validation, future = split_data(df)
    model = train_model(train)
    save_model(model)

    print(f"[5/6] Evaluating on the last {config.HORIZON} days")
    metrics = evaluate_model(model, validation)
    save_metrics(metrics)
    print(f"      {metrics}")

    print(f"[6/6] Forecasting the next {config.HORIZON} days")
    model = load_model()
    validation_daily = daily_totals(predict_sales(model, validation))
    future = predict_sales(model, future)
    forecast_daily, forecast_items = daily_totals(future), top_items(future)
    save_predictions(validation_daily, forecast_daily, forecast_items)
    print(forecast_daily.to_string(index=False))
    print(forecast_items.to_string(index=False))


if __name__ == "__main__":
    main()
