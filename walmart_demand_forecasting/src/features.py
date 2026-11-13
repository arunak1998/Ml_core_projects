from src import config


def add_lag_features(df):
    sales_by_item = df.groupby("id")["sales"]
    for lag in config.LAGS:
        df[f"lag_{lag}"] = sales_by_item.shift(lag)
    for window in config.ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = sales_by_item.transform(
            lambda sales: sales.shift(config.HORIZON).rolling(window).mean()
        )
    return df


def add_price_features(df):
    price_by_item = df.groupby("id")["sell_price"]
    df["price_change"] = df["sell_price"] / price_by_item.shift(7) - 1
    df["price_vs_item_mean"] = df["sell_price"] / price_by_item.transform("mean")
    return df


def encode_categories(df):
    for column in config.CATEGORICAL_FEATURES:
        df[column] = df[column].astype("category")
    return df


def build_features(df):
    df = add_lag_features(df.copy())
    df = add_price_features(df)
    return encode_categories(df)
