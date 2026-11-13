from src import config


def add_past_activity(df, keys, prefix):
    keys = [keys] if isinstance(keys, str) else keys
    by_key = df.groupby(keys)
    same_moment = df.groupby([*keys, "time_stamp"])
    df[f"{prefix}_past_views"] = by_key.cumcount() - same_moment.cumcount()
    df[f"{prefix}_past_clicks"] = by_key[config.TARGET].cumsum() - same_moment[config.TARGET].cumsum()
    return df


def add_price_features(df):
    category_median = df.groupby("category_id")["price"].transform("median")
    df["price_vs_category"] = df["price"] / category_median
    return df


def encode_categories(df):
    for column in config.CATEGORICAL_FEATURES:
        df[column] = df[column].astype("category")
    return df


def build_features(df):
    df = add_past_activity(df, "user_id", "user")
    df = add_past_activity(df, "ad_id", "ad")
    df = add_past_activity(df, ["user_id", "category_id"], "user_category")
    df = add_price_features(df)
    df = encode_categories(df)
    return df[["date", *config.FEATURES, config.TARGET]]
