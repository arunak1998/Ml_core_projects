import pandas as pd

from src import config


def add_time_columns(df):
    china_time = pd.to_datetime(df["time_stamp"], unit="s") + pd.Timedelta(hours=8)
    df["time"] = china_time
    df["date"] = china_time.dt.strftime("%Y-%m-%d")
    df["hour"] = china_time.dt.hour
    return df


def clean_users(users):
    users = users.drop_duplicates("user_id").copy()
    users["gender"] = users["gender"].map(config.GENDER_NAMES)
    columns = ["user_id", "gender", "age_level", "spending_level", "shopping_depth", "is_student", "city_level", "user_group"]
    return users[columns]


def clean_ads(ads):
    ads = ads.drop_duplicates("ad_id")
    return ads[["ad_id", "category_id", "brand_id", "price"]]


def id_to_label(values):
    return values.astype("Int64").astype("string").fillna("unknown")


def group_rare_categories(df):
    for column in ["category_id", "brand_id"]:
        values = id_to_label(df[column])
        counts = values.map(values.value_counts())
        df[column] = values.where(counts >= config.MIN_CATEGORY_COUNT, "rare")
    return df


def clean_data(impressions, ads, users):
    df = impressions.drop(columns="nonclk").drop_duplicates()
    df = df.merge(clean_ads(ads), on="ad_id", how="left")
    df = df.merge(clean_users(users), on="user_id", how="left")
    df = add_time_columns(df)
    df = group_rare_categories(df)
    df["gender"] = df["gender"].fillna("unknown")
    df["user_group"] = id_to_label(df["user_group"])
    return df.sort_values("time").reset_index(drop=True)
