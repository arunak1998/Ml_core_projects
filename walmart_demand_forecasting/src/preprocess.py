import numpy as np
import pandas as pd

from src import config

CALENDAR_COLUMNS = ["d", "date", "wm_yr_wk", "wday", "month", "event_name_1", "event_type_1"]


def filter_store_category(sales):
    keep = (sales["store_id"] == config.STORE_ID) & (sales["cat_id"] == config.CATEGORY_ID)
    return sales[keep]


def add_future_days(sales):
    last_day = int(sales.columns[-1].split("_")[1])
    future_days = [f"d_{day}" for day in range(last_day + 1, last_day + config.HORIZON + 1)]
    future = pd.DataFrame(np.nan, index=sales.index, columns=future_days)
    return pd.concat([sales, future], axis=1)


def to_long_format(sales):
    id_columns = ["id", "item_id", "dept_id", "store_id"]
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    return sales.melt(id_vars=id_columns, value_vars=day_columns, var_name="d", value_name="sales")


def add_calendar_and_prices(df, calendar, prices):
    calendar = calendar[CALENDAR_COLUMNS + [f"snap_{config.STATE_ID}"]]
    calendar = calendar.rename(columns={f"snap_{config.STATE_ID}": "snap"})
    df = df.merge(calendar, on="d").merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    return df[df["sell_price"].notna()]


def clean_data(sales, calendar, prices):
    sales = add_future_days(filter_store_category(sales))
    df = add_calendar_and_prices(to_long_format(sales), calendar, prices)
    df["date"] = pd.to_datetime(df["date"])
    df["day_num"] = df["d"].str[2:].astype(int)
    df[["event_name_1", "event_type_1"]] = df[["event_name_1", "event_type_1"]].fillna("none")
    return df.sort_values(["id", "day_num"]).reset_index(drop=True)
