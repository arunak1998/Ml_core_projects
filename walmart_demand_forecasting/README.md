# 🛒 Walmart Demand Forecasting

Forecasts daily unit sales for every food item in a Walmart store for the next 28 days.

Built on the M5 forecasting competition data: real Walmart sales with prices, holidays and food-stamp (SNAP) days.

## Dataset

[M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) on Kaggle: daily Walmart sales from 10 stores in California, Texas and Wisconsin (2011–2016).

| File | What it has |
|---|---|
| `sales_train_evaluation.csv` | Daily unit sales for 30,490 item-store series (`d_1` … `d_1941`) |
| `calendar.csv` | Date, weekday, month, events (Super Bowl, Easter…), SNAP days |
| `sell_prices.csv` | Weekly price of each item in each store |

## Get the data

1. Open the [competition data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and sign in to Kaggle.
2. Accept the competition rules (one click, required before downloading).
3. Click **Download All** to get `m5-forecasting-accuracy.zip` (≈48 MB).
4. Put the zip in `walmart_demand_forecasting/data/`. It is extracted automatically on the first run, and only the three files above are used.

Optional: with a [Kaggle API token](https://www.kaggle.com/settings) set up and the rules accepted, the first run downloads the data by itself.

## Pipeline

| Step | File | What happens |
|---|---|---|
| 1. Load | `src/data_loader.py` | Extract the zip if needed, read sales, calendar, prices |
| 2. Clean | `src/preprocess.py` | Keep FOODS in store CA_1 (1,437 items), add 28 future days, reshape wide → long, join calendar and prices, drop days before an item was on sale |
| 3. Features | `src/features.py` | Sales lags, rolling means, price and calendar features |
| 4. Train | `src/train.py` | LightGBM with a tweedie objective (suits the many zero-sales days) |
| 5. Evaluate | `src/train.py` | Hold out the last 28 known days and compare with a moving-average baseline |
| 6. Predict | `src/predict.py` | Forecast the next 28 days: daily store total + top-10 items |

Change `STORE_ID` / `CATEGORY_ID` in `src/config.py` to forecast another store or category.

## Features (15)

| Group | Features |
|---|---|
| Item | `item_id`, `dept_id` |
| Calendar | `wday`, `month`, `event_name_1`, `event_type_1`, `snap` |
| Price | `sell_price`, `price_change` (vs last week), `price_vs_item_mean` |
| Past sales | `lag_28`, `lag_35`, `lag_42`, `rolling_mean_7`, `rolling_mean_28` |

All past-sales features look back at least 28 days, so the whole 28-day horizon is forecast in one shot with no recursive loop.

## Model

LightGBM regressor: tweedie objective (variance power 1.1), 800 trees, learning rate 0.05, 127 leaves, min 100 rows per leaf, subsample 0.8, colsample 0.8.

## Results

**1,437 FOODS items in store CA_1**, validated on the last 28 days (25 Apr – 22 May 2016).

| Metric | Model | Baseline (28-day average) |
|---|---|---|
| RMSE (units per item-day) | **2.74** | 3.00 |
| Item WAPE | 65.9% | 70.0% |
| **Store daily WAPE** | **7.6%** | 14.9% |

The model **halves the store-level forecast error**. Item-level error stays high because most items sell only 0–2 units a day.

## Run

```bash
cd walmart_demand_forecasting
pip install -r requirements.txt   # skip if you already installed the main requirements.txt
python main.py
```

Runtime: about 100 seconds. Outputs: `models/model.joblib`, `outputs/metrics.json`, `outputs/validation_daily.csv`, `outputs/forecast_daily.csv`, `outputs/forecast_top_items.csv`.

## Structure

```
walmart_demand_forecasting/
├── data/              M5 files (downloaded)
├── models/            trained model
├── outputs/           metrics + forecasts
├── src/
│   ├── config.py      paths, store, horizon, feature lists
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── main.py            runs the full pipeline
└── requirements.txt
```
