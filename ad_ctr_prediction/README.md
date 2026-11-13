# 📢 Ad CTR Prediction

Predicts the probability that a shopper clicks a product ad on Taobao.com, Alibaba's shopping site.

This is the core model behind e-commerce ad ranking: every ad slot goes to the ad with the best **bid × predicted click probability**.

## Dataset

[Ad Display/Click Data on Taobao.com](https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom) on Kaggle, released by Alibaba: 8 days of real ad traffic (6–13 May 2017).

| File | Rows | What it has |
|---|---|---|
| `raw_sample.csv` | 26.6M | One row per ad shown: user, time, ad, ad slot, clicked (5.14% click rate) |
| `ad_feature.csv` | 847k | Each ad's product: category, brand, price, campaign, advertiser |
| `user_profile.csv` | 1.06M | Each shopper: gender, age level, spending level, shopping depth, student, city level |

## Get the data

1. Open the [Taobao dataset page](https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom) and sign in to Kaggle.
2. Download the dataset zip (≈260 MB, it holds the three files above).
3. Put the zip in `ad_ctr_prediction/data/`. It is extracted automatically on the first run.

If the page also offers `behavior_log.csv` (≈20 GB), skip it; it is not used.
Optional: with a [Kaggle API token](https://www.kaggle.com/settings) set up, the first run downloads the three files by itself.

## Pipeline

| Step | File | What happens |
|---|---|---|
| 1. Load | `src/data_loader.py` | Extract the zip if needed, read the 3 files, readable column names, sample 20% of users |
| 2. Clean | `src/preprocess.py` | Join ads + users onto impressions, China-time hour, group categories/brands seen < 50 times into `rare`, missing values → `unknown` |
| 3. Features | `src/features.py` | Past activity of the user, the ad and the user in this category, price vs the category's median price |
| 4. Train | `src/train.py` | LightGBM with native categorical features |
| 5. Evaluate | `src/train.py` | Train on 6–12 May, test on 13 May (unseen day) |
| 6. Predict | `src/predict.py` | Click probability for sample impressions + feature importance |

To keep it fast, the pipeline samples **20% of users** with all their impressions (~5.3M rows, `USER_SAMPLE_FRACTION` in `src/config.py`).

## Features (19)

| Group | Features |
|---|---|
| Shopper | `gender`, `age_level`, `spending_level`, `shopping_depth`, `is_student`, `city_level`, `user_group` |
| Product | `category_id`, `brand_id`, `price`, `price_vs_category` |
| Context | `ad_slot`, `hour` |
| Past behaviour | `user_past_views`, `user_past_clicks`, `ad_past_views`, `ad_past_clicks`, `user_category_past_views`, `user_category_past_clicks` |

Past-behaviour features count **only impressions with an earlier timestamp**. Ads shown together on the same page load share a timestamp, so they never see each other's clicks.

## Model

LightGBM classifier: 300 trees, learning rate 0.05, 31 leaves, min 200 rows per leaf, category smoothing (`cat_smooth` 50, `cat_l2` 10, `min_data_per_group` 200, `max_cat_threshold` 32), `reg_lambda` 5, subsample 0.8, colsample 0.8.
Feature choices were checked on 12 May as a validation day, so the 13 May test day was never used for decisions.

## Results

Tested on **655,786 impressions from 13 May 2017**, a day the model never saw.

| Metric | Value |
|---|---|
| ROC-AUC | **0.650** |
| Log loss | 0.1937 |
| Baseline log loss (always predict the average CTR) | 0.2008 |
| Improvement over baseline | 3.5% |

For reference, a [2025 study on this dataset](https://arxiv.org/html/2511.21963v1) reports AUC ≈ 0.66 for logistic regression and LightGBM using a random split, and 0.687 for a Transformer using full browsing history.
The strongest signals are behaviour and product (`brand_id`, `user_past_views`, `category_id`, `user_past_clicks`). Demographics on their own move the click rate very little (about 4.9% to 5.6% across groups).

## Run

```bash
cd ad_ctr_prediction
pip install -r requirements.txt
python main.py
```

Runtime: about 2.5 minutes, peak memory about 4 GB. Outputs: `models/model.joblib`, `outputs/metrics.json`, `outputs/predictions.csv`, `outputs/feature_importance.csv`.

## Structure

```
ad_ctr_prediction/
├── data/              Taobao files (downloaded)
├── models/            trained model
├── outputs/           metrics, predictions, feature importance
├── src/
│   ├── config.py      paths, sample size, column names, feature lists
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── main.py            runs the full pipeline
└── requirements.txt
```
