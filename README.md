# ML Core Projects

Four end-to-end machine learning projects on real-world data, each covering a different core ML problem:

| Project | Problem type | Model | Result |
|---|---|---|---|
| [🏏 IPL Winner Prediction](ipl_winner_prediction/) | Classification (sports) | XGBoost | 76.8% accuracy on unseen IPL 2025–26 |
| [🎬 Netflix Movie Recommender](netflix_recommender/) | Recommendation | Truncated SVD | 61.9% hit rate@10 (popular movies: 49.3%) |
| [🛒 Walmart Demand Forecasting](walmart_demand_forecasting/) | Time-series forecasting | LightGBM (tweedie) | Store-level forecast error 7.6% vs 14.9% baseline |
| [📢 Ad CTR Prediction](ad_ctr_prediction/) | Click probability (ads) | LightGBM | ROC-AUC 0.650 on an unseen day |

Every project follows the same clean pipeline, one file per step:

**load → clean → features → train → evaluate → predict**

```
<project>/
├── data/          raw data (not committed, see "Get the data")
├── models/        trained model (created on run)
├── outputs/       metrics.json + prediction CSVs (created on run)
├── src/
│   ├── config.py        paths and settings
│   ├── data_loader.py   1. load
│   ├── preprocess.py    2. clean
│   ├── features.py      3. features
│   ├── train.py         4–5. train + evaluate
│   └── predict.py       6. predict
├── main.py        runs the whole pipeline
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/arunak1998/Ml_core_projects.git
cd Ml_core_projects
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

The datasets are not stored in this repository (they are large, and their licences don't allow re-sharing). Each project's README has step-by-step instructions; in short:

| Project | Where to get it | What to do |
|---|---|---|
| IPL Winner Prediction | [Cricsheet](https://cricsheet.org/downloads/) | Nothing. Downloaded automatically on the first run |
| Netflix Movie Recommender | [Kaggle: Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) | Put the zip (or `combined_data_1.txt` + `movie_titles.csv`) in `netflix_recommender/data/` |
| Walmart Demand Forecasting | [Kaggle: M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) | Accept the rules, Download All, put the zip in `walmart_demand_forecasting/data/` |
| Ad CTR Prediction | [Kaggle: Taobao ad clicks](https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom) | Put the zip in `ad_ctr_prediction/data/` |

Zips are extracted automatically on the first run; no manual unzipping needed.
Optional: with a [Kaggle API token](https://www.kaggle.com/settings) set up, missing files are downloaded automatically.

## Run

Go into a project folder and run its pipeline:

```bash
cd ipl_winner_prediction
python main.py
```

Each run prints the progress of every step, saves the model to `models/` and writes the metrics and predictions to `outputs/`.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SciPy
