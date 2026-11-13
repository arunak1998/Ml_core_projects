# 🎬 Netflix Movie Recommender

Recommends movies to each user and finds "because you watched X" lookalikes, learned only from who rated what.

There are no genres or descriptions in the data. The model discovers hidden taste patterns (like action vs drama) from 4.7 million real Netflix ratings using matrix factorization.

## Dataset

[Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) on Kaggle: the data from Netflix's $1M recommendation challenge (ratings from 1999–2005).

| File | What it has |
|---|---|
| `combined_data_1.txt` | ~24M ratings (1–5 stars) from ~470k users, with dates |
| `movie_titles.csv` | Movie id, release year, title |

## Get the data

1. Open the [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) page and sign in to Kaggle.
2. Download either:
    - just `combined_data_1.txt` and `movie_titles.csv` (≈500 MB), or
    - the whole dataset zip (≈700 MB).
3. Put the files or the zip in `netflix_recommender/data/`. A zip is extracted automatically on the first run, and only the two files above are taken from it.

Optional: with a [Kaggle API token](https://www.kaggle.com/settings) set up, the first run downloads the two files by itself.

## Pipeline

| Step | File | What happens |
|---|---|---|
| 1. Load | `src/data_loader.py` | Extract the zip if needed, parse Netflix's `movie_id:` block format into rows |
| 2. Clean | `src/preprocess.py` | Drop duplicates, keep movies with 500+ ratings and users with 50+ ratings |
| 3. Features | `src/features.py` | Encode users and movies as indexes, build a sparse user × movie rating matrix |
| 4. Train | `src/train.py` | Truncated SVD (24 latent factors) learns a taste vector per user and per movie |
| 5. Evaluate | `src/train.py` | Hide each user's latest rating and check if the model ranks it in the top 10 |
| 6. Predict | `src/predict.py` | Top-10 recommendations per user, and similar movies by cosine similarity |

By default the first 10M rows of `combined_data_1.txt` are read to fit in ~2 GB of memory (`MAX_ROWS` in `src/config.py`; set it to `None` for all 24M).

## Model

Truncated SVD from scikit-learn with **24 factors**. A user's score for a movie is the dot product of their two vectors.
Factor counts from 8 to 128 were tested: 16–32 work best, while 128 memorizes users' histories and falls below the popularity baseline.

## Results

4.7M ratings from **54,464 users × 1,018 movies**, evaluated on 2,000 users.
**Hit rate@10**: each user's latest-watched movie is mixed with 99 movies they never rated; a hit means the model ranks it in its top 10.

| Approach | Hit rate@10 |
|---|---|
| Random | 10.0% |
| Most popular movies | 49.3% |
| **SVD model** | **61.9%** |

Example "because you watched":
- *Speed* → Lethal Weapon, Die Hard 2, Beverly Hills Cop
- *American Beauty* → A Beautiful Mind, High Fidelity, Eternal Sunshine of the Spotless Mind

## Run

```bash
cd netflix_recommender
pip install -r requirements.txt
python main.py
```

Runtime: about 20 seconds. Outputs: `models/model.joblib`, `outputs/metrics.json`, `outputs/recommendations.csv`, `outputs/similar_movies.csv`.

## Structure

```
netflix_recommender/
├── data/              Netflix files (downloaded)
├── models/            trained factors
├── outputs/           metrics + recommendations
├── src/
│   ├── config.py      paths, filters, model size
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── main.py            runs the full pipeline
└── requirements.txt
```
