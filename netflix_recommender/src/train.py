import json

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

from src import config
from src.features import build_rating_matrix


def split_data(ratings):
    latest = ratings.sort_values("date").groupby("user_idx").tail(1)
    return ratings.drop(latest.index), latest


def train_model(train_ratings, user_ids, movie_ids):
    matrix = build_rating_matrix(train_ratings, len(user_ids), len(movie_ids))
    svd = TruncatedSVD(n_components=config.N_FACTORS, random_state=config.RANDOM_STATE)
    return {
        "user_factors": svd.fit_transform(matrix),
        "movie_factors": svd.components_.T,
        "seen": matrix,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
    }


def is_hit(scores):
    rank_of_true_movie = (scores[:-1] > scores[-1]).sum()
    return rank_of_true_movie < config.TOP_K


def sample_candidates(model, user, true_movie, rng):
    n_movies = len(model["movie_ids"])
    unseen = np.setdiff1d(np.arange(n_movies), [*model["seen"][user].indices, true_movie])
    negatives = rng.choice(unseen, config.EVAL_NEGATIVES, replace=False)
    return np.append(negatives, true_movie)


def evaluate_model(model, test_ratings):
    rng = np.random.default_rng(config.RANDOM_STATE)
    test = test_ratings.sample(min(config.EVAL_USERS, len(test_ratings)), random_state=config.RANDOM_STATE)
    popularity = np.asarray((model["seen"] > 0).sum(axis=0)).ravel()
    model_hits, popular_hits = 0, 0
    for user, movie in zip(test["user_idx"], test["movie_idx"]):
        candidates = sample_candidates(model, user, movie, rng)
        model_hits += int(is_hit(model["movie_factors"][candidates] @ model["user_factors"][user]))
        popular_hits += int(is_hit(popularity[candidates]))
    return {
        "hit_rate@10": round(model_hits / len(test), 4),
        "popularity_hit_rate@10": round(popular_hits / len(test), 4),
        "random_hit_rate@10": round(config.TOP_K / (config.EVAL_NEGATIVES + 1), 4),
        "users": len(model["user_ids"]),
        "movies": len(model["movie_ids"]),
    }


def save_model(model):
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_FILE)


def save_metrics(metrics):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2))
