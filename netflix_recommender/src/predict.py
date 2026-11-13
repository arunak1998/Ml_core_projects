import joblib
import numpy as np
import pandas as pd

from src import config


def load_model():
    return joblib.load(config.MODEL_FILE)


def recommend_movies(model, user_idx, k=config.TOP_K):
    scores = model["movie_factors"] @ model["user_factors"][user_idx]
    scores[model["seen"][user_idx].indices] = -np.inf
    return np.argsort(-scores)[:k]


def similar_movies(model, movie_idx, k=5):
    factors = model["movie_factors"]
    unit = factors / np.linalg.norm(factors, axis=1, keepdims=True).clip(min=1e-9)
    similarity = unit @ unit[movie_idx]
    similarity[movie_idx] = -np.inf
    return np.argsort(-similarity)[:k]


def favourite_titles(ratings, user_idx, titles, n=3):
    favourites = ratings[ratings["user_idx"] == user_idx].nlargest(n, "rating")
    return ", ".join(titles[favourites["movie_id"]])


def demo_recommendations(model, ratings, titles):
    users = ratings["user_idx"].drop_duplicates().sample(config.DEMO_USERS, random_state=config.RANDOM_STATE)
    rows = []
    for user in users:
        loved = favourite_titles(ratings, user, titles)
        for rank, movie in enumerate(recommend_movies(model, user), start=1):
            rows.append({"user_id": model["user_ids"][user], "loved": loved, "rank": rank,
                         "recommended": titles[model["movie_ids"][movie]]})
    return pd.DataFrame(rows)


def demo_similar_movies(model, ratings, titles):
    popular = ratings["movie_idx"].value_counts().index[:config.DEMO_MOVIES]
    rows = []
    for movie in popular:
        for rank, other in enumerate(similar_movies(model, movie), start=1):
            rows.append({"because_you_watched": titles[model["movie_ids"][movie]], "rank": rank,
                         "you_may_like": titles[model["movie_ids"][other]]})
    return pd.DataFrame(rows)


def save_predictions(recommendations, similar):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    recommendations.to_csv(config.RECOMMENDATIONS_FILE, index=False)
    similar.to_csv(config.SIMILAR_MOVIES_FILE, index=False)
