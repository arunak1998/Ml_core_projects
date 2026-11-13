import numpy as np
from scipy.sparse import csr_matrix


def encode_ids(ratings):
    ratings = ratings.copy()
    users = ratings["user_id"].astype("category")
    movies = ratings["movie_id"].astype("category")
    ratings["user_idx"] = users.cat.codes
    ratings["movie_idx"] = movies.cat.codes
    return ratings, users.cat.categories.to_numpy(), movies.cat.categories.to_numpy()


def build_rating_matrix(ratings, n_users, n_movies):
    values = ratings["rating"].to_numpy(dtype=np.float32)
    rows, cols = ratings["user_idx"].to_numpy(), ratings["movie_idx"].to_numpy()
    return csr_matrix((values, (rows, cols)), shape=(n_users, n_movies))
