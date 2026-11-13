from src import config


def keep_frequent(ratings, column, min_count):
    counts = ratings[column].value_counts()
    return ratings[ratings[column].isin(counts[counts >= min_count].index)]


def clean_data(ratings, titles):
    ratings = ratings.drop_duplicates(["user_id", "movie_id"])
    ratings = keep_frequent(ratings, "movie_id", config.MIN_MOVIE_RATINGS)
    ratings = keep_frequent(ratings, "user_id", config.MIN_USER_RATINGS)
    titles = titles[titles["movie_id"].isin(ratings["movie_id"].unique())].copy()
    titles["name"] = titles["title"] + " (" + titles["year"] + ")"
    return ratings.reset_index(drop=True), titles.set_index("movie_id")["name"]
