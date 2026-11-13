from src.data_loader import load_data
from src.features import encode_ids
from src.predict import demo_recommendations, demo_similar_movies, load_model, save_predictions
from src.preprocess import clean_data
from src.train import evaluate_model, save_metrics, save_model, split_data, train_model


def main():
    print("[1/6] Loading data")
    ratings, titles = load_data()

    print("[2/6] Cleaning and filtering active users / popular movies")
    ratings, titles = clean_data(ratings, titles)
    print(f"      {len(ratings):,} ratings")

    print("[3/6] Building user-movie features")
    ratings, user_ids, movie_ids = encode_ids(ratings)
    print(f"      {len(user_ids):,} users x {len(movie_ids):,} movies")

    print("[4/6] Training matrix factorization (SVD)")
    train_ratings, test_ratings = split_data(ratings)
    model = train_model(train_ratings, user_ids, movie_ids)
    save_model(model)

    print("[5/6] Evaluating on each user's latest watch")
    metrics = evaluate_model(model, test_ratings)
    save_metrics(metrics)
    print(f"      {metrics}")

    print("[6/6] Recommending movies")
    model = load_model()
    recommendations = demo_recommendations(model, train_ratings, titles)
    similar = demo_similar_movies(model, train_ratings, titles)
    save_predictions(recommendations, similar)
    print(recommendations.to_string(index=False))
    print(similar.to_string(index=False))


if __name__ == "__main__":
    main()
