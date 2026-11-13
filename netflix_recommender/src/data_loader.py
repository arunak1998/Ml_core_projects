import shutil
import subprocess
import zipfile

import pandas as pd

from src import config

DATA_FILES = [config.RATINGS_FILE, config.TITLES_FILE]


def files_ready():
    return all(path.exists() for path in DATA_FILES)


def download_data():
    config.DATA_DIR.mkdir(exist_ok=True)
    for path in DATA_FILES:
        subprocess.run(
            ["kaggle", "datasets", "download", config.KAGGLE_DATASET,
             "-f", path.name, "-p", str(config.DATA_DIR), "--unzip"],
            check=True,
        )


def unzip_downloads():
    wanted = {path.name for path in DATA_FILES}
    for archive in config.DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                file_name = name.split("/")[-1]
                if file_name in wanted:
                    with zf.open(name) as source, open(config.DATA_DIR / file_name, "wb") as target:
                        shutil.copyfileobj(source, target)


def read_ratings(path, max_rows):
    raw = pd.read_csv(path, header=None, names=["user_id", "rating", "date"], nrows=max_rows)
    is_movie_row = raw["rating"].isna()
    raw["movie_id"] = raw["user_id"].where(is_movie_row).str.rstrip(":").ffill().astype(int)
    ratings = raw[~is_movie_row].astype({"user_id": int, "rating": int})
    ratings["date"] = pd.to_datetime(ratings["date"])
    return ratings[["user_id", "movie_id", "rating", "date"]]


def read_titles(path):
    with open(path, encoding="latin-1") as file:
        rows = [line.rstrip("\n").split(",", 2) for line in file]
    titles = pd.DataFrame(rows, columns=["movie_id", "year", "title"])
    titles["movie_id"] = titles["movie_id"].astype(int)
    return titles


def load_data():
    if not files_ready():
        unzip_downloads()
    if not files_ready():
        download_data()
    return read_ratings(config.RATINGS_FILE, config.MAX_ROWS), read_titles(config.TITLES_FILE)
