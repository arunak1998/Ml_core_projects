import shutil
import subprocess
import zipfile

import pandas as pd

from src import config

DATA_FILES = [config.IMPRESSIONS_FILE, config.ADS_FILE, config.USERS_FILE]


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


def read_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df.rename(columns=config.COLUMN_NAMES)


def sample_users(impressions):
    users = impressions["user_id"].drop_duplicates()
    kept = users.sample(frac=config.USER_SAMPLE_FRACTION, random_state=config.RANDOM_STATE)
    return impressions[impressions["user_id"].isin(kept)]


def load_data():
    if not files_ready():
        unzip_downloads()
    if not files_ready():
        download_data()
    impressions = sample_users(read_csv(config.IMPRESSIONS_FILE))
    return impressions, read_csv(config.ADS_FILE), read_csv(config.USERS_FILE)
