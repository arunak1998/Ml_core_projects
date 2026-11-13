import shutil
import subprocess
import zipfile

import pandas as pd

from src import config

DATA_FILES = [config.SALES_FILE, config.CALENDAR_FILE, config.PRICES_FILE]


def files_ready():
    return all(path.exists() for path in DATA_FILES)


def download_data():
    config.DATA_DIR.mkdir(exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", config.KAGGLE_COMPETITION,
         "-p", str(config.DATA_DIR), "--unzip"],
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


def load_data():
    if not files_ready():
        unzip_downloads()
    if not files_ready():
        download_data()
    sales = pd.read_csv(config.SALES_FILE)
    calendar = pd.read_csv(config.CALENDAR_FILE)
    prices = pd.read_csv(config.PRICES_FILE)
    return sales, calendar, prices
