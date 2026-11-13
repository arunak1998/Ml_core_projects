import csv
import io
import urllib.request
import zipfile

import pandas as pd

from src import config


def download_data():
    config.DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(config.CRICSHEET_URL, config.DATA_FILE)


def read_match_info(archive, file_name):
    info = {"match_id": int(file_name.split("_")[0])}
    with archive.open(file_name) as file:
        for row in csv.reader(io.TextIOWrapper(file, encoding="utf-8")):
            if row[0] == "info" and row[1] in config.MATCH_INFO_FIELDS:
                info[row[1]] = row[-1]
    return info


def load_data():
    if not config.DATA_FILE.exists():
        download_data()
    with zipfile.ZipFile(config.DATA_FILE) as archive:
        balls = pd.read_csv(archive.open(config.BALLS_FILE), low_memory=False)
        info_files = [name for name in archive.namelist() if name.endswith("_info.csv")]
        matches = pd.DataFrame([read_match_info(archive, name) for name in info_files])
    return balls, matches
