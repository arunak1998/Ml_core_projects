# 🏏 IPL Winner Prediction

Predicts the chasing team's chance of winning, ball by ball, during the second innings of an IPL match.

Give it the live situation (teams, target, score, overs, wickets) and it returns a win percentage for each side.

## Dataset

[Cricsheet IPL ball-by-ball data](https://cricsheet.org/downloads/): every IPL match from **2008 to 2026**, free to use.

| File (inside `ipl_csv2.zip`) | What it has |
|---|---|
| `all_matches.csv` | ~296k balls: teams, innings, runs, extras (wides, no-balls), wickets |
| `<match_id>_info.csv` | One per match (~1,240): date, winner, target runs, target overs, D/L method |

## Get the data

Nothing to do. The first run downloads `ipl_csv2.zip` (7 MB) from Cricsheet into `data/` automatically.

Offline option: download [ipl_csv2.zip](https://cricsheet.org/downloads/ipl_csv2.zip) yourself and put it in `ipl_winner_prediction/data/`. Keep it zipped; the code reads it directly.

## Pipeline

| Step | File | What happens |
|---|---|---|
| 1. Load | `src/data_loader.py` | Download the Cricsheet zip if missing, read the balls and every match info file |
| 2. Clean | `src/preprocess.py` | Keep completed 20-over matches (no ties, no-results or D/L), use current team names, drop defunct teams |
| 3. Features | `src/features.py` | Ball-by-ball match state |
| 4. Train | `src/train.py` | XGBoost classifier trained on **2008–2024** |
| 5. Evaluate | `src/train.py` | Test on **IPL 2025 + 2026**, seasons the model has never seen |
| 6. Predict | `src/predict.py` | Win % for sample live match situations |

## Features

| Feature | How it is built |
|---|---|
| `batting_team`, `bowling_team` | From the data, one-hot encoded |
| `target` | Runs needed to win, from the match info |
| `runs_left` | target − runs scored so far |
| `balls_left` | 120 − legal balls bowled (wides and no-balls don't count) |
| `wickets_left` | 10 − wickets fallen |
| `current_run_rate` | runs scored × 6 ÷ balls bowled |
| `required_run_rate` | runs left × 6 ÷ balls left |

## Model

XGBoost classifier: 200 trees, max depth 3, learning rate 0.05, subsample 0.8, colsample 0.8.
The depth was chosen on IPL 2024 as a validation season, so the 2025–2026 test seasons were never used for tuning.

## Results

Tested on **15,776 balls from 138 matches in IPL 2025–2026**.

| Metric | Value |
|---|---|
| Accuracy | **76.8%** (2025: 80.6%, 2026: 73.2%) |
| ROC-AUC | 0.885 |
| Log loss | 0.506 |

In 2026 chasing teams won 63% of matches (a normal season is ~48%), which makes it harder to predict from history.

Example: CSK need 45 off 24 balls with 6 wickets left vs MI → **CSK 55% / MI 45%**.

## Run

```bash
cd ipl_winner_prediction
pip install -r requirements.txt   # skip if you already installed the main requirements.txt
python main.py
```

Runtime: about 40 seconds. Outputs: `models/model.joblib`, `outputs/metrics.json`, `outputs/predictions.csv`.
To try your own match situation, edit `SAMPLE_SITUATIONS` in `src/predict.py`.

## Structure

```
ipl_winner_prediction/
├── data/              Cricsheet zip (downloaded)
├── models/            trained model
├── outputs/           metrics + predictions
├── src/
│   ├── config.py      paths, test seasons, teams, feature lists
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── main.py            runs the full pipeline
└── requirements.txt
```
