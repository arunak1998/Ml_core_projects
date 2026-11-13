import pandas as pd

from src import config


def clean_matches(matches):
    completed = (
        matches["winner"].notna()
        & matches["method"].isna()
        & (matches["target_overs"].astype(float) == 20)
    )
    matches = matches[completed].copy()
    matches["season"] = matches["date"].str[:4].astype(int)
    matches["target"] = matches["target_runs"].astype(int)
    return matches[["match_id", "season", "winner", "target"]]


def standardize_balls(balls):
    return pd.DataFrame({
        "match_id": balls["match_id"],
        "inning": balls["innings"],
        "batting_team": balls["batting_team"],
        "bowling_team": balls["bowling_team"],
        "total_runs": balls["runs_off_bat"] + balls["extras"],
        "is_wicket": balls["wicket_type"].notna().astype(int),
        "is_legal_ball": balls["wides"].isna() & balls["noballs"].isna(),
    })


def standardize_team_names(df):
    team_columns = ["batting_team", "bowling_team", "winner"]
    df[team_columns] = df[team_columns].replace(config.TEAM_RENAMES)
    active = ~df["batting_team"].isin(config.INACTIVE_TEAMS) & ~df["bowling_team"].isin(config.INACTIVE_TEAMS)
    return df[active]


def clean_data(balls, matches):
    balls = standardize_balls(balls)
    chase = balls[balls["inning"] == 2]
    chase = chase.merge(clean_matches(matches), on="match_id")
    return standardize_team_names(chase)
