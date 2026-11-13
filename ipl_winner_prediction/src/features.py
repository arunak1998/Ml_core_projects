from src import config


def add_chase_features(df):
    df = df.copy()
    df["runs_left"] = df["target"] - df["runs_scored"]
    df["balls_left"] = 120 - df["balls_bowled"]
    df["wickets_left"] = 10 - df["wickets_out"]
    df["current_run_rate"] = df["runs_scored"] * 6 / df["balls_bowled"].clip(lower=1)
    df["required_run_rate"] = df["runs_left"] * 6 / df["balls_left"].clip(lower=1)
    return df


def add_ball_by_ball_state(chase):
    df = chase.copy()
    by_match = df.groupby("match_id")
    df["runs_scored"] = by_match["total_runs"].cumsum()
    df["wickets_out"] = by_match["is_wicket"].cumsum()
    df["balls_bowled"] = by_match["is_legal_ball"].cumsum()
    return df


def build_features(chase):
    df = add_chase_features(add_ball_by_ball_state(chase))
    df[config.TARGET] = (df["winner"] == df["batting_team"]).astype(int)
    in_play = (df["runs_left"] > 0) & (df["balls_left"] > 0) & (df["wickets_left"] > 0)
    return df.loc[in_play, ["season", *config.FEATURES, config.TARGET]]
