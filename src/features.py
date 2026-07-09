"""
Feature engineering on top of the Elo history: Elo levels/diff, neutral venue,
competition-importance bucket, and short-term form (rolling goal difference
over each team's last 10 matches, computed causally so no future leakage).
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from elo import k_factor

FORM_WINDOW = 10


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    history = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    n = len(df)
    home_form = np.full(n, np.nan)
    away_form = np.full(n, np.nan)

    for i, row in enumerate(df.itertuples(index=False)):
        home, away = row.home_team, row.away_team
        h_hist, a_hist = history[home], history[away]
        if h_hist:
            home_form[i] = np.mean(h_hist)
        if a_hist:
            away_form[i] = np.mean(a_hist)

        if pd.notna(row.home_score) and pd.notna(row.away_score):
            gd = row.home_score - row.away_score
            h_hist.append(gd)
            a_hist.append(-gd)

    out = df.copy()
    out["home_form"] = home_form
    out["away_form"] = away_form
    return out


def build_feature_frame(elo_df: pd.DataFrame) -> pd.DataFrame:
    df = add_form_features(elo_df)
    df["elo_diff"] = (df["elo_home_pre"] + np.where(df["neutral"], 0.0, 100.0)) - df["elo_away_pre"]
    df["neutral_int"] = df["neutral"].astype(int)
    df["k_importance"] = df["tournament"].apply(k_factor)
    df["form_diff"] = df["home_form"].fillna(0.0) - df["away_form"].fillna(0.0)

    result = np.sign(df["home_score"] - df["away_score"])
    df["result"] = result.map({1.0: "H", 0.0: "D", -1.0: "A"})
    return df


def current_form(df: pd.DataFrame, played_mask: pd.Series) -> dict:
    """Latest known rolling-form value (last-10 goal diff avg) per team, after all played matches."""
    history = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    played = df[played_mask]
    for row in played.itertuples(index=False):
        gd = row.home_score - row.away_score
        history[row.home_team].append(gd)
        history[row.away_team].append(-gd)
    return {team: (np.mean(hist) if hist else 0.0) for team, hist in history.items()}


FEATURE_COLUMNS = [
    "elo_home_pre",
    "elo_away_pre",
    "elo_diff",
    "neutral_int",
    "k_importance",
    "home_form",
    "away_form",
    "form_diff",
]

LABEL_ORDER = ["A", "D", "H"]  # away win, draw, home win


if __name__ == "__main__":
    from load_data import load_results, played_mask as pm
    from elo import compute_elo_history

    df = load_results()
    mask = pm(df)
    elo_df = compute_elo_history(df, mask)
    feat_df = build_feature_frame(elo_df)
    played = feat_df[mask]
    print(played[FEATURE_COLUMNS + ["result"]].tail(10))
    print("\nResult distribution:")
    print(played["result"].value_counts(normalize=True))
