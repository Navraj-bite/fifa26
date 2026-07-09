"""
Rolling Elo rating engine for national football teams, following the
long-standing eloratings.net methodology (goal-difference-weighted, tournament-
weighted K-factor), computed match-by-match over the full 1872-2026 history.

This is the core feature engineering step: team strength isn't a column in the
dataset, it's a number we build by replaying every match in order.
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

INITIAL_ELO = 1500.0
HOME_ADVANTAGE = 100.0  # rating-point bonus applied to the home side, if not neutral

# Tournament importance -> K-factor, per eloratings.net conventions.
WORLD_CUP_FINALS_RE = re.compile(r"^FIFA World Cup$")
MAJOR_CONTINENTAL_RE = re.compile(
    r"Copa América|UEFA Euro$|African Cup of Nations$|AFC Asian Cup$|"
    r"Gold Cup$|Confederations Cup|CONCACAF Championship$"
)
QUALIFIER_OR_MAJOR_RE = re.compile(r"qualification|Nations League|CONCACAF Championship qualification")


def k_factor(tournament: str) -> float:
    if WORLD_CUP_FINALS_RE.search(tournament):
        return 60.0
    if MAJOR_CONTINENTAL_RE.search(tournament):
        return 50.0
    if QUALIFIER_OR_MAJOR_RE.search(tournament):
        return 40.0
    if tournament == "Friendly":
        return 20.0
    return 30.0


def goal_diff_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo_history(df: pd.DataFrame, played_mask: pd.Series) -> pd.DataFrame:
    """
    Replay every played match in chronological order, updating Elo ratings.

    Returns a copy of df with four new columns: elo_home_pre, elo_away_pre
    (rating entering the match, used as features) and elo_home_post,
    elo_away_post (rating after the match is applied). Unplayed fixture rows
    get elo_*_pre populated (using the latest known rating for each team) but
    no post-match update.
    """
    ratings = defaultdict(lambda: INITIAL_ELO)
    n = len(df)
    elo_home_pre = np.empty(n)
    elo_away_pre = np.empty(n)
    elo_home_post = np.full(n, np.nan)
    elo_away_post = np.full(n, np.nan)

    for i, row in enumerate(df.itertuples(index=False)):
        home, away = row.home_team, row.away_team
        r_home, r_away = ratings[home], ratings[away]
        elo_home_pre[i] = r_home
        elo_away_pre[i] = r_away

        if not played_mask.iloc[i]:
            continue

        home_score, away_score = row.home_score, row.away_score
        goal_diff = home_score - away_score
        if goal_diff > 0:
            s_home = 1.0
        elif goal_diff == 0:
            s_home = 0.5
        else:
            s_home = 0.0

        home_adj = 0.0 if row.neutral else HOME_ADVANTAGE
        e_home = expected_score(r_home + home_adj, r_away)
        k = k_factor(row.tournament) * goal_diff_multiplier(goal_diff)

        delta = k * (s_home - e_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta
        elo_home_post[i] = ratings[home]
        elo_away_post[i] = ratings[away]

    out = df.copy()
    out["elo_home_pre"] = elo_home_pre
    out["elo_away_pre"] = elo_away_pre
    out["elo_home_post"] = elo_home_post
    out["elo_away_post"] = elo_away_post
    return out


def current_ratings(elo_df: pd.DataFrame, played_mask: pd.Series) -> dict:
    """Latest known Elo rating for every team, after all played matches."""
    ratings = defaultdict(lambda: INITIAL_ELO)
    played = elo_df[played_mask]
    for row in played.itertuples(index=False):
        ratings[row.home_team] = row.elo_home_post
        ratings[row.away_team] = row.elo_away_post
    return dict(ratings)


if __name__ == "__main__":
    from load_data import load_results, played_mask as pm

    df = load_results()
    mask = pm(df)
    elo_df = compute_elo_history(df, mask)
    ratings = current_ratings(elo_df, mask)

    top20 = sorted(ratings.items(), key=lambda kv: -kv[1])[:20]
    print("Top 20 teams by current Elo rating:")
    for team, rating in top20:
        print(f"  {team:20s} {rating:7.1f}")
