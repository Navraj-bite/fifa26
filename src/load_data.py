"""
Load and clean the international match history dataset.

The Kaggle dataset (martj42/international-football-results-from-1872-to-2017,
despite the name, auto-updates through the present) lags real-world results by
a day or two. As of 2026-07-07 it was missing final scores for the entire 2026
World Cup Round of 16, and correctly carries the France vs Morocco
quarterfinal as an unplayed fixture stub. We patch the known results in as
they get confirmed by press coverage.

Penalty shootouts are recorded as the regulation/extra-time draw they actually
were (0-0, 1-1, etc), matching how the source dataset's own shootouts.csv
handles this -- goal difference for Elo purposes reflects play, not the
shootout coin flip. The shootout winner still determines who advances in
simulate.py.

Sources for patched results (cross-checked across 2+ independent outlets each):
  Brazil 1-2 Norway          - Al Jazeera, ESPN, FIFA.com (2026-07-05)
  Mexico 2-3 England         - ESPN, Fox Sports (2026-07-05)
  Portugal 0-1 Spain         - ESPN, CBS Sports, Olympics.com (2026-07-06)
  United States 1-4 Belgium  - ESPN, NPR, ABC News, FIFA.com (2026-07-06)
  Argentina 3-2 Egypt        - ESPN, Yahoo Sports, NPR, NBC News (2026-07-07)
  Switzerland 0-0 Colombia   - ESPN, Al Jazeera, CNN, NBC News, FIFA.com
                                (2026-07-07); Switzerland won 4-3 on penalties
                                and advances to the quarterfinals
  France 2-0 Morocco         - FIFA.com, ESPN, Fox Sports, CNN, NBC News
                                (2026-07-09); France advances to face the
                                Spain/Belgium winner in the semifinal
"""
import pandas as pd

RESULTS_CSV = "data/results.csv"

KNOWN_2026_RESULT_PATCHES = [
    # (date, home_team, away_team, home_score, away_score)
    ("2026-07-05", "Brazil", "Norway", 1, 2),
    ("2026-07-05", "Mexico", "England", 2, 3),
    ("2026-07-06", "Portugal", "Spain", 0, 1),
    ("2026-07-06", "United States", "Belgium", 1, 4),
    ("2026-07-06", "Argentina", "Egypt", 3, 2),
    ("2026-07-06", "Switzerland", "Colombia", 0, 0),  # won 4-3 on penalties
    ("2026-07-09", "France", "Morocco", 2, 0),
]

# Round of 16 matches decided on penalties: (date, home_team, away_team, shootout_winner).
# Used by simulate.py to determine who actually advances, since the goal-based
# score above records the level scoreline, not the shootout outcome.
PENALTY_SHOOTOUT_WINNERS = {
    ("2026-07-06", "Switzerland", "Colombia"): "Switzerland",
}


def load_results(path: str = RESULTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(bool)

    for date, home, away, hs, away_s in KNOWN_2026_RESULT_PATCHES:
        mask = (
            (df["date"] == pd.Timestamp(date))
            & (df["home_team"] == home)
            & (df["away_team"] == away)
        )
        if mask.sum() != 1:
            raise ValueError(f"Expected exactly one row for {date} {home} v {away}, found {mask.sum()}")
        df.loc[mask, "home_score"] = hs
        df.loc[mask, "away_score"] = away_s

    df = df.sort_values("date").reset_index(drop=True)
    return df


def played_mask(df: pd.DataFrame) -> pd.Series:
    """Rows with a confirmed final score (excludes future/unplayed fixtures)."""
    return df["home_score"].notna() & df["away_score"].notna()


if __name__ == "__main__":
    df = load_results()
    print(f"Total rows: {len(df)}")
    print(f"Played matches: {played_mask(df).sum()}")
    print(f"Unplayed fixture stubs: {(~played_mask(df)).sum()}")
    print(df[~played_mask(df)][["date", "home_team", "away_team", "tournament"]])
