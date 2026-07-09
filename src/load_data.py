"""
Load and clean the international match history dataset.

The Kaggle dataset (martj42/international-football-results-from-1872-to-2017,
despite the name, auto-updates through the present) lags real-world results by
a day or two. As of 2026-07-07 it was missing final scores for six 2026 World
Cup Round of 16 matches that have since been confirmed by press coverage, and
correctly carries the France vs Morocco quarterfinal as an unplayed fixture
stub. We patch the known results in; two Round of 16 matches (Argentina vs
Egypt, Switzerland vs Colombia) are genuinely unplayed as of this run and are
left as NaN scores -- those are exactly the matches this project predicts
live and grades once real results land.

Sources for patched results (cross-checked across 2+ independent outlets each):
  Brazil 1-2 Norway        - Al Jazeera, ESPN, FIFA.com (2026-07-05)
  Mexico 2-3 England       - ESPN, Fox Sports (2026-07-05)
  Portugal 0-1 Spain       - ESPN, CBS Sports, Olympics.com (2026-07-06)
  United States 1-4 Belgium - ESPN, NPR, ABC News, FIFA.com (2026-07-06)
"""
import pandas as pd

RESULTS_CSV = "data/results.csv"

KNOWN_2026_RESULT_PATCHES = [
    # (date, home_team, away_team, home_score, away_score)
    ("2026-07-05", "Brazil", "Norway", 1, 2),
    ("2026-07-05", "Mexico", "England", 2, 3),
    ("2026-07-06", "Portugal", "Spain", 0, 1),
    ("2026-07-06", "United States", "Belgium", 1, 4),
]


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
