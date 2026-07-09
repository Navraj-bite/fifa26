"""
Monte Carlo forward simulation of the rest of the 2026 World Cup, using the
trained model's win probabilities and the confirmed bracket tree.

Bracket confirmed via live web search on 2026-07-07 (cross-checked across
ESPN, Fox Sports, Olympics.com, Al Jazeera):

  QF1 (Boston,  Jul 9):  France     vs Morocco
  QF2 (LA,      Jul 10): Spain      vs Belgium
  QF3 (Miami,   Jul 11): England    vs Norway
  QF4 (KC,      Jul 11/12): Winner(Argentina v Egypt) vs Winner(Switzerland v Colombia)
                             [both Round of 16 matches unplayed as of this run -- July 7]

  SF1 (Dallas,   Jul 14): Winner QF1 vs Winner QF2
  SF2 (Atlanta,  Jul 15): Winner QF3 vs Winner QF4

  Final (East Rutherford, Jul 19): Winner SF1 vs Winner SF2

For a knockout match with no replay, a draw isn't a valid final outcome (it
goes to extra time/penalties). We convert the model's 3-way (home/draw/away)
probabilities into a single win probability by (a) treating the match as
neutral-venue and (b) averaging the two home/away orderings of the same
matchup, splitting draw probability 50/50 -- a standard simplification for
penalty-shootout coin flips.
"""
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from elo import compute_elo_history, current_ratings, k_factor
from features import FEATURE_COLUMNS, LABEL_ORDER, current_form
from load_data import load_results, played_mask as pm

N_SIMULATIONS = 20000
RNG_SEED = 42

# Round of 16 matches still to be played as of this run (2026-07-07)
PENDING_R16 = [("Argentina", "Egypt"), ("Switzerland", "Colombia")]

QUARTERFINALS = [
    ("France", "Morocco"),
    ("Spain", "Belgium"),
    ("England", "Norway"),
    # 4th QF slot filled in dynamically once the two pending R16 games resolve
]


def win_probability(team_a, team_b, elo, form, model):
    """P(team_a beats team_b) in a neutral-venue knockout match, draw split 50/50."""
    def build_row(home, away):
        return {
            "elo_home_pre": elo.get(home, 1500.0),
            "elo_away_pre": elo.get(away, 1500.0),
            "elo_diff": elo.get(home, 1500.0) - elo.get(away, 1500.0),  # neutral: no home bonus
            "neutral_int": 1,
            "k_importance": k_factor("FIFA World Cup"),
            "home_form": form.get(home, 0.0),
            "away_form": form.get(away, 0.0),
            "form_diff": form.get(home, 0.0) - form.get(away, 0.0),
        }

    rows = pd.DataFrame([build_row(team_a, team_b), build_row(team_b, team_a)])[FEATURE_COLUMNS]
    proba = model.predict_proba(rows.values)
    idx = {c: i for i, c in enumerate(LABEL_ORDER)}  # xgb was fit with y in LABEL_ORDER index order

    p_a_as_home = proba[0, idx["H"]] + 0.5 * proba[0, idx["D"]]
    p_a_as_away = proba[1, idx["A"]] + 0.5 * proba[1, idx["D"]]
    return 0.5 * (p_a_as_home + p_a_as_away)


def build_matchup_probabilities(teams, elo, form, model):
    """Precompute P(team beats every other relevant team) for the small set of teams in play."""
    probs = {}
    for a in teams:
        for b in teams:
            if a != b and (a, b) not in probs:
                p = win_probability(a, b, elo, form, model)
                probs[(a, b)] = p
                probs[(b, a)] = 1.0 - p
    return probs


def simulate_bracket(rng, probs, pending_r16, quarterfinals):
    def play(a, b):
        return a if rng.random() < probs[(a, b)] else b

    # Resolve the two pending Round of 16 matches
    r16_winners = [play(a, b) for a, b in pending_r16]
    qf4 = (r16_winners[0], r16_winners[1])

    all_qf = quarterfinals + [qf4]
    qf_winners = [play(a, b) for a, b in all_qf]

    sf1 = (qf_winners[0], qf_winners[1])
    sf2 = (qf_winners[2], qf_winners[3])
    sf_winners = [play(*sf1), play(*sf2)]

    champion = play(*sf_winners)
    return {
        "qf4_opponent": qf4,
        "semifinalists": qf_winners,   # reached the semifinal (won their QF)
        "finalists": sf_winners,       # reached the final (won their SF)
        "champion": champion,
    }


def main():
    df = load_results()
    mask = pm(df)
    elo_df = compute_elo_history(df, mask)
    elo = current_ratings(elo_df, mask)
    form = current_form(df, mask)

    with open("results/models.pkl", "rb") as f:
        models = pickle.load(f)
    xgb = models["xgb"]

    teams_in_play = set()
    for a, b in PENDING_R16 + QUARTERFINALS:
        teams_in_play.add(a)
        teams_in_play.add(b)
    # QF4 participants are unknown in advance (winners of pending R16), so include
    # every team that could reach the quarterfinal/semifinal/final stage.
    remaining_teams = list(teams_in_play)

    print("Win probability for today's still-live Round of 16 matches:")
    probs = build_matchup_probabilities(remaining_teams, elo, form, xgb)
    for a, b in PENDING_R16:
        print(f"  {a:12s} vs {b:12s}  ->  {a}: {probs[(a,b)]:.1%}   {b}: {probs[(b,a)]:.1%}")

    print("\nWin probability for confirmed quarterfinals:")
    for a, b in QUARTERFINALS:
        print(f"  {a:12s} vs {b:12s}  ->  {a}: {probs[(a,b)]:.1%}   {b}: {probs[(b,a)]:.1%}")

    rng = np.random.default_rng(RNG_SEED)
    champion_counts = defaultdict(int)
    finalist_counts = defaultdict(int)
    semifinalist_counts = defaultdict(int)

    for _ in range(N_SIMULATIONS):
        sim = simulate_bracket(rng, probs, PENDING_R16, QUARTERFINALS)
        champion_counts[sim["champion"]] += 1
        for t in sim["finalists"]:
            finalist_counts[t] += 1
        for t in sim["semifinalists"]:
            semifinalist_counts[t] += 1

    table = []
    all_teams = set(champion_counts) | set(finalist_counts) | set(semifinalist_counts)
    for team in all_teams:
        table.append({
            "team": team,
            "semifinal_pct": 100 * semifinalist_counts[team] / N_SIMULATIONS,
            "final_pct": 100 * finalist_counts[team] / N_SIMULATIONS,
            "champion_pct": 100 * champion_counts[team] / N_SIMULATIONS,
        })
    table_df = pd.DataFrame(table).sort_values("champion_pct", ascending=False).reset_index(drop=True)

    print(f"\nChampionship probability table ({N_SIMULATIONS:,} simulations):")
    print(table_df.to_string(index=False, float_format=lambda x: f"{x:5.1f}%"))

    table_df.to_csv("results/championship_probabilities.csv", index=False)
    with open("results/simulation_meta.json", "w") as f:
        json.dump({
            "n_simulations": N_SIMULATIONS,
            "rng_seed": RNG_SEED,
            "pending_r16": PENDING_R16,
            "confirmed_quarterfinals": QUARTERFINALS,
        }, f, indent=2)
    print("\nSaved: results/championship_probabilities.csv, results/simulation_meta.json")

    return table_df


if __name__ == "__main__":
    main()
