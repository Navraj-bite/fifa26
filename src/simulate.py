"""
Monte Carlo forward simulation of the rest of the 2026 World Cup, using the
trained model's win probabilities and the confirmed bracket tree.

Bracket confirmed via live web search, cross-checked across ESPN, Fox Sports,
FIFA.com, CNN, and NBC News. Status as of this run (2026-07-10):

  QF1 (Boston,  Jul 9):  France 2-0 Morocco -- DONE, France advances
  QF2 (LA,      Jul 10): Spain      vs Belgium     -- pending
  QF3 (Miami,   Jul 11): England    vs Norway      -- pending
  QF4 (KC,      Jul 11/12): Argentina vs Switzerland -- pending

  SF1 (Dallas,   Jul 14): France vs Winner QF2
  SF2 (Atlanta,  Jul 15): Winner QF3 vs Winner QF4

  Final (East Rutherford, Jul 19): Winner SF1 vs Winner SF2

A "slot" below is either a confirmed team name (a string) or a match still to
be played (a tuple of two team names). Confirmed slots skip straight through;
pending slots get a coin flip using the model's win probability each
simulation run.

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

SF1_SLOTS = ["France", ("Spain", "Belgium")]
SF2_SLOTS = [("England", "Norway"), ("Argentina", "Switzerland")]


def pending_matches(slots):
    return [s for s in slots if isinstance(s, tuple)]


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


def resolve(rng, probs, slot):
    if isinstance(slot, str):
        return slot
    a, b = slot
    return a if rng.random() < probs[(a, b)] else b


def simulate_bracket(rng, probs):
    sf1_teams = [resolve(rng, probs, s) for s in SF1_SLOTS]
    sf2_teams = [resolve(rng, probs, s) for s in SF2_SLOTS]
    semifinalists = sf1_teams + sf2_teams

    finalist_1 = resolve(rng, probs, tuple(sf1_teams))
    finalist_2 = resolve(rng, probs, tuple(sf2_teams))
    finalists = [finalist_1, finalist_2]

    champion = resolve(rng, probs, tuple(finalists))
    return {
        "semifinalists": semifinalists,
        "finalists": finalists,
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

    # Every team that could plausibly meet every other team in play (across QF,
    # SF, and Final matchups) needs a pairwise probability, since who plays whom
    # in the semifinal and final depends on how the pending quarterfinals resolve.
    all_teams = set()
    for slot in SF1_SLOTS + SF2_SLOTS:
        if isinstance(slot, tuple):
            all_teams.update(slot)
        else:
            all_teams.add(slot)
    remaining_teams = list(all_teams)

    print("Win probability for the confirmed quarterfinals still to be played:")
    probs = build_matchup_probabilities(remaining_teams, elo, form, xgb)
    for a, b in pending_matches(SF1_SLOTS) + pending_matches(SF2_SLOTS):
        print(f"  {a:12s} vs {b:12s}  ->  {a}: {probs[(a,b)]:.1%}   {b}: {probs[(b,a)]:.1%}")
    # probs already covers every pair among remaining_teams (built above), which
    # is what resolve() needs for cross-matchup pairings like France vs whoever
    # wins Spain/Belgium, since the semifinal opponent isn't fixed in advance.

    rng = np.random.default_rng(RNG_SEED)
    champion_counts = defaultdict(int)
    finalist_counts = defaultdict(int)
    semifinalist_counts = defaultdict(int)

    for _ in range(N_SIMULATIONS):
        sim = simulate_bracket(rng, probs)
        champion_counts[sim["champion"]] += 1
        for t in sim["finalists"]:
            finalist_counts[t] += 1
        for t in sim["semifinalists"]:
            semifinalist_counts[t] += 1

    table = []
    all_result_teams = set(champion_counts) | set(finalist_counts) | set(semifinalist_counts)
    for team in all_result_teams:
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
            "sf1_slots": [list(s) if isinstance(s, tuple) else s for s in SF1_SLOTS],
            "sf2_slots": [list(s) if isinstance(s, tuple) else s for s in SF2_SLOTS],
        }, f, indent=2)
    print("\nSaved: results/championship_probabilities.csv, results/simulation_meta.json")

    return table_df


if __name__ == "__main__":
    main()
