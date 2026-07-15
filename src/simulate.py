"""
Monte Carlo forward simulation of the rest of the 2026 World Cup, using the
trained model's win probabilities and the confirmed bracket tree.

Bracket confirmed via live web search, cross-checked across ESPN, Fox Sports,
FIFA.com, CNN, and Al Jazeera. Status as of this run (2026-07-11):

  QF1 (Boston,  Jul 9):  France 2-0 Morocco -- DONE, France advances
  QF2 (LA,      Jul 10): Spain 2-1 Belgium  -- DONE, Spain advances
  QF3 (Miami,   Jul 11): England 2-1 Norway (aet) -- DONE, England advances
  QF4 (KC,      Jul 11): Argentina 3-1 Switzerland (aet) -- DONE, Argentina advances

  SF1 (Dallas,   Jul 14): France 0-2 Spain -- DONE, Spain advances to the final
  SF2 (Atlanta,  Jul 15): England vs Argentina -- pending

  Final (East Rutherford, Jul 19): Spain vs Winner SF2

A "slot" below is either a confirmed team name (a string) or a match still to
be played (a tuple of two team names). Confirmed slots skip straight through;
pending slots get a coin flip using the model's win probability each
simulation run. Spain's spot in the final is locked, so the only remaining
randomness is the second semifinal and the final itself.

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

SF1_SLOTS = ["France", "Spain"]  # played Jul 14: Spain won 0-2
SF1_WINNER = "Spain"
SF2_SLOTS = ["England", "Argentina"]


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


def most_likely_winner(probs, slot):
    """The single favorite for a slot, plus the model's win probability for that pick.
    A confirmed slot (a string) returns probability 1.0, it already happened."""
    if isinstance(slot, str):
        return slot, 1.0
    a, b = slot
    return (a, float(probs[(a, b)])) if probs[(a, b)] >= 0.5 else (b, float(probs[(b, a)]))


def most_likely_bracket(probs):
    """Greedily picks the favorite at every stage to build the single most probable
    bracket outcome. This is a different question than the Monte Carlo table above:
    that table asks 'how often does each team win it all across every possible
    branch', this asks 'if the favorite wins every remaining match, who's left
    standing'. The two can disagree, and often do: a team can lead the marginal
    championship table by being a strong favorite across many possible paths to
    the final, while still losing the single specific matchup this greedy walk
    happens to route them into."""
    sf2_winner, sf2_conf = most_likely_winner(probs, tuple(SF2_SLOTS))

    champion, final_conf = most_likely_winner(probs, (SF1_WINNER, sf2_winner))

    return {
        "quarterfinals": {
            "France vs Morocco": {"winner": "France", "confidence": 1.0, "status": "confirmed"},
            "Spain vs Belgium": {"winner": "Spain", "confidence": 1.0, "status": "confirmed"},
            "England vs Norway": {"winner": "England", "confidence": 1.0, "status": "confirmed"},
            "Argentina vs Switzerland": {"winner": "Argentina", "confidence": 1.0, "status": "confirmed"},
        },
        "semifinals": {
            "France vs Spain": {"winner": SF1_WINNER, "confidence": 1.0, "status": "confirmed"},
            f"{SF2_SLOTS[0]} vs {SF2_SLOTS[1]}": {"winner": sf2_winner, "confidence": round(sf2_conf, 3), "status": "predicted"},
        },
        "final": {
            f"{SF1_WINNER} vs {sf2_winner}": {"winner": champion, "confidence": round(final_conf, 3)},
        },
        "champion": champion,
    }


def simulate_bracket(rng, probs):
    sf1_teams = [resolve(rng, probs, s) for s in SF1_SLOTS]
    sf2_teams = [resolve(rng, probs, s) for s in SF2_SLOTS]
    semifinalists = sf1_teams + sf2_teams

    finalist_1 = SF1_WINNER  # France vs Spain already played
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

    print("Win probability for the matches still to be played:")
    probs = build_matchup_probabilities(remaining_teams, elo, form, xgb)
    remaining = [tuple(SF2_SLOTS)] + [(SF1_WINNER, t) for t in SF2_SLOTS]
    for a, b in remaining:
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

    bracket = most_likely_bracket(probs)
    print(f"\nMost likely bracket (favorite wins every remaining match): {bracket['champion']} takes it all")
    for stage in ["quarterfinals", "semifinals", "final"]:
        for match, outcome in bracket[stage].items():
            print(f"  {match:30s} -> {outcome['winner']:12s} ({outcome['confidence']:.1%})")
    with open("results/most_likely_bracket.json", "w") as f:
        json.dump(bracket, f, indent=2)
    print("Saved: results/most_likely_bracket.json")

    return table_df


if __name__ == "__main__":
    main()
