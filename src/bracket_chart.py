"""
Visual bracket diagram of the single most probable outcome: at every remaining
match, the model's favorite advances. This is a different view than the
championship probability chart -- that one asks "how often does each team win
it all across every possible branch", this asks "if the favorite wins every
remaining match, who's left standing". They don't have to agree (see
simulate.py's most_likely_bracket docstring), so both are worth showing.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
ACCENT = "#256abf"
GRIDLINE = "#e1e0d9"

QF_Y = {"qf1": 3.5, "qf2": 2.5, "qf3": 1.5, "qf4": 0.5}
SF_Y = {"sf1": 3.0, "sf2": 1.0}
FINAL_Y = 2.0

QF_X, SF_X, FINAL_X, CHAMPION_X = 0.0, 2.2, 4.4, 6.7


def draw_match(ax, x, y, team_a, team_b, winner, confidence, status, box_h=0.7):
    loser = team_b if winner == team_a else team_a
    is_a_winner = winner == team_a

    top_color = INK_PRIMARY if is_a_winner else INK_MUTED
    bot_color = INK_MUTED if is_a_winner else INK_PRIMARY
    top_weight = "bold" if is_a_winner else "normal"
    bot_weight = "bold" if not is_a_winner else "normal"

    ax.text(x, y + box_h / 4, team_a, fontsize=9.5, color=top_color, weight=top_weight,
             va="center", ha="left", fontfamily="sans-serif")
    ax.text(x, y - box_h / 4, team_b, fontsize=9.5, color=bot_color, weight=bot_weight,
             va="center", ha="left", fontfamily="sans-serif")

    tag = "confirmed" if status == "confirmed" else f"{confidence:.0%}"
    ax.plot([x - 0.15], [y + (box_h / 4 if is_a_winner else -box_h / 4)],
             marker="o", markersize=4, color=ACCENT, zorder=3)
    ax.text(x + 1.55, y, tag, fontsize=7.5, color=INK_MUTED, va="center", ha="right",
             style="italic")

    ax.plot([x - 0.35, x - 0.35], [y - box_h / 4, y + box_h / 4], color=GRIDLINE, linewidth=1.5)
    return winner


def connector(ax, x0, y0, x1, y1):
    mid_x = (x0 + x1) / 2
    ax.plot([x0, mid_x, mid_x, x1], [y0, y0, y1, y1], color=GRIDLINE, linewidth=1.2, zorder=1)


def main():
    with open("results/most_likely_bracket.json") as f:
        bracket = json.load(f)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    qf = bracket["quarterfinals"]
    qf1 = draw_match(ax, QF_X, QF_Y["qf1"], "France", "Morocco",
                      qf["France vs Morocco"]["winner"], qf["France vs Morocco"]["confidence"],
                      qf["France vs Morocco"]["status"])
    qf2 = draw_match(ax, QF_X, QF_Y["qf2"], "Spain", "Belgium",
                      qf["Spain vs Belgium"]["winner"], qf["Spain vs Belgium"]["confidence"],
                      qf["Spain vs Belgium"]["status"])
    qf3 = draw_match(ax, QF_X, QF_Y["qf3"], "England", "Norway",
                      qf["England vs Norway"]["winner"], qf["England vs Norway"]["confidence"],
                      qf["England vs Norway"]["status"])
    qf4 = draw_match(ax, QF_X, QF_Y["qf4"], "Argentina", "Switzerland",
                      qf["Argentina vs Switzerland"]["winner"], qf["Argentina vs Switzerland"]["confidence"],
                      qf["Argentina vs Switzerland"]["status"])

    connector(ax, QF_X + 1.7, QF_Y["qf1"], SF_X, SF_Y["sf1"])
    connector(ax, QF_X + 1.7, QF_Y["qf2"], SF_X, SF_Y["sf1"])
    connector(ax, QF_X + 1.7, QF_Y["qf3"], SF_X, SF_Y["sf2"])
    connector(ax, QF_X + 1.7, QF_Y["qf4"], SF_X, SF_Y["sf2"])

    sf = bracket["semifinals"]
    sf1_match = f"{qf1} vs {qf2}"
    sf2_match = f"{qf3} vs {qf4}"
    sf1 = draw_match(ax, SF_X, SF_Y["sf1"], qf1, qf2,
                      sf[sf1_match]["winner"], sf[sf1_match]["confidence"], "predicted")
    sf2 = draw_match(ax, SF_X, SF_Y["sf2"], qf3, qf4,
                      sf[sf2_match]["winner"], sf[sf2_match]["confidence"], "predicted")

    connector(ax, SF_X + 1.7, SF_Y["sf1"], FINAL_X, FINAL_Y)
    connector(ax, SF_X + 1.7, SF_Y["sf2"], FINAL_X, FINAL_Y)

    final = bracket["final"]
    final_match = f"{sf1} vs {sf2}"
    champion = draw_match(ax, FINAL_X, FINAL_Y, sf1, sf2,
                           final[final_match]["winner"], final[final_match]["confidence"], "predicted")

    ax.plot([FINAL_X + 1.7, CHAMPION_X - 0.3], [FINAL_Y, FINAL_Y], color=GRIDLINE, linewidth=1.2)
    ax.text(CHAMPION_X, FINAL_Y, champion, fontsize=13, color=ACCENT, weight="bold",
            va="center", ha="left")
    ax.text(CHAMPION_X, FINAL_Y - 0.35, "predicted champion", fontsize=8, color=INK_MUTED,
            va="center", ha="left", style="italic")

    stage_labels = [("Quarterfinals", QF_X), ("Semifinals", SF_X), ("Final", FINAL_X), ("", CHAMPION_X)]
    for label, x in stage_labels:
        ax.text(x, 4.15, label, fontsize=10, color=INK_MUTED, ha="left", weight="bold")

    ax.set_xlim(-0.7, CHAMPION_X + 2.2)
    ax.set_ylim(-0.2, 4.5)
    ax.axis("off")
    ax.set_title(
        "Most Likely Bracket\nfavorite wins every remaining match, as of the last update",
        color=INK_PRIMARY, fontsize=12, loc="left", pad=14,
    )

    fig.tight_layout()
    fig.savefig("results/most_likely_bracket.png", facecolor=SURFACE, bbox_inches="tight")
    print("Saved: results/most_likely_bracket.png")


if __name__ == "__main__":
    main()
