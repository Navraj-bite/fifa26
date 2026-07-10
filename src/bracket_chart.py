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
from matplotlib.patches import FancyBboxPatch

PAGE = "#f2f1ed"
CARD_BG = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
ACCENT = "#256abf"
ACCENT_LIGHT = "#e8f0fb"
BORDER = "#e1e0d9"

CARD_W = 1.85
CARD_H = 0.66
ROW_H = CARD_H / 2

QF_Y = {"qf1": 3.6, "qf2": 2.5, "qf3": 1.4, "qf4": 0.3}
SF_Y = {"sf1": 3.05, "sf2": 0.85}
FINAL_Y = 1.95

QF_X, SF_X, FINAL_X, CHAMPION_X = 0.0, 2.55, 5.1, 7.65


def card(ax, x, y, w=CARD_W, h=CARD_H, fc=CARD_BG, ec=BORDER, lw=1.1, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size=0.07",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
    ))


def draw_match(ax, x, y, team_a, team_b, winner, confidence, status):
    card(ax, x, y)

    is_a_winner = winner == team_a
    top_color = INK_PRIMARY if is_a_winner else INK_MUTED
    bot_color = INK_MUTED if is_a_winner else INK_PRIMARY
    top_weight = "bold" if is_a_winner else "normal"
    bot_weight = "bold" if not is_a_winner else "normal"

    top_y, bot_y = y + ROW_H / 2, y - ROW_H / 2

    stripe_y = top_y if is_a_winner else bot_y
    ax.add_patch(FancyBboxPatch(
        (x + 0.05, stripe_y - ROW_H / 2 + 0.07), 0.055, ROW_H - 0.14,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=0, facecolor=ACCENT, zorder=3,
    ))

    ax.text(x + 0.24, top_y, team_a, fontsize=10, color=top_color, weight=top_weight,
            va="center", ha="left", zorder=4)
    ax.text(x + 0.24, bot_y, team_b, fontsize=10, color=bot_color, weight=bot_weight,
            va="center", ha="left", zorder=4)
    ax.plot([x + 0.1, x + CARD_W - 0.1], [y, y], color=BORDER, linewidth=1, zorder=3)

    tag = "confirmed result" if status == "confirmed" else f"{confidence:.0%} confidence"
    ax.text(x + CARD_W / 2, y - CARD_H / 2 - 0.15, tag, fontsize=7.5, color=INK_MUTED,
            ha="center", va="top", style="italic")

    return winner


def connector(ax, x0, y0, x1, y1):
    mid_x = (x0 + x1) / 2
    ax.plot([x0, mid_x, mid_x, x1], [y0, y0, y1, y1], color=BORDER, linewidth=1.4,
            zorder=1, solid_capstyle="round", solid_joinstyle="round")


def stage_header(ax, label, x, w=CARD_W):
    ax.text(x + w / 2, 4.35, label, fontsize=11, color=INK_PRIMARY, ha="center",
            weight="bold")
    ax.plot([x, x + w], [4.15, 4.15], color=BORDER, linewidth=1.5)


def main():
    with open("results/most_likely_bracket.json") as f:
        bracket = json.load(f)

    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=150)
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)

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

    connector(ax, QF_X + CARD_W, QF_Y["qf1"], SF_X, SF_Y["sf1"])
    connector(ax, QF_X + CARD_W, QF_Y["qf2"], SF_X, SF_Y["sf1"])
    connector(ax, QF_X + CARD_W, QF_Y["qf3"], SF_X, SF_Y["sf2"])
    connector(ax, QF_X + CARD_W, QF_Y["qf4"], SF_X, SF_Y["sf2"])

    sf = bracket["semifinals"]
    sf1_match = f"{qf1} vs {qf2}"
    sf2_match = f"{qf3} vs {qf4}"
    sf1 = draw_match(ax, SF_X, SF_Y["sf1"], qf1, qf2,
                      sf[sf1_match]["winner"], sf[sf1_match]["confidence"], "predicted")
    sf2 = draw_match(ax, SF_X, SF_Y["sf2"], qf3, qf4,
                      sf[sf2_match]["winner"], sf[sf2_match]["confidence"], "predicted")

    connector(ax, SF_X + CARD_W, SF_Y["sf1"], FINAL_X, FINAL_Y)
    connector(ax, SF_X + CARD_W, SF_Y["sf2"], FINAL_X, FINAL_Y)

    final = bracket["final"]
    final_match = f"{sf1} vs {sf2}"
    champion = draw_match(ax, FINAL_X, FINAL_Y, sf1, sf2,
                           final[final_match]["winner"], final[final_match]["confidence"], "predicted")

    connector(ax, FINAL_X + CARD_W, FINAL_Y, CHAMPION_X, FINAL_Y)

    champ_w, champ_h = 1.9, 1.0
    ax.add_patch(FancyBboxPatch(
        (CHAMPION_X, FINAL_Y - champ_h / 2), champ_w, champ_h,
        boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=0, facecolor=ACCENT, zorder=2,
    ))
    ax.text(CHAMPION_X + champ_w / 2, FINAL_Y + 0.18, "PREDICTED CHAMPION",
            fontsize=7.5, color="#dbe8fa", ha="center", va="center", weight="bold")
    ax.text(CHAMPION_X + champ_w / 2, FINAL_Y - 0.12, champion,
            fontsize=15, color="white", ha="center", va="center", weight="bold")

    stage_header(ax, "Quarterfinals", QF_X)
    stage_header(ax, "Semifinals", SF_X)
    stage_header(ax, "Final", FINAL_X)
    stage_header(ax, "Champion", CHAMPION_X, w=champ_w)

    ax.set_xlim(-0.35, CHAMPION_X + champ_w + 0.35)
    ax.set_ylim(-0.55, 4.75)
    ax.axis("off")
    ax.set_title(
        "Most Likely Bracket",
        color=INK_PRIMARY, fontsize=16, loc="left", pad=18, weight="bold",
    )
    ax.text(-0.35, 4.66, "What happens if the favorite wins every remaining match, as of the last update",
            fontsize=9.5, color=INK_MUTED, ha="left", va="top")

    fig.tight_layout()
    fig.savefig("results/most_likely_bracket.png", facecolor=PAGE, bbox_inches="tight")
    print("Saved: results/most_likely_bracket.png")


if __name__ == "__main__":
    main()
