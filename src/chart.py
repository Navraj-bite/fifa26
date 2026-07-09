"""
Championship probability chart: horizontal bar, ranked by magnitude, single
sequential hue (darker = higher probability), direct-labeled.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Sequential blue ramp + chrome, from the project's validated default palette.
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def hue_for(pct, max_pct):
    step = min(int((pct / max_pct) * (len(SEQ_RAMP) - 1)), len(SEQ_RAMP) - 1)
    return SEQ_RAMP[max(step, 1)]


def main():
    df = pd.read_csv("results/championship_probabilities.csv").sort_values("champion_pct")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    max_pct = df["champion_pct"].max()
    colors = [hue_for(p, max_pct) for p in df["champion_pct"]]
    bars = ax.barh(df["team"], df["champion_pct"], color=colors, height=0.62)

    for bar, pct in zip(bars, df["champion_pct"]):
        ax.text(
            bar.get_width() + max_pct * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left", fontsize=9, color=INK_PRIMARY,
        )

    ax.set_xlim(0, max_pct * 1.18)
    ax.set_xlabel("Championship probability (%)", color=INK_MUTED, fontsize=9)
    ax.set_title(
        "2026 World Cup Championship Probability\n(20,000-run Monte Carlo, as of 2026-07-07)",
        color=INK_PRIMARY, fontsize=12, loc="left", pad=12,
    )
    ax.tick_params(axis="y", colors=INK_PRIMARY, labelsize=10)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)

    fig.tight_layout()
    fig.savefig("results/championship_probabilities.png", facecolor=SURFACE)
    print("Saved: results/championship_probabilities.png")


if __name__ == "__main__":
    main()
