"""Draw assets/neuron.png: one neuron with the numbers of the worked example in notebook 01.

Run from the repository root: uv run python assets/diagrams/neuron.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent.parent / "neuron.png"

BLUE, PINK, INK, GREY, NODE = "#2f6fb0", "#c2407e", "#222222", "#6b6b6b", "#fdf6ec"

fig = plt.figure(figsize=(10, 4.4), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 0.44)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

R = 0.034
X_IN, X_SUM, X_ACT, X_OUT = 0.1, 0.47, 0.70, 0.89
Y_MID = 0.21
inputs = [(0.355, "$x_1$", "1.5", "$w_1 = 0.4$"), (0.21, "$x_2$", "-0.5", "$w_2 = 1.2$"),
          (0.065, "$x_3$", "2.0", "$w_3 = -0.3$")]


def arrow(start, end, colour=GREY, width=1.4):
    """Draw a straight arrow between two points."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>,head_length=6,head_width=3.5", color=colour,
                                 linewidth=width, shrinkA=0, shrinkB=0, zorder=2))


# inputs and weighted connections
for y, name, value, weight in inputs:
    ax.add_patch(Circle((X_IN, y), R, facecolor=NODE, edgecolor=INK, linewidth=1.6, zorder=3))
    ax.text(X_IN, y + 0.009, name, fontsize=12, ha="center", va="center", zorder=4)
    ax.text(X_IN, y - 0.014, value, fontsize=10, ha="center", va="center", color=GREY, zorder=4)
    arrow((X_IN + R, y), (X_SUM - R - 0.004, Y_MID + (y - Y_MID) * 0.25), PINK, 1.6)
    t = 0.45
    lx, ly = X_IN + R + t * (X_SUM - R - X_IN - R), y + t * ((Y_MID + (y - Y_MID) * 0.25) - y)
    ax.text(lx, ly + 0.012, weight, fontsize=11, color=PINK, ha="center", va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})

# bias
ax.add_patch(FancyBboxPatch((X_SUM - 0.045, 0.375), 0.09, 0.04, boxstyle="round,pad=0,rounding_size=0.008",
                            facecolor="white", edgecolor=PINK, linewidth=1.4, zorder=3))
ax.text(X_SUM, 0.395, "$b = 0.1$", fontsize=11, color=PINK, ha="center", va="center", zorder=4)
arrow((X_SUM, 0.375), (X_SUM, Y_MID + R + 0.003), PINK, 1.6)

# weighted sum
ax.add_patch(Circle((X_SUM, Y_MID), R, facecolor=NODE, edgecolor=BLUE, linewidth=1.8, zorder=3))
ax.text(X_SUM, Y_MID, "Σ", fontsize=17, color=BLUE, ha="center", va="center", zorder=4)
ax.text(X_SUM, Y_MID - R - 0.02, "$z = -0.50$", fontsize=11, color=BLUE, ha="center", va="top")

# activation
arrow((X_SUM + R, Y_MID), (X_ACT - 0.04, Y_MID))
ax.add_patch(FancyBboxPatch((X_ACT - 0.04, Y_MID - 0.04), 0.08, 0.08, boxstyle="round,pad=0,rounding_size=0.012",
                            facecolor=NODE, edgecolor=BLUE, linewidth=1.8, zorder=3))
ax.plot([X_ACT - 0.025, X_ACT, X_ACT + 0.022], [Y_MID - 0.015, Y_MID - 0.015, Y_MID + 0.02], color=BLUE,
        linewidth=2, solid_capstyle="round", zorder=4)
ax.text(X_ACT, Y_MID - 0.06, "$f = $ ReLU", fontsize=11, color=BLUE, ha="center", va="top")

# output
arrow((X_ACT + 0.04, Y_MID), (X_OUT - 0.045, Y_MID))
ax.add_patch(FancyBboxPatch((X_OUT - 0.045, Y_MID - 0.028), 0.09, 0.056, boxstyle="round,pad=0,rounding_size=0.01",
                            facecolor="white", edgecolor=INK, linewidth=1.4, zorder=3))
ax.text(X_OUT, Y_MID, r"$\hat{y} = 0$", fontsize=12, ha="center", va="center", zorder=4)

# column captions
for x, label in [(X_IN, "inputs"), ((X_IN + X_SUM) / 2, "weights"), (X_SUM, "weighted sum"),
                 (X_ACT, "activation"), (X_OUT, "output")]:
    ax.text(x, 0.008, label, fontsize=10, color=GREY, ha="center", va="bottom")

fig.savefig(OUT, facecolor="white")
