"""Draw assets/relu_curve.png: the feed-forward network of four ReLU neurons that builds y = x² in notebook 01.

Every hidden neuron shows its weighted sum z = w·x + b with its own numbers and the ReLU that follows, and the
values for the example x = 2 are written under each step.

Run from the repository root: uv run python assets/diagrams/relu_curve.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent.parent / "relu_curve.png"

INK, GREY, NODE, AMBER = "#222222", "#6b6b6b", "#fdf6ec", "#c47612"
X_EXAMPLE = 2
# per hidden neuron: input weight w, bias b, output weight v, colour (the same colours as the plots in notebook 01)
NEURONS = [
    (+1, 0.0, 1.5, "#2f6fb0"),
    (-1, 0.0, 1.5, "#17a2b8"),
    (+1, -1.5, 3, "#7b52ab"),
    (-1, -1.5, 3, "#8c564b"),
]

fig = plt.figure(figsize=(11, 5.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1.1)
ax.set_ylim(-0.05, 0.51)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

X_IN, X_BOX, BOX_W, BOX_H = 0.07, 0.235, 0.29, 0.068
X_SUM, X_OUT, Y_MID, R = 0.79, 0.985, 0.215, 0.034
hidden_y = [0.395, 0.275, 0.155, 0.035]


def arrow(start, end, colour=GREY, width=1.4, zorder=2):
    """Draw a straight arrow between two points."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>,head_length=6,head_width=3.5", color=colour,
                                 linewidth=width, shrinkA=0, shrinkB=0, zorder=zorder))


def number(value):
    """Format a number without a trailing .0."""
    return f"{value:g}"


# input layer: one number x
ax.add_patch(Circle((X_IN, Y_MID), R, facecolor=NODE, edgecolor=INK, linewidth=1.6, zorder=3))
ax.text(X_IN, Y_MID, "$x$", fontsize=16, ha="center", va="center", zorder=4)
ax.text(X_IN, Y_MID - R - 0.012, f"one number\nhere $x = {X_EXAMPLE}$", fontsize=10, color=GREY, ha="center",
        va="top", linespacing=1.5)

# rule that every hidden neuron follows
ax.text(X_BOX + BOX_W / 2, 0.47, r"each hidden neuron:  $z = w \cdot x + b$,  then  $h = \mathrm{ReLU}(z)$",
        fontsize=11, color=INK, ha="center", va="center")

h_values = []
for y, (w, b, v, colour) in zip(hidden_y, NEURONS):
    arrow((X_IN + R, Y_MID), (X_BOX - 0.003, y))
    ax.add_patch(FancyBboxPatch((X_BOX, y - BOX_H / 2), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.01",
                                facecolor=NODE, edgecolor=colour, linewidth=1.8, zorder=3))

    # weighted sum with this neuron's numbers
    sign = "+" if w > 0 else "-"
    bias = f"+ {number(b)}" if b >= 0 else f"- {number(-b)}"
    ax.text(X_BOX + 0.012, y, f"$z = {sign}1 \\cdot x {bias}$", fontsize=12, color=INK, ha="left", va="center",
            zorder=4)

    # ReLU icon
    xr = X_BOX + BOX_W - 0.045
    arrow((X_BOX + 0.168, y), (xr - 0.026, y), GREY, 1.1, zorder=4)
    ax.add_patch(FancyBboxPatch((xr - 0.025, y - 0.025), 0.05, 0.05, boxstyle="round,pad=0,rounding_size=0.008",
                                facecolor="white", edgecolor=colour, linewidth=1.4, zorder=4))
    ax.plot([xr - 0.016, xr, xr + 0.014], [y - 0.01, y - 0.01, y + 0.013], color=colour, linewidth=1.8,
            solid_capstyle="round", zorder=5)

    # values for the example
    z = w * X_EXAMPLE + b
    h = max(0.0, z)
    h_values.append(h)
    note = f"$z = {number(z)}$,  $h = {number(h)}$" + ("  (ReLU sets it to 0)" if z < 0 else "")
    ax.text(X_BOX + 0.012, y - BOX_H / 2 - 0.011, note, fontsize=9.5, color=GREY, ha="left", va="center")

    # weighted connection to the output neuron
    end = (X_SUM - R - 0.004, Y_MID + (y - Y_MID) * 0.25)
    arrow((X_BOX + BOX_W, y), end, colour, 1.8)
    above = y > Y_MID - 0.1  # the label sits on the side of the arrow away from the line
    ax.text(X_BOX + BOX_W + 0.012, y + (0.006 if above else -0.008), f"× {number(v)}", fontsize=11, color=AMBER,
            ha="left", va="bottom" if above else "top", fontweight="bold", zorder=5)

# output layer: weighted sum of the four hidden outputs plus the output bias c
ax.add_patch(Circle((X_SUM, Y_MID), R, facecolor=NODE, edgecolor=INK, linewidth=1.6, zorder=3))
ax.text(X_SUM, Y_MID, "Σ", fontsize=17, color=INK, ha="center", va="center", zorder=4)
ax.add_patch(FancyBboxPatch((X_SUM - 0.035, 0.335), 0.07, 0.04, boxstyle="round,pad=0,rounding_size=0.008",
                            facecolor="white", edgecolor=AMBER, linewidth=1.4, zorder=3))
ax.text(X_SUM, 0.355, "$c = 0$", fontsize=11, color=AMBER, ha="center", va="center", zorder=4)
arrow((X_SUM, 0.335), (X_SUM, Y_MID + R + 0.003), AMBER, 1.6)
y_hat = sum(v * h for (_, _, v, _), h in zip(NEURONS, h_values))
terms = " + ".join(f"{number(v)} \\cdot {number(h)}" for (_, _, v, _), h in zip(NEURONS, h_values))
ax.text((X_SUM + X_OUT) / 2, 0.035, rf"$\hat{{y}} = {terms} + 0 = {number(y_hat)}$", fontsize=10, color=GREY,
        ha="center", va="center")

# prediction
arrow((X_SUM + R, Y_MID), (X_OUT - 0.055, Y_MID))
ax.add_patch(FancyBboxPatch((X_OUT - 0.055, Y_MID - 0.03), 0.11, 0.06, boxstyle="round,pad=0,rounding_size=0.01",
                            facecolor="white", edgecolor=INK, linewidth=1.4, zorder=3))
ax.text(X_OUT, Y_MID, rf"$\hat{{y}} = {number(y_hat)}$", fontsize=13, ha="center", va="center", zorder=4)
ax.text(X_OUT, Y_MID - 0.042, f"true value\n$x^2 = {X_EXAMPLE**2}$", fontsize=10, color=GREY, ha="center",
        va="top", linespacing=1.5)

# layer captions
for x, label in [(X_IN, "input layer"), (X_BOX + BOX_W / 2, "hidden layer: 4 ReLU neurons"),
                 (X_SUM, "output layer"), (X_OUT, "prediction")]:
    ax.text(x, -0.042, label, fontsize=10, color=GREY, ha="center", va="bottom")
ax.text((X_BOX + BOX_W + X_SUM - R) / 2, -0.042, "output weights $v$", fontsize=10, color=AMBER, ha="center",
        va="bottom")

fig.savefig(OUT, facecolor="white")
