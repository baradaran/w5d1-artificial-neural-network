"""Draw assets/hidden_layer_weights.png: a hidden layer of 4 neurons on 3 inputs, next to its weight matrix W.

Every connection from input k to neuron j is one weight w_kj. The connections into neuron j share that neuron's
colour, and so does column j of W, so the figure shows that a column holds all the weights of one neuron. The
connection from x2 to neuron 3 and the cell w_23 are highlighted as one worked example.

Run from the repository root: uv run python assets/diagrams/hidden_layer_weights.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent.parent / "hidden_layer_weights.png"

INK, GREY, NODE = "#222222", "#6b6b6b", "#fdf6ec"
# one colour per hidden neuron (the same colours as the ReLU neurons in notebook 01)
COLOURS = ["#2f6fb0", "#17a2b8", "#7b52ab", "#8c564b"]
HIGHLIGHT = (2, 3)  # input k, neuron j of the worked example w_23

fig = plt.figure(figsize=(11, 5), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 0.5)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

R = 0.03
X_IN, X_HID = 0.08, 0.40
input_y = [0.36, 0.25, 0.14]
hidden_y = [0.40, 0.29, 0.18, 0.07]

# connections: one weight per pair of input and neuron, coloured by the neuron it feeds
for k, yi in enumerate(input_y, start=1):
    for j, (yh, colour) in enumerate(zip(hidden_y, COLOURS), start=1):
        highlighted = (k, j) == HIGHLIGHT
        ax.plot([X_IN + R, X_HID - R], [yi, yh], color=colour, linewidth=3 if highlighted else 1.4,
                alpha=1 if highlighted else 0.55, zorder=2 if highlighted else 1, solid_capstyle="round")
k, j = HIGHLIGHT
ax.text(0.315, 0.218, "$w_{23}$", fontsize=13, color=COLOURS[j - 1], ha="center", va="center", fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=4)

# input layer
for k, y in enumerate(input_y, start=1):
    ax.add_patch(Circle((X_IN, y), R, facecolor=NODE, edgecolor=INK, linewidth=1.6, zorder=3))
    ax.text(X_IN, y, f"$x_{k}$", fontsize=14, ha="center", va="center", zorder=4)

# hidden layer: each neuron also has one bias
for j, (y, colour) in enumerate(zip(hidden_y, COLOURS), start=1):
    ax.add_patch(Circle((X_HID, y), R, facecolor=NODE, edgecolor=colour, linewidth=2.2, zorder=3))
    ax.text(X_HID, y, str(j), fontsize=14, color=colour, ha="center", va="center", fontweight="bold", zorder=4)
    ax.text(X_HID + R + 0.012, y, f"$+\\,b_{j}$", fontsize=11, color=colour, ha="left", va="center")

ax.text(X_IN, 0.015, "3 inputs", fontsize=10, color=GREY, ha="center", va="center")
ax.text(X_HID, 0.015, "4 neurons", fontsize=10, color=GREY, ha="center", va="center")

# weight matrix W: one row per input, one column per neuron
X0, Y_TOP, CW, CH = 0.66, 0.39, 0.085, 0.06
ax.text(X0 + 2 * CW, Y_TOP + 0.085, "4 columns: one per neuron", fontsize=10, color=GREY, ha="center",
        va="center")
for j, colour in enumerate(COLOURS, start=1):
    ax.text(X0 + (j - 0.5) * CW, Y_TOP + 0.035, f"neuron {j}", fontsize=10, color=colour, ha="center",
            va="center", fontweight="bold")
for k in range(1, 4):
    y = Y_TOP - k * CH
    ax.text(X0 - 0.015, y + CH / 2, f"$x_{k}$", fontsize=13, ha="right", va="center")
    for j, colour in enumerate(COLOURS, start=1):
        highlighted = (k, j) == HIGHLIGHT
        ax.add_patch(Rectangle((X0 + (j - 1) * CW, y), CW, CH, facecolor=to_rgba(colour, 0.35 if highlighted else
                                                                                0.12),
                               edgecolor=colour if highlighted else "white", linewidth=2.4 if highlighted else 1.5,
                               zorder=3 if highlighted else 2))
        ax.text(X0 + (j - 0.5) * CW, y + CH / 2, f"$w_{{{k}{j}}}$", fontsize=13, color=INK, ha="center",
                va="center", zorder=4)
ax.text(X0 - 0.06, Y_TOP - 1.5 * CH, "3 rows:\none per\ninput", fontsize=10, color=GREY, ha="right", multialignment="center",
        va="center", linespacing=1.4)
ax.text(X0 + 4 * CW + 0.012, Y_TOP - 1.5 * CH, "$W$", fontsize=16, color=INK, ha="left", va="center")

# biases: one per neuron, kept apart from W
y_b = Y_TOP - 3 * CH - 0.085
for j, colour in enumerate(COLOURS, start=1):
    ax.add_patch(Rectangle((X0 + (j - 1) * CW, y_b), CW, CH, facecolor=to_rgba(colour, 0.12), edgecolor="white",
                           linewidth=1.5))
    ax.text(X0 + (j - 0.5) * CW, y_b + CH / 2, f"$b_{j}$", fontsize=13, color=INK, ha="center", va="center")
ax.text(X0 - 0.015, y_b + CH / 2, "biases", fontsize=10, color=GREY, ha="right", va="center")

# parameter count
ax.add_patch(FancyBboxPatch((X0 - 0.03, 0.005), 4 * CW + 0.06, 0.05, boxstyle="round,pad=0,rounding_size=0.01",
                            facecolor="white", edgecolor=INK, linewidth=1.2))
ax.text(X0 + 2 * CW, 0.03, r"$3 \times 4 = 12$ weights $+\ 4$ biases $= 16$ parameters", fontsize=11,
        color=INK, ha="center", va="center")

fig.savefig(OUT, facecolor="white")
