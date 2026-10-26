"""Draw assets/training_cycle.png: the four steps of training around the ReLU network of notebook 01.

The cycle layout follows the first figure of the Spiced Academy ds-deep-learning
backpropagation notebook. Run from the repository root: uv run python assets/diagrams/training_cycle.py
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent.parent / "training_cycle.png"

BLUE, GREEN, ORANGE, RED = "#2f6fb0", "#2e9467", "#d9821e", "#c4423f"
INK, GREY, NODE = "#222222", "#6b6b6b", "#fdf6ec"

fig = plt.figure(figsize=(9, 9), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# --- the cycle: four arcs between the step boxes, clockwise
R, CX, CY = 0.335, 0.5, 0.5
arcs = [(120, 60, BLUE), (30, -30, GREEN), (-60, -120, ORANGE), (210, 150, RED)]
for start, end, colour in arcs:
    a, b = math.radians(start), math.radians(end)
    p_start = (CX + R * math.cos(a), CY + R * math.sin(a))
    p_end = (CX + R * math.cos(b), CY + R * math.sin(b))
    ax.add_patch(
        FancyArrowPatch(
            p_start,
            p_end,
            connectionstyle="arc3,rad=-0.27",
            arrowstyle="-|>,head_length=12,head_width=7",
            color=colour,
            linewidth=4,
            zorder=1,
        )
    )

# --- the step boxes in the corners
steps = [
    (0.02, 0.825, BLUE, "1. Forward pass", "Send x through the layers\nto compute the prediction ŷ."),
    (0.60, 0.825, GREEN, "2. Loss", "Compare ŷ with the true value y,\nfor example with the MSE."),
    (0.60, 0.025, ORANGE, "3. Backward pass", "Apply the chain rule from the loss\nbackwards to get every gradient."),
    (0.02, 0.025, RED, "4. Update", "Move every weight and bias\na small step against its gradient."),
]
for x, y, colour, title, text in steps:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.38,
            0.15,
            boxstyle="round,pad=0,rounding_size=0.018",
            facecolor=colour + "14",
            edgecolor=colour,
            linewidth=1.6,
            zorder=2,
        )
    )
    ax.add_patch(Rectangle((x, y + 0.012), 0.008, 0.126, color=colour, zorder=3))
    ax.text(x + 0.03, y + 0.108, title, fontsize=14, fontweight="bold", color=colour, va="center", zorder=3)
    ax.text(x + 0.03, y + 0.05, text, fontsize=10.5, color=INK, va="center", linespacing=1.45, zorder=3)

# --- the network: x -> 4 ReLU neurons -> linear output -> y_hat
X_IN, X_HID, X_OUT, X_PRED = 0.28, 0.45, 0.62, 0.735
hidden_y = [0.635, 0.545, 0.455, 0.365]
r_node = 0.032

# weight bands
ax.add_patch(Rectangle((0.345, 0.335), 0.04, 0.33, color="#e86fa8", alpha=0.18, zorder=3))
ax.add_patch(Rectangle((0.515, 0.335), 0.04, 0.33, color="#f2a43a", alpha=0.22, zorder=3))
ax.text(0.365, 0.678, "w, b", fontsize=12, color="#c2407e", ha="center", fontweight="bold", zorder=5)
ax.text(0.535, 0.678, "v, c", fontsize=12, color="#c47612", ha="center", fontweight="bold", zorder=5)

# connections
for hy in hidden_y:
    ax.plot([X_IN, X_HID], [0.5, hy], color=GREY, linewidth=1.2, zorder=4)
    ax.plot([X_HID, X_OUT], [hy, 0.5], color=GREY, linewidth=1.2, zorder=4)
ax.add_patch(
    FancyArrowPatch((X_OUT + r_node, 0.5), (X_PRED - 0.03, 0.5), arrowstyle="-|>,head_length=6,head_width=3.5",
                    color=GREY, linewidth=1.2, zorder=4)
)


def node(x, y, edge):
    """Draw one round node with a coloured outline."""
    ax.add_patch(Circle((x, y), r_node, facecolor=NODE, edgecolor=edge, linewidth=1.8, zorder=6))


# input node
node(X_IN, 0.5, INK)
ax.text(X_IN, 0.5, "$x$", fontsize=15, ha="center", va="center", zorder=7)

# hidden ReLU neurons with a small ReLU sketch inside
for hy in hidden_y:
    node(X_HID, hy, BLUE)
    ax.plot([X_HID - 0.017, X_HID, X_HID + 0.015], [hy - 0.008, hy - 0.008, hy + 0.012], color=BLUE,
            linewidth=1.6, zorder=7, solid_capstyle="round")

# output neuron: a weighted sum
node(X_OUT, 0.5, BLUE)
ax.text(X_OUT, 0.5, "Σ", fontsize=15, ha="center", va="center", color=BLUE, zorder=7)

# prediction
ax.add_patch(FancyBboxPatch((X_PRED - 0.03, 0.475), 0.06, 0.05, boxstyle="round,pad=0,rounding_size=0.012",
                            facecolor="white", edgecolor=INK, linewidth=1.4, zorder=6))
ax.text(X_PRED, 0.5, r"$\hat{y}$", fontsize=15, ha="center", va="center", zorder=7)

# layer captions
for x, label in [(X_IN, "input"), (X_HID, "hidden layer\nReLU"), (X_OUT, "output\nlinear"), (X_PRED, "prediction")]:
    ax.text(x, 0.322, label, fontsize=9, color=GREY, ha="center", va="top", linespacing=1.2, zorder=5)

# forward and backward pass arrows
ax.add_patch(FancyArrowPatch((0.30, 0.735), (0.70, 0.735), arrowstyle="-|>,head_length=8,head_width=4.5",
                             color=BLUE, linewidth=2, zorder=5))
ax.text(0.5, 0.75, "forward pass", fontsize=11, color=BLUE, ha="center", va="bottom", fontweight="bold")
ax.add_patch(FancyArrowPatch((0.70, 0.262), (0.30, 0.262), arrowstyle="-|>,head_length=8,head_width=4.5",
                             color=ORANGE, linewidth=2, zorder=5))
ax.text(0.5, 0.233, "backward pass", fontsize=11, color=ORANGE, ha="center", va="top", fontweight="bold")

# numbered backward steps I to V (the rows of the chain-rule table), placed under the part of the
# network each one belongs to, in the colours of the table
PINK, AMBER = "#c2407e", "#c47612"
badges = [(0.68, "I", GREEN), (0.585, "II", AMBER), (0.515, "III", AMBER), (0.45, "IV", BLUE), (0.365, "V", PINK)]
for x, numeral, colour in badges:
    ax.add_patch(Circle((x, 0.262), 0.019, facecolor=colour, edgecolor="white", linewidth=1.5, zorder=8))
    ax.text(x, 0.2615, numeral, fontsize=9, color="white", fontweight="bold", ha="center", va="center", zorder=9)

fig.savefig(OUT, facecolor="white")
