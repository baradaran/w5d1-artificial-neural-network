"""Draw assets/gradient_descent.png: the four training steps and gradient descent on a parabola-shaped loss.

Run from the repository root: uv run python assets/diagrams/gradient_descent.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent.parent / "gradient_descent.png"

BLUE, GREEN, ORANGE, RED = "#2f6fb0", "#2e9467", "#d9821e", "#c4423f"
INK, GREY = "#222222", "#6b6b6b"

W_MIN, LEARNING_RATE, W_START, N_STEPS = 2.0, 0.2, -1.2, 5


def loss(w):
    """A parabola-shaped loss with its minimum at w = 2."""
    return (w - W_MIN) ** 2 + 1


def gradient(w):
    """The slope of the loss at w."""
    return 2 * (w - W_MIN)


# the steps of gradient descent
ws = [W_START]
for _ in range(N_STEPS):
    ws.append(ws[-1] - LEARNING_RATE * gradient(ws[-1]))

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=200)
w_grid = np.linspace(-1.6, 4.0, 300)
ax.plot(w_grid, loss(w_grid), color=INK, linewidth=2.2, zorder=1)


def badge(x, y, number, colour):
    """Draw a numbered circle in the colour of a training step."""
    ax.text(x, y, str(number), fontsize=10, fontweight="bold", color="white", ha="center", va="center", zorder=8,
            bbox={"boxstyle": "circle,pad=0.3", "facecolor": colour, "edgecolor": "white", "linewidth": 1.2})


def step_label(x, y, number, colour, text, ha="left"):
    """Write a step label with its numbered badge in front of it."""
    badge(x, y, number, colour)
    ax.text(x + 0.17, y, text, fontsize=10.5, color=colour, ha=ha, va="center", zorder=8)


w0, L0 = ws[0], loss(ws[0])

# 1. predict with the current weight w0
ax.plot([w0, w0], [0, L0], color=BLUE, linestyle="--", linewidth=1.3, zorder=1)
step_label(-2.8, 2.2, 1, BLUE, "predict $\\hat{y}$ with\nthe current\nweight $w_0$")

# 2. measure the loss: the height of the point on the curve
ax.plot([-2.9, w0], [L0, L0], color=GREEN, linestyle="--", linewidth=1.3, zorder=1)
step_label(-2.8, L0 + 0.55, 2, GREEN, "loss $L(w_0)$")

# 3. compute the gradient: the slope of the tangent
tangent_w = np.linspace(w0 - 0.65, w0 + 0.65, 2)
ax.plot(tangent_w, L0 + gradient(w0) * (tangent_w - w0), color=ORANGE, linewidth=2, zorder=2)
ax.annotate("", xy=(w0 + 0.45, L0 + gradient(w0) * 0.45), xytext=(-0.3, 13.35),
            arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 1})
step_label(-0.3, 13.65, 3, ORANGE, "gradient $\\partial L / \\partial w$ = slope of the tangent")
ax.text(-0.13, 13.0, "negative here, so the step goes right", fontsize=10, color=ORANGE, va="center")

# 4. update: the steps as arrows hopping along the curve
for a, b in zip(ws[:-1], ws[1:]):
    ax.add_patch(
        FancyArrowPatch(
            (a, loss(a)),
            (b, loss(b)),
            connectionstyle="arc3,rad=-0.35",
            arrowstyle="-|>,head_length=7,head_width=4",
            color=RED,
            linewidth=1.8,
            shrinkA=6,
            shrinkB=6,
            zorder=3,
        )
    )
step_label(0.55, 7.4, 4, RED, "update: step $= -\\eta \\, \\partial L / \\partial w$")
ax.text(1.62, 3.7, "repeat steps 1 to 4\nfrom $w_1$, $w_2$, ...", fontsize=10, color=GREY, ha="left", va="center")

for k, w in enumerate(ws):
    ax.scatter(w, loss(w), s=70, color=BLUE, edgecolor="white", linewidth=1.5, zorder=4)
    if k < 4:
        ax.text(w + (0.18 if k == 0 else 0), loss(w) - 0.6, f"$w_{k}$", fontsize=12, color=BLUE,
                ha="left" if k == 0 else "center", va="top")

# the minimum
ax.scatter(W_MIN, loss(W_MIN), s=90, marker="*", color=GREEN, zorder=5)
ax.text(W_MIN + 0.12, loss(W_MIN) - 0.25, "minimum", fontsize=11, color=GREEN, ha="left", va="top")

# the four steps with their formulas
X0, X1, Y0, Y1 = 2.75, 5.65, 5.6, 14.4
ax.add_patch(
    FancyBboxPatch((X0, Y0), X1 - X0, Y1 - Y0, boxstyle="round,pad=0,rounding_size=0.15",
                   facecolor="#f7f7f7", edgecolor=GREY, linewidth=1.2, zorder=5)
)
ax.text((X0 + X1) / 2, Y1 - 0.45, "One training step", fontsize=12, fontweight="bold", color=INK, ha="center",
        va="center", zorder=6)
formulas = [
    (1, BLUE, "Predict", r"$\hat{y}_i = w \, x_i + b$"),
    (2, GREEN, "Loss (MSE)", r"$L = \dfrac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$"),
    (3, ORANGE, "Gradient", r"$\dfrac{\partial L}{\partial w} = \dfrac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \, x_i$"),
    (4, RED, "Update", r"$w_{\mathrm{new}} = w_{\mathrm{old}} - \eta \, \dfrac{\partial L}{\partial w}$"),
]
for row, (number, colour, title, formula) in enumerate(formulas):
    y = Y1 - 1.3 - row * 1.9
    badge(X0 + 0.22, y + 0.3, number, colour)
    ax.text(X0 + 0.42, y + 0.3, title, fontsize=10.5, fontweight="bold", color=colour, va="center", zorder=6)
    ax.text(X0 + 0.42, y - 0.62, formula, fontsize=12, color=INK, va="center", zorder=6)
ax.text(X0 + 0.42, Y0 + 0.32, r"$\eta$: learning rate (step size)", fontsize=9.5, color=GREY, va="center", zorder=6)

ax.set_xlim(-2.9, 5.75)
ax.set_ylim(0, 14.5)
ax.set_xlabel("weight $w$", fontsize=12)
ax.set_ylabel("loss $L(w)$", fontsize=12)
ax.set_xticks([])
ax.set_yticks([])
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, facecolor="white")
