"""Draw the two dot-product figures of notebook 01 from the X and the weights of the notebook.

- assets/numpy_dot_product_neuron.png, section 1: np.dot(X, weights) for one neuron with a weight column (3 × 1).
  Each example (row of X) has its own colour, and so does the result it produces, so the figure shows how every
  row of X times the weight column gives one weighted sum.
- assets/numpy_dot_product.png, section 3: np.dot(X, W) for a hidden layer of 4 neurons (3 × 4). Each neuron
  (column of W) has its own colour, the same as in hidden_layer_weights.png, so the figure shows how row 1 of X
  times each column gives one cell of the result. The other rows work the same way.

Both figures write the shape rule underneath: the inner dimensions must be equal and the outer dimensions give the
shape of the result.

Run from the repository root: uv run python assets/diagrams/numpy_dot_product.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle

ASSETS = Path(__file__).parent.parent
INK, GREY, AMBER, PLAIN = "#222222", "#6b6b6b", "#c47612", "#f4f4f4"
EXAMPLE_COLOURS = ["#2e9467", "#c2407e", "#3a6ea5"]  # one colour per example (row of X)
NEURON_COLOURS = ["#2f6fb0", "#17a2b8", "#7b52ab", "#8c564b"]  # one colour per neuron, as in hidden_layer_weights.png
X = np.array([[1.5, -0.5, 2.0], [0.5, 0.5, 1.0], [-1.0, 2.0, 0.5]])
WEIGHTS = np.array([0.4, 1.2, -0.3]).reshape(3, 1)
W = np.array([[0.4, -0.5, 0.3, 1.0], [1.2, 0.2, -0.4, 0.5], [-0.3, 0.1, 0.6, -0.2]])
CW, CH = 0.62, 0.42  # cell width and height


def number(value):
    """Format a number, with brackets around a negative one."""
    return f"({value:g})" if value < 0 else f"{value:g}"


def new_figure(height):
    """Create an empty figure whose data coordinates are in the units of the cells."""
    fig = plt.figure(figsize=(12, height), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def matrix(ax, x0, y_top, values, name, shape, fills, fmt="{:g}", mono=False, bold=None):
    """Draw a matrix of numbers with its name and shape; fills[i][j] is the colour of cell (i, j)."""
    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(Rectangle((x0 + j * CW, y_top - (i + 1) * CH), CW, CH, facecolor=fills[i][j],
                                   edgecolor="white", linewidth=2))
            ax.text(x0 + (j + 0.5) * CW, y_top - (i + 0.5) * CH, fmt.format(values[i, j]), fontsize=11,
                    color=INK, ha="center", va="center",
                    fontweight="bold" if bold is not None and bold[i][j] else "normal")
    centre = x0 + n_cols * CW / 2
    ax.text(centre, y_top + 0.42, name, fontsize=14, color=INK, ha="center", va="center",
            family="monospace" if mono else None)
    ax.text(centre, y_top + 0.14, shape, fontsize=10, color=GREY, ha="center", va="center")
    return x0 + n_cols * CW


def equation(ax, y_top, weights, weights_name, weights_shape, result_name, result_shape, fills_x, fills_w,
             fills_r, gap):
    """Draw X · weights = result, centred on the figure."""
    n_neurons = weights.shape[1]
    total = (3 + 2 * n_neurons) * CW + 2 * gap
    x = 6 - total / 2
    x = matrix(ax, x, y_top, X, "X", "3 examples × 3 inputs", fills_x)
    ax.text(x + gap / 2, y_top - 1.5 * CH, "·", fontsize=26, color=INK, ha="center", va="center")
    x = matrix(ax, x + gap, y_top, weights, weights_name, weights_shape, fills_w)
    ax.text(x + gap / 2, y_top - 1.5 * CH, "=", fontsize=20, color=INK, ha="center", va="center")
    matrix(ax, x + gap, y_top, np.dot(X, weights), result_name, result_shape, fills_r, fmt="{:.2f}", mono=True,
           bold=[[fill != PLAIN for fill in row] for row in fills_r])


def shape_rule(fig, ax, y_rule, n_neurons):
    """Write (3 × 3) · (3 × n) → (3 × n) with the two inner dimensions boxed and linked."""
    renderer = fig.canvas.get_renderer()
    x = 2.2
    inner_boxes = []
    for text, colour in [("(3 ×  ", GREY), ("3", AMBER), ("  )   ·   (  ", GREY), ("3", AMBER),
                         (f"  × {n_neurons})", GREY), (f"    →    (3 × {n_neurons})", GREY)]:
        label = ax.text(x, y_rule, text, fontsize=15, color=colour, ha="left", va="center",
                        fontweight="bold" if colour == AMBER else "normal")
        box = label.get_window_extent(renderer).transformed(ax.transData.inverted())
        if colour == AMBER:
            inner_boxes.append(box)
        x = box.x1  # the next piece starts where this one ends
    for box in inner_boxes:
        ax.add_patch(FancyBboxPatch((box.x0 - 0.08, y_rule - 0.22), box.width + 0.16, 0.44,
                                    boxstyle="round,pad=0,rounding_size=0.06", facecolor="none", edgecolor=AMBER,
                                    linewidth=1.6))
    left, right = [(b.x0 + b.x1) / 2 for b in inner_boxes]
    ax.annotate("", xy=(left, y_rule - 0.27), xytext=(right, y_rule - 0.27),
                arrowprops={"arrowstyle": "<->", "color": AMBER, "linewidth": 1.3,
                            "connectionstyle": "arc3,rad=0.5"})
    ax.text(x + 0.4, y_rule + 0.14, "inner dimensions must be equal", fontsize=11, color=AMBER, ha="left",
            va="center", fontweight="bold")
    ax.text(x + 0.4, y_rule - 0.18, "outer dimensions give the shape of the result", fontsize=11, color=GREY,
            ha="left", va="center")


def neuron_figure(out):
    """Section 1: every row of X times the weight column, one colour per example."""
    height = 5.4
    fig, ax = new_figure(height)
    rows = [to_rgba(colour, 0.25) for colour in EXAMPLE_COLOURS]
    equation(ax, 4.5, WEIGHTS, "weights", "3 inputs × 1 neuron", "np.dot(X, weights)", "3 examples × 1",
             fills_x=[[rows[i]] * 3 for i in range(3)], fills_w=[["#e6e6e6"]] * 3,
             fills_r=[[rows[i]] for i in range(3)], gap=1.8)
    for i, colour in enumerate(EXAMPLE_COLOURS):
        terms = " + ".join(f"{number(X[i, k])} · {number(WEIGHTS[k, 0])}" for k in range(3))
        ax.text(6, 2.55 - i * 0.38, f"row {i + 1} of X · weights  =  {terms}  =  {np.dot(X, WEIGHTS)[i, 0]:.2f}",
                fontsize=11.5, color=colour, ha="center", va="center")
    shape_rule(fig, ax, 0.6, 1)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def layer_figure(out):
    """Section 3: row 1 of X times every column of W, one colour per neuron."""
    height = 5.8
    fig, ax = new_figure(height)
    columns = [to_rgba(colour, 0.25) for colour in NEURON_COLOURS]
    first_row = to_rgba(INK, 0.12)
    equation(ax, 4.9, W, "W", "3 inputs × 4 neurons", "np.dot(X, W)", "3 examples × 4 neurons",
             fills_x=[[first_row] * 3] + [[PLAIN] * 3] * 2, fills_w=[columns] * 3,
             fills_r=[columns] + [[PLAIN] * 4] * 2, gap=0.6)
    result = np.dot(X, W)
    for j, colour in enumerate(NEURON_COLOURS):
        terms = " + ".join(f"{number(X[0, k])} · {number(W[k, j])}" for k in range(3))
        ax.text(6, 2.95 - j * 0.38, f"row 1 of X · column {j + 1} of W  =  {terms}  =  {result[0, j]:.2f}",
                fontsize=11.5, color=colour, ha="center", va="center")
    ax.text(6, 1.4, "Rows 2 and 3 of the result work the same way, with rows 2 and 3 of X.", fontsize=10.5,
            color=GREY, ha="center", va="center")
    shape_rule(fig, ax, 0.6, 4)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


neuron_figure(ASSETS / "numpy_dot_product_neuron.png")
layer_figure(ASSETS / "numpy_dot_product.png")
