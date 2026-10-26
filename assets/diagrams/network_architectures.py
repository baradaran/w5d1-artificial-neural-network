"""Draw the architectures of notebooks 03 and 04 as networks of neurons and connections.

- assets/feedforward_regression.png: 8 features, Normalization, three hidden layers of 50 ReLU neurons, 1 output.
- assets/forest_cover_classifier.png: 54 features, Normalization, hidden layers of 128 and 64 ReLU neurons,
  7 softmax outputs, one per tree species.
- assets/count_parameters_network.png: the network of notebook 03 for the "Count the Parameters" Try It in
  notebook 01, with layer sizes only: no parameter counts (they are the answer) and no Normalization step.

A layer with more neurons than fit on the page shows its first and last neurons with a ⋮ in between, and the
label under every layer gives its real size and its number of parameters, as in model.summary().

Run from the repository root: uv run python assets/diagrams/network_architectures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

ASSETS = Path(__file__).parent.parent
INK, GREY, LINE, NODE = "#222222", "#6b6b6b", "#c9c9c9", "#fdf6ec"
INPUT, HIDDEN, OUTPUT, NORM = "#6b6b6b", "#2f6fb0", "#c47612", "#2e9467"
SLOTS = 8  # places per layer: a larger layer shows its first and last neurons and a ⋮
R = 0.16  # neuron radius


def positions(n_neurons, height):
    """Return the y positions of the neurons drawn for a layer and the y of the ⋮ (or None).

    A layer of up to SLOTS neurons is drawn in full. A larger layer fills the SLOTS places with its first
    four neurons, a ⋮ and its last three neurons.
    """
    gap = height / (SLOTS - 1)
    slots = min(n_neurons, SLOTS)
    ys = [(slots - 1) * gap / 2 - k * gap for k in range(slots)]
    if n_neurons <= SLOTS:
        return ys, None
    return ys[:4] + ys[5:], ys[4]


def draw(layers, out, height=3.6, xlim=(-1.4, 12.4)):
    """Draw one network. layers: list of dicts with x, n, colour, label, detail and optional names."""
    fig = plt.figure(figsize=(12, 5.6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(-3.7, 2.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    for layer in layers:
        ys, dots = positions(layer["n"], height)
        layer["ys"], layer["dots"] = ys, dots

    # connections: every neuron of a layer to every neuron of the next (Normalization keeps one line per feature)
    for left, right in zip(layers, layers[1:]):
        if right.get("norm"):
            continue
        for y1 in left["ys"]:
            for y2 in right["ys"]:
                ax.plot([left["x"] + R, right["x"] - R], [y1, y2], color=LINE, linewidth=0.6, zorder=1)

    for layer in layers:
        x = layer["x"]
        if layer.get("norm"):  # one box over all features: each feature is scaled on its own
            top, bottom = max(layers[0]["ys"]), min(layers[0]["ys"])
            ax.add_patch(FancyBboxPatch((x - 0.32, bottom - 0.35), 0.64, top - bottom + 0.7,
                                        boxstyle="round,pad=0,rounding_size=0.12", facecolor="#eaf5ef",
                                        edgecolor=NORM, linewidth=1.6, zorder=2))
            ax.text(x, 0, "Normalization", rotation=90, fontsize=11, color=NORM, ha="center", va="center",
                    fontweight="bold", zorder=3)
            for y in layers[0]["ys"]:  # each feature passes straight through its own scaling
                ax.plot([layers[0]["x"] + R, x - 0.32], [y, y], color=LINE, linewidth=0.9, zorder=1)
            layer["ys"] = layers[0]["ys"]  # the next layer connects to every scaled feature
        else:
            for y in layer["ys"]:
                ax.add_patch(Circle((x, y), R, facecolor=NODE, edgecolor=layer["colour"], linewidth=1.6, zorder=3))
            if layer["dots"] is not None:
                ax.text(x, layer["dots"], "⋮", fontsize=18, color=layer["colour"], ha="center", va="center")
        for name, y in zip(layer.get("names", []), layer["ys"]):
            ha, dx = ("right", -R - 0.1) if layer is layers[0] else ("left", R + 0.1)
            ax.text(x + dx, y, name, fontsize=8.5, color=GREY, ha=ha, va="center")
        ax.text(x, -2.55, layer["label"], fontsize=10.5, color=layer["colour"], ha="center", va="center",
                fontweight="bold")
        ax.text(x, -3.05, layer["detail"], fontsize=9, color=GREY, ha="center", va="center", linespacing=1.4)

    # the connections from the normalised features to the first hidden layer, if there is a Normalization step
    norm = next((layer for layer in layers if layer.get("norm")), None)
    if norm is not None:
        first_hidden = layers[layers.index(norm) + 1]
        for y1 in norm["ys"]:
            for y2 in first_hidden["ys"]:
                ax.plot([norm["x"] + 0.32, first_hidden["x"] - R], [y1, y2], color=LINE, linewidth=0.6, zorder=1)

    fig.savefig(out, facecolor="white")
    plt.close(fig)


draw(
    [
        {"x": 0.9, "n": 8, "colour": INPUT, "label": "Input", "detail": "8 features",
         "names": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]},
        {"x": 2.4, "n": 8, "colour": NORM, "norm": True, "label": "Normalization", "detail": "mean 0, std 1\n17 non-trainable"},
        {"x": 4.4, "n": 50, "colour": HIDDEN, "label": "Hidden layer 1", "detail": "50 ReLU neurons\n8 × 50 + 50 = 450"},
        {"x": 6.4, "n": 50, "colour": HIDDEN, "label": "Hidden layer 2", "detail": "50 ReLU neurons\n50 × 50 + 50 = 2,550"},
        {"x": 8.4, "n": 50, "colour": HIDDEN, "label": "Hidden layer 3", "detail": "50 ReLU neurons\n50 × 50 + 50 = 2,550"},
        {"x": 10.4, "n": 1, "colour": OUTPUT, "label": "Output", "detail": "1 neuron, no activation\n50 × 1 + 1 = 51",
         "names": ["median house value"]},
    ],
    ASSETS / "feedforward_regression.png",
)
draw(
    [
        {"x": 0.9, "n": 54, "colour": INPUT, "label": "Input", "detail": "54 features",
         "names": ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Soil_Type_37", "Soil_Type_38", "Soil_Type_39"]},
        {"x": 2.6, "n": 54, "colour": NORM, "norm": True, "label": "Normalization", "detail": "mean 0, std 1\n109 non-trainable"},
        {"x": 4.9, "n": 128, "colour": HIDDEN, "label": "Hidden layer 1", "detail": "128 ReLU neurons\n54 × 128 + 128 = 7,040"},
        {"x": 7.2, "n": 64, "colour": HIDDEN, "label": "Hidden layer 2", "detail": "64 ReLU neurons\n128 × 64 + 64 = 8,256"},
        {"x": 9.5, "n": 7, "colour": OUTPUT, "label": "Output", "detail": "7 neurons, softmax\n64 × 7 + 7 = 455",
         "names": ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]},
    ],
    ASSETS / "forest_cover_classifier.png",
    xlim=(-2.6, 12.0),  # room for the long feature names on the left
)
draw(
    [
        {"x": 1.4, "n": 8, "colour": INPUT, "label": "Input", "detail": "8 features",
         "names": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]},
        {"x": 3.9, "n": 50, "colour": HIDDEN, "label": "Hidden layer 1", "detail": "50 ReLU neurons"},
        {"x": 6.1, "n": 50, "colour": HIDDEN, "label": "Hidden layer 2", "detail": "50 ReLU neurons"},
        {"x": 8.3, "n": 50, "colour": HIDDEN, "label": "Hidden layer 3", "detail": "50 ReLU neurons"},
        {"x": 10.5, "n": 1, "colour": OUTPUT, "label": "Output", "detail": "1 neuron", "names": ["house value"]},
    ],
    ASSETS / "count_parameters_network.png",
)
