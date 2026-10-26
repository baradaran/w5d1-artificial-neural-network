# Artificial Neural Networks

A neural network is built from one simple unit, the **neuron**: it multiplies its inputs by weights, adds a bias and passes the result through an activation function. This repository builds that idea up step by step. You compute and train a single neuron by hand with NumPy, run backpropagation through a small network, then hand the same work to Keras and train feed-forward networks on two real tabular datasets, one for regression and one for classification.

## Learning Objectives

By the end of this repository, you should be able to:

- Compute the output of a single neuron by hand and train it with gradient descent written in NumPy.
- Calculate the gradients of a small ReLU network with backpropagation and use them to train it.
- Build, compile, train, evaluate and save a Keras `Sequential` model, and calculate the parameters of each `Dense` layer.
- Train feed-forward networks with Keras for regression and for multiclass classification, and compare them with linear baselines on a validation set.
- Plot learning curves and identify the most confused classes in a confusion matrix.

## Learning Path

Work through the notebooks in this order:

| File / Folder | Description |
|---|---|
| [**01 - How a Neuron Learns: Gradient Descent and Backpropagation**](01_neurons_and_backpropagation.ipynb) | Compute a neuron with NumPy, train it with gradient descent, build a curve from ReLU neurons and train a small network with backpropagation. |
| [**02 - The Keras Workflow: Configure, Compile, Fit, Evaluate, Predict**](02_keras_workflow.ipynb) | Follow the five Keras steps on a two-feature bowl, count the parameters of each layer, and save and reload the trained model. |
| [**03 - Regression with a Feed-Forward Network**](03_regression_with_neural_networks.ipynb) | Predict California house prices with a feed-forward network that standardises its inputs with a `Normalization` layer, and compare it with linear regression. |
| [**04 - Multiclass Classification with a Feed-Forward Network**](04_classification_with_neural_networks.ipynb) | Predict one of seven tree species from terrain data with a softmax classifier, compare it with two baselines and read a confusion matrix. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Solutions**](solutions/) | Worked solutions for all four notebooks. |
| [**Assets**](assets/) | The figures used in the notebooks. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a
> **placeholder**. Replace it, including the `< >` brackets, with your own
> value. For example, `cd <repo-name>` becomes `cd amle-artificial-neural-network`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies, including TensorFlow and Keras, and creates a virtual environment in `.venv/`. TensorFlow is a large package, so the first install takes a few minutes.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [**3Blue1Brown: Neural networks**](https://www.3blue1brown.com/topics/neural-networks): Visual explanations of neurons, gradient descent and backpropagation.
- [**Neural Networks and Deep Learning**](http://neuralnetworksanddeeplearning.com/): Michael Nielsen's free online book, with a full chapter on how backpropagation works.
- [**CS231n: Minimal Neural Network Case Study**](https://cs231n.github.io/neural-networks-case-study/): A complete network written from scratch in NumPy, with a softmax output and the log loss.
- [**Deep Learning with Python**](https://deeplearningwithpython.io/): François Chollet's book on Keras, free to read online.
- [**Keras: The Sequential model**](https://keras.io/guides/sequential_model/): The official guide to building models layer by layer.
- [**Introduction to Keras for engineers**](https://keras.io/getting_started/intro_to_keras_for_engineers/): A short tour of the Keras 3 API, from layers to training.
- [**TensorFlow tutorial: Basic regression**](https://www.tensorflow.org/tutorials/keras/regression): A regression network on real data with a `Normalization` layer, the same recipe as notebook 03.
- [**Keras example: Structured data classification from scratch**](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/): A classifier for a real tabular dataset, from raw columns to predictions.
- [**Covertype dataset**](https://archive.ics.uci.edu/dataset/31/covertype): The UCI page of the forest cover data used in notebook 04.
