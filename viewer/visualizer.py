from matplotlib import pyplot as plt
import numpy as np
from dataset import *
from models.sampled_net import *
from logic import *
from matplotlib.ticker import LogFormatter, LogLocator
import csv
import json
import matplotlib.colors as mcolors
from itertools import zip_longest


def visualize_data(Y_1, Y_2, Y1="Y 1", Y2="Y 2"):
    Y_1, Y_2 = np.array(Y_1), np.array(Y_2)
    n, d = len(Y_1), Y_1.shape[1]

    # Get the sorting indices for each dimension of Y_1
    sorting_indices = np.argsort(Y_1, axis=0)[::-1]

    fig, axs = plt.subplots(d, 1, figsize=(8, d * 4))
    axs = axs.flatten() if d > 1 else [axs]

    for dim, ax in enumerate(axs):
        # Sort Y_1 and Y_2 based on the sorting indices for the current dimension
        Y_1_sorted = Y_1[sorting_indices[:, dim], dim]
        Y_2_sorted = Y_2[sorting_indices[:, dim], dim]

        ax.plot(range(n), Y_2_sorted, label=Y2, color="orange")
        ax.plot(range(n), Y_1_sorted, label=Y1, color="blue")
        ax.set_ylabel(f"Dimension {dim + 1}")
        ax.legend()

    axs[-1].set_xlabel("Sample")
    plt.show()


def plot_weight_biases_differences(weights1, weights2, biases1, biases2):
    weights1 = np.transpose(weights1)
    weights2 = np.transpose(weights2)
    angles = np.degrees(
        [
            np.arccos(
                np.clip(
                    np.dot(vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)),
                    -1.0,
                    1.0,
                )
            )
            for vec1, vec2 in zip(weights1, weights2)
        ]
    )

    # Get the indices that would sort the angles
    sorted_indices = np.argsort(angles)

    # Use the indices to order the angles and biases
    angles = angles[sorted_indices]
    biases1_sorted = np.array(biases1)[sorted_indices]
    biases2_sorted = np.array(biases2)[sorted_indices]

    x = np.arange(len(angles))

    # Plot angles between corresponding vectors
    plt.figure()
    plt.plot(x, angles, marker="o")
    plt.xlabel("Vector Index")
    plt.ylabel("Angle (degrees)")
    plt.title("Angles between corresponding vectors")
    plt.grid(True)
    plt.show()

    # Plot sorted biases
    fig, ax = plt.subplots()
    ax.plot(x, biases1_sorted, label="biases1")
    ax.plot(x, biases2_sorted, label="biases2")
    plt.xlabel("Vector Index (sorted by angle)")
    plt.ylabel("Bias Value")
    plt.title("Biases from biases1 and biases2 (sorted by angle)")
    plt.grid(True)
    ax.legend()
    plt.show()


def plot_vector_differences(weights1, weights2, radii=[], losses=[]):
    fig, ax = plt.subplots()

    for w1, w2, r, l in zip_longest(weights1, weights2, radii, losses):
        ww1 = np.transpose(w1)
        ww2 = np.transpose(w2)
        angles = np.degrees(
            [
                np.arccos(
                    np.clip(
                        np.dot(vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)),
                        -1.0,
                        1.0,
                    )
                )
                for vec1, vec2 in zip(ww1, ww2)
            ]
        )
        angles = np.sort(angles)
        x = np.arange(len(angles))
        label = "Mean: {:0.2f}".format(float(np.mean(angles)))
        if r is not None and l is not None:
            label = "r: {:0.2f}".format(float(r)) + " loss: {:0.3e}".format(float(l)) + " " + label
        ax.plot(x, angles, marker="o", label=label)

    plt.xlabel("Vector Index")
    plt.ylabel("Angle (degrees)")
    plt.title("Angles between corresponding vectors")
    plt.grid(True)
    ax.legend()
    plt.show()


def write_to_file(output_file, data):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([str(data)])


def write_to_file(output_file, data):
    with open(output_file, "w") as f:
        f.write(json.dumps(data))


def read_from_file(input_file):
    with open(input_file, "r") as f:
        return json.loads(f.read())


def plot_loss_vs_alpha_radius(losses):
    fig, ax = plt.subplots()
    for radius, inner_dict in losses["sampled_net"].items():
        x = list(inner_dict.keys())
        x = [float(s) for s in x]
        y = list(inner_dict.values())

        radius_str = "{:0.2f}".format(float(radius))
        ax.plot(x, y, marker="o", label=f"Radius: {radius_str}")

    ax.axhline(y=losses["given_model"], color="r", linestyle="--", label="loss given_model")

    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("MSE")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatter())
    ax.legend()

    plt.show()


def plot_loss_f_num_samples(losses):
    x = [int(k) for k in losses.keys()]
    x_ticks = x
    x = np.log10(x)

    y1 = [v["mse"] for v in losses.values()]
    y2 = [v["r2"] for v in losses.values()]

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("MSE", color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("R2", color=color)
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_ticks)

    fig.tight_layout()
    plt.show()


def plot_weight_vectors_and_point_pairs(X, x_1_x2_tuples, weights, num_vectors=np.inf):
    X = np.array(X)
    plt.scatter(X[:, 0], X[:, 1])  # plot points in X
    from matplotlib import cm

    weight_norms = np.array([np.linalg.norm(w) for w in weights])
    sorted_indices = np.argsort(weight_norms)[::-1]

    if num_vectors > len(weights):
        num_vectors = len(weights)

    # Select the num_vectors largest weights and corresponding points
    x_1_x2_tuples = [x_1_x2_tuples[i] for i in sorted_indices[:num_vectors]]
    weights = [weights[i] for i in sorted_indices[:num_vectors]]
    original_indices = sorted_indices[:num_vectors]

    for (x1, x2), w, original_index in zip(x_1_x2_tuples, weights, original_indices):
        import matplotlib.colors as mcolors

        colors_list = list(mcolors.TABLEAU_COLORS.values())

        color = colors_list[original_index % len(colors_list)]
        plt.scatter(x1[0], x1[1], color=color)
        # Plot line segment
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color=color, linewidth=0.5)

        w_norm = np.linalg.norm(w)
        x_1_x_2_norm = np.linalg.norm(np.array(x2) - np.array(x1))
        w = w * 0.5 * x_1_x_2_norm / w_norm
        offset_arrow = 0.03 * (w / np.linalg.norm(w))
        # Plot arrow
        plt.arrow(x1[0], x1[1], w[0], w[1], head_width=0.02, color=color, linewidth=0.5)
        plt.text(
            x1[0] + w[0] + offset_arrow[0],
            x1[1] + w[1] + offset_arrow[1],
            str(original_index + 1),
            color="black",
            alpha=0.8,
            fontsize=7,
            ha="center",
            va="center",
        )

    plt.show()
