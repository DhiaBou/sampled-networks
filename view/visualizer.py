import json
from itertools import zip_longest
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter, LogLocator

from dataset.dataset import Dataset
from models.base_model import BaseModel
from models.sampled_net import SampledNet
from utils.utilities import loss_mse, loss_r2


def visualize_data(Y_1, Y_2, Y1="Y 1", Y2="Y 2"):
    Y_1, Y_2 = np.array(Y_1), np.array(Y_2)
    n, d = len(Y_1), Y_1.shape[1]

    # Get the sorting indices for each dimension of Y_1
    sorting_indices = np.argsort(Y_1, axis=0)[::-1]

    # Adjusting the subplots for horizontal layout
    fig, axs = plt.subplots(1, d, figsize=(d * 5, 3))
    axs = axs.flatten() if d > 1 else [axs]

    for dim, ax in enumerate(axs):
        # Sort Y_1 and Y_2 based on the sorting indices for the current dimension
        Y_1_sorted = Y_1[sorting_indices[:, dim], dim]
        Y_2_sorted = Y_2[sorting_indices[:, dim], dim]

        ax.plot(range(n), Y_1_sorted, label=Y1)
        ax.plot(range(n), Y_2_sorted, label=Y2)
        ax.set_title(f"Dimension {dim + 1}")
        ax.legend()

    axs[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.show()


def plot_weight_biases_differences(weights1, weights2, biases1, biases2):
    weights1 = np.transpose(weights1)
    weights2 = np.transpose(weights2)
    print("are given weight matrices equal: ", np.array_equal(weights1, weights2))
    angles = map_to_angle_differences(weights1, weights2)
    norms = map_to_norm_ratio(weights1, weights2)

    # Get the indices that would sort the angles
    sorted_indices = np.argsort(angles)

    # Use the indices to order the angles and biases
    angles = angles[sorted_indices]
    biases1_sorted = np.array(biases1)[sorted_indices]
    biases2_sorted = np.array(biases2)[sorted_indices]
    norms = np.array(norms)[sorted_indices]

    x = np.arange(len(angles))
    label_angle = "Angle mean: {:0.2f}".format(float(np.mean(angles)))
    label_norm_of_difference = "Euclidean distance mean: {:0.2f}".format(float(np.mean(norms)))

    fig, ax2 = plt.subplots()

    ax2.plot(x, norms, linestyle='--', marker='o', label=label_norm_of_difference, color='#606060')
    ax2.set_ylabel("Ratio ||ŵᵢ|| / ||wᵢ||")
    ax2.tick_params(axis='y')
    ax2.legend(loc="upper left")
    # Plotting angles on the primary y-axis
    ax1 = ax2.twinx()
    ax1.plot(x, angles, marker="o", label=label_angle, color='#1f77b4')
    ax1.set_ylabel("Angle (degrees)")
    ax1.tick_params(axis='y')
    ax1.set_xlabel("Vector Index")
    ax1.set_title("Comparison of Weight Vector Differences to the Base Model")
    ax1.legend(loc="upper right")

    # Creating a secondary y-axis for norms

    # Grid and legend
    ax1.grid(True)
    fig.tight_layout()  # To make sure the right y-label is not clipped

    print(norms)
    # Plot sorted biases
    fig, ax = plt.subplots()
    ax.plot(x, biases1_sorted, label="bᵢ")
    ax.plot(x, biases2_sorted, label="b̂ᵢ")
    plt.xlabel("Vector Index (sorted by angle)")
    plt.ylabel("Bias Value")
    plt.title("Comparison of Bias Differences to the Base Model")
    plt.grid(True)
    ax.legend()
    plt.show()


def map_to_norm_ratio(weights1, weights2):
    return [np.linalg.norm(vec2) / np.linalg.norm(vec1) for vec1, vec2 in zip(weights1, weights2)]


def map_to_angle_differences(weights1, weights2):
    angles = np.degrees(
        [
            np.arccos(
                np.clip(
                    np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)),
                    -1.0,
                    1.0,
                )
            )
            for vec1, vec2 in zip(weights1, weights2)
        ]
    )
    return angles


def plot_vector_differences(weights1, weights2, biases_base, biases2_sampled, radii=[], losses=[]):
    fig, ax = plt.subplots()
    x = np.arange(len(biases_base))

    for w1, w2, b2, r, l in zip_longest(weights1, weights2, biases2_sampled, radii, losses):
        ww1 = np.transpose(w1)
        ww2 = np.transpose(w2)
        angles = map_to_angle_differences(ww1, ww2)
        angles = np.sort(angles)
        label = "Mean Angle: {:0.2f}".format(float(np.mean(angles)))
        if r is not None and l is not None:
            label = "r: {:0.2f}".format(float(r)) + " loss: {:0.3e}".format(float(l)) + " " + label
        ax.plot(x, angles, marker="o", label=label)

    plt.xlabel("Vector Index")
    plt.ylabel("Angle (degrees)")
    plt.title("Angles between corresponding vectors")
    plt.grid(True)
    ax.legend()
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(radii, [np.mean(np.abs(b2 - biases_base)) for b2 in biases2_sampled], color='#1f77b4', marker='o',
             linestyle='-', label='Bias difference mean')
    ax1.set_xlabel('radius')
    ax1.set_ylabel('Bias difference mean')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()

    ax2.plot(radii, losses, color='#ff7f0e', marker='o', linestyle='--', label='mse')

    ax2.set_ylabel('mse')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')

    plt.title('Comparison of mse loss and mean differences in biases against radii')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_vector_differences_biases(biases_base, biases2_sampled, radii=[], losses=[]):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(biases_base))
    sorted_indices = np.argsort(biases_base)
    biases_base = biases_base[sorted_indices]
    zorder = len(biases2_sampled)

    for b2, r, l in zip_longest(biases2_sampled, radii, losses):
        b2 = np.array(b2)[sorted_indices]
        label = ""
        if r is not None and l is not None:
            label = "r: {:0.2f}".format(float(r)) + " loss: {:0.3e}".format(float(l)) + " " + label
        ax.plot(x, b2, label=label, zorder=zorder)
        zorder -= 1

    ax.plot(x, biases_base, label="", color="black", zorder=len(biases2_sampled) + 1)

    plt.xlabel("Vector Index")
    plt.ylabel("Angle (degrees)")
    plt.title("Angles between corresponding vectors")
    plt.grid(True)
    ax.legend()
    plt.show()


def write_to_file(output_file, data):
    with open(output_file, "w") as f:
        f.write(json.dumps(data))


def read_from_file(input_file):
    with open(input_file, "r") as f:
        return json.loads(f.read())


def plot_loss_f_alpha_radius(losses):
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
    plt.scatter(X[:, 0], X[:, 1], s=10)  # plot points in X

    weight_norms = np.array([np.linalg.norm(w) for w in weights])
    sorted_indices = np.argsort(weight_norms)[::-1]

    if num_vectors > len(weights):
        num_vectors = len(weights)

    x_1_x2_tuples = [x_1_x2_tuples[i] for i in sorted_indices[:num_vectors]]
    weights = [weights[i] for i in sorted_indices[:num_vectors]]
    original_indices = sorted_indices[:num_vectors]

    for (x1, x2), w, original_index in zip(x_1_x2_tuples, weights, original_indices):
        import matplotlib.colors as mcolors

        colors_list = list(mcolors.TABLEAU_COLORS.values())

        color = colors_list[original_index % len(colors_list)]
        plt.scatter(x1[0], x1[1], color=color, s=5, marker='x')
        # Plot line segment
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color=color, linewidth=0.5)
        plt.scatter([x2[0]], [x2[1]], color=color, marker='x', s=5)

        offset_arrow = 0.1 * (w / np.linalg.norm(w))
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


def model_base_vs_model_sampled(dataset: Dataset, model_base: BaseModel, model_sampled: SampledNet,
                                x_1_X2_tuples: List = None):
    y_base_model_train = model_base.predict(dataset.X_train)
    y_base_model_test = model_base.predict(dataset.X_test)

    y_sampled_test = model_sampled.predict(dataset.X_test)
    y_sampled_train = model_sampled.predict(dataset.X_train)
    print(f"train: loss(y_base_model, y_sampled)")
    print(f"r2: {loss_r2(y_sampled_train, y_base_model_train)}\tmse: {loss_mse(y_sampled_train, y_base_model_train)}")
    print(f"test: loss(y_base_model, y_sampled)")
    print(f"r2: {loss_r2(y_sampled_test, y_base_model_test)}\tmse: {loss_mse(y_sampled_test, y_base_model_test)}")
    plot_weight_biases_differences(
        model_base.weights[0], model_sampled.weights[0], model_base.biases[0], model_sampled.biases[0]
    )
    if x_1_X2_tuples:
        plot_weight_vectors_and_point_pairs(dataset.X_train, x_1_X2_tuples, np.transpose(model_base.weights[0]), 30)
    visualize_data(y_base_model_train, y_sampled_train, "y_base_model_train", "y_sampled_train")
    visualize_data(y_base_model_test, y_sampled_test, "y_base_model_test", "y_sampled_test")
