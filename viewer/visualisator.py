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

colors_list = list(mcolors.TABLEAU_COLORS.values())


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

        ax.plot(range(n), Y_1_sorted, label=Y1)
        ax.plot(range(n), Y_2_sorted, label=Y2)
        ax.set_ylabel(f"Dimension {dim + 1}")
        ax.legend()

    axs[-1].set_xlabel("Sample")
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

    # # Calculate and plot histogram of angle ranges
    # range_start = 0
    # range_end = 90
    # bin_size = 10

    # hist, bins = np.histogram(angles, bins=np.arange(range_start, range_end + bin_size, bin_size))

    # plt.figure()
    # plt.bar(bins[:-1], hist, width=bin_size, align="edge", edgecolor="black")
    # plt.xlabel("Angle Range")
    # plt.ylabel("Count")
    # plt.title("Number of Angles in Each Range")
    # plt.xticks(np.arange(range_start, range_end, bin_size))
    # plt.grid(True)
    # mean_angle = np.mean(angles)
    # sd_angle = np.std(angles)

    # plt.annotate(
    #     f"Mean: {mean_angle:.2f}\nSD: {sd_angle:.2f}",
    #     xy=(1, 0),
    #     xycoords="axes fraction",
    #     xytext=(-20, -20),
    #     textcoords="offset points",
    #     horizontalalignment="right",
    #     verticalalignment="top",
    #     fontsize=12,
    # )

    # plt.show()


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


def plot_weight_vectors_and_point_pairs(X, x_1_x2_tuples, weights):
    X = np.array(X)
    plt.scatter(X[:, 0], X[:, 1])  # plot points in X
    from matplotlib import cm

    for (x1, x2), w in zip(x_1_x2_tuples, weights):
        import matplotlib.colors as mcolors
        import random

        color = random.choice(colors_list)
        plt.scatter(x1[0], x1[1], color=color)
        # Plot line segment
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color=color, linewidth=0.5)

        w_norm = np.linalg.norm(w)
        x_1_x_2_norm = np.linalg.norm(np.array(x2) - np.array(x1))
        w = w * 0.5 * x_1_x_2_norm / w_norm

        # Plot arrow
        plt.arrow(x1[0], x1[1], w[0], w[1], head_width=0.02, color=color, linewidth=0.5)

    plt.show()
