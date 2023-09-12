import json
from itertools import zip_longest, count
from typing import List

import matplotlib
import matplotlib_inline
from matplotlib.lines import Line2D

from dataset.dataset import Dataset
from models.base_model import BaseModel
from models.sampled_net import SampledNet
from utils.utilities import loss_mse, loss_r2

save_to_pgf = False
path_to_histogram_directory = "C:/Users/daydo/Downloads/tum-thesis-latex-master-4/tum-thesis-latex-master/histograms/histogram"
matplotlib.rcParams['legend.fontsize'] = 'small'

if save_to_pgf:
    matplotlib_inline.backend_inline.set_matplotlib_formats('png')
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

counter = count(start=1)  # Creates a counter starting at 1


def save_fig(plt, txt=""):
    if save_to_pgf:
        plt.savefig(
            path_to_histogram_directory + str(next(counter)) + txt + ".pgf")


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
        mse = loss_mse(Y_1_sorted, Y_2_sorted)
        r2 = loss_r2(Y_1_sorted, Y_2_sorted)

        ax.plot(range(n), Y_1_sorted, label=Y1)
        ax.plot(range(n), Y_2_sorted, label=Y2)
        ax.text(0.05, 0.95, f'MSE: {mse:.4e}\n$R^2$: {r2:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        ax.set_title(f"Dimension {dim + 1}")
        ax.legend()

    axs[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.grid(True)

    plt.show()
    save_fig(plt, "visualize_data")


def plot_weight_biases_differences(weights1, weights2, biases1, biases2):
    weights1 = np.transpose(weights1)
    weights2 = np.transpose(weights2)
    print("are given weight matrices equal: ", np.array_equal(weights1, weights2))
    angles = map_to_angle_differences(weights1, weights2)
    # norms = map_to_norm_ratio(weights1, weights2)
    normsw1 = np.array([np.linalg.norm(w) for w in weights1])
    normsw2 = np.array([np.linalg.norm(w) for w in weights2])

    sorted_indices = np.argsort(angles)

    angles = angles[sorted_indices]
    biases1_sorted = np.array(biases1)[sorted_indices]
    biases2_sorted = np.array(biases2)[sorted_indices]
    normsw1 = np.array(normsw1)[sorted_indices]
    normsw2 = np.array(normsw2)[sorted_indices]

    x = np.arange(len(angles))
    label_angle = "Angle mean: {:0.2f}".format(float(np.mean(angles)))

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [8, 5]})

    weight_norms_MAE = np.mean(np.abs(normsw1 - normsw2))
    correlation = np.corrcoef(normsw1, normsw2)[0, 1]
    print(f"Correlation{correlation:.4f}")
    ax1.plot(x, angles, marker="o", label=label_angle, color='#606060', markersize=3)
    ax1.set_ylabel("Angle (degrees)")
    ax1.tick_params(axis='y', labelright=True, labelleft=False)
    ax1.set_xlabel("Node Index $i$ (sorted by weights' angle)")
    ax1.set_title("Weight differences")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, normsw1, label=r"$\| w_{1,i}  \|$", color='#1f77b4')
    ax2.plot(x, normsw2, label=r"$\| \hat{w}_{1,i}  \|$", color='#ff7f0e')
    ax2.plot([], [], alpha=0, label=f"MAE={weight_norms_MAE:.3f}")
    ax2.set_ylabel(r"Weight Norm")
    ax2.legend(loc="upper left")

    ax2.tick_params(axis='y', labelleft=True, labelright=False)
    ax2.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")
    ax1.legend(loc="upper right")
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")

    bias_MAE = np.mean(np.abs(biases1_sorted - biases2_sorted))
    ax3.plot(x, biases1_sorted, label="$b_{1,i}$")
    ax3.plot(x, biases2_sorted, label="$\hat{b}_{1,i}$")
    ax3.plot([], [], alpha=0, label=f"MAE={bias_MAE:.3f}")
    ax3.set_xlabel("Node Index $i$ (sorted by weights' angle)")
    ax3.set_ylabel("Bias Value")
    ax3.set_title("Bias differences")
    ax3.grid(True)
    ax3.legend()

    fig.tight_layout()
    plt.show()
    save_fig(plt, "plot_weight_biases_differences")

    # label_norm_of_difference = "Euclidean distance mean: {:0.2f}".format(float(np.mean(normsw1 - normsw2)))
    # norms = np.array(norms)[sorted_indices]
    # ax2.plot(x, norms, linestyle='--', marker='o', label=label_norm_of_difference, color='#606060')
    # if not any(n > 2 for n in norms):
    #     ax2.set_ylim(top=2)


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
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(5, 7.5))
    x = np.arange(len(biases_base))

    for w1, w2, b2, r, l in zip_longest(weights1, weights2, biases2_sampled, radii, losses, fillvalue=None):
        ww1 = np.transpose(w1)
        ww2 = np.transpose(w2)
        angles = map_to_angle_differences(ww1, ww2)  # make sure this function is correctly defined
        angles = np.sort(angles)
        label = "mean angle={:0.2f}".format(float(np.mean(angles)))
        if r is not None and l is not None:
            label = "$r$={:0.2f}".format(float(r))
        ax.plot(x, angles, marker="o", label=label, markersize=3)

    ax.set_xlabel("Vector Index")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Angles between corresponding vectors")
    ax.grid(True)
    ax.legend(fontsize='small')

    ax1.plot(radii, [np.mean(np.abs(b2 - biases_base)) for b2 in biases2_sampled], color='#1f77b4', marker='o',
             markersize=3,
             linestyle='-', label='Bias difference mean')
    ax1.set_xlabel('ratio $r$')
    ax1.set_ylabel('Bias difference mean')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left', fontsize='small')
    ax2 = ax1.twinx()

    ax2.plot(radii, losses, color='#ff7f0e', marker='o', linestyle='--', label='mse', markersize=3)
    ax2.set_ylabel('mse')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right', fontsize='small')

    ax1.set_title('Comparison of mse loss and mean differences in biases against ratios')
    ax1.grid(True)
    plt.tight_layout()
    plt.show()
    save_fig(plt, "plot_vector_differences")


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


import numpy as np
import matplotlib.pyplot as plt


def plot_weight_vectors_and_point_pairs(X, x_1_x2_tuples, weights, num_vectors=np.inf):
    X = np.array(X)
    plt.figure(figsize=(3, 3))
    plt.scatter(X[:, 0], X[:, 1], s=7)  # plot points in X

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
        plt.scatter(x1[0], x1[1], color=color, s=10, marker='x')
        # Plot line segment
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color=color, linewidth=0.5)
        plt.scatter([x2[0]], [x2[1]], color=color, marker='^', s=10)

        offset_arrow = 0.1 * (w / np.linalg.norm(w))
        c = plt.arrow(x1[0], x1[1], w[0], w[1], head_width=0.04, color=color, linewidth=0.5)

        plt.text(
            x1[0] + w[0] + offset_arrow[0],
            x1[1] + w[1] + offset_arrow[1],
            str(original_index + 1),
            color="black",
            alpha=0.9,
            fontsize=7,
            ha="center",
            va="center",
        )
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plot_for_x1, = plt.plot([], [], 'x', color='black', label='$x^{(1)}$', markersize=3)
    plot_for_x2, = plt.plot([], [], '^', color='black', label='$x^{(2)}$', markersize=3)
    plot_for_w = Line2D([], [], color='black', marker=r'$\rightarrow$', linestyle='None', markersize=4, label='w')

    # Creating a legend with specified marker styles
    plt.legend(handles=[plot_for_x1, plot_for_x2, plot_for_w], labels=['$x^{(1)}$', '$x^{(2)}$', 'w'],
               markerfirst=True, fontsize='small')
    plt.show()

    save_fig(plt, "plot_weight_vectors_and_point_pairs")


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

    visualize_data(y_base_model_train, y_sampled_train, "y base model train", "y sampled train")
    # visualize_data(y_base_model_test, y_sampled_test, "y base model test", "y sampled test")
