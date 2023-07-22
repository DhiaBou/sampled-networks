from matplotlib import pyplot as plt
import numpy as np
from dataset import *
from sampled_net import *
from logic import *
from matplotlib.ticker import LogLocator


def visualize_data(Y_1, Y_2):
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

        ax.plot(range(n), Y_1_sorted, label="Y 1")
        ax.plot(range(n), Y_2_sorted, label="Y 2")
        ax.set_ylabel(f"Dimension {dim + 1}")
        ax.legend()

    axs[-1].set_xlabel("Sample")
    plt.show()

def plot_vector_differences(weights1, weights2):
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
    angles = np.sort(angles)
    x = np.arange(len(angles))

    # Plot angles between corresponding vectors
    plt.figure()
    plt.plot(x, angles, marker="o")
    plt.xlabel("Vector Index")
    plt.ylabel("Angle (degrees)")
    plt.title("Angles between corresponding vectors")
    plt.grid(True)
    plt.show()

    # Calculate and plot histogram of angle ranges
    range_start = 0
    range_end = 90
    bin_size = 10

    hist, bins = np.histogram(
        angles, bins=np.arange(range_start, range_end + bin_size, bin_size)
    )

    plt.figure()
    plt.bar(bins[:-1], hist, width=bin_size, align="edge", edgecolor="black")
    plt.xlabel("Angle Range")
    plt.ylabel("Count")
    plt.title("Number of Angles in Each Range")
    plt.xticks(np.arange(range_start, range_end, bin_size))
    plt.grid(True)
    mean_angle = np.mean(angles)
    sd_angle = np.std(angles)

    plt.annotate(
        f"Mean: {mean_angle:.2f}\nSD: {sd_angle:.2f}",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-20, -20),
        textcoords="offset points",
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=12,
    )

    plt.show()

def plot_loss_f_alpha_radius_mse(mses):      
    fig, ax = plt.subplots()
    for radius, inner_dict in mses.items():
        x = list(inner_dict.keys())
        y = list(inner_dict.values())
        
        radius_str = "{:0.2f}".format(radius)   
        ax.plot(x, y, marker='o', label=f'Radius: {radius_str}')

    ax.set_xlabel('Alpha (log scale)')
    ax.set_ylabel('MSE')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10)) 
    ax.legend()

    plt.show()



