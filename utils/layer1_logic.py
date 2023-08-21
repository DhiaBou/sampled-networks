from typing import Literal

import numpy as np
from tqdm import tqdm

from utils.utilities import augment_by_sampling_gaussian_noise


def validate_radius(radius):
    if radius < 0 or radius > 1:
        raise ValueError(f"Expected radius to be in [0, 1], got {radius}")


def project_onto_plane(x_1, w, b):
    # Calculate the distance d from x1 to the plane
    d = (np.dot(x_1, w) - b) / np.linalg.norm(w)
    # Calculate the projection x2
    x_1_projection = x_1 - d * (w / np.linalg.norm(w))
    return x_1_projection


def find_max_distance_to_bias_origin(X, weight, bias):
    return np.max(np.abs(np.dot(X, weight) - bias))


def choose_x1_x2(X, weight, bias, radius=0, project_onto_boundary=False,
                 choose_x_2: Literal["angle", "norm"] = "norm"):
    validate_radius(radius)

    X_closest_indices = choose_x1_in_radius(X, bias, radius, weight)
    x_1, x_2 = None, None
    if choose_x_2 == "norm":
        x_1, x_2 = choose_x2_closest_to_weight_scaled(X, X_closest_indices, weight)
    if choose_x_2 == "angle":
        x_1, x_2 = choose_x2_min_angle_to_weight(X, X_closest_indices, weight)

    if project_onto_boundary:
        x_1, x_2 = shift_x_1_x_2_to_bias_origin(x_1, x_2, weight, bias)

    return x_1, x_2


def shift_x_1_x_2_to_bias_origin(x_1, x_2, weight, bias):
    x_1_projection = project_onto_plane(x_1, weight, bias)
    delta = x_1_projection - x_1
    x_2 += delta
    x_1 += delta
    return x_1, x_2


def choose_x1_in_radius(X, bias, radius, weight):
    max_dist = find_max_distance_to_bias_origin(X, weight, bias)
    x_min_activation_index = np.argmin(np.abs(np.dot(X, weight) - bias))
    distances_to_bias_origin = np.abs(np.dot(X, weight) - bias)
    X_closest_indices = np.where(
        (distances_to_bias_origin <= radius * max_dist) | (np.arange(X.shape[0]) == x_min_activation_index)
    )[0]
    return X_closest_indices


def choose_x2_min_angle_to_weight(X, X_closest_indices, weight):
    weight_norm = np.linalg.norm(weight)
    min_value = float("inf")
    x_1_retur = None
    x_2_retur = None
    for i in X_closest_indices:
        x_1 = X[i]
        X_other = np.delete(X, i, axis=0)  # remove x_1 from X
        diffs = X_other - X[i]
        angles = np.arccos(np.clip((diffs @ weight) / (np.linalg.norm(diffs, axis=1) * weight_norm), -1, 1))
        min_angle_i = np.argmin(angles)
        d = angles[min_angle_i]
        if d < min_value:
            min_value = d
            x_1_retur = x_1
            x_2_retur = X_other[min_angle_i]
    return x_1_retur.copy(), x_2_retur.copy()


def choose_x2_closest_to_weight_scaled(X, X_candidate_indices, weight):
    min_distance = float("inf")
    x_1_retur = None
    x_2_retur = None

    for i in X_candidate_indices:
        x_1 = X[i]
        X_other = np.delete(X, i, axis=0)  # remove x_1 from X

        diffs = X_other - x_1
        normalized_diffs = diffs / (np.linalg.norm(diffs, axis=1) ** 2).reshape(-1, 1)
        distances = np.linalg.norm(normalized_diffs - weight, axis=1)

        min_distance_i = np.argmin(distances)
        d = distances[min_distance_i]

        if d < min_distance:
            min_distance = d
            x_1_retur = x_1
            x_2_retur = X_other[min_distance_i]

    return x_1_retur.copy(), x_2_retur.copy()


def compute_weights_biases_layer1(
        X, weights, biases, radius=0, project_onto_boundary=False, augment_data=None,
        choose_x_2: Literal["angle", "norm"] = "norm"):
    validate_radius(radius)

    if augment_data is not None:
        X = augment_by_sampling_gaussian_noise(X, augment_data[0], augment_data[1])

    N1 = len(weights[1])
    weights_l1 = []
    biases_l1 = []
    x_pairs = []
    for i in tqdm(range(N1), desc="Layer1 sampling: "):
        x_1, x_2 = choose_x1_x2(X, weights[0][:, i], biases[0][i], radius, project_onto_boundary, choose_x_2)

        w1i_hat = (
                (x_2 - x_1)
                / (np.linalg.norm(x_2 - x_1)) ** 2
        )
        b1_hat = np.dot(x_1, w1i_hat)

        x_pairs.append((x_1, x_2))
        weights_l1.append(w1i_hat)
        biases_l1.append(b1_hat)

    return np.transpose(weights_l1), biases_l1, x_pairs
