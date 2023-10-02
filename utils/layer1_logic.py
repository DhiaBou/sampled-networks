from typing import Literal

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from utils.utilities import augment_by_sampling_gaussian_noise


def validate_ratio(r):
    if r < 0 or r > 1:
        raise ValueError(f"Expected ratio to be in [0, 1], got {r}")


def project_onto_hyperplane(x_1, w, b):
    d = (np.dot(x_1, w) - b) / np.linalg.norm(w)
    x_1_projection = x_1 - d * (w / np.linalg.norm(w))
    return x_1_projection


def find_max_absolute_output_before_activation(X, weight, bias):
    return np.max(np.abs(np.dot(X, weight) - bias))


def choose_x1_x2(X, trained_weight_vector, trained_bias, r=0, project_onto_boundary=False,
                 choose_x_2: Literal["angle", "norm", "norm_kdtree"] = "norm"):
    validate_ratio(r)

    X_closest_indices = choose_x1_in_threshold_ratio(X, trained_weight_vector, trained_bias, r)
    x_1, x_2 = None, None

    if choose_x_2 == "norm":
        x_1, x_2 = choose_x2_weight_norm_preservation(X, X_closest_indices, trained_weight_vector)
    elif choose_x_2 == "norm_kdtree":
        X_tree = KDTree(X)
        x_1, x_2 = choose_x2_proximity_optimization(X_tree, X_closest_indices, trained_weight_vector)
    elif choose_x_2 == "angle":
        x_1, x_2 = choose_x2_lowest_angle_approach(X, X_closest_indices, trained_weight_vector)

    if project_onto_boundary:
        x_1, x_2 = shift_x_1_x_2_to_activation_boundary(x_1, x_2, trained_weight_vector, trained_bias)

    return x_1, x_2


def shift_x_1_x_2_to_activation_boundary(x_1, x_2, weight_vector, bias):
    x_1_proj = project_onto_hyperplane(x_1, weight_vector, bias)
    delta = x_1_proj - x_1
    x_2 += delta
    x_1 += delta
    return x_1, x_2


def choose_x1_in_threshold_ratio(X, weight_vector, bias, r):
    max_dist = find_max_absolute_output_before_activation(X, weight_vector, bias)
    x_min_activation_index = np.argmin(np.abs(np.dot(X, weight_vector) - bias))
    distances_to_bias_origin = np.abs(np.dot(X, weight_vector) - bias)
    X_closest_indices = np.where(
        (distances_to_bias_origin <= r * max_dist) | (np.arange(X.shape[0]) == x_min_activation_index)
    )[0]
    return X_closest_indices


def choose_x2_lowest_angle_approach(X, X_closest_indices, weight_vector):
    weight_norm = np.linalg.norm(weight_vector)
    min_value = float("inf")
    x_1_retur = None
    x_2_retur = None
    for i in X_closest_indices:
        x_1 = X[i]
        X_other = np.delete(X, i, axis=0)  # remove x_1 from X
        diffs = X_other - X[i]
        angles = np.arccos(np.clip((diffs @ weight_vector) / (np.linalg.norm(diffs, axis=1) * weight_norm), -1, 1))
        min_angle_index = np.argmin(angles)
        d = angles[min_angle_index]
        if d < min_value:
            min_value = d
            x_1_retur = x_1
            x_2_retur = X_other[min_angle_index]
    return x_1_retur.copy(), x_2_retur.copy()


def choose_x2_weight_norm_preservation(X, X_candidate_indices, weight_vector):
    min_distance = float("inf")
    x_1_retur = None
    x_2_retur = None

    for i in X_candidate_indices:
        x_1 = X[i]
        X_other = np.delete(X, i, axis=0)  # remove x_1 from X

        diffs = X_other - x_1
        w_1_hat = diffs / (np.linalg.norm(diffs, axis=1) ** 2).reshape(-1, 1)
        distances = np.linalg.norm(w_1_hat - weight_vector, axis=1)

        min_distance_index = np.argmin(distances)
        d = distances[min_distance_index]

        if d < min_distance:
            min_distance = d
            x_1_retur = x_1
            x_2_retur = X_other[min_distance_index]

    return x_1_retur.copy(), x_2_retur.copy()


def choose_x2_proximity_optimization(X: KDTree, X_candidate_indices, weight_vector):
    min_distance = np.inf
    x_1_retur, x_2_retur = None, None

    for i in X_candidate_indices:
        x_1 = X.data[i]
        x_hat = x_1 + weight_vector / (np.linalg.norm(weight_vector) ** 2)
        dists, idxs = X.query(x_hat, 2)

        chosen_idx = 1 if np.array_equal(X.data[idxs[0]], x_1) else 0
        if dists[chosen_idx] < min_distance:
            min_distance = dists[chosen_idx]
            x_1_retur, x_2_retur = x_1, X.data[idxs[chosen_idx]]

    return x_1_retur.copy(), x_2_retur.copy()


def layer_1_conversion(
        X, trained_weights, trained_biases, r=0, project_onto_boundary=False, augment_data=None,
        choose_x_2: Literal["angle", "norm", "norm_kdtree"] = "norm"):
    validate_ratio(r)

    if augment_data is not None:
        X = augment_by_sampling_gaussian_noise(X, augment_data[0], augment_data[1])

    N1 = len(trained_weights[1])
    W_1_hat = []
    b_1_hat = []
    x_1_x_2_pairs = []
    for i in tqdm(range(N1), desc="Layer1 sampling: "):
        x_1, x_2 = choose_x1_x2(X, trained_weights[0][:, i], trained_biases[0][i], r, project_onto_boundary, choose_x_2)

        w1i_hat = (
                (x_2 - x_1)
                / (np.linalg.norm(x_2 - x_1)) ** 2
        )
        b1_hat = np.dot(x_1, w1i_hat)

        x_1_x_2_pairs.append((x_1, x_2))
        W_1_hat.append(w1i_hat)
        b_1_hat.append(b1_hat)

    return np.transpose(W_1_hat), b_1_hat, x_1_x_2_pairs
