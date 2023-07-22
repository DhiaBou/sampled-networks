import math
import numpy as np
import sklearn
from tqdm import tqdm
from sklearn.linear_model import Ridge
import pytest

from dataset import Dataset
from neural_net import NeuralNet
from sampled_net import SampledNet


def loss(y1, y2):
    return sklearn.metrics.mean_squared_error(y1, y2)


def predict_output(X, weights, biases):
    num_layers = len(weights)
    output = X

    for i in range(num_layers):
        output = np.dot(output, weights[i]) - biases[i]

        if i < num_layers - 1:
            output = np.maximum(output, 0)  # Apply ReLU activation for hidden layers

    return output


def choose_x1_x2_lowest_activation(X, weight, bias, radius):
    x_min_activation = X[np.argmin(np.dot(X, weight) - bias)]
    distances = np.linalg.norm(X - x_min_activation, axis=1)
    X_closest_indices = np.where(distances <= radius)[0]
    weight_norm = np.linalg.norm(weight)

    if len(X_closest_indices) == 0:
        return None, None

    min_value = float("inf")
    x_1_retur = None
    x_2_retur = None
    for i in X_closest_indices:
        x_1 = X[i]
        X_other = np.delete(X, i, axis=0)  # remove x_1 from X
        diffs = X_other - X[i]
        angles = np.arccos(
            (diffs @ weight) / (np.linalg.norm(diffs, axis=1) * weight_norm)
        )
        j = np.argmin(angles)
        d = angles[j]
        if d < min_value:
            min_value = d
            x_1_retur = x_1
            x_2_retur = X_other[j]

    return x_1_retur, x_2_retur


def find_max_distance_dataset(X):
    diffs = X[:, None, :] - X[None, :, :]
    distances = np.sqrt((diffs ** 2).sum(-1))
    np.fill_diagonal(distances, np.nan)
    max_distance = np.nanmax(distances)
    return max_distance


def compute_weights_biases_layer2_classic(X, y, weights, biases, weights_l1, biases_l1):
    N2 = 1 if np.ndim(y) == 1 else y.shape[1]
    N1 = len(weights[1])
    weights_l2 = weights[1].copy()
    biases_l2 = biases[1].copy()
    for i in range(N1):
        for j in range(N2):
            w1i_hat = weights_l1[:, i]
            b1_hat = biases_l1[i]
            min_x = min(X, key=lambda x: np.dot(x, w1i_hat))
            if b1_hat < np.dot(min_x, w1i_hat) and biases[0][i] < b1_hat:
                biases_l2[j] = biases_l2[j] + weights[1][:, j][i] * (
                    biases[0][i] - b1_hat
                )
    return weights_l2, biases_l2


def compute_weights_biases_layer2_lstsq(X, y, weights_l1, biases_l1):
    X_l1 = np.dot(X, weights_l1) - biases_l1
    X_l1 = np.maximum(X_l1, 0)
    X_c = np.c_[np.ones(X_l1.shape[0]), X_l1]
    w_c, residuals, rank, singular_values = np.linalg.lstsq(X_c, y, rcond=None)
    biases_l2 = -w_c[0]
    weights_l2 = w_c[1:]
    return weights_l2, biases_l2


def compute_weights_biases_layer2_ridge(X, y, weights_l1, biases_l1, alpha=1):
    X_l1 = np.dot(X, weights_l1) - biases_l1
    X_l1 = np.maximum(X_l1, 0)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_l1, y)

    weights_l2 = ridge.coef_.T
    biases_l2 = -ridge.intercept_
    return weights_l2, biases_l2


def compute_weights_biases_layer1(X, y, weights, biases, radius=0):
    if radius<0:
        raise ValueError("Radius negative")
    N1 = len(weights[1])
    weights_l1 = []
    biases_l1 = []
    for i in tqdm(range(N1)):
        # Find x(1) ∈ X which minimizes {⟨w1,i, x⟩ − b1,i : x ∈ X};
        x_1, x_2 = choose_x1_x2_lowest_activation(
            X, weights[0][:, i], biases[0][i], radius
        )
        w1i_hat = (x_2 - x_1) / (np.linalg.norm(x_2 - x_1)) ** 2
        b1_hat = np.dot(x_1, w1i_hat)
        weights_l1.append(w1i_hat)
        biases_l1.append(b1_hat)
    return np.transpose(weights_l1), biases_l1


def choose_best_alpha(X_train, y_train, X_test, y_test, weights_nn, biases_nn, radius):
    weights_l1, biases_l1 = compute_weights_biases_layer1(
        X_train, y_train, weights_nn, biases_nn, radius
    )
    alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    min_mse = np.inf
    alpha_r, weights_r, biases_r = 0, None, None
    print("alpha \tloss")
    for alpha in alpha_values:
        weights_l2, biases_l2 = compute_weights_biases_layer2_ridge(
            X_train, y_train, weights_l1, biases_l1, alpha=alpha
        )
        w, b = [weights_l1, weights_l2], [biases_l1, biases_l2]
        y_pred = predict_output(X_test, w, b)
        mse = loss(y_pred, y_test)
        print(f"{alpha} \t{mse:.3e}")

        if mse < min_mse:
            min_mse = mse
            alpha_r = alpha
            weights_r = w
            biases_r = b
    return  alpha_r, weights_r, biases_r

def loss_f_alpha_radius_mse(data: Dataset, model_nn: NeuralNet):
    max_radius = find_max_distance_dataset(data.X_train)
    radiuses = np.linspace(0, max_radius, 10)
    alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    mses = {}
    for radius in radiuses:
        print(radius)
        mses[radius] = {}
        weights_l1, biases_l1 = compute_weights_biases_layer1(
            data.X_train, data.y_train, model_nn.weights, model_nn.biases, radius
        )
        for alpha in alpha_values:
            weights_l2, biases_l2 = compute_weights_biases_layer2_ridge(
                data.X_train, data.y_train, weights_l1, biases_l1, alpha=alpha
            )
            model_sampled = SampledNet()
            model_sampled.weights = [weights_l1, weights_l2]
            model_sampled.biases = [biases_l1, biases_l2]
            y_sampled = model_sampled.predict(data.X_test)
            mse = loss(y_sampled, data.y_test)
            mses[radius][alpha]=mse
    return mses



def mse_f_num_samples(Xs_train, ys_train, Xs_test, ys_test, weights_nn, biases_nn):
    for i in range(len(Xs_train)):
        num_samples_i = len(Xs_train[i])
        alpha, radius, weight, bias, mses = choose_best_radius_alpha(
            Xs_train[i],
            ys_train[i],
            Xs_test[i],
            ys_test[i],
            weights_nn[i],
            biases_nn[i],
        )


def choose_best_radius_alpha(
    X_train, y_train, X_test, y_test, weights_nn, biases_nn, num_intervals=10
):
    max_radius = find_max_distance_dataset(X_train)
    radiuses = np.linspace(0, max_radius, num_intervals)
    alpha_r, radius_r, radius_r, weights_r, biases_r = 0, 0, 0, [], []
    min_mse = np.inf
    for r in radiuses:
        print()
        print(f"radius: {r:.3f}")
        alpha, w, b = choose_best_alpha(
            X_train, y_train, X_test, y_test, weights_nn, biases_nn, r
        )
        y_pred = predict_output(X_test, w, b)
        mse = loss(y_pred, y_test)
        if mse < min_mse:
            min_mse = mse
            alpha_r = alpha
            weights_r = w
            biases_r = b
            radius_r = r
    return alpha_r, radius_r, weights_r, biases_r
