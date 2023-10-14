from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from utils.layer1_logic import layer_1_conversion
from utils.utilities import predict_output, loss_mse


def layer_2_conversion(
        X, trained_weights, trained_biases, W_1_hat, b_1_hat, alpha=1,
        layer2: Literal["bias_only", "ridge", "lstsq"] = "bias_only"
):
    y = predict_output(X, trained_weights, trained_biases)
    if layer2 == "bias_only":
        return layer_2_conversion_only_bias_update(X, y, trained_weights, trained_biases, W_1_hat, b_1_hat)
    elif layer2 == "lstsq":
        return layer_2_conversion_lstsq(X, y, W_1_hat, b_1_hat)
    elif layer2 == "ridge":
        return layer_2_conversion_ridge(X, y, W_1_hat, b_1_hat, alpha=alpha)
    else:
        raise ValueError(f"Invalid layer2 value: {layer2}. Expected 'classic', 'lstsq', or 'ridge'")


def layer_2_conversion_only_bias_update(X, y_old, trained_weights, trained_biases, W_1_hat, b_1_hat):
    weights_l2 = trained_weights[1].copy()
    y_new = np.dot((np.maximum(0, np.dot(X, W_1_hat) - b_1_hat)), weights_l2)
    delta = y_new - y_old
    delta_avg = np.average(delta, axis=0)
    biases_l2 = delta_avg
    return weights_l2, biases_l2


def layer_2_conversion_lstsq(X, y, W_1_hat, b_1_hat):
    X_l1 = np.dot(X, W_1_hat) - b_1_hat
    X_l1 = np.maximum(X_l1, 0)
    X_c = np.c_[np.ones(X_l1.shape[0]), X_l1]
    w_c, residuals, rank, singular_values = np.linalg.lstsq(X_c, y, rcond=None)
    biases_l2 = -w_c[0]
    weights_l2 = w_c[1:]
    return weights_l2, biases_l2


def layer_2_conversion_ridge(X, y, W_1_hat, b_1_hat, alpha=1):
    X_l1 = np.dot(X, W_1_hat) - b_1_hat
    X_l1 = np.maximum(X_l1, 0)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_l1, y)

    weights_l2 = ridge.coef_.T
    biases_l2 = -ridge.intercept_
    return weights_l2, biases_l2


def choose_best_alpha_for_ridge(X_train, W_1_hat, b_1_hat, trained_weights, trained_biases, verbose=1,
                                validation_split=0.2):
    y_train = predict_output(X_train, trained_weights, trained_biases)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split)

    alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    min_loss = np.inf
    alpha_ret, weights_ret, biases_ret = 0, None, None
    if verbose == 1:
        print("alpha \tloss validation")

    for alpha in alpha_values:
        weights_l2, biases_l2 = layer_2_conversion_ridge(
            X_train, y_train, W_1_hat, b_1_hat, alpha=alpha
        )

        w, b = [W_1_hat, weights_l2], [b_1_hat, biases_l2]

        y_pred = predict_output(X_val, w, b)
        loss_alpha = loss_mse(y_pred, y_val)

        if verbose == 1:
            print(f"{alpha} \t{loss_alpha:.3e}")

        if loss_alpha <= min_loss:
            min_loss = loss_alpha
            alpha_ret = alpha
            weights_ret = w
            biases_ret = b
    return alpha_ret, weights_ret, biases_ret


def choose_best_threshold_ratio_and_alpha(
        X, trained_weights, trained_biases, num_intervals=10, verbose=1, validation_split=0.2,
        choose_x_2: Literal["angle", "norm_kdtree", "norm"] = "norm"
):
    y = predict_output(X, trained_weights, trained_biases)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split)

    ratios = np.linspace(0, 0.5, num_intervals)
    alpha_ret, ratio_ret, ratio_ret, weights_ret, biases_ret = 0, 0, 0, [], []
    min_mse = np.inf
    for r in ratios:
        if verbose == 1:
            print()
            print(f"ratio: {r:.3f}")
        W_1_hat, b_1_hat, x_1_x_2_pairs = layer_1_conversion(
            X, trained_weights, trained_biases, r=r, choose_x_2=choose_x_2)

        alpha, w, b = choose_best_alpha_for_ridge(X_train, W_1_hat, b_1_hat, trained_weights, trained_biases,
                                                  verbose=verbose)

        y_pred = predict_output(X_val, w, b)
        mse = loss_mse(y_pred, y_val)
        if mse <= min_mse:
            min_mse = mse
            alpha_ret = alpha
            weights_ret = w
            biases_ret = b
            ratio_ret = r
    return alpha_ret, ratio_ret, weights_ret, biases_ret
