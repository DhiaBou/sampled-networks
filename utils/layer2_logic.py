import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from dataset.dataset import Dataset
from models.neural_net import NeuralNet
from utils.layer1_logic import compute_weights_biases_layer1
from utils.utilities import predict_output, loss_mse, loss_r2, loss_model_on_test


def compute_weights_biases_layer2_classic_old(X, y, weights, biases, weights_l1, biases_l1):
    N2 = len(weights[1][0])
    N1 = len(weights[1])
    weights_l2 = weights[1].copy()
    biases_l2 = biases[1].copy()
    for i in range(N1):
        for j in range(N2):
            w1i_hat = weights_l1[:, i]
            b1_hat = biases_l1[i]
            min_x = min(X, key=lambda x: np.dot(x, w1i_hat))
            if b1_hat < np.dot(min_x, w1i_hat) and biases[0][i] < b1_hat:
                print("****************************")
                biases_l2[j] = biases_l2[j] + weights[1][:, j][i] * (biases[0][i] - b1_hat)
    return weights_l2, biases_l2


def compute_weights_biases_layer2_classic(X, y, weights, biases, weights_l1, biases_l1):
    weights_l2 = weights[1].copy()
    biases_l2 = biases[1].copy()
    y_old = np.dot((np.maximum(0, np.dot(X, weights[0]) - biases[0])), weights[1]) - biases[1]
    y_new = np.dot((np.maximum(0, np.dot(X, weights_l1) - biases_l1)), weights_l2)
    delta = y_new - y_old
    delta_avg = np.average(delta, axis=0)
    biases_l2 = delta_avg
    return weights_l2, biases_l2


def compute_weights_biases_layer2_lstsq(X, y, weights_l1, biases_l1):
    X_l1 = np.dot(X, weights_l1) - biases_l1
    X_l1 = np.maximum(X_l1, 0)
    X_c = np.c_[np.ones(X_l1.shape[0]), X_l1]  # add column of 1s to count for the biases
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


def choose_best_alpha(X_train, y_train, X_val, y_val, weights_l1, biases_l1, verbose=1):
    alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    min_loss = np.inf
    alpha_r, weights_r, biases_r = 0, None, None
    if verbose == 1:
        print("alpha \tloss validation")

    for alpha in alpha_values:
        weights_l2, biases_l2 = compute_weights_biases_layer2_ridge(
            X_train, y_train, weights_l1, biases_l1, alpha=alpha
        )

        w, b = [weights_l1, weights_l2], [biases_l1, biases_l2]

        y_pred = predict_output(X_val, w, b)
        loss_alpha = loss_mse(y_pred, y_val)

        if verbose == 1:
            print(f"{alpha} \t{loss_alpha:.3e}")

        if loss_alpha <= min_loss:
            min_loss = loss_alpha
            alpha_r = alpha
            weights_r = w
            biases_r = b
    return alpha_r, weights_r, biases_r


def choose_best_radius_alpha(
        X, y, weights_nn, biases_nn, num_intervals=10, verbose=1, validation_split=0.2
):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split)

    radii = np.linspace(0, 0.5, num_intervals)
    alpha_r, radius_r, radius_r, weights_r, biases_r = 0, 0, 0, [], []
    min_mse = np.inf
    for r in radii:
        if verbose == 1:
            print()
            print(f"radius: {r:.3f}")
        weights_l1, biases_l1, x_1_x_2_pairs = compute_weights_biases_layer1(
            X, weights_nn, biases_nn, radius=r, choose_x_2="angle")

        alpha, w, b = choose_best_alpha(
            X_train,
            y_train,
            X_val,
            y_val,
            weights_l1,
            biases_l1,
            verbose=verbose,
        )

        y_pred = predict_output(X_val, w, b)
        mse = loss_mse(y_pred, y_val)
        if mse <= min_mse:
            min_mse = mse
            alpha_r = alpha
            weights_r = w
            biases_r = b
            radius_r = r
    return alpha_r, radius_r, weights_r, biases_r


def loss_vs_aslpha_radius__sample_trained_on_dataset(data: Dataset, model_nn: NeuralNet):
    from models.sampled_net import SampledNet

    radiuses = np.linspace(0, 1, 10)
    alpha_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
    mses = {}

    given_model_loss = loss_model_on_test(model_nn, data.X_test, data.y_test)
    mses["given_model"] = given_model_loss

    # calculate loss of SampledNet on radius and alpha combinations
    mses["sampled_net"] = {}
    for radius in radiuses:
        print(f"radius: {radius:.3f}")
        mses["sampled_net"][radius] = {}
        weights_l1, biases_l1, _ = compute_weights_biases_layer1(
            data.X_train, model_nn.weights, model_nn.biases, radius
        )
        for alpha in alpha_values:
            weights_l2, biases_l2 = compute_weights_biases_layer2_ridge(
                data.X_train, data.y_train, weights_l1, biases_l1, alpha=alpha
            )
            model_sampled = SampledNet()
            model_sampled.weights = [weights_l1, weights_l2]
            model_sampled.biases = [biases_l1, biases_l2]
            y_sampled = model_sampled.predict(data.X_test)
            mse = loss_mse(y_sampled, data.y_test)
            mses["sampled_net"][radius][alpha] = mse
    return mses


def loss_vs_num_samples(datasets, models_nn):
    from models.sampled_net import SampledNet

    losses = {}
    for dataset, model in zip(datasets, models_nn):
        num_training = len(dataset.X_train)
        model_sampled = SampledNet()
        alpha, radius = model_sampled.fit(dataset.X_train, dataset.y_train, model)
        y_sampled = model_sampled.predict(dataset.X_test)
        mse = loss_mse(dataset.y_test, y_sampled)
        r2_score = loss_r2(dataset.y_test, y_sampled)
        losses[num_training] = {"mse": mse, "r2": r2_score}
    return losses
