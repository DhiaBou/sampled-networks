import numpy as np
import sklearn
from tqdm import tqdm
from sklearn.linear_model import Ridge

from dataset import Dataset
from models.neural_net import NeuralNet


def loss_mse(y_true, y_pred):
    return sklearn.metrics.mean_squared_error(y_true, y_pred)


def loss_r2(y_true, y_pred):
    return sklearn.metrics.r2_score(y_true, y_pred)


def validate_radius(radius):
    if radius < 0 or radius > 1:
        raise ValueError(f"Expected radius to be in [0, 1], got {radius}")


def predict_output(X, weights, biases):
    num_layers = len(weights)
    output = X

    for i in range(num_layers):
        output = np.dot(output, weights[i]) - biases[i]

        if i < num_layers - 1:
            output = np.maximum(output, 0)  # Apply ReLU activation for hidden layers

    return output


def loss_model_on_test(model, X_test, y_test):
    y_predict = model.predict(X_test)
    loss = loss_mse(y_test, y_predict)
    return loss


def choose_x1_x2(X, weight, bias, radius=0):
    validate_radius(radius)

    max_dist = find_max_distance_to_bias_origin(X, weight, bias)

    x_min_activation_index = np.argmin(np.abs(np.dot(X, weight) - bias))

    distances_to_bias_origin = np.abs(np.dot(X, weight) - bias)
    X_closest_indices = np.where(
        (distances_to_bias_origin <= radius * max_dist) | (np.arange(X.shape[0]) == x_min_activation_index)
    )[0]
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

    return x_1_retur, x_2_retur


def find_max_distance_to_bias_origin(X, weight, bias):
    return np.max(np.abs(np.dot(X, weight) - bias))


def compute_weights_biases_layer2_classico(X, y, weights, biases, weights_l1, biases_l1):
    N2 = len(weights[1][0])
    N1 = len(weights[1])
    weights_l2 = weights[1].copy()
    biases_l2 = biases[1].copy()
    for i in range(N1):
        for j in range(N2):
            w1i_hat = weights_l1[:, i]
            b1_hat = biases_l1[i]
            min_x = min(X, key=lambda x: np.dot(x, w1i_hat))
            if b1_hat <= np.dot(min_x, w1i_hat):  # and biases[0][i] < b1_hat:
                print("****************************")
                biases_l2[j] = biases_l2[j] + weights[1][:, j][i] * (biases[0][i] - b1_hat)
    return weights_l2, biases_l2


def compute_weights_biases_layer2_classic(X, y, weights, biases, weights_l1, biases_l1):
    N2 = len(weights[1][0])
    N1 = len(weights[1])
    weights_l2 = weights[1].copy()
    biases_l2 = biases[1].copy()
    y_old = np.dot((np.maximum(0, np.dot(X, weights[0]) - biases[0])), weights[1]) - biases[1]
    y_new = np.dot((np.maximum(0, np.dot(X, weights_l1) - biases_l1)), weights_l2) - biases_l2
    delta = y_new - y_old
    delta_avg = np.average(delta, axis=0)
    biases_l2 = biases_l2 + delta_avg
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


def compute_weights_biases_layer1(X, weights, biases, radius=0):
    validate_radius(radius)

    N1 = len(weights[1])
    weights_l1 = []
    biases_l1 = []
    x_pairs = []
    for i in tqdm(range(N1), desc="Layer1 sampling: "):
        x_1, x_2 = choose_x1_x2(X, weights[0][:, i], biases[0][i], radius)
        x_2_x_1_norm = np.linalg.norm(x_2 - x_1)
        weight_norm = np.linalg.norm(weights[0][:, i])
        w1i_hat = (x_2 - x_1) * weight_norm / x_2_x_1_norm  # / (np.linalg.norm(x_2 - x_1))# ** 2
        b1_hat = np.dot(x_1, w1i_hat)
        x_pairs.append((x_1, x_2))
        weights_l1.append(w1i_hat)
        biases_l1.append(b1_hat)
    return np.transpose(weights_l1), biases_l1, x_pairs


def choose_best_alpha(X_train, y_train, X_val, y_val, weights_nn, biases_nn, radius, verbose=1):
    weights_l1, biases_l1, _ = compute_weights_biases_layer1(X_train, weights_nn, biases_nn, radius)
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


def choose_best_radius_alpha(
    X_train, y_train, X_test, y_test, weights_nn, biases_nn, num_intervals=10, verbose=1
):
    radiuses = np.linspace(0, 1, num_intervals)
    alpha_r, radius_r, radius_r, weights_r, biases_r = 0, 0, 0, [], []
    min_mse = np.inf
    for r in radiuses:
        if verbose == 1:
            print()
            print(f"radius: {r:.3f}")
        alpha, w, b = choose_best_alpha(
            X_train, y_train, X_test, y_test, weights_nn, biases_nn, r, verbose=verbose
        )
        y_pred = predict_output(X_test, w, b)
        mse = loss_mse(y_pred, y_test)
        if mse <= min_mse:
            min_mse = mse
            alpha_r = alpha
            weights_r = w
            biases_r = b
            radius_r = r
    return alpha_r, radius_r, weights_r, biases_r
