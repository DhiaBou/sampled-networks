import numpy as np
import sklearn


def loss_mse(y_true, y_pred):
    return sklearn.metrics.mean_squared_error(y_true, y_pred)


def loss_r2(y_true, y_pred):
    return sklearn.metrics.r2_score(y_true, y_pred)


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


def augment_by_sampling_gaussian_noise(X, sigma=0.1, num_samples_per_point=1):
    n, D = X.shape
    augmented_data = []

    for i in range(n):
        for _ in range(num_samples_per_point):
            noise = np.random.normal(0, sigma, D)
            new_data_point = X[i] + noise
            augmented_data.append(new_data_point)

    # Combine the original and augmented data
    X_augmented = np.vstack([X, np.array(augmented_data)])
    return X_augmented
