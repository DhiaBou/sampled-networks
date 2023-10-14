import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
    """Handles loading and preprocessing data.

    Attributes:
        X: Input features
        y: Target labels
        X_train: Train inputs
        X_test: Test inputs
        y_train: Train targets
        y_test: Test targets
    """

    def __init__(self):
        """Initialize an empty dataset"""
        self.X = np.array([])
        self.y = np.array([])

        self.X_train = np.array([])
        self.X_test = np.array([])

        self.y_train = np.array([])
        self.y_test = np.array([])

    def load_data_from_csv(self, csv_file, x_cols, y_cols, skiprows=0):
        data = np.loadtxt(csv_file, delimiter=",", skiprows=skiprows)
        self.X = data[:, 0:x_cols]
        self.y = data[:, x_cols: x_cols + y_cols]
        return self

    def get_subset(self, num_samples):
        dataset = Dataset()
        selected_indices = np.random.choice(self.X.shape[0], num_samples, replace=False)

        dataset.X = self.X[selected_indices]
        dataset.y = self.y[selected_indices]
        return dataset

    def create_dataset_Barron(self, xd, num_samples):
        X = np.random.uniform(-1, 1, size=(num_samples, xd))
        a = np.array([2 * j / xd - 1 for j in range(1, xd + 1)])

        y = np.array([np.sqrt(3 / 2) * (np.linalg.norm(x - a) - np.linalg.norm(x + a)) for x in X])
        y = y.reshape(-1, 1)
        self.X = X
        self.y = y
        return self

    def create_dataset_sinus_2d(self, num_samples):
        X = np.random.uniform(-np.pi, np.pi, size=(num_samples, 2))
        y = np.array([[np.sin(x[0]), np.cos(x[1])] for x in X])
        self.X = X
        self.y = 2 * y
        return self

    def create_dataset_laplacian_of_gaussian(self, num_samples, sigma=0.5):
        X = np.random.uniform(-2, 2, size=(num_samples, 2))
        term1 = -1 / (np.pi * sigma ** 4)
        term2 = 1 - ((X[:, 0] ** 2 + X[:, 1] ** 2) / (2 * sigma ** 2))
        term3 = np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2) / (2 * sigma ** 2))
        self.X = X
        self.y = np.array(term1 * term2 * term3).reshape(-1, 1)
        return self

    def scale(self, scaler):
        self.X = scaler.fit_transform(self.X)
        return self

    def split_train_test(self, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size
        )
        return self

    def get_random_subset(self, num_samples):
        if num_samples > len(self.X):
            raise ValueError("Num samples exceeded dataset size")

        indices = np.random.choice(self.X.shape[0], size=num_samples, replace=False)
        X_subset = self.X[indices]
        y_subset = self.y[indices]

        return X_subset, y_subset
