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
        self.X = []
        self.y = []

        self.X_train = []
        self.X_test = []

        self.y_train = []
        self.y_test = []

    def load_data_from_csv(self, csv_file, y_cols, x_cols):
        """Load data from CSV file."""
        data = np.loadtxt(csv_file, delimiter=",")
        self.X = data[:, 0:x_cols]
        self.y = data[:, x_cols : x_cols + y_cols - 1]
        return self

    def create_dataset_Barron(self, xd, num_samples):
        """Create synthetic Barron dataset."""
        X = np.random.uniform(-1, 1, size=(num_samples, xd))
        a = np.array([2 * j / xd - 1 for j in range(1, xd + 1)])

        y = np.array(
            [np.sqrt(3 / 2) * (np.linalg.norm(x - a) - np.linalg.norm(x + a)) for x in X]
        )
        y = y.reshape(-1, 1)
        self.X = X
        self.y = y
        return self

    def scale(self, scaler):
        """Scale the input features."""
        self.X = scaler.fit_transform(self.X)
        return self

    def split_train_test(self, test_size):
        """Split data into train and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size
        )
        return self
