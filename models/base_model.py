import numpy as np
import tensorflow as tf


class Model:
    """The base neural networks model"""

    def __init__(self):
        """Initializes the weights and biases as empty lists"""
        self.weights = []
        self.biases = []

    def predict(self, X):
        """Make predictions on data X using the model's weights and biases"""
        num_layers = len(self.weights)
        output = X

        for i in range(num_layers):
            output = np.dot(output, self.weights[i]) - self.biases[i]

            if i < num_layers - 1:
                output = np.maximum(output, 0)  # Apply ReLU activation for hidden layers

        return output
