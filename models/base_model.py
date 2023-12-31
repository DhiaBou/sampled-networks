import numpy as np


class BaseModel:
    """The base neural networks model"""

    def __init__(self):
        """Initializes the weights and biases as empty lists"""
        self.weights = []
        self.biases = []

    def predict(self, X, activation="relu"):
        """Make predictions on data X using the model's weights and biases"""
        num_layers = len(self.weights)
        output = X

        for i in range(num_layers):
            output = np.dot(output, self.weights[i]) - self.biases[i]

            if i < num_layers - 1:
                if activation == "relu":
                    output = np.maximum(output, 0)
                elif activation == "tanh":
                    output = np.tanh(output)
                elif activation == "sigmoid":
                    output = 1 / (1 + np.exp(-output))
                else:
                    raise ValueError("Unsupported activation function:", activation)

        return output
