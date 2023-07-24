import numpy as np
from base_model import Model
from logic import *
from neural_net import NeuralNet
from sklearn.model_selection import train_test_split


class SampledNet(Model):
    def __init__(self):
        super().__init__()

    def fit(
        self,
        X_train,
        y_train,
        model: NeuralNet,
        radius=-1,
        validation_split=0.2,
        layer2="ridge",
        alpha=-1,
        num_intervals=10,
    ):
        """
        Trains the sampled network model with provided training data by learning the weights and biases.

        Parameters:
        - X_train: Input training data
        - y_train: Output training data
        - model: Instance of NeuralNet class representing the trained neural network
        - radius: Radius parameter for the distance to the least activator data point (Default: 0)
        - validation_split: Proportion of the dataset to include in the validation split (Default: 0.2)
        - layer2: Configuration of the second layer ("classic" or "ridge" or "lstsq") (Default: "classic")
        - alpha: Regularization strength for Ridge regression (Default: 1)
        - num_intervals: Number of radius, equal distances between 0 and max_distance between X_train pairs (Default: 10)

        Returns:
        - Alpha: Optimal or chosen alpha value
        - Radius: Optimal or chosen radius value
        """
        if layer2 == "ridge" and radius == -1 and alpha == -1:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=validation_split
            )
            alpha, radius, self.weights, self.biases = choose_best_radius_alpha(
                X_train,
                y_train,
                X_val,
                y_val,
                model.weights,
                model.biases,
                num_intervals=num_intervals,
            )

        elif layer2 == "ridge" and radius != -1 and alpha == -1:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=validation_split
            )
            alpha, self.weights, self.biases = choose_best_alpha(
                X_train, y_train, X_val, y_val, model.weights, model.biases, radius=radius
            )

        elif layer2 == "ridge" and radius == -1 and alpha != 1:
            raise ValueError("cannot determine best radius given alpha")

        else:
            weights_l1, biases_l1 = compute_weights_biases_layer1(
                X_train, y_train, model.weights, model.biases, radius
            )
            weights_l2, biases_l2 = self._compute_weights_biases_layer2(
                X_train,
                y_train,
                model.weights,
                model.biases,
                weights_l1,
                biases_l1,
                layer2=layer2,
                alpha=alpha,
            )
            self.weights, self.biases = [weights_l1, weights_l2], [biases_l1, biases_l2]

        return alpha, radius

    def _compute_weights_biases_layer2(
        self, X, y, weights, biases, weights_l1, biases_l1, alpha=1, layer2="classic"
    ):
        if layer2 == "classic":
            return compute_weights_biases_layer2_classic(
                X, y, weights, biases, weights_l1, biases_l1
            )
        elif layer2 == "lstsq":
            return compute_weights_biases_layer2_lstsq(X, y, weights_l1, biases_l1)
        elif layer2 == "ridge":
            return compute_weights_biases_layer2_ridge(
                X, y, weights_l1, biases_l1, alpha=alpha
            )
        else:
            raise ValueError(
                f"Invalid layer2 value: {layer2}. Expected 'classic', 'lstsq', or 'ridge'"
            )
