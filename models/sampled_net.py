from typing import Literal

from models.base_model import BaseModel
from utils.layer1_logic import layer_1_conversion
from utils.layer2_logic import choose_best_threshold_ratio_and_alpha, choose_best_alpha_for_ridge, layer_2_conversion


class SampledNet(BaseModel):
    def __init__(self):
        super().__init__()

    def fit(
            self,
            X_train,
            model,
            r=0,
            validation_split=0.2,
            layer2: Literal["bias_only", "ridge", "lstsq"] = "bias_only",
            alpha=-1,
            num_intervals=10,
            verbose=1,
            project_onto_boundary=False,
            augment_data=None,
            choose_x_2: Literal["angle", "norm_kdtree", "norm"] = "norm"
    ):
        """
        Trains the sampled network model with provided training data by learning the weights and biases.

        Parameters:
        - X_train: Input training data
        - model: Instance of BaseModel class representing the trained neural network
        - r: threshold ratio r (Default: -1, find best value of r)
        - validation_split: Proportion of the dataset to include in the validation split (Default: 0.2)
        - layer2: Configuration of the second layer ("bias_only", "ridge", "lstsq") (Default: "bias_only")
        - alpha: Regularization strength for Ridge regression (Default: -1, find best alpha)
        - num_intervals: Number of r values for the hyperparameter search for r
        - verbose: Flag to print progress in finding hyperparameters
        - project_onto_boundary: Project x_1 onto the activation boundary of the neuron (Default: False)
        - augment_data: is a pair (n, sigma) to augment the input dataset using Gaussian noise. (Default: None)
        - choose_x_2: Configuration to choose the second point x_2  ("angle", "norm_kdtree", "norm") (Default: "norm")

        Returns:
        - Alpha: Optimal or chosen alpha value
        - r: Optimal or chosen radius value
        - x_1_x_2_pairs: A list of the sampling pairs for each neuron's parameters
        """
        if layer2 == "ridge" and r == -1 and alpha == -1:
            alpha, r, self.weights, self.biases = choose_best_threshold_ratio_and_alpha(
                X_train,
                model.weights,
                model.biases,
                num_intervals=num_intervals,
                verbose=verbose,
                validation_split=validation_split,
                choose_x_2=choose_x_2
            )
            return alpha, r

        elif layer2 == "ridge" and r != -1 and alpha == -1:
            W_1_hat, b_1_hat, x_1_x_2_pairs = layer_1_conversion(
                X_train, model.weights, model.biases, r, project_onto_boundary,
                augment_data, choose_x_2
            )
            alpha, self.weights, self.biases = choose_best_alpha_for_ridge(
                X_train,
                W_1_hat,
                b_1_hat,
                model.weights,
                model.biases,
                verbose=verbose,
                validation_split=validation_split,
            )
            return alpha

        elif layer2 == "ridge" and r == -1 and alpha != 1:
            raise ValueError("cannot determine best r given alpha")

        else:
            W_1_hat, b_1_hat, x_1_x_2_pairs = layer_1_conversion(
                X_train, model.weights, model.biases, r, project_onto_boundary,
                augment_data, choose_x_2
            )
            weights_l2, biases_l2 = layer_2_conversion(
                X_train,
                model.weights,
                model.biases,
                W_1_hat,
                b_1_hat,
                layer2=layer2,
                alpha=alpha,
            )
            self.weights, self.biases = [W_1_hat, weights_l2], [b_1_hat, biases_l2]
            return x_1_x_2_pairs
