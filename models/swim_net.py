from sklearn.pipeline import Pipeline
from swimnetworks import Dense, Linear

from models.base_model import BaseModel


class SWIMNetwork(BaseModel):
    """Neural network model built using TensorFlow Keras and optimizer Adam."""

    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train, layers):
        steps = [
            (
                "dense",
                Dense(
                    layer_width=layers[0],
                    activation="relu",
                    parameter_sampler="relu",
                    random_seed=42,
                ),
            ),
            ("linear", Linear(regularization_scale=1e-10)),
        ]
        model_swim = Pipeline(steps)

        model_swim.fit(X_train, y_train)

        model_base_swim = SWIMNetwork()

        model_base_swim.weights = [
            model_swim.get_params()["steps"][0][1].weights,
            model_swim.get_params()["steps"][1][1].weights,
        ]
        model_base_swim.biases = [
            -model_swim.get_params()["steps"][0][1].biases[0],
            -model_swim.get_params()["steps"][1][1].biases[0],
        ]
        return model_base_swim
