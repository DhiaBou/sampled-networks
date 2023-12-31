import numpy as np
import tensorflow as tf
from keras import Sequential

from models.base_model import BaseModel


class NeuralNet(BaseModel):
    """Neural network model built using TensorFlow Keras and optimizer Adam."""

    def __init__(self):
        super().__init__()
        self.model: Sequential = None

    def fit(self, X_train, y_train, layers, validation_split=0.2, epochs=200):
        """Train the neural network on data X_train and targets y_train.

        Args:
            X_train: Input data for training
            y_train: Target values for training
            layers: List of layer widths
            validation_split: Validation set fraction of training data
            epochs: Number of training epochs
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))

        for layer_size in layers:
            model.add(tf.keras.layers.Dense(layer_size, activation="relu", kernel_initializer="he_normal",
                                            ))

        output_dim = 1 if np.ndim(y_train) == 1 else y_train.shape[1]
        model.add(tf.keras.layers.Dense(output_dim, kernel_initializer="he_normal",
                                        ))

        model.compile(optimizer="adam", loss="mse")

        model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            verbose=1,
        )

        self.model = model

        # Extract weights and biases
        self.weights = [layer.get_weights()[0] for layer in model.layers]
        self.biases = [-layer.get_weights()[1] for layer in model.layers]

    def resume_training(self, X_train, y_train, initial_epoch=0, epochs=200, validation_split=0.2):
        self.model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            initial_epoch=initial_epoch,
            epochs=epochs,
            verbose=0,
        )

        # Extract weights and biases
        self.weights = [layer.get_weights()[0] for layer in self.model.layers]
        self.biases = [-layer.get_weights()[1] for layer in self.model.layers]
