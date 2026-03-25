"""
Circle-counting model for 32x32 binary (black/white) images.

The model takes a flattened 32x32 = 1024-element input vector where each
value is 0 (black) or 1 (white), and predicts the number of circles in the
image as a non-negative integer via regression.
"""

import tensorflow as tf
from tensorflow import keras


IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH  # 1024


def build_model(hidden_units: int = 128) -> keras.Model:
    """Build and return the circle-counting model.

    Architecture:
        Input  -> Flatten (32x32 -> 1024)
               -> Dense(hidden_units, relu)   # one hidden layer
               -> Dense(1, relu)              # output: predicted circle count

    Args:
        hidden_units: Number of neurons in the single hidden layer.

    Returns:
        A compiled Keras model ready for training.
    """
    model = keras.Sequential(
        [
            # Accept either a (32, 32) or (32, 32, 1) image tensor.
            keras.layers.InputLayer(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)),
            # Flatten 32x32x1 -> 1024
            keras.layers.Flatten(),
            # Single hidden layer with ReLU activation
            keras.layers.Dense(hidden_units, activation="relu"),
            # Output: a single continuous value representing the circle count
            keras.layers.Dense(1, activation="relu"),
        ],
        name="circle_counter",
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"],
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
