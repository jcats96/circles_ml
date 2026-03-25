"""
CNN circle-counting model for 32x32 binary (black/white) images.

The model takes a 32x32x1 input tensor and predicts the number of circles in
the image as a non-negative integer-like value via regression.
"""

import tensorflow as tf
from tensorflow import keras


IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
INPUT_CHANNELS = 1


def build_model() -> keras.Model:
    """Build and return a CNN-based circle-counting model.

    Architecture:
        Input (32x32x1)
            -> Conv2D(32, 3x3, relu)
            -> MaxPooling2D(2x2)
            -> Conv2D(64, 3x3, relu)
            -> MaxPooling2D(2x2)
            -> Flatten
            -> Dense(64, relu)
            -> Dense(1, relu)
    """
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_CHANNELS)),
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="relu"),
        ],
        name="circle_counter_cnn",
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
