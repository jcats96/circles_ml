"""
Dense circle-counting model with two hidden layers.

This variant extends the baseline dense model by adding one additional hidden
Dense layer before the output.
"""

from tensorflow import keras


IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH  # 1024


def build_model(hidden_units_1: int = 128, hidden_units_2: int = 64) -> keras.Model:
    """Build and return a dense model with two hidden layers.

    Architecture:
        Input  -> Flatten (32x32 -> 1024)
               -> Dense(hidden_units_1, relu)
               -> Dense(hidden_units_2, relu)
               -> Dense(1, relu)

    Args:
        hidden_units_1: Number of neurons in the first hidden layer.
        hidden_units_2: Number of neurons in the second hidden layer.

    Returns:
        A compiled Keras model ready for training.
    """
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(hidden_units_1, activation="relu"),
            keras.layers.Dense(hidden_units_2, activation="relu"),
            keras.layers.Dense(1, activation="relu"),
        ],
        name="circle_counter_dense_two_hidden",
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
