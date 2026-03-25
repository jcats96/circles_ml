import argparse
import os

import tensorflow as tf

from models import (
    build_cnn_extra_hidden_model,
    build_cnn_model,
    build_dense_model,
    build_dense_two_hidden_model,
)


def print_model_info(label: str, model) -> None:
    print(f"{label} model name: {model.name}")
    print(f"{label} trainable params: {model.count_params()}")
    print("-" * 40)


def maybe_load_weights(label: str, model, weights_path: str) -> None:
    if not os.path.exists(weights_path):
        print(f"{label} weights not found at: {weights_path}")
        return

    model.load_weights(weights_path)
    print(f"{label} weights loaded from: {weights_path}")


def load_image_for_prediction(image_path: str):
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_png(img_bytes, channels=1)
    img = tf.image.resize(img, [32, 32], method="nearest")
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, axis=0)


def predict_directory(
    dense_model,
    dense_two_hidden_model,
    cnn_model,
    cnn_extra_hidden_model,
    predict_dir: str,
) -> None:
    if not os.path.isdir(predict_dir):
        print(f"Prediction directory not found: {predict_dir}")
        return

    image_names = sorted(
        name for name in os.listdir(predict_dir) if name.lower().endswith(".png")
    )

    if not image_names:
        print(f"No PNG files found in: {predict_dir}")
        return

    print("Predictions")
    print("=" * 40)
    for name in image_names:
        image_path = os.path.join(predict_dir, name)
        x = load_image_for_prediction(image_path)

        dense_pred = float(dense_model.predict(x, verbose=0)[0][0])
        dense_two_hidden_pred = float(dense_two_hidden_model.predict(x, verbose=0)[0][0])
        cnn_pred = float(cnn_model.predict(x, verbose=0)[0][0])
        cnn_extra_hidden_pred = float(cnn_extra_hidden_model.predict(x, verbose=0)[0][0])

        print(f"{name}")
        print(f"  Dense: {dense_pred:.3f} (rounded: {max(0, round(dense_pred))})")
        print(
            f"  Dense2Hidden: {dense_two_hidden_pred:.3f} "
            f"(rounded: {max(0, round(dense_two_hidden_pred))})"
        )
        print(f"  CNN:   {cnn_pred:.3f} (rounded: {max(0, round(cnn_pred))})")
        print(
            f"  CNNExtraHidden: {cnn_extra_hidden_pred:.3f} "
            f"(rounded: {max(0, round(cnn_extra_hidden_pred))})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build circle-counting models and optionally load weights.")
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load saved weights for both models from --weights-dir.",
    )
    parser.add_argument(
        "--weights-dir",
        default="weights",
        help="Directory that contains dense.weights.h5, dense_two_hidden.weights.h5, cnn.weights.h5, and cnn_extra_hidden.weights.h5.",
    )
    parser.add_argument(
        "--predict-dir",
        default="test_data",
        help="Directory containing PNG images to run predictions on.",
    )
    args = parser.parse_args()

    dense_model = build_dense_model()
    dense_two_hidden_model = build_dense_two_hidden_model()
    cnn_model = build_cnn_model()
    cnn_extra_hidden_model = build_cnn_extra_hidden_model()

    if args.load_weights:
        maybe_load_weights("Dense", dense_model, os.path.join(args.weights_dir, "dense.weights.h5"))
        maybe_load_weights(
            "DenseTwoHidden",
            dense_two_hidden_model,
            os.path.join(args.weights_dir, "dense_two_hidden.weights.h5"),
        )
        maybe_load_weights("CNN", cnn_model, os.path.join(args.weights_dir, "cnn.weights.h5"))
        maybe_load_weights(
            "CNNExtraHidden",
            cnn_extra_hidden_model,
            os.path.join(args.weights_dir, "cnn_extra_hidden.weights.h5"),
        )

    print("Model import/build check")
    print("=" * 40)
    print_model_info("Dense", dense_model)
    print_model_info("DenseTwoHidden", dense_two_hidden_model)
    print_model_info("CNN", cnn_model)
    print_model_info("CNNExtraHidden", cnn_extra_hidden_model)
    predict_directory(
        dense_model,
        dense_two_hidden_model,
        cnn_model,
        cnn_extra_hidden_model,
        args.predict_dir,
    )


if __name__ == "__main__":
    main()
