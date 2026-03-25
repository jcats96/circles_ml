import argparse
import csv
import os
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import (
    build_cnn_extra_hidden_model,
    build_cnn_model,
    build_dense_model,
    build_dense_two_hidden_model,
)


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    labels_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "images")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    images = []
    labels = []
    skipped = 0

    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames or "circles" not in reader.fieldnames:
            raise ValueError("labels.csv must have headers: filename,circles")

        for row in reader:
            filename = (row.get("filename") or "").strip()
            circles_raw = (row.get("circles") or "").strip()

            if not filename:
                skipped += 1
                continue

            try:
                circles = float(int(circles_raw))
            except ValueError:
                skipped += 1
                continue

            img_path = os.path.join(images_dir, filename)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            img_bytes = tf.io.read_file(img_path)
            img = tf.image.decode_png(img_bytes, channels=1)
            img = tf.image.resize(img, [32, 32], method="nearest")
            img = tf.cast(img, tf.float32) / 255.0

            images.append(img.numpy())
            labels.append(circles)

    if not images:
        raise ValueError("No valid labeled samples found. Check labels.csv and image paths.")

    x = np.stack(images).astype(np.float32)
    y = np.array(labels, dtype=np.float32)

    print(f"Loaded {len(x)} samples from {data_dir}")
    if skipped:
        print(f"Skipped {skipped} invalid/missing rows from labels.csv")

    return x, y


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    val_split: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(x) < 2:
        return x, y, x, y

    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)

    val_size = int(round(len(x) * val_split))
    val_size = max(1, min(val_size, len(x) - 1))

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def train_one_model(
    label: str,
    weight_name: str,
    builder: Callable[[], keras.Model],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    weights_dir: str,
) -> str:
    model = builder()

    print("=" * 60)
    print(f"Training {label} model")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=min(batch_size, len(x_train)),
        verbose=2,
    )

    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{weight_name}.weights.h5")
    model.save_weights(weights_path)

    final_train_mae = history.history.get("mae", [None])[-1]
    final_val_mae = history.history.get("val_mae", [None])[-1]

    print(f"Saved {label} weights to: {weights_path}")
    print(f"{label} final train MAE: {final_train_mae}")
    print(f"{label} final val MAE: {final_val_mae}")

    return weights_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train circle-counting model variants.")
    parser.add_argument("--data-dir", default="training_data", help="Directory containing labels.csv and images/")
    parser.add_argument("--weights-dir", default="weights", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--val-split", type=float, default=0.25, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    x, y = load_dataset(args.data_dir)
    x_train, y_train, x_val, y_val = split_dataset(x, y, args.val_split, args.seed)

    print(f"Train samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")

    dense_weights = train_one_model(
        label="Dense",
        weight_name="dense",
        builder=build_dense_model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weights_dir=args.weights_dir,
    )

    dense_two_hidden_weights = train_one_model(
        label="DenseTwoHidden",
        weight_name="dense_two_hidden",
        builder=build_dense_two_hidden_model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weights_dir=args.weights_dir,
    )

    cnn_weights = train_one_model(
        label="CNN",
        weight_name="cnn",
        builder=build_cnn_model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weights_dir=args.weights_dir,
    )

    cnn_extra_hidden_weights = train_one_model(
        label="CNNExtraHidden",
        weight_name="cnn_extra_hidden",
        builder=build_cnn_extra_hidden_model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weights_dir=args.weights_dir,
    )

    print("=" * 60)
    print("Training complete")
    print(f"Dense weights: {dense_weights}")
    print(f"DenseTwoHidden weights: {dense_two_hidden_weights}")
    print(f"CNN weights: {cnn_weights}")
    print(f"CNNExtraHidden weights: {cnn_extra_hidden_weights}")
    print("To load later, run: python main.py --load-weights")


if __name__ == "__main__":
    main()
