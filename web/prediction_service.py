"""
prediction_service.py – wraps the prediction logic from main.py so it can
be called by the web server without spawning a subprocess.
"""

import io
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


# ──────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────


def preprocess_image_bytes(data: bytes) -> np.ndarray:
    """Convert raw PNG/JPEG bytes to a (1, 32, 32, 1) float32 numpy array."""
    img = Image.open(io.BytesIO(data)).convert("L").resize((32, 32), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 32, 32, 1)


def preprocess_image_path(path: str) -> np.ndarray:
    """Load a PNG from *path* and return a (1, 32, 32, 1) float32 array."""
    with open(path, "rb") as f:
        return preprocess_image_bytes(f.read())


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────


def load_models(weights_dir: str) -> Dict:
    """Load CNN model variants with their weights.

    Returns a dict keyed by model label.  Models whose weight files are
    missing are still included but with weights=None (untrained).
    """
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from models import (
        build_cnn_extra_hidden_model,
        build_cnn_model,
        build_cnn_one_hidden_model,
    )

    specs = [
        ("CNN", "cnn", build_cnn_model),
        ("CNNOneHidden", "cnn_one_hidden", build_cnn_one_hidden_model),
        ("CNNExtraHidden", "cnn_extra_hidden", build_cnn_extra_hidden_model),
    ]

    models = {}
    for label, weight_name, builder in specs:
        model = builder()
        weights_path = os.path.join(weights_dir, f"{weight_name}.weights.h5")
        loaded = False
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            loaded = True
        models[label] = {"model": model, "weights_loaded": loaded, "weights_path": weights_path}

    return models


# ──────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────


def predict_single(x: np.ndarray, models: Dict) -> List[dict]:
    """Run *x* through all loaded models and return a list of predictions.

    Each item in the returned list has:
        model, raw, rounded, weights_loaded
    """
    results = []
    for label, info in models.items():
        model = info["model"]
        raw = float(model.predict(x, verbose=0)[0][0])
        results.append(
            {
                "model": label,
                "raw": raw,
                "rounded": max(0, round(raw)),
                "weights_loaded": info["weights_loaded"],
            }
        )
    return results


def predict_image_bytes(data: bytes, models: Dict) -> List[dict]:
    """Predict on a single image supplied as raw bytes."""
    x = preprocess_image_bytes(data)
    return predict_single(x, models)


def predict_directory(directory: str, models: Dict) -> List[dict]:
    """Predict on every PNG in *directory*.

    Returns a list of dicts, one per image file, each containing:
        filename, predictions (list from predict_single)
    """
    if not os.path.isdir(directory):
        return []

    results = []
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(".png"):
            continue
        path = os.path.join(directory, name)
        x = preprocess_image_path(path)
        preds = predict_single(x, models)
        results.append({"filename": name, "predictions": preds})

    return results
