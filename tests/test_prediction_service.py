"""
tests/test_prediction_service.py – tests for web/prediction_service.py

We mock the Keras models so TensorFlow does not need to run during tests.
"""

import io
import os
import sys
import unittest.mock as mock

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web.prediction_service import (
    predict_directory,
    predict_image_bytes,
    predict_single,
    preprocess_image_bytes,
    preprocess_image_path,
)


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def make_png_bytes(width=32, height=32, color=128) -> bytes:
    img = Image.new("L", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_fake_model(return_value: float = 3.0):
    """Return a mock that behaves like a Keras model."""
    m = mock.MagicMock()
    m.predict.return_value = np.array([[return_value]])
    return m


def make_fake_models(values=None) -> dict:
    """Build a models-dict like load_models() returns."""
    if values is None:
        values = {"Dense": 1.0, "DenseTwoHidden": 2.0, "CNN": 3.0, "CNNExtraHidden": 4.0}
    return {
        label: {"model": make_fake_model(v), "weights_loaded": True, "weights_path": f"/tmp/{label}.h5"}
        for label, v in values.items()
    }


# ──────────────────────────────────────────────────────────
# preprocess_image_bytes
# ──────────────────────────────────────────────────────────


class TestPreprocessImageBytes:
    def test_output_shape(self):
        arr = preprocess_image_bytes(make_png_bytes())
        assert arr.shape == (1, 32, 32, 1)

    def test_dtype_float32(self):
        arr = preprocess_image_bytes(make_png_bytes())
        assert arr.dtype == np.float32

    def test_values_normalized_0_to_1(self):
        arr = preprocess_image_bytes(make_png_bytes(color=255))
        assert float(arr.max()) <= 1.0 + 1e-6

        arr_black = preprocess_image_bytes(make_png_bytes(color=0))
        assert float(arr_black.min()) >= -1e-6

    def test_large_image_resized(self):
        arr = preprocess_image_bytes(make_png_bytes(100, 100))
        assert arr.shape == (1, 32, 32, 1)

    def test_small_image_resized(self):
        arr = preprocess_image_bytes(make_png_bytes(8, 8))
        assert arr.shape == (1, 32, 32, 1)


# ──────────────────────────────────────────────────────────
# preprocess_image_path
# ──────────────────────────────────────────────────────────


class TestPreprocessImagePath:
    def test_output_shape(self, tmp_path):
        p = tmp_path / "img.png"
        p.write_bytes(make_png_bytes())
        arr = preprocess_image_path(str(p))
        assert arr.shape == (1, 32, 32, 1)

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(Exception):
            preprocess_image_path(str(tmp_path / "missing.png"))


# ──────────────────────────────────────────────────────────
# predict_single
# ──────────────────────────────────────────────────────────


class TestPredictSingle:
    def test_returns_one_entry_per_model(self):
        models = make_fake_models()
        x = np.zeros((1, 32, 32, 1), dtype=np.float32)
        results = predict_single(x, models)
        assert len(results) == 4

    def test_result_has_required_keys(self):
        models = make_fake_models()
        x = np.zeros((1, 32, 32, 1), dtype=np.float32)
        for r in predict_single(x, models):
            assert "model" in r
            assert "raw" in r
            assert "rounded" in r
            assert "weights_loaded" in r

    def test_rounded_is_non_negative(self):
        # Even if the model outputs a negative raw value, rounded should be ≥ 0.
        models = make_fake_models({"Dense": -2.5})
        x = np.zeros((1, 32, 32, 1), dtype=np.float32)
        results = predict_single(x, models)
        assert results[0]["rounded"] >= 0

    def test_correct_raw_value(self):
        models = make_fake_models({"Dense": 3.7})
        x = np.zeros((1, 32, 32, 1), dtype=np.float32)
        results = predict_single(x, models)
        assert abs(results[0]["raw"] - 3.7) < 1e-5

    def test_rounded_value_correct(self):
        models = make_fake_models({"Dense": 2.6})
        x = np.zeros((1, 32, 32, 1), dtype=np.float32)
        results = predict_single(x, models)
        assert results[0]["rounded"] == 3  # round(2.6) = 3

    def test_weights_loaded_flag_propagated(self):
        models = {
            "Dense": {
                "model": make_fake_model(1.0),
                "weights_loaded": False,
                "weights_path": "",
            }
        }
        x = np.zeros((1, 32, 32, 1), dtype=np.float32)
        results = predict_single(x, models)
        assert results[0]["weights_loaded"] is False


# ──────────────────────────────────────────────────────────
# predict_image_bytes
# ──────────────────────────────────────────────────────────


class TestPredictImageBytes:
    def test_basic(self):
        models = make_fake_models()
        results = predict_image_bytes(make_png_bytes(), models)
        assert len(results) == 4
        for r in results:
            assert isinstance(r["raw"], float)

    def test_model_predict_called_once_per_model(self):
        models = make_fake_models()
        predict_image_bytes(make_png_bytes(), models)
        for info in models.values():
            info["model"].predict.assert_called_once()


# ──────────────────────────────────────────────────────────
# predict_directory
# ──────────────────────────────────────────────────────────


class TestPredictDirectory:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        models = make_fake_models()
        results = predict_directory(str(tmp_path), models)
        assert results == []

    def test_missing_dir_returns_empty_list(self, tmp_path):
        models = make_fake_models()
        results = predict_directory(str(tmp_path / "nope"), models)
        assert results == []

    def test_non_png_files_ignored(self, tmp_path):
        (tmp_path / "file.txt").write_bytes(b"text")
        models = make_fake_models()
        results = predict_directory(str(tmp_path), models)
        assert results == []

    def test_predicts_each_png(self, tmp_path):
        for i in range(3):
            (tmp_path / f"img_{i}.png").write_bytes(make_png_bytes())
        models = make_fake_models()
        results = predict_directory(str(tmp_path), models)
        assert len(results) == 3

    def test_result_filenames_match(self, tmp_path):
        names = ["a.png", "b.png"]
        for n in names:
            (tmp_path / n).write_bytes(make_png_bytes())
        models = make_fake_models()
        results = predict_directory(str(tmp_path), models)
        returned_names = [r["filename"] for r in results]
        assert set(returned_names) == set(names)

    def test_each_result_has_predictions_list(self, tmp_path):
        (tmp_path / "x.png").write_bytes(make_png_bytes())
        models = make_fake_models()
        results = predict_directory(str(tmp_path), models)
        assert "predictions" in results[0]
        assert isinstance(results[0]["predictions"], list)

    def test_results_sorted_by_filename(self, tmp_path):
        for n in ["z.png", "a.png", "m.png"]:
            (tmp_path / n).write_bytes(make_png_bytes())
        models = make_fake_models()
        results = predict_directory(str(tmp_path), models)
        names = [r["filename"] for r in results]
        assert names == sorted(names)
