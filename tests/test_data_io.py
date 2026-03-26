"""
tests/test_data_io.py – tests for web/data_io.py
"""

import base64
import csv
import io
import os
import sys
import threading

import pytest
from PIL import Image

# ── ensure project root is on the path ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web.data_io import (
    _decode_base64_or_raw,
    _generate_filename,
    _normalize_image,
    list_prediction_samples,
    list_training_samples,
    read_image_as_png_bytes,
    save_prediction_sample,
    save_training_sample,
)


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def make_png_bytes(width=16, height=16, color=128) -> bytes:
    """Return a small in-memory grayscale PNG."""
    img = Image.new("L", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_b64_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode()
    return f"data:image/png;base64,{b64}"


# ──────────────────────────────────────────────────────────
# _decode_base64_or_raw
# ──────────────────────────────────────────────────────────


class TestDecodeBase64OrRaw:
    def test_raw_bytes_returned_as_is(self):
        png = make_png_bytes()
        result = _decode_base64_or_raw(png)
        assert result == png

    def test_base64_string_decoded(self):
        png = make_png_bytes()
        b64 = base64.b64encode(png).decode()
        result = _decode_base64_or_raw(b64)
        assert result == png

    def test_data_uri_string_decoded(self):
        png = make_png_bytes()
        uri = make_b64_data_uri(png)
        result = _decode_base64_or_raw(uri)
        assert result == png

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            _decode_base64_or_raw(12345)


# ──────────────────────────────────────────────────────────
# _normalize_image
# ──────────────────────────────────────────────────────────


class TestNormalizeImage:
    def test_output_size_is_32x32(self):
        img = Image.new("RGB", (64, 128), color=(200, 100, 50))
        result = _normalize_image(img)
        assert result.size == (32, 32)

    def test_output_mode_is_grayscale(self):
        img = Image.new("RGB", (16, 16))
        result = _normalize_image(img)
        assert result.mode == "L"

    def test_already_32x32_grayscale_unchanged(self):
        img = Image.new("L", (32, 32), color=255)
        result = _normalize_image(img)
        assert result.size == (32, 32)
        assert result.mode == "L"


# ──────────────────────────────────────────────────────────
# _generate_filename
# ──────────────────────────────────────────────────────────


class TestGenerateFilename:
    def test_ends_with_png(self):
        assert _generate_filename().endswith(".png")

    def test_uses_prefix(self):
        name = _generate_filename("myprefix")
        assert name.startswith("myprefix_")

    def test_two_calls_differ(self):
        import time
        n1 = _generate_filename()
        time.sleep(0.01)
        n2 = _generate_filename()
        assert n1 != n2


# ──────────────────────────────────────────────────────────
# save_training_sample
# ──────────────────────────────────────────────────────────


class TestSaveTrainingSample:
    def test_creates_image_file(self, tmp_path):
        png = make_png_bytes()
        filename = save_training_sample(png, 3, str(tmp_path))
        assert (tmp_path / "images" / filename).exists()

    def test_appends_to_labels_csv(self, tmp_path):
        png = make_png_bytes()
        filename = save_training_sample(png, 2, str(tmp_path))

        labels_path = tmp_path / "labels.csv"
        assert labels_path.exists()
        with open(labels_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["filename"] == filename
        assert rows[0]["circles"] == "2"

    def test_multiple_saves_append_rows(self, tmp_path):
        png = make_png_bytes()
        save_training_sample(png, 1, str(tmp_path))
        save_training_sample(png, 4, str(tmp_path))

        labels_path = tmp_path / "labels.csv"
        with open(labels_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_header_written_once(self, tmp_path):
        png = make_png_bytes()
        for _ in range(3):
            save_training_sample(png, 0, str(tmp_path))

        labels_path = tmp_path / "labels.csv"
        with open(labels_path) as f:
            lines = f.readlines()
        # Only one header line
        headers = [l for l in lines if l.startswith("filename")]
        assert len(headers) == 1

    def test_negative_circles_raises(self, tmp_path):
        with pytest.raises(ValueError, match="non-negative"):
            save_training_sample(make_png_bytes(), -1, str(tmp_path))

    def test_accepts_base64_data_uri(self, tmp_path):
        uri = make_b64_data_uri(make_png_bytes())
        filename = save_training_sample(uri, 0, str(tmp_path))
        assert (tmp_path / "images" / filename).exists()

    def test_saved_image_is_32x32(self, tmp_path):
        png = make_png_bytes(64, 64)
        filename = save_training_sample(png, 1, str(tmp_path))
        saved = Image.open(tmp_path / "images" / filename)
        assert saved.size == (32, 32)

    def test_concurrent_saves_no_corruption(self, tmp_path):
        """Multiple threads should produce exactly one header row."""
        png = make_png_bytes()
        errors = []

        def _save():
            try:
                save_training_sample(png, 1, str(tmp_path))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_save) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        labels_path = tmp_path / "labels.csv"
        with open(labels_path, newline="") as f:
            lines = f.readlines()
        headers = [l for l in lines if l.startswith("filename")]
        assert len(headers) == 1

        with open(labels_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10


# ──────────────────────────────────────────────────────────
# save_prediction_sample
# ──────────────────────────────────────────────────────────


class TestSavePredictionSample:
    def test_creates_file_in_test_dir(self, tmp_path):
        png = make_png_bytes()
        test_dir = str(tmp_path / "test_data")
        filename = save_prediction_sample(png, test_dir)
        assert os.path.exists(os.path.join(test_dir, filename))

    def test_filename_starts_with_pred(self, tmp_path):
        png = make_png_bytes()
        test_dir = str(tmp_path / "test_data")
        filename = save_prediction_sample(png, test_dir)
        assert filename.startswith("pred_")

    def test_creates_directory_if_missing(self, tmp_path):
        png = make_png_bytes()
        test_dir = str(tmp_path / "new_dir" / "nested")
        save_prediction_sample(png, test_dir)
        assert os.path.isdir(test_dir)

    def test_saved_image_is_32x32(self, tmp_path):
        png = make_png_bytes(100, 100)
        test_dir = str(tmp_path / "test_data")
        filename = save_prediction_sample(png, test_dir)
        saved = Image.open(os.path.join(test_dir, filename))
        assert saved.size == (32, 32)


# ──────────────────────────────────────────────────────────
# list_training_samples
# ──────────────────────────────────────────────────────────


class TestListTrainingSamples:
    def test_returns_empty_when_no_csv(self, tmp_path):
        assert list_training_samples(str(tmp_path)) == []

    def test_returns_correct_entries(self, tmp_path):
        # Write a real image and CSV.
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        img = Image.new("L", (32, 32), 0)
        img.save(str(images_dir / "a.png"))

        labels = tmp_path / "labels.csv"
        labels.write_text("filename,circles\na.png,3\n")

        samples = list_training_samples(str(tmp_path))
        assert len(samples) == 1
        assert samples[0]["filename"] == "a.png"
        assert samples[0]["circles"] == 3
        assert samples[0]["exists"] is True

    def test_missing_image_marked_not_exists(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels.csv").write_text("filename,circles\nghost.png,1\n")
        samples = list_training_samples(str(tmp_path))
        assert samples[0]["exists"] is False

    def test_invalid_circles_returns_none(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels.csv").write_text("filename,circles\nbad.png,notanumber\n")
        samples = list_training_samples(str(tmp_path))
        assert samples[0]["circles"] is None

    def test_skips_rows_with_empty_filename(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "labels.csv").write_text("filename,circles\n,3\nvalid.png,1\n")
        samples = list_training_samples(str(tmp_path))
        assert len(samples) == 1
        assert samples[0]["filename"] == "valid.png"


# ──────────────────────────────────────────────────────────
# list_prediction_samples
# ──────────────────────────────────────────────────────────


class TestListPredictionSamples:
    def test_returns_empty_when_dir_missing(self, tmp_path):
        assert list_prediction_samples(str(tmp_path / "nonexistent")) == []

    def test_returns_only_png_files(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"")
        (tmp_path / "b.PNG").write_bytes(b"")
        (tmp_path / "c.txt").write_bytes(b"")
        result = list_prediction_samples(str(tmp_path))
        assert set(result) == {"a.png", "b.PNG"}

    def test_returns_sorted(self, tmp_path):
        for name in ["z.png", "a.png", "m.png"]:
            (tmp_path / name).write_bytes(b"")
        result = list_prediction_samples(str(tmp_path))
        assert result == sorted(result)


# ──────────────────────────────────────────────────────────
# read_image_as_png_bytes
# ──────────────────────────────────────────────────────────


class TestReadImageAsPngBytes:
    def test_returns_bytes(self, tmp_path):
        img = Image.new("L", (32, 32), 200)
        p = tmp_path / "test.png"
        img.save(str(p))
        data = read_image_as_png_bytes(str(p))
        assert isinstance(data, bytes)
        assert data[:4] == b"\x89PNG"

    def test_output_is_32x32_png(self, tmp_path):
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        p = tmp_path / "test.png"
        img.save(str(p))
        data = read_image_as_png_bytes(str(p))
        result = Image.open(io.BytesIO(data))
        assert result.size == (32, 32)
