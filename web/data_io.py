"""
data_io.py – helpers for saving and listing training/prediction images.

All writes to labels.csv are serialized through a module-level lock so that
two concurrent HTTP requests cannot corrupt the file.
"""

import base64
import csv
import io
import os
import re
import threading
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from PIL import Image

# One global lock for all CSV append operations.
_csv_lock = threading.Lock()

# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _decode_image_bytes(data: bytes) -> Image.Image:
    """Return a PIL Image from raw PNG/JPEG bytes."""
    return Image.open(io.BytesIO(data))


def _normalize_image(img: Image.Image) -> Image.Image:
    """Resize to 32×32 and convert to grayscale."""
    img = img.convert("L")  # grayscale
    img = img.resize((32, 32), Image.NEAREST)
    return img


def _generate_filename(prefix: str = "drawn") -> str:
    """Generate a timestamped filename that is very unlikely to collide."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}.png"


def _decode_base64_or_raw(data) -> bytes:
    """Accept either raw bytes or a base64-encoded string/bytes."""
    if isinstance(data, str):
        # strip optional data-URI header, e.g. "data:image/png;base64,..."
        if "," in data:
            data = data.split(",", 1)[1]
        return base64.b64decode(data)
    if isinstance(data, bytes):
        # Try to detect if this is base64-encoded rather than raw image bytes
        try:
            decoded = base64.b64decode(data)
            # Verify decoded data looks like a known image format using PIL
            try:
                Image.open(io.BytesIO(decoded)).verify()
                return decoded
            except Exception:
                pass
        except Exception:
            pass
        return data
    raise TypeError(f"Unsupported image data type: {type(data)}")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def save_training_sample(
    image_data,
    circles: int,
    data_dir: str,
) -> str:
    """Save a drawn image to training_data/images/ and append a row to labels.csv.

    Args:
        image_data: Raw PNG bytes, JPEG bytes, or a base64-encoded string.
        circles:    Number of circles in the image (non-negative integer).
        data_dir:   Root training data directory (contains labels.csv and images/).

    Returns:
        The filename of the saved image (not the full path).

    Raises:
        ValueError: If *circles* is negative.
        OSError:    If the directory cannot be created or the file cannot be written.
    """
    if circles < 0:
        raise ValueError(f"circles must be non-negative, got {circles}")

    raw = _decode_base64_or_raw(image_data)
    img = _decode_image_bytes(raw)
    img = _normalize_image(img)

    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    filename = _generate_filename("drawn")
    img_path = os.path.join(images_dir, filename)
    img.save(img_path, format="PNG")

    labels_path = os.path.join(data_dir, "labels.csv")

    with _csv_lock:
        file_exists = os.path.exists(labels_path)
        with open(labels_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(labels_path) == 0:
                writer.writerow(["filename", "circles"])
            writer.writerow([filename, circles])

    return filename


def save_prediction_sample(
    image_data,
    test_dir: str,
) -> str:
    """Save a drawn image to test_data/ for later prediction.

    Args:
        image_data: Raw PNG bytes, JPEG bytes, or a base64-encoded string.
        test_dir:   Directory to save prediction images (e.g. 'test_data/').

    Returns:
        The filename of the saved image (not the full path).
    """
    raw = _decode_base64_or_raw(image_data)
    img = _decode_image_bytes(raw)
    img = _normalize_image(img)

    os.makedirs(test_dir, exist_ok=True)

    filename = _generate_filename("pred")
    img_path = os.path.join(test_dir, filename)
    img.save(img_path, format="PNG")

    return filename


def list_training_samples(data_dir: str) -> List[dict]:
    """Return a list of {'filename': ..., 'circles': ...} dicts from labels.csv.

    Missing image files are included in the list but their 'exists' key is False.
    """
    labels_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "images")

    if not os.path.exists(labels_path):
        return []

    samples = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            circles_raw = (row.get("circles") or "").strip()
            if not filename:
                continue
            try:
                circles = int(circles_raw)
            except ValueError:
                circles = None

            img_path = os.path.join(images_dir, filename)
            samples.append(
                {
                    "filename": filename,
                    "circles": circles,
                    "exists": os.path.exists(img_path),
                }
            )
    return samples


def list_prediction_samples(test_dir: str) -> List[str]:
    """Return sorted list of PNG filenames in *test_dir*."""
    if not os.path.isdir(test_dir):
        return []
    return sorted(f for f in os.listdir(test_dir) if f.lower().endswith(".png"))


def update_training_label(data_dir: str, filename: str, circles: int) -> bool:
    """Update the circle label for a specific filename in labels.csv.

    Returns True if a row was updated, otherwise False.
    """
    if circles < 0:
        raise ValueError(f"circles must be non-negative, got {circles}")

    labels_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        return False

    with _csv_lock:
        with open(labels_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or ["filename", "circles"]

    updated = False
    for row in rows:
        if (row.get("filename") or "").strip() == filename:
            row["circles"] = str(circles)
            updated = True

    if not updated:
        return False

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return True


def read_image_as_png_bytes(path: str) -> bytes:
    """Return the raw PNG bytes for an image file (re-encoding if necessary)."""
    img = Image.open(path).convert("L").resize((32, 32), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ──────────────────────────────────────────────
# Dataset management
# ──────────────────────────────────────────────

#: Allowed characters for a custom dataset name.
_DATASET_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]{1,50}$')


def list_datasets(training_data_dir: str, custom_datasets_dir: str) -> List[dict]:
    """Return metadata for every available dataset.

    Always includes the built-in ``circle`` dataset (backed by *training_data_dir*),
    followed by any user-created custom datasets found under *custom_datasets_dir*.

    Returns a list of dicts with keys ``id``, ``type``, ``name``, ``sample_count``.
    """
    datasets: List[dict] = []

    # Built-in circle dataset
    circle_count = len(list_training_samples(training_data_dir))
    datasets.append(
        {"id": "circle", "type": "circle", "name": "Circle", "sample_count": circle_count}
    )

    # User-created custom datasets
    if os.path.isdir(custom_datasets_dir):
        for name in sorted(os.listdir(custom_datasets_dir)):
            ds_dir = os.path.join(custom_datasets_dir, name)
            if os.path.isdir(ds_dir):
                count = len(list_training_samples(ds_dir))
                datasets.append(
                    {
                        "id": f"custom_{name}",
                        "type": "custom",
                        "name": name,
                        "sample_count": count,
                    }
                )

    return datasets


def create_custom_dataset(name: str, custom_datasets_dir: str) -> dict:
    """Create a new empty custom dataset directory.

    Args:
        name:               Dataset name (letters, digits, underscores, hyphens; 1–50 chars).
        custom_datasets_dir: Root directory that holds all custom datasets.

    Returns:
        Metadata dict with ``id``, ``type``, ``name``, ``sample_count``.

    Raises:
        ValueError: If *name* is invalid or a dataset with that name already exists.
        OSError:    If the directory cannot be created.
    """
    if not _DATASET_NAME_RE.match(name):
        raise ValueError(
            "Dataset name must be 1–50 characters containing only letters, "
            "digits, underscores, or hyphens."
        )

    ds_dir = os.path.join(custom_datasets_dir, name)
    if os.path.exists(ds_dir):
        raise ValueError(f"A dataset named '{name}' already exists.")

    os.makedirs(os.path.join(ds_dir, "images"), exist_ok=True)

    return {"id": f"custom_{name}", "type": "custom", "name": name, "sample_count": 0}
