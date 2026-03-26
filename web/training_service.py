"""
training_service.py – runs model training in a background thread and streams
per-epoch metrics back to the caller via the TrainingJob object.

This module wraps the existing train_models.py logic so the web server can
import it directly without shelling out to a subprocess.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from web.training_jobs import EpochMetric, JobStatus, TrainingJob


MODEL_SPECS = [
    ("CNN", "cnn", "build_cnn_model"),
    ("CNNOneHidden", "cnn_one_hidden", "build_cnn_one_hidden_model"),
    ("CNNExtraHidden", "cnn_extra_hidden", "build_cnn_extra_hidden_model"),
]


def _clear_existing_weights(weights_dir: str) -> None:
    """Remove any existing web-model weight files before a new run starts."""
    for _, weight_name, _ in MODEL_SPECS:
        weights_path = os.path.join(weights_dir, f"{weight_name}.weights.h5")
        if os.path.exists(weights_path):
            os.remove(weights_path)


# ──────────────────────────────────────────────
# Keras callback
# ──────────────────────────────────────────────


def _make_callback(job: TrainingJob, model_label: str):
    """Return a Keras callback that pushes epoch metrics into *job*."""
    # Import inside function to avoid loading TF at module import time.
    from tensorflow import keras

    class StreamingCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if job.status == JobStatus.CANCELLED:
                self.model.stop_training = True
                return
            logs = logs or {}
            metric = EpochMetric(
                model=model_label,
                epoch=epoch + 1,  # 1-based for display
                loss=float(logs.get("loss", 0.0)),
                mae=float(logs.get("mae", 0.0)),
                val_loss=float(logs["val_loss"]) if "val_loss" in logs else None,
                val_mae=float(logs["val_mae"]) if "val_mae" in logs else None,
            )
            job.add_metric(metric)

    return StreamingCallback()


# ──────────────────────────────────────────────
# Core training function
# ──────────────────────────────────────────────


def _train_models(
    job: TrainingJob,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    selected_models: list[str],
    weights_dir: str,
    runs_dir: str,
    transfer_learning: bool = False,
) -> None:
    """Train all four model variants and update *job* with metrics/summary.

    This function is meant to be called in a background thread.
    """
    # Lazy imports so TF is only loaded when actually training.
    from models import (
        build_cnn_extra_hidden_model,
        build_cnn_model,
        build_cnn_one_hidden_model,
    )
    from tensorflow import keras

    job.status = JobStatus.RUNNING

    builders = {
        "build_cnn_model": build_cnn_model,
        "build_cnn_one_hidden_model": build_cnn_one_hidden_model,
        "build_cnn_extra_hidden_model": build_cnn_extra_hidden_model,
    }
    model_specs = [
        (label, weight_name, builders[builder_name])
        for label, weight_name, builder_name in MODEL_SPECS
        if label in selected_models
    ]

    os.makedirs(weights_dir, exist_ok=True)
    if not transfer_learning:
        _clear_existing_weights(weights_dir)

    run_dir = os.path.join(runs_dir, job.job_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    summary_models = {}
    job_start = time.monotonic()

    for label, weight_name, builder in model_specs:
        if job.status == JobStatus.CANCELLED:
            break

        model_start = time.monotonic()
        model = builder()

        # ── Transfer learning: load circle weights & freeze conv layers ──
        if transfer_learning:
            pretrained_path = os.path.join(weights_dir, f"{weight_name}.weights.h5")
            if os.path.exists(pretrained_path):
                model.load_weights(pretrained_path)
                for layer in model.layers:
                    if isinstance(layer, (keras.layers.Conv2D, keras.layers.MaxPooling2D)):
                        layer.trainable = False
                # Re-compile so the optimizer only tracks trainable params
                model.compile(
                    optimizer="adam",
                    loss="mean_squared_error",
                    metrics=["mae"],
                )

        effective_batch = min(batch_size, len(x_train))

        from tensorflow import keras as _keras
        early_stop = _keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=10, restore_best_weights=True
        )

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=effective_batch,
            verbose=0,
            callbacks=[_make_callback(job, label), early_stop],
        )

        if job.status == JobStatus.CANCELLED:
            break

        weights_path = os.path.join(weights_dir, f"{weight_name}.weights.h5")
        model.save_weights(weights_path)

        final_train_mae = float(history.history.get("mae", [None])[-1] or 0.0)
        final_val_mae = float(history.history.get("val_mae", [None])[-1] or 0.0)

        model_elapsed = time.monotonic() - model_start
        summary_models[label] = {
            "weights_path": weights_path,
            "final_train_mae": final_train_mae,
            "final_val_mae": final_val_mae,
            "elapsed_seconds": round(model_elapsed, 2),
        }

        # Persist per-epoch metrics to disk.
        with open(metrics_path, "a", encoding="utf-8") as fh:
            for i, (loss, mae) in enumerate(
                zip(
                    history.history.get("loss", []),
                    history.history.get("mae", []),
                )
            ):
                val_loss = history.history.get("val_loss", [None] * (i + 1))[i]
                val_mae = history.history.get("val_mae", [None] * (i + 1))[i]
                record = {
                    "job_id": job.job_id,
                    "model": label,
                    "epoch": i + 1,
                    "loss": loss,
                    "mae": mae,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                }
                fh.write(json.dumps(record) + "\n")

    if job.status != JobStatus.CANCELLED:
        total_elapsed = time.monotonic() - job_start
        summary = {
            "job_id": job.job_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(total_elapsed, 2),
            "config": job.config,
            "models": summary_models,
        }
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        job.finish(summary)


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────


def start_training_job(
    job: TrainingJob,
    x: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.25,
    seed: int = 42,
    weights_dir: str = "weights",
    runs_dir: str = "runs",
    on_complete: Optional[Callable[[], None]] = None,
) -> threading.Thread:
    """Split the dataset and launch training in a daemon thread.

    Returns the Thread object so callers can join it if they want.
    """
    # Lazy import so we don't drag in train_models at server start.
    import sys
    import os as _os
    # Ensure the project root is importable.
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from train_models import split_dataset, augment_training_data

    x_train, y_train, x_val, y_val = split_dataset(x, y, val_split, seed)
    x_train, y_train = augment_training_data(x_train, y_train)

    epochs = int(job.config.get("epochs", 10))
    batch_size = int(job.config.get("batch_size", 8))
    selected_models = list(job.config.get("models") or [label for label, _, _ in MODEL_SPECS])
    transfer_learning = bool(job.config.get("transfer_learning", False))

    def _run():
        try:
            _train_models(
                job=job,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                selected_models=selected_models,
                weights_dir=weights_dir,
                runs_dir=runs_dir,
                transfer_learning=transfer_learning,
            )
        except Exception as exc:
            job.fail(str(exc))
        finally:
            if on_complete is not None:
                on_complete()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
