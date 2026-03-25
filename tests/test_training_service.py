"""
tests/test_training_service.py – tests for web/training_service.py

TensorFlow is mocked via tests/conftest.py so no real TF is needed.
"""

import os
import sys
import types
import unittest.mock as mock

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from web.training_jobs import EpochMetric, JobStatus, TrainingJob, create_job
from web.training_service import _make_callback, start_training_job


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def _make_fake_model(history_data):
    """Return a mock Keras model whose fit() returns a History-like object
    AND calls the epoch callbacks so metrics get emitted."""
    hist = mock.MagicMock()
    hist.history = history_data

    def fake_fit(x, y, validation_data=None, epochs=1, batch_size=8, verbose=0, callbacks=None):
        n_epochs = len(history_data.get("loss", []))
        for i in range(n_epochs):
            logs = {
                "loss": history_data["loss"][i],
                "mae":  history_data["mae"][i],
            }
            if "val_loss" in history_data:
                logs["val_loss"] = history_data["val_loss"][i]
                logs["val_mae"]  = history_data["val_mae"][i]
            if callbacks:
                for cb in callbacks:
                    cb.model = model
                    cb.on_epoch_end(i, logs)
        return hist

    model = mock.MagicMock()
    model.fit.side_effect = fake_fit
    return model


_DUMMY_HISTORY = {
    "loss":    [0.5, 0.4],
    "mae":     [0.3, 0.25],
    "val_loss": [0.6, 0.5],
    "val_mae":  [0.4, 0.35],
}


@pytest.fixture(autouse=True)
def patch_model_builders(monkeypatch):
    """Replace all four model builders so no real Keras model is built."""
    # models/__init__.py re-exports these; patch them there.
    # We also need to ensure `models` itself doesn't fail to import.
    # Patch each builder in the models package.
    models_mock = types.ModuleType("models")
    models_mock.build_dense_model            = lambda: _make_fake_model(_DUMMY_HISTORY)
    models_mock.build_dense_two_hidden_model = lambda: _make_fake_model(_DUMMY_HISTORY)
    models_mock.build_cnn_model              = lambda: _make_fake_model(_DUMMY_HISTORY)
    models_mock.build_cnn_extra_hidden_model = lambda: _make_fake_model(_DUMMY_HISTORY)

    monkeypatch.setitem(sys.modules, "models", models_mock)

    # Also patch train_models.split_dataset (used by start_training_job)
    train_models_mock = types.ModuleType("train_models")

    def _split(x, y, val_split, seed):
        n = max(1, int(len(x) * (1 - val_split)))
        return x[:n], y[:n], x[n:], y[n:]

    train_models_mock.split_dataset = _split
    monkeypatch.setitem(sys.modules, "train_models", train_models_mock)


# ──────────────────────────────────────────────────────────
# _make_callback
# ──────────────────────────────────────────────────────────


class TestMakeCallback:
    def test_epoch_end_adds_metric(self):
        job = TrainingJob(job_id="cb-test-001")
        cb = _make_callback(job, "TestModel")
        logs = {"loss": 0.5, "mae": 0.3, "val_loss": 0.6, "val_mae": 0.4}
        cb.on_epoch_end(0, logs)
        assert len(job.metrics) == 1
        m = job.metrics[0]
        assert m.model == "TestModel"
        assert m.epoch == 1  # 1-based
        assert abs(m.loss - 0.5) < 1e-6
        assert abs(m.mae - 0.3) < 1e-6

    def test_epoch_end_val_metrics_optional(self):
        job = TrainingJob(job_id="cb-test-002")
        cb = _make_callback(job, "TestModel")
        cb.on_epoch_end(0, {"loss": 0.2, "mae": 0.1})
        m = job.metrics[0]
        assert m.val_loss is None
        assert m.val_mae is None

    def test_cancelled_job_stops_training(self):
        job = TrainingJob(job_id="cb-test-003")
        job.status = JobStatus.CANCELLED
        cb = _make_callback(job, "TestModel")
        cb.model = mock.MagicMock()
        cb.model.stop_training = False
        cb.on_epoch_end(0, {"loss": 0.2, "mae": 0.1})
        assert cb.model.stop_training is True
        assert len(job.metrics) == 0

    def test_epoch_number_is_one_based(self):
        job = TrainingJob(job_id="cb-test-004")
        cb = _make_callback(job, "TestModel")
        for i in range(3):
            cb.on_epoch_end(i, {"loss": 0.1, "mae": 0.1})
        epochs = [m.epoch for m in job.metrics]
        assert epochs == [1, 2, 3]


# ──────────────────────────────────────────────────────────
# start_training_job
# ──────────────────────────────────────────────────────────


class TestStartTrainingJob:
    def _make_data(self, n=10):
        rng = np.random.default_rng(0)
        x = rng.random((n, 32, 32, 1)).astype(np.float32)
        y = rng.integers(0, 5, n).astype(np.float32)
        return x, y

    def test_job_completes(self, tmp_path):
        x, y = self._make_data()
        job = create_job({"epochs": 2, "batch_size": 4})
        thread = start_training_job(
            job=job, x=x, y=y, val_split=0.25, seed=42,
            weights_dir=str(tmp_path / "weights"),
            runs_dir=str(tmp_path / "runs"),
        )
        thread.join(timeout=15)
        assert job.status == JobStatus.COMPLETED

    def test_job_has_summary(self, tmp_path):
        x, y = self._make_data()
        job = create_job({"epochs": 2, "batch_size": 4})
        thread = start_training_job(
            job=job, x=x, y=y,
            weights_dir=str(tmp_path / "weights"),
            runs_dir=str(tmp_path / "runs"),
        )
        thread.join(timeout=15)
        assert job.summary is not None
        assert "models" in job.summary
        assert len(job.summary["models"]) == 4

    def test_metrics_are_collected(self, tmp_path):
        x, y = self._make_data()
        job = create_job({"epochs": 2, "batch_size": 4})
        thread = start_training_job(
            job=job, x=x, y=y,
            weights_dir=str(tmp_path / "weights"),
            runs_dir=str(tmp_path / "runs"),
        )
        thread.join(timeout=15)
        # 4 models × 2 epochs = 8 metrics
        assert len(job.metrics) == 8

    def test_metrics_jsonl_written(self, tmp_path):
        x, y = self._make_data()
        job = create_job({"epochs": 2, "batch_size": 4})
        thread = start_training_job(
            job=job, x=x, y=y,
            weights_dir=str(tmp_path / "weights"),
            runs_dir=str(tmp_path / "runs"),
        )
        thread.join(timeout=15)
        metrics_path = tmp_path / "runs" / job.job_id / "metrics.jsonl"
        assert metrics_path.exists()
        lines = metrics_path.read_text().strip().splitlines()
        assert len(lines) == 8  # 4 models × 2 epochs

    def test_summary_json_written(self, tmp_path):
        import json

        x, y = self._make_data()
        job = create_job({"epochs": 2, "batch_size": 4})
        thread = start_training_job(
            job=job, x=x, y=y,
            weights_dir=str(tmp_path / "weights"),
            runs_dir=str(tmp_path / "runs"),
        )
        thread.join(timeout=15)
        summary_path = tmp_path / "runs" / job.job_id / "summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["job_id"] == job.job_id

    def test_fail_when_builder_raises(self, tmp_path, monkeypatch):
        """If a model builder raises, training_service should call job.fail()."""
        import sys
        import types

        bad_models = types.ModuleType("models")
        bad_models.build_dense_model            = mock.MagicMock(side_effect=RuntimeError("boom"))
        bad_models.build_dense_two_hidden_model = lambda: _make_fake_model(_DUMMY_HISTORY)
        bad_models.build_cnn_model              = lambda: _make_fake_model(_DUMMY_HISTORY)
        bad_models.build_cnn_extra_hidden_model = lambda: _make_fake_model(_DUMMY_HISTORY)
        monkeypatch.setitem(sys.modules, "models", bad_models)

        x, y = self._make_data()
        job = create_job({"epochs": 1, "batch_size": 4})
        thread = start_training_job(
            job=job, x=x, y=y,
            weights_dir=str(tmp_path / "weights"),
            runs_dir=str(tmp_path / "runs"),
        )
        thread.join(timeout=10)
        assert job.status == JobStatus.FAILED
        assert "boom" in job.error

