"""
tests/test_app.py – FastAPI endpoint tests using the httpx AsyncClient.

Heavy TF dependencies are mocked so the full test suite stays fast.
"""

import base64
import io
import json
import os
import sys
import unittest.mock as mock

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── helpers ──────────────────────────────────────────────────────────────────


def make_png_bytes(width=32, height=32, color=128) -> bytes:
    img = Image.new("L", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_b64_png() -> str:
    return "data:image/png;base64," + base64.b64encode(make_png_bytes()).decode()


def fake_predict_result(n_models=4):
    return [
        {
            "model": f"Model{i}",
            "raw": float(i),
            "rounded": i,
            "weights_loaded": True,
        }
        for i in range(n_models)
    ]


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def patch_directories(tmp_path, monkeypatch):
    """Redirect all directory constants in web.app to tmp_path."""
    import web.app as app_module

    monkeypatch.setattr(app_module, "TRAINING_DATA_DIR", str(tmp_path / "training_data"))
    monkeypatch.setattr(app_module, "TEST_DATA_DIR",     str(tmp_path / "test_data"))
    monkeypatch.setattr(app_module, "WEIGHTS_DIR",       str(tmp_path / "weights"))
    monkeypatch.setattr(app_module, "RUNS_DIR",          str(tmp_path / "runs"))
    # Also reset the models cache so tests don't share it.
    monkeypatch.setattr(app_module, "_models_cache", None)


@pytest.fixture
def mock_models(monkeypatch):
    """Provide a fake models dict so TF is never loaded during tests."""
    import web.app as app_module

    fake = {
        label: {
            "model": mock.MagicMock(**{"predict.return_value": np.array([[float(i)]])}),
            "weights_loaded": True,
            "weights_path": f"/tmp/{label}.h5",
        }
        for i, label in enumerate(["CNN", "CNNOneHidden", "CNNExtraHidden"])
    }

    async def _get_models():
        return fake

    monkeypatch.setattr(app_module, "get_models", _get_models)
    monkeypatch.setattr(app_module, "_models_cache", fake)
    return fake


@pytest_asyncio.fixture
async def client(mock_models):
    from web.app import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── root / static ─────────────────────────────────────────────────────────────


class TestRoot:
    async def test_root_returns_html(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# ── training samples ──────────────────────────────────────────────────────────


class TestTrainingSamples:
    async def test_post_training_sample_201(self, client):
        resp = await client.post(
            "/api/training-samples",
            json={"image": make_b64_png(), "circles": 3},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "filename" in body
        assert body["circles"] == 3

    async def test_post_training_sample_negative_circles_422(self, client):
        resp = await client.post(
            "/api/training-samples",
            json={"image": make_b64_png(), "circles": -1},
        )
        assert resp.status_code == 422

    async def test_get_training_samples_empty(self, client):
        resp = await client.get("/api/training-samples")
        assert resp.status_code == 200
        body = resp.json()
        assert body["samples"] == []
        assert body["count"] == 0

    async def test_get_training_samples_after_save(self, client):
        await client.post(
            "/api/training-samples",
            json={"image": make_b64_png(), "circles": 2},
        )
        resp = await client.get("/api/training-samples")
        body = resp.json()
        assert body["count"] == 1
        assert body["samples"][0]["circles"] == 2

    async def test_post_multiple_saves(self, client):
        for i in range(5):
            await client.post(
                "/api/training-samples",
                json={"image": make_b64_png(), "circles": i},
            )
        resp = await client.get("/api/training-samples")
        assert resp.json()["count"] == 5


# ── prediction samples ────────────────────────────────────────────────────────


class TestPredictionSamples:
    async def test_post_prediction_sample_201(self, client):
        resp = await client.post(
            "/api/prediction-samples",
            json={"image": make_b64_png()},
        )
        assert resp.status_code == 201
        assert "filename" in resp.json()

    async def test_get_prediction_samples_empty(self, client):
        resp = await client.get("/api/prediction-samples")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    async def test_get_prediction_samples_after_save(self, client):
        await client.post(
            "/api/prediction-samples",
            json={"image": make_b64_png()},
        )
        resp = await client.get("/api/prediction-samples")
        assert resp.json()["count"] == 1


# ── predict-image ─────────────────────────────────────────────────────────────


class TestPredictImage:
    async def test_predict_image_returns_predictions(self, client, mock_models):
        # Patch predict_image_bytes in the web.app namespace
        import web.app as app_module

        with mock.patch.object(app_module, "predict_image_bytes", return_value=fake_predict_result()):
            resp = await client.post(
                "/api/predict-image",
                json={"image": make_b64_png()},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 4

    async def test_predict_image_has_correct_fields(self, client):
        import web.app as app_module

        with mock.patch.object(
            app_module,
            "predict_image_bytes",
            return_value=fake_predict_result(1),
        ):
            resp = await client.post(
                "/api/predict-image",
                json={"image": make_b64_png()},
            )
        pred = resp.json()["predictions"][0]
        assert "model" in pred
        assert "raw" in pred
        assert "rounded" in pred
        assert "weights_loaded" in pred

    async def test_predict_image_invalid_base64_returns_400(self, client):
        resp = await client.post(
            "/api/predict-image",
            json={"image": "data:image/png;base64,not-valid-base64!!!"},
        )
        assert resp.status_code == 400

    async def test_predict_image_invalid_image_bytes_returns_400(self, client, monkeypatch):
        import web.app as app_module

        async def _get_models():
            return mock_models_payload()

        monkeypatch.setattr(app_module, "get_models", _get_models)
        bad_bytes = base64.b64encode(b"not an image").decode()
        resp = await client.post(
            "/api/predict-image",
            json={"image": bad_bytes},
        )
        assert resp.status_code == 400

    async def test_predict_image_missing_model_file_returns_404(self, client, monkeypatch):
        import web.app as app_module

        async def _raise_missing_models():
            raise FileNotFoundError("Missing model weights")

        monkeypatch.setattr(app_module, "get_models", _raise_missing_models)
        resp = await client.post(
            "/api/predict-image",
            json={"image": make_b64_png()},
        )
        assert resp.status_code == 404


# ── predict-directory ─────────────────────────────────────────────────────────


class TestPredictDirectory:
    async def test_predict_directory_missing_returns_404(self, client, tmp_path):
        resp = await client.post(
            "/api/predict-directory",
            params={"directory": str(tmp_path / "nonexistent")},
        )
        assert resp.status_code == 404

    async def test_predict_directory_empty(self, client, tmp_path, monkeypatch):
        import web.app as app_module

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        monkeypatch.setattr(app_module, "TEST_DATA_DIR", str(empty_dir))

        with mock.patch.object(app_module, "predict_directory", return_value=[]):
            resp = await client.post("/api/predict-directory")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    async def test_predict_directory_returns_results(self, client, tmp_path, monkeypatch):
        import web.app as app_module

        d = tmp_path / "imgs"
        d.mkdir()
        for i in range(3):
            (d / f"img{i}.png").write_bytes(make_png_bytes())
        monkeypatch.setattr(app_module, "TEST_DATA_DIR", str(d))

        fake_results = [
            {"filename": f"img{i}.png", "predictions": fake_predict_result(1)}
            for i in range(3)
        ]
        with mock.patch.object(app_module, "predict_directory", return_value=fake_results):
            resp = await client.post("/api/predict-directory")
        assert resp.json()["count"] == 3

    async def test_predict_directory_missing_model_file_returns_404(self, client, monkeypatch):
        import web.app as app_module

        async def _raise_missing_models():
            raise FileNotFoundError("Missing model weights")

        monkeypatch.setattr(app_module, "get_models", _raise_missing_models)
        resp = await client.post("/api/predict-directory")
        assert resp.status_code == 404


def mock_models_payload():
    return {
        "CNN": {
            "model": mock.MagicMock(**{"predict.return_value": np.array([[1.0]])}),
            "weights_loaded": True,
            "weights_path": "/tmp/CNN.h5",
        }
    }


# ── training endpoints ────────────────────────────────────────────────────────


class TestTrainingEndpoints:
    async def test_start_training_returns_job_id(self, client, monkeypatch):
        import web.app as app_module

        # Prevent actual training from running
        mocked_start = mock.MagicMock()
        monkeypatch.setattr(app_module, "start_training_job", mocked_start)

        # Mock train_models so it doesn't try to load TF
        import types
        dummy_x = np.zeros((4, 32, 32, 1), dtype=np.float32)
        dummy_y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        train_models_mock = types.ModuleType("train_models")
        train_models_mock.load_dataset = mock.MagicMock(return_value=(dummy_x, dummy_y))
        train_models_mock.split_dataset = lambda x, y, v, s: (x, y, x[:1], y[:1])
        train_models_mock.augment_training_data = lambda x, y: (x, y)
        monkeypatch.setitem(sys.modules, "train_models", train_models_mock)

        resp = await client.post(
            "/api/train",
            json={"epochs": 2, "batch_size": 4, "val_split": 0.25, "seed": 42, "models": ["CNN", "CNNOneHidden"]},
        )

        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        mocked_start.assert_called_once()
        assert mocked_start.call_args.kwargs["job"].config["models"] == ["CNN", "CNNOneHidden"]

    async def test_start_training_rejects_unknown_model(self, client):
        resp = await client.post(
            "/api/train",
            json={"epochs": 2, "batch_size": 4, "val_split": 0.25, "seed": 42, "models": ["NopeNet"]},
        )
        assert resp.status_code == 422

    async def test_get_training_status_404_unknown(self, client):
        resp = await client.get("/api/train/nonexistent-job")
        assert resp.status_code == 404

    async def test_cancel_training_404_unknown(self, client):
        resp = await client.post("/api/train/nonexistent-job/cancel")
        assert resp.status_code == 404

    async def test_get_jobs_list(self, client):
        resp = await client.get("/api/jobs")
        assert resp.status_code == 200
        assert "jobs" in resp.json()

    async def test_get_training_status_after_create(self, client, monkeypatch):
        import web.app as app_module
        from web.training_jobs import create_job

        job = create_job({"epochs": 1})

        resp = await client.get(f"/api/train/{job.job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job.job_id

    async def test_cancel_running_job(self, client):
        from web.training_jobs import JobStatus, create_job

        job = create_job({})
        job.status = JobStatus.RUNNING

        resp = await client.post(f"/api/train/{job.job_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == JobStatus.CANCELLED
