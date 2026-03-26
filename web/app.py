"""
app.py – FastAPI web server for the circles-ML local interface.

Start with:
    uvicorn web.app:app --reload --port 8000
"""

import asyncio
import base64
import binascii
import json
import os
import queue
import sys
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

# ── ensure project root is importable when this module is loaded ──────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from web.data_io import (
    create_custom_dataset,
    list_datasets,
    list_prediction_samples,
    list_training_samples,
    read_image_as_png_bytes,
    save_prediction_sample,
    save_training_sample,
    update_training_label,
)
from web.prediction_service import predict_directory, predict_image_bytes
from web.training_jobs import JobStatus, create_job, get_job, list_jobs
from web.training_service import MODEL_SPECS, start_training_job

# ──────────────────────────────────────────────
# Directories (resolved relative to project root)
# ──────────────────────────────────────────────

TRAINING_DATA_DIR = os.path.join(_ROOT, "training_data")
TEST_DATA_DIR = os.path.join(_ROOT, "test_data")
WEIGHTS_DIR = os.path.join(_ROOT, "weights")
RUNS_DIR = os.path.join(_ROOT, "runs")
CUSTOM_DATASETS_DIR = os.path.join(_ROOT, "custom_datasets")

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(title="Circles ML", version="1.0.0")

# Serve static files (frontend).
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

_training_images_dir = os.path.join(TRAINING_DATA_DIR, "images")
if os.path.isdir(_training_images_dir):
    app.mount("/training_data/images", StaticFiles(directory=_training_images_dir), name="training_images")


# ──────────────────────────────────────────────
# Lazy-loaded prediction models (avoid loading TF at import time)
# ──────────────────────────────────────────────

_models_cache: Optional[dict] = None
_models_lock = asyncio.Lock()


async def get_models() -> dict:
    global _models_cache
    if _models_cache is None:
        async with _models_lock:
            if _models_cache is None:
                from web.prediction_service import load_models

                _models_cache = load_models(WEIGHTS_DIR)
    return _models_cache


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────


class TrainingSampleRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded PNG image data (may include data-URI prefix)")
    circles: int = Field(..., ge=0, description="Number of circles in the image")


class PredictionSampleRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded PNG image data")


class UpdateTrainingLabelRequest(BaseModel):
    circles: int = Field(..., ge=0, description="Updated number of circles")


class TrainRequest(BaseModel):
    epochs: int = Field(10, ge=1, le=1000)
    batch_size: int = Field(8, ge=1, le=512)
    val_split: float = Field(0.25, ge=0.0, le=0.9)
    seed: int = Field(42)
    models: list[str] = Field(
        default_factory=lambda: [label for label, _, _ in MODEL_SPECS],
        min_length=1,
    )
    dataset: str = Field("circle", description="Dataset ID to train on (e.g. 'circle' or 'custom_stars')")


class CreateDatasetRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Custom dataset name (letters, digits, underscores, hyphens)",
    )


class PredictImageRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded PNG image data")
    save_to_test: bool = Field(False, description="Also save the image to test_data/")


# ──────────────────────────────────────────────
# Root – serve the SPA
# ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Circles ML</h1><p>Static files not found.</p>")


# ──────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────


def _require_dataset_dir(dataset_id: str) -> str:
    """Resolve a dataset ID to its directory, raising HTTPException if unknown."""
    if dataset_id == "circle":
        return TRAINING_DATA_DIR
    if dataset_id.startswith("custom_"):
        name = dataset_id[len("custom_"):]
        ds_dir = os.path.join(CUSTOM_DATASETS_DIR, name)
        if os.path.isdir(ds_dir):
            return ds_dir
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    raise HTTPException(status_code=422, detail=f"Invalid dataset ID: '{dataset_id}'")


# ──────────────────────────────────────────────
# Dataset management endpoints
# ──────────────────────────────────────────────


@app.get("/api/datasets")
async def get_datasets():
    """List all available datasets (built-in circle + user-created custom)."""
    return {"datasets": list_datasets(TRAINING_DATA_DIR, CUSTOM_DATASETS_DIR)}


@app.post("/api/datasets", status_code=201)
async def post_dataset(req: CreateDatasetRequest):
    """Create a new custom dataset."""
    try:
        ds = create_custom_dataset(req.name, CUSTOM_DATASETS_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ds


@app.get("/api/dataset-images/{dataset_id}/{filename}")
async def serve_dataset_image(dataset_id: str, filename: str):
    """Serve a training image from the specified dataset."""
    from fastapi.responses import FileResponse

    data_dir = _require_dataset_dir(dataset_id)
    # Prevent path traversal
    safe_name = os.path.basename(filename)
    img_path = os.path.join(data_dir, "images", safe_name)
    if not os.path.isfile(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path, media_type="image/png")


@app.post("/api/training-samples", status_code=201)
async def post_training_sample(req: TrainingSampleRequest, dataset: str = Query("circle")):
    """Save a drawn image and label to the training set."""
    data_dir = _require_dataset_dir(dataset)
    try:
        filename = save_training_sample(req.image, req.circles, data_dir)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"filename": filename, "circles": req.circles}


@app.get("/api/training-samples")
async def get_training_samples(dataset: str = Query("circle")):
    """List all training samples with their labels."""
    data_dir = _require_dataset_dir(dataset)
    samples = list_training_samples(data_dir)
    return {"samples": samples, "count": len(samples)}


@app.patch("/api/training-samples/{filename}")
async def patch_training_sample(
    filename: str, req: UpdateTrainingLabelRequest, dataset: str = Query("circle")
):
    """Update the label (circle count) for a training sample filename."""
    data_dir = _require_dataset_dir(dataset)
    try:
        updated = update_training_label(data_dir, filename, req.circles)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not updated:
        raise HTTPException(status_code=404, detail=f"Training sample not found: {filename}")

    return {"filename": filename, "circles": req.circles}


@app.post("/api/prediction-samples", status_code=201)
async def post_prediction_sample(req: PredictionSampleRequest):
    """Save a drawn image to the prediction/test set."""
    try:
        filename = save_prediction_sample(req.image, TEST_DATA_DIR)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"filename": filename}


@app.get("/api/prediction-samples")
async def get_prediction_samples():
    """List all images in the test/prediction directory."""
    files = list_prediction_samples(TEST_DATA_DIR)
    return {"files": files, "count": len(files)}


@app.post("/api/weights/drop")
async def drop_weights():
    """Delete all known model weight files from weights/ directory."""
    global _models_cache
    removed = []
    missing = []

    for _, weight_name, _ in MODEL_SPECS:
        weight_path = Path(WEIGHTS_DIR) / f"{weight_name}.weights.h5"
        if weight_path.exists():
            try:
                weight_path.unlink()
                removed.append(weight_name)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Failed deleting {weight_name}: {exc}")
        else:
            missing.append(weight_name)

    # Invalidate cached models so next prediction reloads from disk
    _models_cache = None

    return {
        "removed": removed,
        "removed_count": len(removed),
        "missing": missing,
        "missing_count": len(missing),
    }


# ──────────────────────────────────────────────
# Training endpoints
# ──────────────────────────────────────────────


@app.post("/api/train", status_code=202)
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Start a background training job and return the job_id."""
    valid_models = {label for label, _, _ in MODEL_SPECS}
    invalid_models = sorted(set(req.models) - valid_models)
    if invalid_models:
        raise HTTPException(status_code=422, detail=f"Unknown model selection: {', '.join(invalid_models)}")

    # Validate the dataset ID up front so the response is synchronous.
    training_dir = _require_dataset_dir(req.dataset)

    config = {
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "val_split": req.val_split,
        "seed": req.seed,
        "models": req.models,
        "dataset": req.dataset,
    }
    job = create_job(config)

    def _run():
        from train_models import load_dataset

        try:
            x, y = load_dataset(training_dir)
        except Exception as exc:
            job.fail(str(exc))
            return

        def _invalidate_cache():
            global _models_cache
            _models_cache = None

        start_training_job(
            job=job,
            x=x,
            y=y,
            val_split=req.val_split,
            seed=req.seed,
            weights_dir=WEIGHTS_DIR,
            runs_dir=RUNS_DIR,
            on_complete=_invalidate_cache,
        )

    background_tasks.add_task(_run)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/train/{job_id}")
async def get_training_status(job_id: str):
    """Get the current status and metrics of a training job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    metrics = [
        {
            "model": m.model,
            "epoch": m.epoch,
            "loss": m.loss,
            "mae": m.mae,
            "val_loss": m.val_loss,
            "val_mae": m.val_mae,
        }
        for m in job.metrics
    ]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "config": job.config,
        "metrics": metrics,
        "summary": job.summary,
        "error": job.error,
    }


@app.get("/api/train/{job_id}/events")
async def training_events(job_id: str):
    """Server-Sent Events stream for live training metrics."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    q: queue.Queue = queue.Queue()
    job.subscribe(q)

    async def event_generator():
        try:
            # First, replay already-collected metrics so a late subscriber
            # gets the full history.
            for metric in list(job.metrics):
                payload = json.dumps(
                    {
                        "model": metric.model,
                        "epoch": metric.epoch,
                        "loss": metric.loss,
                        "mae": metric.mae,
                        "val_loss": metric.val_loss,
                        "val_mae": metric.val_mae,
                    }
                )
                yield f"data: {payload}\n\n"

            # If job already finished, send done immediately.
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                yield "data: {\"done\": true}\n\n"
                return

            # Stream new metrics as they arrive.
            loop = asyncio.get_event_loop()
            while True:
                try:
                    item = await loop.run_in_executor(None, lambda: q.get(timeout=30))
                except Exception:
                    # Timeout – send keep-alive comment.
                    yield ": keep-alive\n\n"
                    continue

                if item is None:
                    yield "data: {\"done\": true}\n\n"
                    break

                payload = json.dumps(
                    {
                        "model": item.model,
                        "epoch": item.epoch,
                        "loss": item.loss,
                        "mae": item.mae,
                        "val_loss": item.val_loss,
                        "val_mae": item.val_mae,
                    }
                )
                yield f"data: {payload}\n\n"
        finally:
            job.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/train/{job_id}/cancel")
async def cancel_training(job_id: str):
    """Request cancellation of a running training job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancel()
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/jobs")
async def get_all_jobs():
    """List all training jobs."""
    jobs = list_jobs()
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status,
                "config": j.config,
                "metric_count": len(j.metrics),
            }
            for j in jobs
        ]
    }


# ──────────────────────────────────────────────
# Prediction endpoints
# ──────────────────────────────────────────────


@app.post("/api/predict-image")
async def predict_image_endpoint(req: PredictImageRequest):
    """Predict the circle count for a single drawn image."""
    raw = req.image
    if "," in raw:
        raw = raw.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload") from exc

    try:
        if req.save_to_test:
            save_prediction_sample(image_bytes, TEST_DATA_DIR)

        models = await get_models()
        predictions = predict_image_bytes(image_bytes, models)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {"predictions": predictions}


@app.post("/api/predict-directory")
async def predict_directory_endpoint(directory: Optional[str] = Query(None)):
    """Predict on all PNGs in the given directory (defaults to test_data/)."""
    predict_dir = directory or TEST_DATA_DIR
    if not os.path.isdir(predict_dir):
        raise HTTPException(status_code=404, detail=f"Directory not found: {predict_dir}")

    try:
        models = await get_models()
        results = predict_directory(predict_dir, models)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {"results": results, "count": len(results)}
