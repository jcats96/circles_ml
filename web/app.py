"""
app.py – FastAPI web server for the circles-ML local interface.

Start with:
    uvicorn web.app:app --reload --port 8000
"""

import asyncio
import json
import os
import queue
import sys
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── ensure project root is importable when this module is loaded ──────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from web.data_io import (
    list_prediction_samples,
    list_training_samples,
    read_image_as_png_bytes,
    save_prediction_sample,
    save_training_sample,
)
from web.prediction_service import predict_directory, predict_image_bytes
from web.training_jobs import JobStatus, create_job, get_job, list_jobs
from web.training_service import start_training_job

# ──────────────────────────────────────────────
# Directories (resolved relative to project root)
# ──────────────────────────────────────────────

TRAINING_DATA_DIR = os.path.join(_ROOT, "training_data")
TEST_DATA_DIR = os.path.join(_ROOT, "test_data")
WEIGHTS_DIR = os.path.join(_ROOT, "weights")
RUNS_DIR = os.path.join(_ROOT, "runs")

# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(title="Circles ML", version="1.0.0")

# Serve static files (frontend).
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


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


class TrainRequest(BaseModel):
    epochs: int = Field(10, ge=1, le=1000)
    batch_size: int = Field(8, ge=1, le=512)
    val_split: float = Field(0.25, ge=0.0, le=0.9)
    seed: int = Field(42)


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
# Dataset endpoints
# ──────────────────────────────────────────────


@app.post("/api/training-samples", status_code=201)
async def post_training_sample(req: TrainingSampleRequest):
    """Save a drawn image and label to the training set."""
    try:
        filename = save_training_sample(req.image, req.circles, TRAINING_DATA_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"filename": filename, "circles": req.circles}


@app.get("/api/training-samples")
async def get_training_samples():
    """List all training samples with their labels."""
    samples = list_training_samples(TRAINING_DATA_DIR)
    return {"samples": samples, "count": len(samples)}


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


# ──────────────────────────────────────────────
# Training endpoints
# ──────────────────────────────────────────────


@app.post("/api/train", status_code=202)
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Start a background training job and return the job_id."""
    config = {
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "val_split": req.val_split,
        "seed": req.seed,
    }
    job = create_job(config)

    def _run():
        import numpy as np
        from train_models import load_dataset, split_dataset

        try:
            x, y = load_dataset(TRAINING_DATA_DIR)
        except Exception as exc:
            job.fail(str(exc))
            return

        start_training_job(
            job=job,
            x=x,
            y=y,
            val_split=req.val_split,
            seed=req.seed,
            weights_dir=WEIGHTS_DIR,
            runs_dir=RUNS_DIR,
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
    import base64

    raw = req.image
    if "," in raw:
        raw = raw.split(",", 1)[1]
    image_bytes = base64.b64decode(raw)

    if req.save_to_test:
        save_prediction_sample(image_bytes, TEST_DATA_DIR)

    models = await get_models()
    predictions = predict_image_bytes(image_bytes, models)
    return {"predictions": predictions}


@app.post("/api/predict-directory")
async def predict_directory_endpoint(directory: Optional[str] = Query(None)):
    """Predict on all PNGs in the given directory (defaults to test_data/)."""
    predict_dir = directory or TEST_DATA_DIR
    if not os.path.isdir(predict_dir):
        raise HTTPException(status_code=404, detail=f"Directory not found: {predict_dir}")

    models = await get_models()
    results = predict_directory(predict_dir, models)
    return {"results": results, "count": len(results)}
