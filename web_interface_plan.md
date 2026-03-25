# Local Web Interface Plan

## Goal

Add a local-only web application to this repo that lets you:

1. Draw new 32x32 images in the browser.
2. Save drawn images into either the training set or prediction/test set.
3. Assign labels for training samples.
4. Trigger model training from the browser.
5. Trigger predictions from the browser.
6. Visualize training progress and metrics live while training runs.

This should remain a local development tool, not a deployed multi-user service.

## Current Repo Constraints

The current project already has a clean offline pipeline:

1. [train_models.py](train_models.py) loads `training_data/labels.csv` plus PNG files from `training_data/images/`.
2. [train_models.py](train_models.py) trains four Keras models and writes weights into `weights/`.
3. [main.py](main.py) loads PNG files from `test_data/` and prints predictions.

That means the web interface should not replace the ML code. It should sit on top of the existing file-based workflow.

## Recommended Architecture

Use a small local Python web server plus a lightweight browser UI.

Recommended stack:

1. Backend: FastAPI
2. Frontend: simple HTML/CSS/JavaScript, or React if you want a richer UI
3. Training execution: background Python thread or subprocess
4. Progress updates: WebSocket or Server-Sent Events
5. Charts: Chart.js or a similarly small browser charting library

Why this fits well:

1. FastAPI is easy to wire into the existing Python/TensorFlow code.
2. The current project is local-only, so there is no need for a heavy production architecture.
3. Training progress can be streamed from Python to the browser with minimal complexity.

## Proposed User Flows

### 1. Draw and Save Training Example

User flow:

1. Open the local web app.
2. Draw on a 32x32 canvas using the mouse.
3. Enter the number of circles in the image.
4. Click `Save to Training Set`.
5. The app writes a new PNG into `training_data/images/`.
6. The app appends a row to `training_data/labels.csv`.

Backend behavior:

1. Accept the canvas image as PNG or base64 image data.
2. Normalize it to exactly 32x32 grayscale PNG.
3. Generate a unique filename, for example `drawn_20260325_153012_001.png`.
4. Save the image in `training_data/images/`.
5. Append `filename,circles` to `training_data/labels.csv`.

Important detail:

The write to `labels.csv` must be serialized so two quick saves do not corrupt the file. A simple process lock or file lock is enough.

### 2. Draw and Save Prediction Example

User flow:

1. Draw on the same 32x32 canvas.
2. Click `Save to Prediction Set`.
3. The app writes a PNG into `test_data/`.
4. Click `Run Prediction`.
5. The UI displays predictions from all models.

This fits the current behavior of [main.py](main.py), which already predicts over a directory of PNG files.

### 3. Run Training from the Browser

User flow:

1. Choose training parameters such as epochs, batch size, validation split, and seed.
2. Click `Start Training`.
3. Watch live progress in the browser.
4. See final summary metrics and saved weight paths.

Backend behavior:

1. Start training in a background job.
2. Stream epoch-by-epoch metrics to the frontend.
3. Save final weights exactly as the current script already does.
4. Keep a training status object in memory so the UI can reconnect and still see the current state.

### 4. Run Prediction from the Browser

User flow:

1. Select whether to predict a single freshly drawn image or all images in `test_data/`.
2. Click `Run Prediction`.
3. See each model's raw output and rounded count.

Backend behavior:

1. Load the current weights.
2. Reuse the current preprocessing logic from [main.py](main.py).
3. Return structured JSON instead of printing text.

## Minimal Backend Refactor

The current scripts are usable, but the web app will be much cleaner if training and prediction logic are moved from CLI scripts into reusable functions.

Recommended refactor:

1. Extract dataset save helpers into a new module, for example `data_io.py`.
2. Extract training into a service module, for example `training_service.py`.
3. Extract prediction into a service module, for example `prediction_service.py`.
4. Keep [train_models.py](train_models.py) and [main.py](main.py) as thin CLI wrappers around those modules.

That gives you both:

1. Command-line workflows for quick local runs.
2. Importable functions for the web server.

## Training Progress Design

This is the most important backend change.

Right now, [train_models.py](train_models.py) prints final MAE values but does not persist or stream per-epoch history in a structured way.

To support a UI, training should expose epoch metrics in real time.

Recommended implementation:

1. Add a custom Keras callback.
2. On each epoch end, collect:
   - epoch
   - loss
   - mae
   - val_loss
   - val_mae
   - model label
3. Push those metrics into:
   - an in-memory job state object
   - a JSONL or CSV log file for persistence
   - a WebSocket/SSE stream to the UI

Example internal event shape:

```json
{
  "job_id": "train_20260325_1530",
  "model": "CNN",
  "epoch": 12,
  "loss": 0.184,
  "mae": 0.291,
  "val_loss": 0.227,
  "val_mae": 0.334
}
```

This enables:

1. A live line chart for training and validation curves.
2. A status panel showing current model and epoch.
3. A saved training history for later comparison.

## Suggested API Endpoints

### Dataset endpoints

1. `POST /api/training-samples`
   - body: image + label
   - saves PNG to `training_data/images/`
   - appends row to `training_data/labels.csv`

2. `POST /api/prediction-samples`
   - body: image
   - saves PNG to `test_data/`

3. `GET /api/training-samples`
   - returns recent saved training samples and labels

4. `GET /api/prediction-samples`
   - returns current files in `test_data/`

### Training endpoints

1. `POST /api/train`
   - body: epochs, batch_size, val_split, seed
   - starts a background training job
   - returns `job_id`

2. `GET /api/train/{job_id}`
   - returns current status, current model, epoch progress, final summary if complete

3. `GET /api/train/{job_id}/events`
   - SSE or WebSocket stream for live metrics

4. `POST /api/train/{job_id}/cancel`
   - optional, cancels training if you want stop control

### Prediction endpoints

1. `POST /api/predict-image`
   - body: one image from the canvas
   - returns predictions from all models without saving to disk unless requested

2. `POST /api/predict-directory`
   - body: optional directory or default to `test_data/`
   - returns all predictions in structured JSON

## Frontend Layout

Recommended page sections:

### A. Drawing panel

1. 32x32 editable canvas shown enlarged with crisp nearest-neighbor rendering.
2. Brush and erase tools.
3. Clear button.
4. Optional grid overlay.
5. Optional threshold toggle to force binary output.

### B. Sample actions panel

1. Label input for circle count.
2. `Save to Training Set` button.
3. `Save to Prediction Set` button.
4. `Predict This Image` button.

### C. Training controls panel

1. Epoch count input.
2. Batch size input.
3. Validation split input.
4. Seed input.
5. `Start Training` button.
6. Optional `Cancel Training` button.

### D. Training progress panel

1. Current job status.
2. Current model being trained.
3. Current epoch / total epochs.
4. Live line charts for:
   - training loss
   - validation loss
   - training MAE
   - validation MAE
5. Final summary table per model.

### E. Prediction results panel

1. Raw output for each model.
2. Rounded integer prediction.
3. Optional side-by-side preview of the image.

## Data Format Decisions

### Image format

Keep PNG. It already matches the existing pipeline.

### Training labels format

Keep `training_data/labels.csv` for now. It is simple and already works.

Potential future improvement:

If you later want metadata like notes, source, timestamp, or user tags, switch to a richer format such as JSONL or add columns to the CSV.

### Training history format

Add a new artifact directory, for example:

1. `runs/`
2. `runs/<job_id>/metrics.jsonl`
3. `runs/<job_id>/summary.json`

This is preferable to relying on terminal output because the UI can load historical runs after refresh.

## Recommended Repository Additions

One reasonable structure is:

```text
web/
  app.py
  training_jobs.py
  data_io.py
  prediction_api.py
  static/
    index.html
    styles.css
    app.js
runs/
```

Alternative if you want frontend/backend separation:

```text
backend/
  app.py
  services/
frontend/
  src/
runs/
```

For this repo, the first option is probably enough.

## Recommended Implementation Phases

### Phase 1: Make backend logic reusable

1. Move training logic out of [train_models.py](train_models.py) into importable functions.
2. Move prediction logic out of [main.py](main.py) into importable functions.
3. Add a function that saves a drawn training image and appends to CSV.

### Phase 2: Ship a minimal local UI

1. Add FastAPI server.
2. Add one HTML page with a 32x32 drawing canvas.
3. Support save-to-training, save-to-test, and predict-image.

### Phase 3: Add training control and progress visualization

1. Run training in the background.
2. Stream epoch metrics to the browser.
3. Persist run history into `runs/`.
4. Show charts and summaries.

### Phase 4: Quality improvements

1. Add sample gallery previews.
2. Add delete/edit for mislabeled samples.
3. Add comparison between runs.
4. Add model selection so you can train only one model if desired.

## Risks and Trade-Offs

### 1. Training blocks the server if done naively

If training runs directly inside the request handler, the UI will freeze or time out. Use a background worker or subprocess.

### 2. CSV writes can become fragile

Appending to `labels.csv` is fine locally, but use locking and validation.

### 3. Reproducibility can drift

If users can freely add samples from the browser, results will change over time. That is useful for experimentation, but you should log exactly what was trained and when.

### 4. TensorFlow logs can be noisy

Training output should be captured and converted into structured events, not just dumped into the UI.

## Best First Version

If the goal is to get something useful quickly, the best first version is:

1. FastAPI backend.
2. Single-page local UI.
3. Canvas drawing with save-to-training and save-to-test.
4. Predict current drawing immediately.
5. Start training with configurable epochs and batch size.
6. Show live epoch metrics for one training job at a time.

That gets you a practical interactive workflow without overbuilding.

## Recommendation

Build this as a thin local app around the existing file-based repo.

Do not rewrite the ML code first. Instead:

1. Extract reusable training and prediction functions.
2. Add a small web server.
3. Add structured training-history capture.
4. Add the browser UI last, on top of those backend primitives.

That path keeps the project understandable and gives you a clear progression from the current CLI workflow to a local interactive tool.