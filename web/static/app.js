/* app.js – Circles ML frontend logic */
"use strict";

// ── Canvas setup ─────────────────────────────────────────
const canvas      = document.getElementById("draw-canvas");
const ctx         = canvas.getContext("2d");
const NATIVE_SIZE = 32;

// Fill white initially
ctx.fillStyle = "#fff";
ctx.fillRect(0, 0, NATIVE_SIZE, NATIVE_SIZE);

let drawing   = false;
let tool      = "brush"; // "brush" | "erase"

// ── Dataset / mode state ─────────────────────────────────
// currentDataset: "circle" in Circle mode; "custom_<name>" in Custom mode.
let currentDataset = "circle";
let currentMode    = "circle"; // "circle" | "custom"

// ── Tool buttons ─────────────────────────────────────────
document.getElementById("btn-brush").addEventListener("click", () => setTool("brush"));
document.getElementById("btn-erase").addEventListener("click", () => setTool("erase"));
document.getElementById("btn-clear").addEventListener("click", clearCanvas);

function setTool(t) {
  tool = t;
  document.getElementById("btn-brush").classList.toggle("active", t === "brush");
  document.getElementById("btn-erase").classList.toggle("active", t === "erase");
}

function clearCanvas() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, NATIVE_SIZE, NATIVE_SIZE);
}

// ── Draw on canvas ───────────────────────────────────────
function getCanvasPos(e) {
  const rect  = canvas.getBoundingClientRect();
  const scaleX = NATIVE_SIZE / rect.width;
  const scaleY = NATIVE_SIZE / rect.height;
  return {
    x: Math.floor((e.clientX - rect.left) * scaleX),
    y: Math.floor((e.clientY - rect.top)  * scaleY),
  };
}

function drawPixel(x, y) {
  if (x < 0 || y < 0 || x >= NATIVE_SIZE || y >= NATIVE_SIZE) return;
  ctx.fillStyle = tool === "brush" ? "#000" : "#fff";
  ctx.fillRect(x, y, 1, 1);
}

canvas.addEventListener("mousedown", e => { drawing = true; const p = getCanvasPos(e); drawPixel(p.x, p.y); });
canvas.addEventListener("mousemove", e => { if (!drawing) return; const p = getCanvasPos(e); drawPixel(p.x, p.y); });
canvas.addEventListener("mouseup",   () => { drawing = false; });
canvas.addEventListener("mouseleave", () => { drawing = false; });

// Touch support
canvas.addEventListener("touchstart",  e => { e.preventDefault(); drawing = true;  const p = getCanvasPos(e.touches[0]); drawPixel(p.x, p.y); }, { passive: false });
canvas.addEventListener("touchmove",   e => { e.preventDefault(); if (!drawing) return; const p = getCanvasPos(e.touches[0]); drawPixel(p.x, p.y); }, { passive: false });
canvas.addEventListener("touchend",    e => { e.preventDefault(); drawing = false; }, { passive: false });

// ── Get canvas as base64 PNG ─────────────────────────────
function getImageBase64() {
  return canvas.toDataURL("image/png");
}

// ── Feedback helpers ─────────────────────────────────────
function showFeedback(elId, msg, type = "info") {
  const el = document.getElementById(elId);
  el.textContent = msg;
  el.className   = `feedback ${type}`;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 5000);
}

function showPersistentFeedback(elId, msg, type = "info") {
  const el = document.getElementById(elId);
  el.textContent = msg;
  el.className = `feedback ${type}`;
  el.classList.remove("hidden");
}

// ── Mode toggle ──────────────────────────────────────────
document.getElementById("btn-mode-circle").addEventListener("click", () => switchMode("circle"));
document.getElementById("btn-mode-custom").addEventListener("click", () => switchMode("custom"));

function switchMode(mode) {
  currentMode = mode;
  const isCustom = mode === "custom";

  document.getElementById("btn-mode-circle").classList.toggle("active", !isCustom);
  document.getElementById("btn-mode-circle").setAttribute("aria-pressed", String(!isCustom));
  document.getElementById("btn-mode-custom").classList.toggle("active", isCustom);
  document.getElementById("btn-mode-custom").setAttribute("aria-pressed", String(isCustom));

  document.getElementById("custom-dataset-bar").classList.toggle("hidden", !isCustom);
  document.getElementById("count-label").textContent = isCustom ? "Pattern count" : "Circle count";

  if (isCustom) {
    // Refresh the dataset dropdown; select the first custom dataset if none selected
    loadDatasets().then(() => {
      if (!currentDataset.startsWith("custom_")) {
        const sel = document.getElementById("dataset-select");
        if (sel.options.length > 0) {
          currentDataset = sel.value;
        }
      }
      refreshAfterDatasetChange();
    });
  } else {
    currentDataset = "circle";
    refreshAfterDatasetChange();
  }
}

function refreshAfterDatasetChange() {
  refreshTrainingSampleCount();
  loadTrainingDataTable();
}

// ── Custom dataset management ────────────────────────────
async function loadDatasets() {
  try {
    const res  = await fetch("/api/datasets");
    const data = await res.json();
    if (!res.ok) return;

    const sel = document.getElementById("dataset-select");
    const prevValue = sel.value;
    sel.innerHTML = "";

    const customDatasets = (data.datasets || []).filter(d => d.type === "custom");
    if (customDatasets.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(no custom datasets yet)";
      opt.disabled = true;
      sel.appendChild(opt);
    } else {
      for (const ds of customDatasets) {
        const opt = document.createElement("option");
        opt.value = ds.id;
        opt.textContent = `${ds.name} (${ds.sample_count} samples)`;
        sel.appendChild(opt);
      }
      // Restore previous selection if still available, else pick first
      if ([...sel.options].some(o => o.value === prevValue)) {
        sel.value = prevValue;
      }
      currentDataset = sel.value || customDatasets[0].id;
      sel.value = currentDataset;
    }
  } catch (e) {
    // Non-fatal – dataset bar will just not populate
  }
}

document.getElementById("dataset-select").addEventListener("change", () => {
  currentDataset = document.getElementById("dataset-select").value;
  refreshAfterDatasetChange();
});

document.getElementById("btn-new-dataset").addEventListener("click", async () => {
  const name = window.prompt("New dataset name\n(letters, digits, underscores, hyphens; 1–50 chars):");
  if (!name) return;

  const feedbackEl = document.getElementById("dataset-bar-feedback");
  feedbackEl.classList.add("hidden");

  try {
    const res  = await fetch("/api/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();
    if (!res.ok) {
      feedbackEl.textContent = `Error: ${data.detail}`;
      feedbackEl.className = "dataset-bar-feedback error";
      feedbackEl.classList.remove("hidden");
      return;
    }

    // Select the new dataset
    currentDataset = data.id;
    await loadDatasets();
    document.getElementById("dataset-select").value = currentDataset;
    feedbackEl.textContent = `✅ Created dataset "${data.name}"`;
    feedbackEl.className = "dataset-bar-feedback success";
    feedbackEl.classList.remove("hidden");
    refreshAfterDatasetChange();
  } catch (e) {
    feedbackEl.textContent = `Request failed: ${e}`;
    feedbackEl.className = "dataset-bar-feedback error";
    feedbackEl.classList.remove("hidden");
  }
});

async function refreshTrainingSampleCount() {
  const el = document.getElementById("training-sample-count");
  try {
    const res = await fetch(`/api/training-samples?dataset=${encodeURIComponent(currentDataset)}`);
    const data = await res.json();
    if (!res.ok) {
      el.textContent = "Training samples: unavailable";
      return;
    }
    el.textContent = `Training samples: ${data.count}`;
  } catch {
    el.textContent = "Training samples: unavailable";
  }
}

async function loadTrainingDataTable() {
  const tbody = document.getElementById("training-data-table").querySelector("tbody");
  tbody.innerHTML = "<tr><td colspan='4' class='muted'>Loading…</td></tr>";

  try {
    const res = await fetch(`/api/training-samples?dataset=${encodeURIComponent(currentDataset)}`);
    const data = await res.json();
    if (!res.ok) {
      tbody.innerHTML = `<tr><td colspan='4' class='file-missing'>${escapeHtml(data.detail || "Failed to load samples")}</td></tr>`;
      return;
    }

    const samples = data.samples || [];
    if (samples.length === 0) {
      tbody.innerHTML = "<tr><td colspan='4' class='muted'>No training samples found.</td></tr>";
      return;
    }

    tbody.innerHTML = "";
    for (const sample of samples) {
      const tr = document.createElement("tr");
      const safeFilename = escapeHtml(sample.filename);
      const circlesVal = Number.isInteger(sample.circles) ? sample.circles : 0;
      const imageCell = sample.exists
        ? `<img class="training-thumb" src="/api/dataset-images/${encodeURIComponent(currentDataset)}/${encodeURIComponent(sample.filename)}" alt="${safeFilename}" loading="lazy" decoding="async" />`
        : "<span class='file-missing'>missing</span>";
      tr.innerHTML = `
        <td><code>${safeFilename}</code></td>
        <td>${imageCell}</td>
        <td>
          <input
            type="number"
            class="circle-input"
            min="0"
            value="${circlesVal}"
            data-filename="${safeFilename}"
          />
        </td>
        <td>
          <button class="secondary-btn btn-save-label" data-filename="${safeFilename}">Save</button>
        </td>
      `;
      tbody.appendChild(tr);
    }

    bindTrainingDataSaveButtons();
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan='4' class='file-missing'>Request failed: ${escapeHtml(String(e))}</td></tr>`;
  }
}

function bindTrainingDataSaveButtons() {
  const buttons = document.querySelectorAll(".btn-save-label");
  buttons.forEach(btn => {
    btn.addEventListener("click", async () => {
      const filename = btn.dataset.filename;
      const input = document.querySelector(`input.circle-input[data-filename="${CSS.escape(filename)}"]`);
      if (!input) return;

      const circles = parseInt(input.value, 10);
      if (isNaN(circles) || circles < 0) {
        showPersistentFeedback("training-data-feedback", "Count must be a non-negative integer.", "error");
        return;
      }

      btn.disabled = true;
      try {
        const res = await fetch(`/api/training-samples/${encodeURIComponent(filename)}?dataset=${encodeURIComponent(currentDataset)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ circles }),
        });
        const data = await res.json();
        if (!res.ok) {
          showPersistentFeedback("training-data-feedback", `Error updating ${filename}: ${data.detail}`, "error");
          return;
        }

        showPersistentFeedback("training-data-feedback", `Updated ${filename} to ${data.circles}.`, "success");
      } catch (e) {
        showPersistentFeedback("training-data-feedback", `Request failed: ${e}`, "error");
      } finally {
        btn.disabled = false;
      }
    });
  });
}

// ── Sample actions ───────────────────────────────────────
document.getElementById("btn-save-training").addEventListener("click", async () => {
  const circles = parseInt(document.getElementById("circle-count").value, 10);
  if (isNaN(circles) || circles < 0) { showFeedback("save-feedback", "Enter a valid count (≥ 0)", "error"); return; }
  try {
    const res = await fetch(`/api/training-samples?dataset=${encodeURIComponent(currentDataset)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: getImageBase64(), circles }),
    });
    const data = await res.json();
    if (res.ok) {
      showFeedback("save-feedback", `✅ Saved as ${data.filename}`, "success");
      refreshTrainingSampleCount();
    }
    else        showFeedback("save-feedback", `Error: ${data.detail}`, "error");
  } catch (e) { showFeedback("save-feedback", `Request failed: ${e}`, "error"); }
});

document.getElementById("btn-save-prediction").addEventListener("click", async () => {
  try {
    const res = await fetch("/api/prediction-samples", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: getImageBase64() }),
    });
    const data = await res.json();
    if (res.ok) showFeedback("save-feedback", `✅ Saved as ${data.filename}`, "success");
    else        showFeedback("save-feedback", `Error: ${data.detail}`, "error");
  } catch (e) { showFeedback("save-feedback", `Request failed: ${e}`, "error"); }
});

document.getElementById("btn-predict-image").addEventListener("click", async () => {
  try {
    const res = await fetch("/api/predict-image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: getImageBase64() }),
    });
    const data = await res.json();
    if (res.ok) renderPredictions(data.predictions, "prediction-results");
    else        showFeedback("save-feedback", `Error: ${data.detail}`, "error");
  } catch (e) { showFeedback("save-feedback", `Request failed: ${e}`, "error"); }
});

document.getElementById("btn-predict-dir").addEventListener("click", async () => {
  const el = document.getElementById("dir-prediction-results");
  el.innerHTML = "<p class='muted'>Running…</p>";
  try {
    const res  = await fetch("/api/predict-directory", { method: "POST" });
    const data = await res.json();
    if (!res.ok) { el.innerHTML = `<p class='feedback error'>${data.detail}</p>`; return; }
    if (!data.results || data.results.length === 0) { el.innerHTML = "<p class='muted'>No images found.</p>"; return; }

    let html = "<table class='pred-table'><thead><tr><th>File</th><th>Model</th><th>Raw</th><th>Rounded</th></tr></thead><tbody>";
    for (const item of data.results) {
      for (const p of item.predictions) {
        html += `<tr><td>${escapeHtml(item.filename)}</td><td>${escapeHtml(p.model)}</td><td>${p.raw.toFixed(3)}</td><td>${p.rounded}</td></tr>`;
      }
    }
    html += "</tbody></table>";
    el.innerHTML = html;
  } catch (e) { el.innerHTML = `<p class='feedback error'>Request failed: ${e}</p>`; }
});

function renderPredictions(predictions, containerId) {
  const el = document.getElementById(containerId);
  if (!predictions || predictions.length === 0) { el.innerHTML = "<p class='muted'>No results.</p>"; return; }
  let html = "<table class='pred-table'><thead><tr><th>Model</th><th>Raw</th><th>Rounded</th><th>Weights</th></tr></thead><tbody>";
  for (const p of predictions) {
    const wLabel = p.weights_loaded ? "✅" : "⚠️ none";
    html += `<tr><td>${escapeHtml(p.model)}</td><td>${p.raw.toFixed(3)}</td><td>${p.rounded}</td><td>${wLabel}</td></tr>`;
  }
  html += "</tbody></table>";
  el.innerHTML = html;
}

function escapeHtml(str) {
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function formatDuration(seconds) {
  if (seconds == null) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return `${m}m ${s}s`;
}

// ── Training ─────────────────────────────────────────────
let currentJobId  = null;
let trainingSSE   = null;

// Chart.js instances
const chartLoss = new Chart(document.getElementById("chart-loss"), {
  type: "line",
  data: { labels: [], datasets: [] },
  options: chartOptions("Loss"),
});

const chartMAE = new Chart(document.getElementById("chart-mae"), {
  type: "line",
  data: { labels: [], datasets: [] },
  options: chartOptions("MAE"),
});

const MIN_LOG_VALUE = 1e-4;

function chartOptions(title) {
  return {
    responsive: true,
    animation: false,
    plugins: { legend: { labels: { color: "#8b949e", font: { size: 10 } } } },
    scales: {
      x: { ticks: { color: "#8b949e", font: { size: 9 } }, grid: { color: "#30363d" } },
      y: {
        type: "logarithmic",
        ticks: { color: "#8b949e", font: { size: 9 } },
        grid: { color: "#30363d" },
      },
    },
  };
}

function toLogScaleValue(value) {
  if (value == null) return null;
  return Math.max(value, MIN_LOG_VALUE);
}

// Color palette for chart lines
const MODEL_COLORS = {
  CNN:             { train: "#58a6ff", val: "#1f6feb" },
  CNNOneHidden:    { train: "#3fb950", val: "#238636" },
  CNNExtraHidden:  { train: "#ffa657", val: "#f0883e" },
};

// Per-model, per-metric accumulators: { [model]: { loss:[], mae:[], val_loss:[], val_mae:[], epochs:[] } }
let metricsBuffer = {};

function ensureDataset(chart, key, label, color, dash = false) {
  let ds = chart.data.datasets.find(d => d.label === label);
  if (!ds) {
    ds = {
      label,
      data: [],
      borderColor: color,
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
      borderDash: dash ? [4, 3] : [],
      tension: 0.3,
    };
    chart.data.datasets.push(ds);
  }
  return ds;
}

function pushMetric(event) {
  const { model, epoch, loss, mae, val_loss, val_mae } = event;
  if (!metricsBuffer[model]) metricsBuffer[model] = { epochs: [], loss: [], mae: [], val_loss: [], val_mae: [] };
  const buf = metricsBuffer[model];
  buf.epochs.push(epoch);
  buf.loss.push(loss);
  buf.mae.push(mae);
  buf.val_loss.push(val_loss);
  buf.val_mae.push(val_mae);

  const colors = MODEL_COLORS[model] || { train: "#ccc", val: "#999" };

  // Loss chart
  const trainLossDS = ensureDataset(chartLoss, `${model}-loss`, `${model} train`, colors.train);
  const valLossDS   = ensureDataset(chartLoss, `${model}-val-loss`, `${model} val`, colors.val, true);
  trainLossDS.data  = buf.loss.map(toLogScaleValue);
  valLossDS.data    = buf.val_loss.map(toLogScaleValue);

  // MAE chart
  const trainMaeDS  = ensureDataset(chartMAE, `${model}-mae`, `${model} train`, colors.train);
  const valMaeDS    = ensureDataset(chartMAE, `${model}-val-mae`, `${model} val`, colors.val, true);
  trainMaeDS.data   = buf.mae.map(toLogScaleValue);
  valMaeDS.data     = buf.val_mae.map(toLogScaleValue);

  // Labels (shared x-axis across all models – use max epochs seen so far)
  const maxEpoch  = Math.max(...Object.values(metricsBuffer).flatMap(b => b.epochs));
  const newLabels = Array.from({ length: maxEpoch }, (_, i) => String(i + 1));
  chartLoss.data.labels = newLabels;
  chartMAE.data.labels  = newLabels;

  chartLoss.update("none");
  chartMAE.update("none");

  // Status label
  document.getElementById("current-epoch-label").textContent = `${model} – Epoch ${epoch}`;
}

function resetCharts() {
  metricsBuffer = {};
  chartLoss.data.labels   = [];
  chartLoss.data.datasets = [];
  chartMAE.data.labels    = [];
  chartMAE.data.datasets  = [];
  chartLoss.update();
  chartMAE.update();
  document.getElementById("summary-table-wrap").classList.add("hidden");
  document.getElementById("summary-table").querySelector("tbody").innerHTML = "";
}

document.getElementById("btn-start-training").addEventListener("click", async () => {
  const epochs    = parseInt(document.getElementById("epochs").value, 10);
  const batchSize = parseInt(document.getElementById("batch-size").value, 10);
  const valSplit  = parseFloat(document.getElementById("val-split").value);
  const seed      = parseInt(document.getElementById("seed").value, 10);
  const models    = Array.from(document.querySelectorAll(".model-checkbox:checked"), el => el.value);

  if (models.length === 0) {
    showFeedback("train-feedback", "Select at least one model to train.", "error");
    return;
  }

  resetCharts();

  try {
    const res  = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs, batch_size: batchSize, val_split: valSplit, seed, models, dataset: currentDataset }),
    });
    const data = await res.json();
    if (!res.ok) { showFeedback("train-feedback", `Error: ${data.detail}`, "error"); return; }

    currentJobId = data.job_id;
    document.getElementById("current-job-label").textContent = `Job: ${currentJobId}`;
    document.getElementById("btn-cancel-training").classList.remove("hidden");
    showFeedback("train-feedback", `Training started (${currentJobId})`, "info");

    startSSE(currentJobId);
  } catch (e) { showFeedback("train-feedback", `Request failed: ${e}`, "error"); }
});

document.getElementById("btn-cancel-training").addEventListener("click", async () => {
  if (!currentJobId) return;
  await fetch(`/api/train/${currentJobId}/cancel`, { method: "POST" });
  stopSSE();
  showFeedback("train-feedback", "Training cancellation requested.", "info");
  document.getElementById("btn-cancel-training").classList.add("hidden");
});

document.getElementById("btn-drop-weights").addEventListener("click", async () => {
  const confirmed = window.confirm("Delete all saved model weight files in weights/? This cannot be undone.");
  if (!confirmed) return;

  const btn = document.getElementById("btn-drop-weights");
  btn.disabled = true;
  try {
    const res = await fetch("/api/weights/drop", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      showFeedback("train-feedback", `Error dropping weights: ${data.detail}`, "error");
      return;
    }

    showFeedback(
      "train-feedback",
      `Removed ${data.removed_count} weight file(s). ${data.missing_count} already missing.`,
      "success"
    );
  } catch (e) {
    showFeedback("train-feedback", `Request failed: ${e}`, "error");
  } finally {
    btn.disabled = false;
  }
});

function startSSE(jobId) {
  stopSSE();
  trainingSSE = new EventSource(`/api/train/${jobId}/events`);
  trainingSSE.onmessage = e => {
    const event = JSON.parse(e.data);
    if (event.done) {
      stopSSE();
      document.getElementById("btn-cancel-training").classList.add("hidden");
      loadSummary(jobId);
      return;
    }
    pushMetric(event);
  };
  trainingSSE.onerror = () => {
    stopSSE();
    document.getElementById("btn-cancel-training").classList.add("hidden");
  };
}

function stopSSE() {
  if (trainingSSE) { trainingSSE.close(); trainingSSE = null; }
}

async function loadSummary(jobId) {
  const res  = await fetch(`/api/train/${jobId}`);
  const data = await res.json();
  if (!data.summary || !data.summary.models) return;

  const tbody = document.getElementById("summary-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const [model, info] of Object.entries(data.summary.models)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(model)}</td>
      <td>${(info.final_train_mae ?? 0).toFixed(4)}</td>
      <td>${(info.final_val_mae  ?? 0).toFixed(4)}</td>
      <td>${formatDuration(info.elapsed_seconds)}</td>
      <td><code>${escapeHtml(info.weights_path)}</code></td>
    `;
    tbody.appendChild(tr);
  }
  const totalSec = data.summary.elapsed_seconds;
  const totalStr = totalSec != null ? ` (${formatDuration(totalSec)} total)` : "";
  document.getElementById("summary-table-wrap").classList.remove("hidden");
  showFeedback("train-feedback", `✅ Training complete!${totalStr}`, "success");
}

// ── Server status ─────────────────────────────────────────
async function checkServerStatus() {
  const badge = document.getElementById("server-status");
  try {
    const res = await fetch("/api/training-samples");
    if (res.ok) {
      badge.textContent = "✅ Connected";
      badge.style.background = "rgba(63,185,80,.2)";
      badge.style.color      = "#3fb950";
    }
  } catch {
    badge.textContent = "❌ Offline";
    badge.style.background = "rgba(248,81,73,.2)";
    badge.style.color      = "#f85149";
  }
}

checkServerStatus();
setInterval(checkServerStatus, 30000);
refreshTrainingSampleCount();
loadTrainingDataTable();
        grid: { color: "#30363d" },
      },
    },
  };
}

function toLogScaleValue(value) {
  if (value == null) return null;
  return Math.max(value, MIN_LOG_VALUE);
}

// Color palette for chart lines
const MODEL_COLORS = {
  CNN:             { train: "#58a6ff", val: "#1f6feb" },
  CNNOneHidden:    { train: "#3fb950", val: "#238636" },
  CNNExtraHidden:  { train: "#ffa657", val: "#f0883e" },
};

// Per-model, per-metric accumulators: { [model]: { loss:[], mae:[], val_loss:[], val_mae:[], epochs:[] } }
let metricsBuffer = {};

function ensureDataset(chart, key, label, color, dash = false) {
  let ds = chart.data.datasets.find(d => d.label === label);
  if (!ds) {
    ds = {
      label,
      data: [],
      borderColor: color,
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
      borderDash: dash ? [4, 3] : [],
      tension: 0.3,
    };
    chart.data.datasets.push(ds);
  }
  return ds;
}

function pushMetric(event) {
  const { model, epoch, loss, mae, val_loss, val_mae } = event;
  if (!metricsBuffer[model]) metricsBuffer[model] = { epochs: [], loss: [], mae: [], val_loss: [], val_mae: [] };
  const buf = metricsBuffer[model];
  buf.epochs.push(epoch);
  buf.loss.push(loss);
  buf.mae.push(mae);
  buf.val_loss.push(val_loss);
  buf.val_mae.push(val_mae);

  const colors = MODEL_COLORS[model] || { train: "#ccc", val: "#999" };

  // Loss chart
  const trainLossDS = ensureDataset(chartLoss, `${model}-loss`, `${model} train`, colors.train);
  const valLossDS   = ensureDataset(chartLoss, `${model}-val-loss`, `${model} val`, colors.val, true);
  trainLossDS.data  = buf.loss.map(toLogScaleValue);
  valLossDS.data    = buf.val_loss.map(toLogScaleValue);

  // MAE chart
  const trainMaeDS  = ensureDataset(chartMAE, `${model}-mae`, `${model} train`, colors.train);
  const valMaeDS    = ensureDataset(chartMAE, `${model}-val-mae`, `${model} val`, colors.val, true);
  trainMaeDS.data   = buf.mae.map(toLogScaleValue);
  valMaeDS.data     = buf.val_mae.map(toLogScaleValue);

  // Labels (shared x-axis across all models – use max epochs seen so far)
  const maxEpoch  = Math.max(...Object.values(metricsBuffer).flatMap(b => b.epochs));
  const newLabels = Array.from({ length: maxEpoch }, (_, i) => String(i + 1));
  chartLoss.data.labels = newLabels;
  chartMAE.data.labels  = newLabels;

  chartLoss.update("none");
  chartMAE.update("none");

  // Status label
  document.getElementById("current-epoch-label").textContent = `${model} – Epoch ${epoch}`;
}

function resetCharts() {
  metricsBuffer = {};
  chartLoss.data.labels   = [];
  chartLoss.data.datasets = [];
  chartMAE.data.labels    = [];
  chartMAE.data.datasets  = [];
  chartLoss.update();
  chartMAE.update();
  document.getElementById("summary-table-wrap").classList.add("hidden");
  document.getElementById("summary-table").querySelector("tbody").innerHTML = "";
}

document.getElementById("btn-start-training").addEventListener("click", async () => {
  const epochs    = parseInt(document.getElementById("epochs").value, 10);
  const batchSize = parseInt(document.getElementById("batch-size").value, 10);
  const valSplit  = parseFloat(document.getElementById("val-split").value);
  const seed      = parseInt(document.getElementById("seed").value, 10);
  const models    = Array.from(document.querySelectorAll(".model-checkbox:checked"), el => el.value);

  if (models.length === 0) {
    showFeedback("train-feedback", "Select at least one model to train.", "error");
    return;
  }

  resetCharts();

  try {
    const res  = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs, batch_size: batchSize, val_split: valSplit, seed, models }),
    });
    const data = await res.json();
    if (!res.ok) { showFeedback("train-feedback", `Error: ${data.detail}`, "error"); return; }

    currentJobId = data.job_id;
    document.getElementById("current-job-label").textContent = `Job: ${currentJobId}`;
    document.getElementById("btn-cancel-training").classList.remove("hidden");
    showFeedback("train-feedback", `Training started (${currentJobId})`, "info");

    startSSE(currentJobId);
  } catch (e) { showFeedback("train-feedback", `Request failed: ${e}`, "error"); }
});

document.getElementById("btn-cancel-training").addEventListener("click", async () => {
  if (!currentJobId) return;
  await fetch(`/api/train/${currentJobId}/cancel`, { method: "POST" });
  stopSSE();
  showFeedback("train-feedback", "Training cancellation requested.", "info");
  document.getElementById("btn-cancel-training").classList.add("hidden");
});

document.getElementById("btn-drop-weights").addEventListener("click", async () => {
  const confirmed = window.confirm("Delete all saved model weight files in weights/? This cannot be undone.");
  if (!confirmed) return;

  const btn = document.getElementById("btn-drop-weights");
  btn.disabled = true;
  try {
    const res = await fetch("/api/weights/drop", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      showFeedback("train-feedback", `Error dropping weights: ${data.detail}`, "error");
      return;
    }

    showFeedback(
      "train-feedback",
      `Removed ${data.removed_count} weight file(s). ${data.missing_count} already missing.`,
      "success"
    );
  } catch (e) {
    showFeedback("train-feedback", `Request failed: ${e}`, "error");
  } finally {
    btn.disabled = false;
  }
});

function startSSE(jobId) {
  stopSSE();
  trainingSSE = new EventSource(`/api/train/${jobId}/events`);
  trainingSSE.onmessage = e => {
    const event = JSON.parse(e.data);
    if (event.done) {
      stopSSE();
      document.getElementById("btn-cancel-training").classList.add("hidden");
      loadSummary(jobId);
      return;
    }
    pushMetric(event);
  };
  trainingSSE.onerror = () => {
    stopSSE();
    document.getElementById("btn-cancel-training").classList.add("hidden");
  };
}

function stopSSE() {
  if (trainingSSE) { trainingSSE.close(); trainingSSE = null; }
}

async function loadSummary(jobId) {
  const res  = await fetch(`/api/train/${jobId}`);
  const data = await res.json();
  if (!data.summary || !data.summary.models) return;

  const tbody = document.getElementById("summary-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const [model, info] of Object.entries(data.summary.models)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(model)}</td>
      <td>${(info.final_train_mae ?? 0).toFixed(4)}</td>
      <td>${(info.final_val_mae  ?? 0).toFixed(4)}</td>
      <td>${formatDuration(info.elapsed_seconds)}</td>
      <td><code>${escapeHtml(info.weights_path)}</code></td>
    `;
    tbody.appendChild(tr);
  }
  const totalSec = data.summary.elapsed_seconds;
  const totalStr = totalSec != null ? ` (${formatDuration(totalSec)} total)` : "";
  document.getElementById("summary-table-wrap").classList.remove("hidden");
  showFeedback("train-feedback", `✅ Training complete!${totalStr}`, "success");
}

// ── Server status ─────────────────────────────────────────
async function checkServerStatus() {
  const badge = document.getElementById("server-status");
  try {
    const res = await fetch("/api/training-samples");
    if (res.ok) {
      badge.textContent = "✅ Connected";
      badge.style.background = "rgba(63,185,80,.2)";
      badge.style.color      = "#3fb950";
    }
  } catch {
    badge.textContent = "❌ Offline";
    badge.style.background = "rgba(248,81,73,.2)";
    badge.style.color      = "#f85149";
  }
}

checkServerStatus();
setInterval(checkServerStatus, 30000);
refreshTrainingSampleCount();
loadTrainingDataTable();
