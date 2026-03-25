/* app.js – Circles ML frontend logic */
"use strict";

// ── Canvas setup ─────────────────────────────────────────
const canvas      = document.getElementById("draw-canvas");
const ctx         = canvas.getContext("2d");
const NATIVE_SIZE = 32;

// Fill black initially
ctx.fillStyle = "#000";
ctx.fillRect(0, 0, NATIVE_SIZE, NATIVE_SIZE);

let drawing   = false;
let tool      = "brush"; // "brush" | "erase"
let showGrid  = false;
let threshold = false;

// ── Tool buttons ─────────────────────────────────────────
document.getElementById("btn-brush").addEventListener("click", () => setTool("brush"));
document.getElementById("btn-erase").addEventListener("click", () => setTool("erase"));
document.getElementById("btn-clear").addEventListener("click", clearCanvas);
document.getElementById("chk-grid").addEventListener("change", e => { showGrid = e.target.checked; redrawGrid(); });
document.getElementById("chk-threshold").addEventListener("change", e => { threshold = e.target.checked; applyThreshold(); });

function setTool(t) {
  tool = t;
  document.getElementById("btn-brush").classList.toggle("active", t === "brush");
  document.getElementById("btn-erase").classList.toggle("active", t === "erase");
}

function clearCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, NATIVE_SIZE, NATIVE_SIZE);
  redrawGrid();
}

function applyThreshold() {
  if (!threshold) return;
  const imgData = ctx.getImageData(0, 0, NATIVE_SIZE, NATIVE_SIZE);
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = (d[i] + d[i+1] + d[i+2]) / 3;
    const v    = gray > 127 ? 255 : 0;
    d[i] = d[i+1] = d[i+2] = v;
  }
  ctx.putImageData(imgData, 0, 0);
  redrawGrid();
}

function redrawGrid() {
  if (!showGrid) return;
  ctx.strokeStyle = "rgba(88,166,255,0.3)";
  ctx.lineWidth   = 0.05;
  for (let i = 0; i <= NATIVE_SIZE; i++) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, NATIVE_SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(NATIVE_SIZE, i); ctx.stroke();
  }
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
  ctx.fillStyle = tool === "brush" ? "#fff" : "#000";
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

// ── Sample actions ───────────────────────────────────────
document.getElementById("btn-save-training").addEventListener("click", async () => {
  const circles = parseInt(document.getElementById("circle-count").value, 10);
  if (isNaN(circles) || circles < 0) { showFeedback("save-feedback", "Enter a valid circle count (≥ 0)", "error"); return; }
  try {
    const res = await fetch("/api/training-samples", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: getImageBase64(), circles }),
    });
    const data = await res.json();
    if (res.ok) showFeedback("save-feedback", `✅ Saved as ${data.filename}`, "success");
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

function chartOptions(title) {
  return {
    responsive: true,
    animation: false,
    plugins: { legend: { labels: { color: "#8b949e", font: { size: 10 } } } },
    scales: {
      x: { ticks: { color: "#8b949e", font: { size: 9 } }, grid: { color: "#30363d" } },
      y: { ticks: { color: "#8b949e", font: { size: 9 } }, grid: { color: "#30363d" } },
    },
  };
}

// Color palette for chart lines
const MODEL_COLORS = {
  Dense:           { train: "#58a6ff", val: "#1f6feb" },
  DenseTwoHidden:  { train: "#3fb950", val: "#238636" },
  CNN:             { train: "#d2a8ff", val: "#8957e5" },
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
  trainLossDS.data  = buf.loss;
  valLossDS.data    = buf.val_loss.map(v => v ?? null);

  // MAE chart
  const trainMaeDS  = ensureDataset(chartMAE, `${model}-mae`, `${model} train`, colors.train);
  const valMaeDS    = ensureDataset(chartMAE, `${model}-val-mae`, `${model} val`, colors.val, true);
  trainMaeDS.data   = buf.mae;
  valMaeDS.data     = buf.val_mae.map(v => v ?? null);

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

  resetCharts();

  try {
    const res  = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs, batch_size: batchSize, val_split: valSplit, seed }),
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
      <td><code>${escapeHtml(info.weights_path)}</code></td>
    `;
    tbody.appendChild(tr);
  }
  document.getElementById("summary-table-wrap").classList.remove("hidden");
  showFeedback("train-feedback", "✅ Training complete!", "success");
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
