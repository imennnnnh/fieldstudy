"use strict";

const byId = (id) => document.getElementById(id);
const files = byId("files");
const measuredStatus = byId("measured-status");
const predictionForm = byId("prediction-form");
const number = new Intl.NumberFormat("zh-TW");
const percent = new Intl.NumberFormat("zh-TW", { style: "percent", maximumFractionDigits: 2 });
const phases = {
  uploading: "上傳資料",
  reading: "讀取、驗證與展開",
  splitting: "五折切分",
  cross_validation: "五折訓練與驗證",
  evaluating: "選模與整批評估",
  refitting: "正式模型重訓",
  publishing: "保存與發布",
};
let trainingActive = false;
let submitting = false;
let predicting = false;
let displayedRun = null;
let polling = null;

const formatLanding = (x, y) => "(" + x + "," + y + ")";
const formatRate = (value) => value === null ? "無可計算資料" : percent.format(value);
const activeJob = (job) => job.status === "uploading" || job.status === "running";

async function api(path, options) {
  const response = await fetch(path, options);
  const isJson = response.headers.get("content-type")?.includes("application/json");
  const payload = isJson ? await response.json() : null;
  if (!response.ok || !payload) {
    throw new Error(payload?.error ?? ("請求失敗（HTTP " + response.status + "）"));
  }
  return payload;
}

function updateButtons() {
  files.disabled = trainingActive || submitting;
  byId("train-button").disabled = files.disabled || files.files.length !== 3;
  byId("predict-button").disabled = predicting || !displayedRun;
}

function showStatus(id, message, error = false) {
  const element = byId(id);
  element.textContent = message;
  element.classList.toggle("error", error);
}

function renderJob(job) {
  trainingActive = activeJob(job);
  updateButtons();
  const progress = byId("train-progress");
  progress.hidden = !trainingActive || !job.total;
  if (!progress.hidden) {
    progress.max = job.total;
    progress.value = job.completed;
  }
  const stage = phases[job.phase];
  let message = job.message;
  if (stage) message = stage + " · " + message;
  if (!progress.hidden) message += "（" + job.completed + " / " + job.total + "）";
  if (job.status === "failed") message = "訓練失敗｜" + message;
  showStatus("train-status", message, job.status === "failed");
  const counts = job.counts;
  byId("data-counts").textContent = counts
    ? "共 " + number.format(counts.total_balls) + " 球位，有效 " + number.format(counts.valid_balls) +
      " 球、無效 " + number.format(counts.invalid_balls) + " 球、缺測排除 " + number.format(counts.missing_balls) + " 球。"
    : "";
}

function matrixTable(title, cells) {
  const wrapper = document.createElement("div");
  wrapper.className = "table-wrap";
  const table = document.createElement("table");
  table.createCaption().textContent = title;
  const header = table.createTHead().insertRow();
  const headings = ["實際＼預測", "有效", "無效"];
  for (const heading of headings) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = heading;
    header.append(cell);
  }
  const body = table.createTBody();
  for (const [index, counts] of cells.entries()) {
    const row = body.insertRow();
    const cell = document.createElement("th");
    cell.scope = "row";
    cell.textContent = headings[index + 1];
    row.append(cell);
    for (const count of counts) row.insertCell().textContent = number.format(count);
  }
  wrapper.append(table);
  return wrapper;
}

function renderReport(report) {
  if (!displayedRun) showStatus("prediction-status", "");
  displayedRun = report.run_id;
  updateButtons();
  byId("report-status").hidden = true;
  byId("report-note").textContent = report.note;
  byId("selected-models").textContent = "勝出模型　X 軸 " + report.winners.x + "　Y 軸 " + report.winners.y;
  const scores = byId("scores");
  scores.replaceChildren();
  for (const [model, candidate] of Object.entries(report.selection)) {
    const row = scores.insertRow();
    const cell = document.createElement("th");
    cell.scope = "row";
    cell.textContent = model;
    row.append(cell);
    for (const fold of "ABCDE") row.insertCell().textContent = candidate.folds[fold].toFixed(4);
    row.insertCell().textContent = candidate.mean.toFixed(4);
  }
  const evaluation = report.evaluation;
  byId("validation-count").textContent = "驗證球數 " + number.format(evaluation.validation_balls);
  const distance = evaluation.mean_distance === null ? "無可計算資料" : evaluation.mean_distance.toFixed(3) + " 格";
  const metrics = [
    ["X 軸命中率", formatRate(evaluation.hit_rates.x)],
    ["Y 軸命中率", formatRate(evaluation.hit_rates.y)],
    ["雙軸命中率", formatRate(evaluation.hit_rates.ball)],
    ["預測有效中實際有效", formatRate(evaluation.actual_valid_given_predicted_valid)],
    ["預測無效中實際無效", formatRate(evaluation.actual_invalid_given_predicted_invalid)],
    ["平均距離", distance + "（" + number.format(evaluation.distance_balls) + " 球）"],
  ];
  const list = byId("metrics");
  list.replaceChildren();
  for (const [label, value] of metrics) {
    const item = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    item.append(term, description);
    list.append(item);
  }
  byId("matrices").replaceChildren(
    matrixTable("X 軸有效性", evaluation.validity_matrices.x),
    matrixTable("Y 軸有效性", evaluation.validity_matrices.y),
    matrixTable("整球有效性", evaluation.validity_matrices.ball),
  );
  byId("report").hidden = false;
}

async function refreshJob() {
  if (polling !== null) return polling;
  polling = (async () => {
    try {
      const job = await api("/api/training/status");
      renderJob(job);
      if (job.latest_success_run_id && job.latest_success_run_id !== displayedRun) {
        renderReport(await api("/api/reports/latest"));
      }
    } catch (error) {
      showStatus("train-status", "無法更新訓練狀態，將重新連線。 " + error.message, true);
    } finally {
      polling = null;
    }
  })();
  return polling;
}

files.addEventListener("change", () => {
  const list = byId("selected-files");
  list.replaceChildren();
  for (const file of files.files) {
    const item = document.createElement("li");
    item.textContent = file.name;
    list.append(item);
  }
  list.hidden = !files.files.length;
  files.setCustomValidity(files.files.length === 3 ? "" : "請選擇三份 Excel。");
  updateButtons();
});

byId("training-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submitting || trainingActive || files.files.length !== 3) return;
  const body = new FormData(event.currentTarget);
  submitting = true;
  updateButtons();
  showStatus("train-status", "正在上傳 Excel…");
  try {
    renderJob(await api("/api/training", { method: "POST", body }));
  } catch (error) {
    showStatus("train-status", error.message, true);
  } finally {
    submitting = false;
    updateButtons();
  }
  await refreshJob();
});

measuredStatus.addEventListener("change", () => {
  const enabled = measuredStatus.value === "valid";
  byId("measured-coordinates").hidden = !enabled;
  for (const axis of ["x", "y"]) {
    predictionForm.elements[axis].disabled = !enabled;
    predictionForm.elements[axis].required = enabled;
  }
});

predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!displayedRun || predicting) return;
  const values = new FormData(predictionForm);
  const payload = { measured: { status: measuredStatus.value } };
  for (const key of ["angle", "position", "spin", "strength"]) {
    payload[key] = values.get(key);
  }
  if (measuredStatus.value === "valid") {
    payload.measured.x = values.get("x");
    payload.measured.y = values.get("y");
  }
  predicting = true;
  updateButtons();
  showStatus("prediction-status", "正在預測…");
  byId("comparison").textContent = "";
  try {
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const label = result.valid ? "有效球" : "無效球";
    showStatus("prediction-status", label + " " + formatLanding(result.x, result.y));
    const comparison = result.comparison;
    if (comparison) {
      const detail = comparison.status === "both_valid"
        ? (comparison.same_cell ? "座標相同" : "座標不同") + "，距離 " + comparison.distance.toFixed(3) + " 格"
        : comparison.message;
      byId("comparison").textContent = "實測 " + formatLanding(...comparison.measured) + " · " + detail;
    }
  } catch (error) {
    showStatus("prediction-status", error.message, true);
  } finally {
    predicting = false;
    updateButtons();
  }
});

refreshJob();
setInterval(refreshJob, 1500);
