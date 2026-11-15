"use strict";

const byId = (id) => document.getElementById(id);
const files = byId("files");
const testFile = byId("test-file");
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
let lastJobStatus = null;
let submitting = false;
let predicting = false;
let testing = false;
let displayedRun = null;
let latestRun = null;
let testedRun = null;
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

function excelSelectionError(input, single = false) {
  if (!input.files.length) {
    return single ? "請選擇一份測試用 .xlsx 檔案。" : "請至少選擇一份訓練用 .xlsx 檔案。";
  }
  if (single && input.files.length !== 1) {
    return "每次只能上傳一份測試用 .xlsx 檔案。";
  }
  const invalidFile = Array.from(input.files).find((file) => !/\.xlsx$/i.test(file.name));
  return invalidFile ? "「" + invalidFile.name + "」須為 .xlsx 檔案。" : "";
}

function updateButtons() {
  files.disabled = trainingActive || submitting;
  byId("train-button").disabled = files.disabled || Boolean(excelSelectionError(files));
  byId("predict-button").disabled = predicting || !displayedRun;

  const modelReady = Boolean(latestRun) && latestRun === displayedRun;
  const testHint = byId("test-model-hint");
  if (modelReady) {
    testHint.textContent = testFile.files.length ? "" : "請上傳一份測試 Excel（.xlsx）。";
  } else if (latestRun) {
    testHint.textContent = "正在載入正式模型報表…";
  } else if (lastJobStatus === "completed") {
    testHint.textContent = "訓練已完成，但目前無法載入正式模型。請查看後端終端機訊息。";
  } else {
    testHint.textContent = lastJobStatus === null ? "正在確認正式模型…" : "請先完成訓練";
  }

  testFile.disabled = testing;
  byId("evaluate-button").disabled = testing || !modelReady || Boolean(excelSelectionError(testFile, true));
}

function showStatus(id, message, error = false) {
  const element = byId(id);
  element.textContent = message;
  element.classList.toggle("error", error);
}

function renderJob(job) {
  lastJobStatus = job.status;
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

function renderEvaluation(evaluation, metricsId, matricesId, modelMetrics = []) {
  const distance = evaluation.mean_distance === null
    ? "無可計算資料"
    : evaluation.mean_distance.toFixed(3) + " 格";
  const metrics = modelMetrics.concat([
    ["X 軸命中率", formatRate(evaluation.hit_rates.x)],
    ["Y 軸命中率", formatRate(evaluation.hit_rates.y)],
    ["雙軸命中率", formatRate(evaluation.hit_rates.ball)],
    ["預測有效中實際有效", formatRate(evaluation.actual_valid_given_predicted_valid)],
    ["預測無效中實際無效", formatRate(evaluation.actual_invalid_given_predicted_invalid)],
    ["平均距離", distance + "（" + number.format(evaluation.distance_balls) + " 球）"],
  ]);
  const list = byId(metricsId);
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
  byId(matricesId).replaceChildren(
    matrixTable("X 軸有效性", evaluation.validity_matrices.x),
    matrixTable("Y 軸有效性", evaluation.validity_matrices.y),
    matrixTable("整球有效性", evaluation.validity_matrices.ball),
  );
}

function renderReport(report) {
  if (!displayedRun) showStatus("prediction-status", "");
  displayedRun = report.run_id;
  updateButtons();
  byId("report-status").hidden = true;
  byId("report-note").textContent = report.note;
  byId("selected-models").textContent =
    "勝出模型　X 軸 " + report.winners.x + "　Y 軸 " + report.winners.y + "　訓練批次 " + report.run_id;
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
  renderEvaluation(evaluation, "metrics", "matrices");
  byId("report").hidden = false;
}

function clearTestResult() {
  testedRun = null;
  byId("test-result").hidden = true;
}

function renderTestReport(result) {
  byId("test-source").textContent =
    "測試檔案 " + result.filename + "　正式模型批次 " + result.run_id;
  byId("test-count").textContent =
    "測試球數 " + number.format(result.evaluation.test_balls) +
    "（含無效球，缺測排除 " + number.format(result.missing_balls) + " 球）";
  const { x, y } = result.models;
  renderEvaluation(result.evaluation, "test-metrics", "test-matrices", [
    ["X 軸 Macro F1 · " + x.algorithm, x.macro_f1.toFixed(4)],
    ["Y 軸 Macro F1 · " + y.algorithm, y.macro_f1.toFixed(4)],
  ]);
  testedRun = result.run_id;
  byId("test-result").hidden = false;
}

function observePublishedRun(runId) {
  latestRun = runId;
  if (testedRun && testedRun !== latestRun) {
    clearTestResult();
    showStatus("test-status", "正式模型已更新，請重新執行獨立測試。");
  }
}

async function refreshJob() {
  if (polling !== null) return polling;
  polling = (async () => {
    try {
      const job = await api("/api/training/status");
      observePublishedRun(job.latest_success_run_id);
      renderJob(job);
      if (latestRun && latestRun !== displayedRun) {
        const report = await api("/api/reports/latest");
        observePublishedRun(report.run_id);
        renderReport(report);
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
  const error = excelSelectionError(files);
  files.setCustomValidity(error);
  byId("train-file-error").textContent = files.files.length ? error : "";
  updateButtons();
});

byId("training-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submitting || trainingActive || excelSelectionError(files)) return;
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

testFile.addEventListener("change", () => {
  clearTestResult();
  const error = testFile.files.length ? excelSelectionError(testFile, true) : "";
  showStatus("test-status", error, Boolean(error));
  updateButtons();
});

byId("test-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (testing || excelSelectionError(testFile, true)) return;
  if (!latestRun) {
    showStatus("test-status", "請先完成訓練", true);
    return;
  }
  if (latestRun !== displayedRun) {
    showStatus("test-status", "正在載入最新正式模型的報表，載入後請重新評估。", true);
    return;
  }

  const requestedRun = latestRun;
  const body = new FormData(event.currentTarget);
  clearTestResult();
  testing = true;
  updateButtons();
  showStatus("test-status", "正在讀取測試 Excel 並評估…");
  try {
    const result = await api("/api/evaluation", { method: "POST", body });
    await refreshJob();
    if (result.run_id !== requestedRun || result.run_id !== latestRun || result.run_id !== displayedRun) {
      throw new Error("正式模型已更新，請重新執行獨立測試。");
    }
    renderTestReport(result);
    showStatus("test-status", "評估完成");
  } catch (error) {
    clearTestResult();
    showStatus("test-status", "評估失敗｜" + error.message, true);
  } finally {
    testing = false;
    updateButtons();
  }
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

updateButtons();
refreshJob();
setInterval(refreshJob, 1500);
