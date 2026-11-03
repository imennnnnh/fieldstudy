'use strict';
const $ = (id) => document.getElementById(id);
const titles = {
  train: ['01', '從實測資料開始。', '匯入發球紀錄，建立四參數與落點之間的關係。'],
  predict: ['02', '看看這組設定，球會落在哪。', '輸入四個控制參數，檢視模型預測的落點與機率。'],
  reverse: ['03', '選定落點，尋找發球設定。', '以單一目標格為條件，完整搜尋四參數組合。'],
  multi: ['04', '同一落點，探索不同球路。', '先符合固定模型機率條件，再比較你偏好的機器設定。'],
  sampling: ['05', '下一輪實測，補在關鍵位置。', '產生 300 組新的補測參數，讓每一輪實驗更有方向。'],
};
const state = {hasModel: false, busy: false, canSample: false, files: {train: []}, targets: {reverse: '(4,3)', multi: '(4,3)'}};
const features = ['仰角角度', '左右位置', '旋球角度', '出球強度'];
const recommendationColumns = ['仰角角度', '左右位置', '旋球角度', '出球強度', '預測落點', '目標落點機率', '無效總機率'];
const routeDefaults = ['高速球', '低速球', '均衡方案'];
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function switchView(view) {
  if (!titles[view]) return;
  document.querySelectorAll('.view').forEach(node => {node.hidden = node.id !== `view-${view}`;});
  document.querySelectorAll('[data-view]').forEach(button => {
    button.classList.toggle('active', button.dataset.view === view);
    if (button.dataset.view === view) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  const [number, title, description] = titles[view];
  $('page-kicker').textContent = `MODEL WORKSPACE / ${number}`;
  $('page-title').textContent = title;
  $('page-description').textContent = description;
}
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => switchView(button.dataset.view)));
function updateButtons() {
  document.querySelectorAll('button[type="submit"]').forEach(button => {
    button.disabled = state.busy || (button.hasAttribute('data-model-required') && !state.hasModel);
    button.title = button.hasAttribute('data-model-required') && !state.hasModel ? '請先上傳資料並訓練模型' : '';
    if (button.id === 'sampling-submit' && document.querySelector('[name="generate"]:checked').value === 'yes') {
      button.disabled = state.busy || !state.canSample;
      button.title = state.canSample ? '' : '請先完成模型訓練，並保留原訓練資料';
    }
  });
}
function notice(target, message, error = true) {
  const box = element('div', message, `notice${error ? ' error' : ''}`);
  box.setAttribute('role', error ? 'alert' : 'status');
  target.replaceChildren(box);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); }
  catch { throw new Error(`伺服器回應無法讀取（HTTP ${response.status}）。請確認後端正在執行。`); }
  if (!response.ok || !data.ok) throw new Error(data.error || `操作失敗（HTTP ${response.status}）`);
  return data;
}
const jsonOptions = (body) => ({method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
function table(rows, columns) {
  if (!rows.length) return emptyState();
  const wrapper = element('div', undefined, 'result-table-wrap');
  wrapper.tabIndex = 0;
  wrapper.setAttribute('aria-label', '結果表格，可橫向捲動');
  const node = document.createElement('table');
  const head = node.createTHead().insertRow();
  for (const col of columns) {
    const th = element('th', col === '旋球角度' ? '旋球設定值' : col);
    th.scope = 'col'; head.append(th);
  }
  const body = node.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const col of columns) {
      const value = row[col];
      const td = tr.insertCell();
      td.textContent = value === null || value === undefined ? '' : col.includes('機率') ? percentage(value) : String(value);
    }
  }
  wrapper.append(node); return wrapper;
}
function percentage(value) {
  const p = Number(value);
  // 未達 100% 的值不會因格式四捨五入而顯示為 100%。
  if (p === 1) return '100%';
  if (p === 0) return '0%';
  if (p >= 0.999995) return '<100%';
  if (p < 0.000005 && p > 0) return '<0.001%';
  return `${(p * 100).toFixed(3).replace(/\.?0+$/, '')}%`;
}
function emptyState() {
  const box = element('div', undefined, 'empty-state');
  box.append(element('strong', '沒有符合條件的方案'), element('span', '模型預測目標格機率須為 100%，且無效球機率為 0%。請補充實測資料後重新訓練與搜尋。'));
  return box;
}
function resultCard(title, description, rows, columns = recommendationColumns) {
  const card = element('div', undefined, 'card');
  card.append(element('p', 'RESULTS', 'eyebrow'), element('h2', title));
  if (description) card.append(element('p', description, 'muted'));
  card.append(table(rows, columns));
  return card;
}
function metrics(entries) {
  const row = element('div', undefined, 'metric-row');
  entries.forEach(([label, value]) => {
    const cell = element('div', undefined, 'metric');
    cell.append(element('strong', value), element('small', label)); row.append(cell);
  });
  return row;
}
async function refreshModel() {
  const {info} = await api('/api/model-info');
  state.hasModel = info.has_model;
  const sampling = info.sampling || {};
  state.canSample = Boolean(sampling.ready);
  const source = $('sampling-source'); source.replaceChildren();
  if (sampling.ready) {
    source.append(element('p', `${info.best_model_name} · ${sampling.observed_settings.toLocaleString()} 組實測設定 · ${sampling.observed_shots.toLocaleString()} 球訓練紀錄`, 'muted'));
    if (sampling.source_files?.length) {
      const list = element('ul', undefined, 'source-files');
      for (const name of sampling.source_files) list.append(element('li', name));
      source.append(list);
    }
    source.append(element('p', `將排除訓練資料中的 ${sampling.excluded_settings.toLocaleString()} 組已有參數。無需再次上傳檔案。`, 'footnote'));
  } else source.append(element('p', sampling.message || '請先在上傳與訓練頁完成訓練。', 'muted'));
  $('model-status').textContent = info.has_model ? '模型已就緒' : '尚未建立模型';
  $('model-status').classList.toggle('ready', info.has_model);
  const detail = $('model-detail'); detail.replaceChildren();
  if (info.has_model) {
    detail.append(element('strong', info.best_model_name));
    const training = info.training_info || {};
    detail.append(element('p', `${Number(info.tested_parameter_count).toLocaleString()} 組設定 · ${info.num_classes} 個已學類別`));
    detail.append(element('p', `交叉驗證 Macro F1：${Number(training.cv_mean_macro_f1).toFixed(4)}`, 'footnote'));
  } else {
    detail.append(element('p', '上傳資料並訓練完成後，即可使用預測與參數搜尋。補測資料可獨立產生。'));
  }
  if (!$('route-options').children.length) buildRoutes(info.route_names);
  updateButtons();
  return info;
}
function buildRoutes(names) {
  for (const name of names || []) {
    const label = element('label', undefined, 'route-option');
    const input = document.createElement('input');
    input.type = 'checkbox'; input.name = 'route'; input.value = name; input.checked = routeDefaults.includes(name);
    label.append(input, element('span', name)); $('route-options').append(label);
  }
}
$('routes-all').addEventListener('click', () => document.querySelectorAll('[name="route"]').forEach(i => {i.checked = true;}));
$('routes-clear').addEventListener('click', () => document.querySelectorAll('[name="route"]').forEach(i => {i.checked = false;}));
function buildGrid(kind) {
  const grid = $(`${kind}-grid`);
  for (let x = 1; x <= 5; x++) {
    grid.append(element('span', `X${x}`, 'grid-axis'));
    for (let y = 1; y <= 5; y++) {
      const target = `(${x},${y})`;
      const button = element('button', target, 'grid-cell');
      button.type = 'button'; button.dataset.target = target;
      button.setAttribute('aria-label', `目標落點 ${target}`);
      button.setAttribute('aria-pressed', String(target === state.targets[kind]));
      button.classList.toggle('selected', target === state.targets[kind]);
      button.addEventListener('click', () => {
        state.targets[kind] = target;
        grid.querySelectorAll('button').forEach(b => {
          b.classList.toggle('selected', b === button); b.setAttribute('aria-pressed', String(b === button));
        });
        $(`${kind}-target`).textContent = target;
      });
      grid.append(button);
    }
  }
  grid.append(element('span', '', 'grid-axis'));
  for (let y = 1; y <= 5; y++) grid.append(element('span', `Y${y}`, 'grid-axis'));
}
['reverse', 'multi'].forEach(buildGrid);
function setupDrop(kind, extensions) {
  const input = $(`${kind}-files`), drop = $(`${kind}-drop`);
  function renderFiles() {
    const list = $(`${kind}-file-list`); list.replaceChildren();
    state.files[kind].forEach((file, index) => {
      const item = element('li');
      const remove = element('button', '移除'); remove.type = 'button'; remove.setAttribute('aria-label', `移除 ${file.name}`);
      remove.addEventListener('click', () => {state.files[kind].splice(index, 1); renderFiles();});
      item.append(element('span', `${file.name} · ${(file.size / 1024).toFixed(1)} KB`), remove); list.append(item);
    });
  }
  function select(files) {
    const incoming = Array.from(files);
    if (incoming.some(file => !extensions.some(ext => file.name.toLowerCase().endsWith(ext)))) {
      notice($(`${kind}-job`), `檔案格式不支援；請選擇 ${extensions.join('、')}。`); return;
    }
    const merged = [...state.files[kind], ...incoming];
    state.files[kind] = merged.filter((file, index) => merged.findIndex(f => f.name === file.name && f.size === file.size && f.lastModified === file.lastModified) === index);
    renderFiles();
  }
  input.addEventListener('change', () => {select(input.files); input.value = '';});
  drop.addEventListener('keydown', event => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); input.click();}});
  let depth = 0;
  drop.addEventListener('dragenter', event => {event.preventDefault(); depth++; drop.classList.add('dragover');});
  drop.addEventListener('dragover', event => {event.preventDefault();});
  drop.addEventListener('dragleave', event => {event.preventDefault(); if (--depth <= 0) drop.classList.remove('dragover');});
  drop.addEventListener('drop', event => {event.preventDefault(); depth = 0; drop.classList.remove('dragover'); select(event.dataTransfer.files);});
}
setupDrop('train', ['.xlsx', '.xls']);
function progressBox(kind) {
  const card = element('div', undefined, 'card job-card');
  const line = element('div', undefined, 'job-line');
  const message = element('strong', '準備執行'); const elapsed = element('span', '0 秒');
  line.append(message, elapsed);
  const progress = document.createElement('progress'); progress.max = 100; progress.value = 0; progress.setAttribute('aria-label', '工作進度');
  const details = document.createElement('details'); details.append(element('summary', '查看執行紀錄'));
  const log = element('pre', '', 'log'); details.append(log); details.hidden = true;
  card.append(line, progress, details); $(`${kind}-job`).replaceChildren(card);
  return {message, elapsed, progress, log, details};
}
async function watchJob(kind, id) {
  const ui = progressBox(kind);
  while (true) {
    const {job} = await api(`/api/job/${id}`);
    ui.message.textContent = job.message;
    ui.elapsed.textContent = `${job.progress}% · ${job.elapsed_seconds} 秒`;
    ui.progress.value = job.progress;
    if (job.log?.length) {ui.details.hidden = false; ui.log.textContent = job.log.join('\n');}
    if (job.status === 'error') {localStorage.removeItem('fieldstudy-job'); throw new Error(job.error || '工作執行失敗');}
    if (job.status === 'done') {
      localStorage.removeItem('fieldstudy-job');
      renderResult(kind, job.result);
      if (kind === 'train') await refreshModel();
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}
async function run(kind, action) {
  if (state.busy) return;
  state.busy = true; updateButtons();
  $(`${kind}-result`).replaceChildren();
  const ui = progressBox(kind); ui.message.textContent = '正在傳送資料，請稍候'; ui.progress.removeAttribute('value');
  try {
    const data = await action();
    if (data.skipped) {notice($(`${kind}-job`), data.message, false); return;}
    localStorage.setItem('fieldstudy-job', JSON.stringify({kind, id: data.job_id}));
    await watchJob(kind, data.job_id);
  } catch (error) {notice($(`${kind}-job`), error.message);}
  finally {state.busy = false; updateButtons();}
}
$('train-form').addEventListener('submit', event => {
  event.preventDefault();
  if (!state.files.train.length) {notice($('train-job'), '請先選擇至少一個 Excel 實測檔案。'); return;}
  const form = new FormData(); state.files.train.forEach(file => form.append('files', file));
  form.append('replace_old', String($('replace-old').checked));
  run('train', () => api('/api/upload-train', {method: 'POST', body: form}));
});
$('predict-form').addEventListener('submit', async event => {
  event.preventDefault(); state.busy = true; updateButtons();
  $('predict-result').replaceChildren(); $('predict-job').replaceChildren();
  try {
    const body = Object.fromEntries(new FormData(event.target).entries());
    const data = await api('/api/predict', jsonOptions(body));
    const card = element('div', undefined, 'card'); card.append(element('p', 'PREDICTED LANDING', 'eyebrow'), element('h2', data.prediction, 'result-highlight'), element('p', '模型機率最高的落點；仍需透過實測確認。', 'muted'));
    card.append(table(data.top, ['落點', '預測機率'])); $('predict-result').append(card);
  } catch (error) {notice($('predict-job'), error.message);}
  finally {state.busy = false; updateButtons();}
});
$('reverse-form').addEventListener('submit', event => {
  event.preventDefault();
  run('reverse', () => api('/api/reverse-search', jsonOptions({targets: [state.targets.reverse], top_n: Number($('reverse-count').value)})));
});
$('multi-form').addEventListener('submit', event => {
  event.preventDefault();
  const routes = Array.from(document.querySelectorAll('[name="route"]:checked'), input => input.value);
  if (!routes.length) {notice($('multi-job'), '請至少選擇一種球路。'); return;}
  run('multi', () => api('/api/multi-route', jsonOptions({targets: [state.targets.multi], routes, n_solutions: Number($('multi-count').value)})));
});
document.querySelectorAll('[name="generate"]').forEach(input => input.addEventListener('change', () => {
  const yes = document.querySelector('[name="generate"]:checked').value === 'yes';
  $('sampling-fields').hidden = !yes;
  $('sampling-submit').textContent = yes ? '產生 300 組補測資料' : '保留到下次';
  updateButtons();
}));
$('sampling-form').addEventListener('submit', event => {
  event.preventDefault();
  const yes = document.querySelector('[name="generate"]:checked').value === 'yes';
  if (yes && !state.canSample) {
    notice($('sampling-job'), '請先在上傳與訓練頁完成模型訓練，補測會直接使用該模型的訓練資料。'); return;
  }
  run('sampling', () => api('/api/sampling', jsonOptions({generate: yes})));
});
function renderResult(kind, result) {
  const target = $(`${kind}-result`); target.replaceChildren();
  if (kind === 'reverse') {
    target.append(resultCard('四參數推薦結果', `${result.status_text} 已掃描 ${result.searched_count.toLocaleString()} 組。`, result.rows));
  } else if (kind === 'multi') {
    for (const route of result.routes) target.append(resultCard(route.route, route.status_text, route.rows, [...recommendationColumns, '球路條件分數']));
  } else if (kind === 'train') {
    const card = element('div', undefined, 'card');
    card.append(element('p', 'TRAINING COMPLETE', 'eyebrow'), element('h2', `訓練完成 · ${result.best_model_name}`));
    card.append(metrics([['有效訓練球數', result.total_samples.toLocaleString()], ['四參數組合', result.total_groups.toLocaleString()], ['流程耗時（秒）', result.time_total_seconds]]));
    card.append(element('p', `交叉驗證 ${result.time_cv_seconds} 秒；全資料重訓 ${result.time_retrain_seconds} 秒。空白落點略過 ${result.blank_count} 球。`, 'muted'));
    card.append(table(result.cv_summary, ['模型', '平均 Macro F1', '平均 Balanced Accuracy']));
    if (result.missing_classes.length) card.append(element('p', `模型尚未學到的類別：${result.missing_classes.join('、')}`, 'footnote'));
    const details = document.createElement('details'); details.append(element('summary', '查看匯入報告、驗證與參數重要程度'));
    if (result.import_report?.length) details.append(table(result.import_report, ['檔案', '工作表', '狀態', '資料列數', '錯誤內容']));
    if (result.cv_folds?.length) details.append(table(result.cv_folds, Object.keys(result.cv_folds[0])));
    if (result.feature_importance?.length) details.append(table(result.feature_importance, ['參數', '重要程度']));
    card.append(details);
    const link = element('button', '前往選擇是否產生 300 組補測資料', 'secondary'); link.type = 'button'; link.addEventListener('click', () => switchView('sampling'));
    const actions = element('div', undefined, 'form-actions'); actions.append(element('span', '下一步：補測與驗證', 'muted'), link); card.append(actions); target.append(card);
  } else if (kind === 'sampling') {
    const card = element('div', undefined, 'card'); const toolbar = element('div', undefined, 'result-toolbar');
    toolbar.append(element('h2', '300 組補測參數已準備好'));
    const downloads = element('div', undefined, 'download-actions');
    for (const [extension, url] of Object.entries(result.downloads)) {
      const a = element('a', `下載 ${extension.toUpperCase()}`, 'secondary'); a.href = url; a.download = ''; downloads.append(a);
    }
    toolbar.append(downloads); card.append(toolbar);
    const s = result.summary;
    card.append(metrics([['空間探索', s.spatial_count], ['分類邊界補強', s.boundary_count], ['隨機驗證', s.random_count]]));
    card.append(element('p', `已排除 ${(s.excluded_count ?? s.measured_count).toLocaleString()} 組已有設定，找到 ${s.boundary_pair_count.toLocaleString()} 對近鄰異分布點對。${s.boundary_fallback_count ? `邊界候選不足，${s.boundary_fallback_count} 個名額轉為空間探索。` : ''}`, 'muted'));
    card.append(element('p', '下載檔包含全部 300 列。Excel 另附採樣策略工作表；五球落點欄留白，供後續實測填寫。', 'footnote'));
    const preview = element('div'); const pager = element('div', undefined, 'form-actions');
    const label = element('span', '', 'muted'); const actions = element('div', undefined, 'download-actions');
    const prev = element('button', '上一頁', 'secondary'), next = element('button', '下一頁', 'secondary'); prev.type = next.type = 'button';
    let page = 0;
    function renderPage() {
      const start = page * 30;
      preview.replaceChildren(table(result.rows.slice(start, start + 30), ['測試編號', ...features, ...[1,2,3,4,5].map(i => `發球${i}_落點`)]));
      label.textContent = `預覽 ${start + 1}–${Math.min(start + 30, result.rows.length)} / ${result.rows.length} 組`;
      prev.disabled = page === 0; next.disabled = start + 30 >= result.rows.length;
    }
    prev.addEventListener('click', () => {page--; renderPage();}); next.addEventListener('click', () => {page++; renderPage();});
    actions.append(prev, next); pager.append(label, actions); card.append(preview, pager); renderPage(); target.append(card);
  }
}
async function initialize() {
  updateButtons();
  try {
    const info = await refreshModel();
    let saved;
    try {saved = JSON.parse(localStorage.getItem('fieldstudy-job'));} catch {localStorage.removeItem('fieldstudy-job');}
    const active = saved || info.active_job;
    if (active && titles[active.kind]) {
      switchView(active.kind); state.busy = true; updateButtons();
      try {await watchJob(active.kind, active.id);}
      catch (error) {notice($(`${active.kind}-job`), error.message);}
      finally {state.busy = false; updateButtons();}
    }
  } catch (error) {const box = $('connection-error'); box.hidden = false; box.textContent = `無法讀取後端：${error.message} 請確認已從本機伺服器開啟此頁面。`;}
}
initialize();
