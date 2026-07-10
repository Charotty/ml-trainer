const API = "/api/v1/cases";

// UI-P0-02: человекочитаемые русские подписи + единицы для всех полей QA
const QA_LABELS = {
  kidney_left_center_x_rel: { label: "Левая почка, X (медиолатеральная)", unit: "мм" },
  kidney_left_center_y_rel: { label: "Левая почка, Y (дорсовентральная)", unit: "мм" },
  kidney_left_center_z_rel: { label: "Левая почка, Z (краниокаудальная)", unit: "мм" },
  kidney_right_center_x_rel: { label: "Правая почка, X (медиолатеральная)", unit: "мм" },
  kidney_right_center_y_rel: { label: "Правая почка, Y (дорсовентральная)", unit: "мм" },
  kidney_right_center_z_rel: { label: "Правая почка, Z (краниокаудальная)", unit: "мм" },
  spine_center_x: { label: "Позвоночник, центр X", unit: "мм" },
  spine_center_y: { label: "Позвоночник, центр Y", unit: "мм" },
  spine_center_z: { label: "Позвоночник, центр Z", unit: "мм" },
  body_com_x: { label: "Центр масс тела, X", unit: "мм" },
  body_com_y: { label: "Центр масс тела, Y", unit: "мм" },
  body_com_z: { label: "Центр масс тела, Z", unit: "мм" },
  body_width_mm: { label: "Ширина тела", unit: "мм" },
  body_depth_mm: { label: "Глубина тела", unit: "мм" },
  body_area_mm2: { label: "Площадь сечения тела", unit: "мм²" },
  kidney_left_volume_cm3: { label: "Объём левой почки", unit: "см³" },
  kidney_right_volume_cm3: { label: "Объём правой почки", unit: "см³" },
};

// UI-P1-06: группировка полей QA по анатомическим секциям
const QA_GROUPS = [
  {
    title: "Левая почка",
    fields: ["kidney_left_center_x_rel", "kidney_left_center_y_rel", "kidney_left_center_z_rel"],
  },
  {
    title: "Правая почка",
    fields: ["kidney_right_center_x_rel", "kidney_right_center_y_rel", "kidney_right_center_z_rel"],
  },
  { title: "Позвоночник", fields: ["spine_center_x", "spine_center_y", "spine_center_z"] },
  { title: "Центр масс тела", fields: ["body_com_x", "body_com_y", "body_com_z"] },
  { title: "Размеры тела", fields: ["body_width_mm", "body_depth_mm", "body_area_mm2"] },
  { title: "Объёмы", fields: ["kidney_left_volume_cm3", "kidney_right_volume_cm3"] },
];

// UI-P2-02: русские подписи таргетов прогноза
const PRED_LABELS = {
  kidney_left_delta_x: "Левая почка, ΔX",
  kidney_left_delta_y: "Левая почка, ΔY",
  kidney_left_delta_z: "Левая почка, ΔZ",
  kidney_right_delta_x: "Правая почка, ΔX",
  kidney_right_delta_y: "Правая почка, ΔY",
  kidney_right_delta_z: "Правая почка, ΔZ",
};

// UI-P1-07: человекочитаемые статусы и стадии extraction
const STATUS_LABELS = {
  created: "Создан",
  uploaded: "DICOM загружен",
  extracting: "Извлечение признаков…",
  features_ready: "Признаки готовы",
  qa_pending: "Ожидает QA",
  predicted: "Прогноз готов",
  reported: "Отчёт сформирован",
  failed: "Ошибка",
};

const WIZARD_STEPS = ["upload", "analyze", "qa", "predict", "report"];
// Статусы кейса, при которых доступен шаг QA/Прогноз
const QA_READY_STATUSES = ["features_ready", "qa_pending", "predicted", "reported"];

let caseId = null;
let caseStatus = "created";
let pollTimer = null;
let qaBaseline = {}; // TD-05: исходные значения для dirty tracking

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Уведомления (UI-P1-02): toast вместо alert()
// ---------------------------------------------------------------------------

function showToast(message, type = "success") {
  const container = el("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("toast-out");
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function showError(message) {
  showToast(message, "error");
}

// UI-P1-03: disable + текст «Загрузка…» на время запроса
async function withBusy(btn, busyText, fn) {
  if (btn.disabled) return;
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = busyText;
  try {
    await fn();
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
}

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
    throw new Error(detail || res.statusText);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Навигация: Dashboard (UI-P0-03) ↔ Wizard
// ---------------------------------------------------------------------------

function showDashboard() {
  el("viewDashboard").classList.remove("hidden");
  el("viewWorkflow").classList.add("hidden");
}

function showWorkflow(step = "upload") {
  el("viewDashboard").classList.add("hidden");
  el("viewWorkflow").classList.remove("hidden");
  showStep(step);
}

el("btnNewCase").onclick = () => {
  caseId = null;
  caseStatus = "created";
  el("caseInfo").textContent = "";
  el("patientLabel").value = "";
  el("dicomFile").value = "";
  el("btnUpload").disabled = true;
  el("btnAnalyze").disabled = true;
  el("btnPredict").disabled = true;
  el("btnReport").disabled = true;
  el("btnReportPdf").disabled = true;
  el("predTable").classList.add("hidden");
  el("coverageBlock").classList.add("hidden");
  el("progress").classList.add("hidden");
  el("reportPreview").classList.add("hidden");
  el("reportSummary").textContent = "Сформируйте отчёт после шага «Прогноз».";
  buildQaForm();
  updateWizardLocks();
  showWorkflow("upload");
};

el("btnBackToCases").onclick = () => {
  showDashboard();
  refreshCaseList();
};

// ---------------------------------------------------------------------------
// Wizard (UI-P1-01)
// ---------------------------------------------------------------------------

function showStep(step) {
  document.querySelectorAll(".wizard-step").forEach((s) => {
    s.classList.toggle("hidden", s.dataset.step !== step);
  });
  document.querySelectorAll(".wizard-tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.step === step);
  });
}

function updateWizardLocks() {
  const qaReady = QA_READY_STATUSES.includes(caseStatus);
  const predicted = ["predicted", "reported"].includes(caseStatus);
  document.querySelectorAll(".wizard-tab").forEach((tab) => {
    const step = tab.dataset.step;
    let locked = false;
    if (step === "qa" || step === "predict") locked = !qaReady;
    if (step === "report") locked = !predicted;
    tab.disabled = locked;
    tab.classList.toggle("wizard-tab-locked", locked);
  });
}

document.querySelectorAll(".wizard-tab").forEach((tab) => {
  tab.onclick = () => showStep(tab.dataset.step);
});

// ---------------------------------------------------------------------------
// Кейс и статусы
// ---------------------------------------------------------------------------

function setCase(id, status = "created") {
  caseId = id;
  caseStatus = status;
  el("caseInfo").textContent = `case_id: ${caseId}`;
  el("btnUpload").disabled = !caseId;
  el("btnAnalyze").disabled = !caseId;
  updateWizardLocks();
}

function setStatus(status) {
  caseStatus = status;
  updateWizardLocks();
}

// ---------------------------------------------------------------------------
// QA форма (UI-P0-02, UI-P1-06, TD-05, TD-06)
// ---------------------------------------------------------------------------

function buildQaForm(base = {}, manualFields = new Set()) {
  const form = el("qaForm");
  form.innerHTML = "";
  qaBaseline = {};
  QA_GROUPS.forEach((group) => {
    const section = document.createElement("fieldset");
    section.className = "qa-group";
    const legend = document.createElement("legend");
    legend.textContent = group.title;
    section.appendChild(legend);
    const grid = document.createElement("div");
    grid.className = "qa-grid";
    group.fields.forEach((name) => {
      const meta = QA_LABELS[name] || { label: name, unit: "" };
      const label = document.createElement("label");
      label.title = name;
      const span = document.createElement("span");
      span.className = "qa-label";
      span.textContent = meta.unit ? `${meta.label}, ${meta.unit}` : meta.label;
      const tech = document.createElement("span");
      tech.className = "qa-tech";
      tech.textContent = name;
      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.name = name;
      const value = base[name] ?? "";
      input.value = value;
      qaBaseline[name] = String(value);
      // TD-06: подсветка невалидных значений
      input.oninput = () => {
        input.classList.toggle("invalid", input.value !== "" && !Number.isFinite(parseFloat(input.value)));
      };
      // UI-P2-06: подсветка полей, изменённых вручную
      if (manualFields.has(name)) label.classList.add("manual-override");
      label.appendChild(span);
      label.appendChild(tech);
      label.appendChild(input);
      grid.appendChild(label);
    });
    section.appendChild(grid);
    form.appendChild(section);
  });
  el("btnSaveQa").disabled = !caseId;
}

// TD-05: отправляем только изменённые поля
function readQaOverrides() {
  const overrides = {};
  el("qaForm").querySelectorAll("input[name]").forEach((input) => {
    if (input.value === "" || input.value === qaBaseline[input.name]) return;
    const num = parseFloat(input.value);
    if (Number.isFinite(num)) overrides[input.name] = num;
  });
  return overrides;
}

// ---------------------------------------------------------------------------
// Coverage (UI-P1-05)
// ---------------------------------------------------------------------------

function showCoverage(coveragePct, missing) {
  el("coverageBlock").classList.remove("hidden");
  const badge = el("coverageBadge");
  badge.textContent = `Coverage: ${coveragePct.toFixed(1)}%`;
  badge.className = "coverage-badge " + (coveragePct >= 90 ? "cov-ok" : coveragePct >= 80 ? "cov-warn" : "cov-bad");
  el("coverageWarning").classList.toggle("hidden", coveragePct >= 80);

  const details = el("missingDetails");
  const list = el("missingList");
  list.innerHTML = "";
  if (missing.length) {
    missing.forEach((name) => {
      const li = document.createElement("li");
      li.textContent = name;
      list.appendChild(li);
    });
    details.classList.remove("hidden");
    details.querySelector("summary").textContent = `Недостающие признаки (${missing.length})`;
  } else {
    details.classList.add("hidden");
  }
}

// ---------------------------------------------------------------------------
// Прогноз (UI-P2-02)
// ---------------------------------------------------------------------------

function showPredictions(predictions) {
  const table = el("predTable");
  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";
  Object.entries(predictions).forEach(([key, val]) => {
    const tr = document.createElement("tr");
    if (key.endsWith("_z")) tr.classList.add("z-row");
    const name = document.createElement("td");
    name.textContent = PRED_LABELS[key] || key;
    name.title = key;
    const value = document.createElement("td");
    value.textContent = Number(val).toFixed(2);
    tr.appendChild(name);
    tr.appendChild(value);
    tbody.appendChild(tr);
  });
  table.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Health / status polling (UI-P1-07, UI-P2-07)
// ---------------------------------------------------------------------------

async function refreshHealth() {
  try {
    const h = await fetch("/health").then((r) => r.json());
    const node = el("health");
    if (h.model_loaded) {
      node.textContent = `API OK · ${h.model_id} · ${h.feature_count} признаков`;
      node.className = "health ok";
    } else {
      node.textContent = "API OK · модель не загружена (обучите clinical_honest.pkl)";
      node.className = "health err";
    }
  } catch {
    el("health").textContent = "API недоступен";
    el("health").className = "health err";
  }
}

async function pollStatus() {
  if (!caseId) return;
  try {
    const st = await api(`/${caseId}/status`);
    setStatus(st.status);
    el("progress").classList.remove("hidden");
    el("progressBar").style.width = `${st.progress_pct}%`;
    const statusLabel = STATUS_LABELS[st.status] || st.status;
    const parts = [statusLabel];
    if (st.stage) parts.push(st.stage);
    if (st.message) parts.push(st.message);
    el("progressText").textContent = parts.join(" · ");
    el("progressText").classList.remove("progress-error");

    if (st.status === "features_ready" || st.status === "qa_pending") {
      clearInterval(pollTimer);
      pollTimer = null;
      await loadFeatures();
      el("btnPredict").disabled = false;
      showToast("Извлечение признаков завершено");
      showStep("qa");
    }
    if (st.status === "failed") {
      clearInterval(pollTimer);
      pollTimer = null;
      // UI-P1-07: текст ошибки в карточке, не только в консоли
      el("progressText").textContent = `Ошибка extraction: ${st.error || st.message || "неизвестная ошибка"}`;
      el("progressText").classList.add("progress-error");
      showError("Extraction завершился с ошибкой");
    }
  } catch (e) {
    console.error(e);
  }
}

async function loadFeatures() {
  const feat = await api(`/${caseId}/features`);
  const manualFields = new Set();
  (feat.manual_overrides || []).forEach((entry) => {
    Object.keys(entry.overrides || {}).forEach((k) => manualFields.add(k));
  });
  buildQaForm(feat.base_features, manualFields);
  showCoverage(feat.coverage_pct, feat.missing_features || []);
}

// UI-P1-04: подгрузка сохранённого прогноза при открытии кейса
async function loadPrediction() {
  try {
    const pred = await api(`/${caseId}/prediction`);
    if (pred && pred.predictions) {
      showPredictions(pred.predictions);
      updateReportPreview(pred.predictions);
      el("btnReport").disabled = false;
      el("btnReportPdf").disabled = false;
    }
  } catch {
    // Прогноз ещё не сохранён — не ошибка
  }
}

// ---------------------------------------------------------------------------
// Список кейсов
// ---------------------------------------------------------------------------

async function refreshCaseList() {
  try {
    const data = await api("");
    const tbody = el("caseList");
    tbody.innerHTML = "";
    data.cases.forEach((c) => {
      const tr = document.createElement("tr");
      const date = (c.created_at || "").slice(0, 10);
      const coverage = c.coverage_pct != null ? `${c.coverage_pct.toFixed(0)}%` : "—";
      tr.innerHTML =
        `<td title="${c.case_id}">${c.case_id.slice(0, 8)}…</td>` +
        `<td>${c.patient_label || "—"}</td>` +
        `<td>${STATUS_LABELS[c.status] || c.status}</td>` +
        `<td>${coverage}</td>` +
        `<td>${date}</td>`;
      const td = document.createElement("td");
      const btn = document.createElement("button");
      btn.textContent = "Открыть";
      btn.onclick = () =>
        withBusy(btn, "Открытие…", async () => {
          setCase(c.case_id, c.status);
          showWorkflow("upload");
          if (QA_READY_STATUSES.includes(c.status)) {
            await loadFeatures();
            el("btnPredict").disabled = false;
            showStep("qa");
          }
          if (c.status === "predicted" || c.status === "reported") {
            el("btnReport").disabled = false;
            el("btnReportPdf").disabled = false;
            await loadPrediction();
            showStep("predict");
          }
        });
      td.appendChild(btn);
      tr.appendChild(td);
      tbody.appendChild(tr);
    });
  } catch (e) {
    showError(`Не удалось загрузить кейсы: ${e.message}`);
  }
}

// ---------------------------------------------------------------------------
// Обработчики кнопок
// ---------------------------------------------------------------------------

el("btnCreate").onclick = () =>
  withBusy(el("btnCreate"), "Создание…", async () => {
    const label = el("patientLabel").value.trim() || null;
    const res = await api("", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_label: label }),
    });
    setCase(res.case_id, res.status);
    buildQaForm();
    el("predTable").classList.add("hidden");
    el("coverageBlock").classList.add("hidden");
    el("btnPredict").disabled = true;
    el("btnReport").disabled = true;
    el("btnReportPdf").disabled = true;
    showToast("Кейс создан");
    await refreshCaseList();
    showWorkflow("upload");
  });

el("btnUpload").onclick = () =>
  withBusy(el("btnUpload"), "Загрузка…", async () => {
    const file = el("dicomFile").files[0];
    if (!file) throw new Error("Выберите файл DICOM (.zip)");
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`${API}/${caseId}/upload`, { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    setStatus("uploaded");
    el("btnAnalyze").disabled = false;
    showToast("DICOM загружен");
    showStep("analyze");
  });

el("btnAnalyze").onclick = () =>
  withBusy(el("btnAnalyze"), "Запуск…", async () => {
    await api(`/${caseId}/analyze`, { method: "POST" });
    el("progress").classList.remove("hidden");
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 2000);
    pollStatus();
  });

el("btnSaveQa").onclick = () =>
  withBusy(el("btnSaveQa"), "Сохранение…", async () => {
    const overrides = readQaOverrides();
    if (!Object.keys(overrides).length) {
      showToast("Нет изменённых полей", "error");
      return;
    }
    await api(`/${caseId}/features/manual`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ overrides, reason: "manual QA from UI" }),
    });
    setStatus("qa_pending");
    await loadFeatures();
    el("btnPredict").disabled = false;
    showToast("Правки сохранены");
  });

el("btnPredict").onclick = () =>
  withBusy(el("btnPredict"), "Прогноз…", async () => {
    const res = await api(`/${caseId}/predict`, { method: "POST" });
    setStatus("predicted");
    showPredictions(res.predictions);
    el("btnReport").disabled = false;
    el("btnReportPdf").disabled = false;
    updateReportPreview(res.predictions);
    showToast("Прогноз рассчитан");
  });

// ---------------------------------------------------------------------------
// Отчёт: превью + JSON / PDF
// ---------------------------------------------------------------------------

function vectorNorm3(dx, dy, dz) {
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function updateReportPreview(predictions) {
  if (!predictions) return;
  const left = {
    x: Number(predictions.kidney_left_delta_x || 0),
    y: Number(predictions.kidney_left_delta_y || 0),
    z: Number(predictions.kidney_left_delta_z || 0),
  };
  const right = {
    x: Number(predictions.kidney_right_delta_x || 0),
    y: Number(predictions.kidney_right_delta_y || 0),
    z: Number(predictions.kidney_right_delta_z || 0),
  };
  const leftNorm = vectorNorm3(left.x, left.y, left.z);
  const rightNorm = vectorNorm3(right.x, right.y, right.z);

  el("reportSummary").textContent =
    "Клиническое резюме: прогноз смещения почек при переводе из supine в lateral. " +
    "PDF содержит текст для врача и инженера, таблицы Δ и графики (столбцы, нормы векторов, 3D-проекции).";

  const preview = el("reportPreview");
  preview.innerHTML = `
    <h3>Краткое резюме прогноза</h3>
    <ul>
      <li><strong>Левая почка:</strong> ΔX=${left.x.toFixed(1)}, ΔY=${left.y.toFixed(1)}, ΔZ=${left.z.toFixed(1)} мм; ‖Δ‖=${leftNorm.toFixed(1)} мм</li>
      <li><strong>Правая почка:</strong> ΔX=${right.x.toFixed(1)}, ΔY=${right.y.toFixed(1)}, ΔZ=${right.z.toFixed(1)} мм; ‖Δ‖=${rightNorm.toFixed(1)} мм</li>
    </ul>
    <p class="report-note">Оси: X — медиолатерально (L→R), Y — передне-задне (P→A), Z — краниокаудально (I→S).</p>
  `;
  preview.classList.remove("hidden");
}

async function downloadReportJson() {
  const report = await api(`/${caseId}/report.json`);
  setStatus("reported");
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `report_${caseId}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

async function downloadReportPdf() {
  const res = await fetch(`${API}/${caseId}/report.pdf`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  setStatus("reported");
  const blob = await res.blob();
  const label = el("patientLabel")?.value?.trim() || caseId.slice(0, 8);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ct_workbench_report_${label}.pdf`;
  a.click();
  URL.revokeObjectURL(a.href);
}

el("btnReportPdf").onclick = () =>
  withBusy(el("btnReportPdf"), "Формирование PDF…", async () => {
    await downloadReportPdf();
    showToast("PDF-отчёт скачан");
  });

el("btnReport").onclick = () =>
  withBusy(el("btnReport"), "Формирование…", async () => {
    await downloadReportJson();
    showToast("JSON-отчёт скачан");
  });

el("btnRefreshCases").onclick = refreshCaseList;

refreshHealth();
refreshCaseList();
buildQaForm();
showDashboard();
