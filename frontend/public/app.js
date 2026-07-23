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
  kidney_left_delta_x: "Левая почка, вправо–влево",
  kidney_left_delta_y: "Левая почка, вперёд–назад",
  kidney_left_delta_z: "Левая почка, вверх–вниз",
  kidney_right_delta_x: "Правая почка, вправо–влево",
  kidney_right_delta_y: "Правая почка, вперёд–назад",
  kidney_right_delta_z: "Правая почка, вверх–вниз",
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
let polledCaseId = null;
let modelLoaded = false;
let patientLabel = null;
let qaBaseline = {}; // TD-05: исходные значения для dirty tracking

const el = (id) => document.getElementById(id);

const ANALYZE_ALLOWED = new Set(["uploaded", "failed"]);
const PREDICT_ALLOWED = new Set(["features_ready", "qa_pending", "predicted", "reported"]);

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  polledCaseId = null;
}

function startPolling(id) {
  stopPolling();
  if (!id) return;
  polledCaseId = id;
  pollTimer = setInterval(pollStatus, 2000);
  pollStatus();
}

function updateActionButtons() {
  const hasCase = Boolean(caseId);
  el("btnUpload").disabled = !hasCase;
  const canAnalyze =
    hasCase && modelLoaded && ANALYZE_ALLOWED.has(caseStatus) && caseStatus !== "extracting";
  el("btnAnalyze").disabled = !canAnalyze;
  const canPredict = hasCase && modelLoaded && PREDICT_ALLOWED.has(caseStatus);
  el("btnPredict").disabled = !canPredict;
  const canReport = hasCase && ["predicted", "reported"].includes(caseStatus);
  el("btnReport").disabled = !canReport;
  el("btnReportPdf").disabled = !canReport;
  el("btnSaveQa").disabled = !hasCase;
  el("btnReloadFeatures").disabled = !hasCase || !QA_READY_STATUSES.includes(caseStatus);
}

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

// UI-P1-03: disable + текст «Загрузка…» на время запроса.
// keepDisabled=true leaves the button disabled after success (e.g. Analyze while extracting).
async function withBusy(btn, busyText, fn, { keepDisabled = false } = {}) {
  if (btn.disabled) return;
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = busyText;
  let ok = false;
  try {
    await fn();
    ok = true;
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    btn.textContent = originalText;
    if (keepDisabled && ok) {
      btn.disabled = true;
    } else {
      btn.disabled = false;
      updateActionButtons();
    }
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
  stopPolling();
  caseId = null;
  caseStatus = "created";
  patientLabel = null;
  el("caseInfo").textContent = "";
  el("patientLabel").value = "";
  el("dicomFile").value = "";
  el("predTable").classList.add("hidden");
  el("coverageBlock").classList.add("hidden");
  el("progress").classList.add("hidden");
  el("reportPreview").classList.add("hidden");
  el("reportSummary").textContent = "Сформируйте отчёт после шага «Прогноз».";
  buildQaForm();
  updateWizardLocks();
  updateActionButtons();
  showWorkflow("upload");
};

el("btnBackToCases").onclick = () => {
  stopPolling();
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

function setCase(id, status = "created", label = null) {
  if (caseId && caseId !== id) {
    stopPolling();
  }
  caseId = id;
  caseStatus = status;
  if (label != null) patientLabel = label;
  el("caseInfo").textContent = caseId
    ? `Кейс: ${patientLabel || caseId.slice(0, 8)}…`
    : "";
  if (patientLabel && el("patientLabel")) {
    el("patientLabel").value = patientLabel;
  }
  updateWizardLocks();
  updateActionButtons();
}

function setStatus(status) {
  caseStatus = status;
  updateWizardLocks();
  updateActionButtons();
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
  badge.textContent = `Полнота данных КТ: ${coveragePct.toFixed(0)}%`;
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
    const num = Number(val);
    value.textContent = Number.isFinite(num) ? num.toFixed(2) : "н/д";
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
    modelLoaded = Boolean(h.model_loaded);
    if (modelLoaded) {
      node.textContent = `API OK · модель загружена`;
      node.className = "health ok";
    } else {
      node.textContent = "API OK · модель не загружена — анализ и прогноз недоступны";
      node.className = "health err";
    }
  } catch {
    modelLoaded = false;
    el("health").textContent = "API недоступен";
    el("health").className = "health err";
  }
  updateActionButtons();
}

async function pollStatus() {
  if (!caseId || !polledCaseId || caseId !== polledCaseId) return;
  const watchedId = polledCaseId;
  try {
    const st = await api(`/${watchedId}/status`);
    if (caseId !== watchedId || polledCaseId !== watchedId) return;
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
      stopPolling();
      showStep("qa");
      try {
        await loadFeatures();
        showToast("Извлечение признаков завершено");
      } catch (loadErr) {
        el("progressText").textContent =
          `Признаки извлечены, но не загрузились: ${loadErr.message || loadErr}`;
        el("progressText").classList.add("progress-error");
        showError("Extraction завершён, но QA не загрузилась — нажмите «Повторить загрузку признаков»");
        el("btnReloadFeatures").disabled = false;
      }
      updateActionButtons();
    }
    if (st.status === "failed") {
      stopPolling();
      el("progressText").textContent = `Ошибка extraction: ${st.error || st.message || "неизвестная ошибка"}`;
      el("progressText").classList.add("progress-error");
      showError("Extraction завершился с ошибкой");
      updateActionButtons();
    }
  } catch (e) {
    console.error(e);
    showError(`Ошибка опроса статуса: ${e.message || e}`);
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
  updateActionButtons();
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
          stopPolling();
          setCase(c.case_id, c.status, c.patient_label || null);
          showWorkflow("upload");
          if (c.status === "extracting") {
            el("progress").classList.remove("hidden");
            showStep("analyze");
            startPolling(c.case_id);
            return;
          }
          if (c.status === "uploaded" || c.status === "failed") {
            showStep("analyze");
          }
          if (QA_READY_STATUSES.includes(c.status)) {
            try {
              await loadFeatures();
            } catch (e) {
              showError(`Не удалось загрузить признаки: ${e.message || e}`);
            }
            showStep("qa");
          }
          if (c.status === "predicted" || c.status === "reported") {
            await loadPrediction();
            showStep("predict");
          }
          updateActionButtons();
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
    stopPolling();
    const label = el("patientLabel").value.trim() || null;
    const res = await api("", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_label: label }),
    });
    setCase(res.case_id, res.status, label);
    buildQaForm();
    el("predTable").classList.add("hidden");
    el("coverageBlock").classList.add("hidden");
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
      const detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
      throw new Error(detail || res.statusText);
    }
    setStatus("uploaded");
    showToast("DICOM загружен");
    showStep("analyze");
  });

el("btnAnalyze").onclick = () =>
  withBusy(
    el("btnAnalyze"),
    "Запуск…",
    async () => {
      if (!modelLoaded) throw new Error("Модель не загружена — анализ недоступен");
      await api(`/${caseId}/analyze`, { method: "POST" });
      setStatus("extracting");
      el("progress").classList.remove("hidden");
      showStep("analyze");
      startPolling(caseId);
    },
    { keepDisabled: true },
  );

el("btnReloadFeatures").onclick = () =>
  withBusy(el("btnReloadFeatures"), "Загрузка…", async () => {
    await loadFeatures();
    showToast("Признаки загружены");
    showStep("qa");
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
    showToast("Правки сохранены");
  });

el("btnPredict").onclick = () =>
  withBusy(el("btnPredict"), "Прогноз…", async () => {
    if (!modelLoaded) throw new Error("Модель не загружена — прогноз недоступен");
    const res = await api(`/${caseId}/predict`, { method: "POST" });
    setStatus("predicted");
    showPredictions(res.predictions);
    updateReportPreview(res.predictions);
    showToast("Прогноз рассчитан");
  });

// ---------------------------------------------------------------------------
// Отчёт: превью + JSON / PDF
// ---------------------------------------------------------------------------

function vectorNorm3(dx, dy, dz) {
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function fmtMm(val) {
  const num = Number(val);
  return Number.isFinite(num) ? num.toFixed(1) : "н/д";
}

function updateReportPreview(predictions) {
  if (!predictions) return;
  const left = {
    x: Number(predictions.kidney_left_delta_x),
    y: Number(predictions.kidney_left_delta_y),
    z: Number(predictions.kidney_left_delta_z),
  };
  const right = {
    x: Number(predictions.kidney_right_delta_x),
    y: Number(predictions.kidney_right_delta_y),
    z: Number(predictions.kidney_right_delta_z),
  };
  const leftNorm = Number.isFinite(left.x + left.y + left.z)
    ? vectorNorm3(left.x || 0, left.y || 0, left.z || 0)
    : NaN;
  const rightNorm = Number.isFinite(right.x + right.y + right.z)
    ? vectorNorm3(right.x || 0, right.y || 0, right.z || 0)
    : NaN;

  el("reportSummary").textContent =
    "Прогноз смещения почек при переводе пациента со спины на бок. " +
    "PDF-отчёт рассчитан для клинического чтения (мм, анатомические направления).";

  const preview = el("reportPreview");
  preview.innerHTML = `
    <h3>Краткое резюме для врача</h3>
    <ul>
      <li><strong>Левая почка:</strong> вправо–влево ${fmtMm(left.x)} мм,
        вперёд–назад ${fmtMm(left.y)} мм, вверх–вниз ${fmtMm(left.z)} мм;
        суммарно ${fmtMm(leftNorm)} мм</li>
      <li><strong>Правая почка:</strong> вправо–влево ${fmtMm(right.x)} мм,
        вперёд–назад ${fmtMm(right.y)} мм, вверх–вниз ${fmtMm(right.z)} мм;
        суммарно ${fmtMm(rightNorm)} мм</li>
    </ul>
    <p class="report-note">Направления: вправо — к правой стороне тела; вперёд — к животу;
      вверх — к голове. Исходное положение — лёжа на спине (МСКТ).</p>
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
  const label = (patientLabel || el("patientLabel")?.value?.trim() || caseId.slice(0, 8)).replace(
    /[^\w\-а-яА-ЯёЁ]+/gi,
    "_",
  );
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
