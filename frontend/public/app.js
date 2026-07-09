const API = "/api/v1/cases";

const QA_FIELDS = [
  "kidney_left_center_x_rel",
  "kidney_left_center_y_rel",
  "kidney_left_center_z_rel",
  "kidney_right_center_x_rel",
  "kidney_right_center_y_rel",
  "kidney_right_center_z_rel",
  "spine_center_x",
  "spine_center_y",
  "spine_center_z",
  "body_com_x",
  "body_com_y",
  "body_com_z",
  "body_width_mm",
  "body_depth_mm",
  "body_area_mm2",
  "kidney_left_volume_cm3",
  "kidney_right_volume_cm3",
];

let caseId = null;
let pollTimer = null;

const el = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function setCase(id) {
  caseId = id;
  el("caseInfo").textContent = `case_id: ${caseId}`;
  el("btnUpload").disabled = !caseId;
  el("btnAnalyze").disabled = !caseId;
}

function buildQaForm(base = {}) {
  const form = el("qaForm");
  form.innerHTML = "";
  QA_FIELDS.forEach((name) => {
    const label = document.createElement("label");
    label.textContent = name;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.name = name;
    input.value = base[name] ?? "";
    label.appendChild(input);
    form.appendChild(label);
  });
  el("btnSaveQa").disabled = !caseId;
}

function readQaOverrides() {
  const overrides = {};
  el("qaForm").querySelectorAll("input[name]").forEach((input) => {
    if (input.value !== "") overrides[input.name] = parseFloat(input.value);
  });
  return overrides;
}

function showPredictions(predictions) {
  const table = el("predTable");
  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";
  Object.entries(predictions).forEach(([key, val]) => {
    const tr = document.createElement("tr");
    if (key.endsWith("_z")) tr.classList.add("z-row");
    tr.innerHTML = `<td>${key}</td><td>${Number(val).toFixed(2)}</td>`;
    tbody.appendChild(tr);
  });
  table.classList.remove("hidden");
}

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
    el("progress").classList.remove("hidden");
    el("progressBar").style.width = `${st.progress_pct}%`;
    el("progressText").textContent = `${st.status} · ${st.stage || ""} · ${st.message || ""}`;
    if (st.error) el("progressText").textContent += ` · ОШИБКА: ${st.error}`;

    if (st.status === "features_ready" || st.status === "qa_pending") {
      clearInterval(pollTimer);
      pollTimer = null;
      await loadFeatures();
      el("btnPredict").disabled = false;
    }
    if (st.status === "failed") {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (e) {
    console.error(e);
  }
}

async function loadFeatures() {
  const feat = await api(`/${caseId}/features`);
  buildQaForm(feat.base_features);
  el("coverage").textContent = `Coverage: ${feat.coverage_pct.toFixed(1)}% · missing: ${feat.missing_features.length}`;
}

async function refreshCaseList() {
  const data = await api("");
  const list = el("caseList");
  list.innerHTML = "";
  data.cases.forEach((c) => {
    const li = document.createElement("li");
    li.innerHTML = `${c.case_id.slice(0, 8)}… · ${c.status} · ${c.patient_label || "—"}`;
    const btn = document.createElement("button");
    btn.textContent = "Открыть";
    btn.onclick = async () => {
      setCase(c.case_id);
      if (["features_ready", "qa_pending", "predicted", "reported"].includes(c.status)) {
        await loadFeatures();
        el("btnPredict").disabled = false;
      }
      if (c.status === "predicted" || c.status === "reported") {
        el("btnReport").disabled = false;
      }
    };
    li.appendChild(btn);
    list.appendChild(li);
  });
}

el("btnCreate").onclick = async () => {
  const label = el("patientLabel").value.trim() || null;
  const res = await api("", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_label: label }),
  });
  setCase(res.case_id);
  buildQaForm();
  await refreshCaseList();
};

el("btnUpload").onclick = async () => {
  const file = el("dicomFile").files[0];
  if (!file) return alert("Выберите файл");
  const fd = new FormData();
  fd.append("file", file);
  await fetch(`${API}/${caseId}/upload`, { method: "POST", body: fd }).then(async (r) => {
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  });
  el("btnAnalyze").disabled = false;
  alert("DICOM загружен");
};

el("btnAnalyze").onclick = async () => {
  await api(`/${caseId}/analyze`, { method: "POST" });
  el("progress").classList.remove("hidden");
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollStatus, 2000);
  pollStatus();
};

el("btnSaveQa").onclick = async () => {
  const overrides = readQaOverrides();
  await api(`/${caseId}/features/manual`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ overrides, reason: "manual QA from UI" }),
  });
  await loadFeatures();
  el("btnPredict").disabled = false;
};

el("btnPredict").onclick = async () => {
  const res = await api(`/${caseId}/predict`, { method: "POST" });
  showPredictions(res.predictions);
  el("btnReport").disabled = false;
};

el("btnReport").onclick = async () => {
  const report = await api(`/${caseId}/report.json`);
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `report_${caseId}.json`;
  a.click();
};

el("btnRefreshCases").onclick = refreshCaseList;

refreshHealth();
refreshCaseList();
buildQaForm();
