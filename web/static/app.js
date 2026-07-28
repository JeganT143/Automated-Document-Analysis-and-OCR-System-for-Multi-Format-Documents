// Document OCR + AI Pipeline — static frontend.
//
// A pure HTTP client of the FastAPI backend (see ../../api/), mirroring what
// web/api_client.py + web/app.py used to do as a Streamlit app. No build
// step, no framework — plain fetch calls against the endpoints documented
// in the README.

const API_BASE_URL = (window.API_BASE_URL || "http://localhost:8010").replace(/\/$/, "");
const TIMEOUT_MS = 60000;

class APIError extends Error {
  constructor(statusCode, detail) {
    super(`[${statusCode}] ${detail}`);
    this.statusCode = statusCode;
    this.detail = detail;
  }
}

function sessionId() {
  let id = sessionStorage.getItem("session_id");
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem("session_id", id);
  }
  return id;
}

function getDocuments() {
  return JSON.parse(sessionStorage.getItem("documents") || "[]");
}

function addDocument(doc) {
  const docs = getDocuments();
  docs.push(doc);
  sessionStorage.setItem("documents", JSON.stringify(docs));
  return docs;
}

async function withTimeout(promise) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    return await promise(controller.signal);
  } finally {
    clearTimeout(t);
  }
}

async function handleJSON(resp) {
  if (resp.status >= 400) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail || detail;
    } catch (_) { /* not JSON */ }
    throw new APIError(resp.status, detail);
  }
  return resp.json();
}

async function isHealthy() {
  try {
    const resp = await withTimeout((signal) =>
      fetch(`${API_BASE_URL}/healthz`, { signal }));
    return resp.status === 200;
  } catch (_) {
    return false;
  }
}

async function listModels() {
  const resp = await withTimeout((signal) =>
    fetch(`${API_BASE_URL}/v1/models`, { signal }));
  return handleJSON(resp);
}

async function sampleDocument(degrade) {
  const resp = await withTimeout((signal) =>
    fetch(`${API_BASE_URL}/v1/sample-document?degrade=${encodeURIComponent(degrade)}`, { signal }));
  if (resp.status >= 400) throw new APIError(resp.status, await resp.text());
  return resp.blob();
}

async function processDocument(file, filename, analyzeLayout, extract, model) {
  const form = new FormData();
  form.append("file", file, filename);
  form.append("analyze_layout", String(analyzeLayout));
  form.append("extract", String(extract));
  if (model) form.append("model", model);
  const resp = await withTimeout((signal) =>
    fetch(`${API_BASE_URL}/v1/documents`, {
      method: "POST", body: form,
      headers: { "X-Session-Id": sessionId() },
      signal,
    }));
  return handleJSON(resp);
}

async function askDocument(documentId, question, model) {
  const resp = await withTimeout((signal) =>
    fetch(`${API_BASE_URL}/v1/documents/${documentId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Session-Id": sessionId() },
      body: JSON.stringify({ question, model: model || null }),
      signal,
    }));
  return handleJSON(resp);
}

async function searchDocuments(query, model, topK = 3) {
  const resp = await withTimeout((signal) =>
    fetch(`${API_BASE_URL}/v1/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Session-Id": sessionId() },
      body: JSON.stringify({ query, model: model || null, top_k: topK }),
      signal,
    }));
  return handleJSON(resp);
}

// ── rendering helpers ──────────────────────────────────────────────────

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function csvEscape(v) {
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function exportResult(result, fmt) {
  if (fmt === "txt") return result.text || "";
  if (fmt === "csv") {
    const rows = ["index,word"];
    (result.word_boxes || []).forEach((wb, i) => rows.push(`${i},${csvEscape(wb.text)}`));
    return rows.join("\n");
  }
  const clean = { ...result };
  delete clean.stages;
  return JSON.stringify(clean, null, 2);
}

function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function renderStage(i, stage) {
  const meta = Object.entries(stage.meta).map(([k, v]) => `${k}: ${v}`).join(" &middot; ");
  return `
    <div class="stg"><span class="no">${String(i).padStart(2, "0")}</span>
      <span class="nm">${esc(stage.title)}</span><span class="mt">${meta}</span></div>
    <div class="ds">${esc(stage.desc)}</div>
    <div class="stage-img"><img src="data:image/jpeg;base64,${stage.image_jpeg_b64}"></div>`;
}

function renderResult(result, outputFormat) {
  const s = result.mean_confidence;
  let html = `<hr class="hr">
    <p class="summary"><b>${result.word_count}</b> words &middot;
      mean confidence <b>${s.toFixed(1)}%</b> &middot; PSM <b>${result.psm_used}</b> &middot;
      <b>${result.regions.length}</b> regions &middot;
      <b>${Object.keys(result.fields).length}</b> regex fields &middot;
      <b>${(result.metrics.total_s || 0).toFixed(2)}s</b></p>
    <p class="seclabel">Pipeline walkthrough</p>`;

  result.stages.forEach((stage, i) => { html += renderStage(i + 1, stage); });

  const n = result.stages.length + 1;
  html += `<div class="stg"><span class="no">${String(n).padStart(2, "0")}</span>
      <span class="nm">Structured output</span><span class="mt">${outputFormat.toUpperCase()}</span></div>
    <div class="ds">Regex clean-up, field extraction and serialisation.</div>`;
  for (const [k, v] of Object.entries(result.fields)) {
    html += `<div class="kv"><span class="k">${esc(k.replace(/_/g, " "))}</span><span class="v">${esc(v)}</span></div>`;
  }
  html += `<div class="ds" style="margin-top:.8rem">Recognised text</div>
    <div class="ocr">${esc(result.text) || "&mdash;"}</div>
    <div style="margin-top:.6rem"><button id="download-btn">Download .${outputFormat}</button></div>`;

  if (result.extraction) {
    const ex = result.extraction;
    html += `<p class="seclabel">LLM structured extraction</p>`;
    html += ex.source === "llm"
      ? `<span class="badge llm">LLM &middot; ${esc(ex.model)}</span>`
      : `<span class="badge fallback">regex fallback &middot; ${esc(ex.warning)}</span>`;
    const data = ex.data;
    const cols = ["vendor", "invoice_no", "date", "currency", "subtotal", "tax", "total"];
    for (const k of cols) {
      if (data[k] != null) {
        html += `<div class="kv"><span class="k">${esc(k.replace(/_/g, " "))}</span><span class="v">${esc(data[k])}</span></div>`;
      }
    }
    if (data.line_items && data.line_items.length) {
      html += `<table class="line-items"><thead><tr><th>Description</th><th>Qty</th><th>Unit price</th><th>Amount</th></tr></thead><tbody>`;
      for (const li of data.line_items) {
        html += `<tr><td>${esc(li.description)}</td><td>${esc(li.quantity)}</td><td>${esc(li.unit_price)}</td><td>${esc(li.amount)}</td></tr>`;
      }
      html += `</tbody></table>`;
    }
  }

  html += `<p class="seclabel">Ask about this document</p>
    <input type="text" id="qa-question" placeholder="e.g. What is the tax rate?">
    <div style="margin-top:.6rem"><button id="qa-ask-btn">Ask</button></div>
    <div id="qa-answer"></div>`;

  return html;
}

// ── page wiring ──────────────────────────────────────────────────────────

let currentResult = null;
let sampleBlobUrl = null;
let llmReady = false;

const $ = (id) => document.getElementById(id);

function updateDocCount() {
  const docs = getDocuments();
  $("doc-count").textContent =
    `${docs.length} document(s) processed in this session ` +
    `(in-memory on the server, cleared on restart).`;
  $("search-btn").disabled = !(docs.length && $("search-query").value.trim());
}

async function init() {
  const up = await isHealthy();
  if (!up) {
    $("api-status").className = "status-bad";
    $("api-status").textContent = "API unreachable";
    $("api-down").style.display = "block";
    $("api-down").textContent = `Cannot reach the API at ${API_BASE_URL}. Is it running?`;
    $("input-section").style.display = "none";
    return;
  }
  $("api-status").className = "status-ok";
  $("api-status").textContent = "API online";

  let modelsInfo;
  try {
    modelsInfo = await listModels();
  } catch (_) {
    modelsInfo = { models: [], llm_configured: false };
  }
  llmReady = modelsInfo.llm_configured;

  const select = $("model-select");
  select.innerHTML = "";
  if (modelsInfo.models.length) {
    select.disabled = false;
    for (const m of modelsInfo.models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.label} (${m.provider})`;
      select.appendChild(opt);
    }
  } else {
    select.disabled = true;
    select.innerHTML = "<option>(none configured)</option>";
  }

  if (!llmReady) {
    $("llm-warning").style.display = "block";
  }
  $("run-extraction").checked = llmReady;
  $("run-extraction").disabled = !llmReady;

  updateDocCount();
}

document.querySelectorAll('input[name="source"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const isUpload = document.querySelector('input[name="source"]:checked').value === "upload";
    $("upload-block").style.display = isUpload ? "block" : "none";
    $("sample-block").style.display = isUpload ? "none" : "block";
    $("run-btn").disabled = true;
    $("preview-img").style.display = "none";
    if (isUpload) {
      $("file-input").value = "";
      $("dropzone-hint").textContent = "up to 10MB · JPG, PNG, BMP, TIF";
      document.querySelector(".dropzone").classList.remove("has-file");
    }
  });
});

$("file-input").addEventListener("change", () => {
  const file = $("file-input").files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  $("preview-img").src = url;
  $("preview-img").style.display = "block";
  $("run-btn").disabled = false;
  $("dropzone-hint").textContent = `${file.name} &middot; ${(file.size / 1e6).toFixed(1)} MB`.replace("&middot;", "·");
  document.querySelector(".dropzone").classList.add("has-file");
});

$("generate-btn").addEventListener("click", async () => {
  $("generate-btn").disabled = true;
  try {
    const blob = await sampleDocument($("degrade-level").value);
    if (sampleBlobUrl) URL.revokeObjectURL(sampleBlobUrl);
    sampleBlobUrl = URL.createObjectURL(blob);
    $("preview-img").src = sampleBlobUrl;
    $("preview-img").style.display = "block";
    $("run-btn").disabled = false;
    $("result-section").innerHTML = "";
    currentResult = null;
  } catch (e) {
    alert(`Couldn't generate a sample invoice: ${e.detail || e.message}`);
  } finally {
    $("generate-btn").disabled = false;
  }
});

$("run-btn").addEventListener("click", async () => {
  const isUpload = document.querySelector('input[name="source"]:checked').value === "upload";
  let file, filename;
  if (isUpload) {
    file = $("file-input").files[0];
    filename = file ? file.name : "document.png";
  } else {
    const resp = await fetch($("preview-img").src);
    file = await resp.blob();
    filename = "sample_invoice.png";
  }
  if (!file) return;

  $("run-btn").disabled = true;
  $("run-btn").textContent = "Processing…";
  $("result-section").innerHTML = "";
  try {
    const model = $("model-select").disabled ? null : $("model-select").value;
    const result = await processDocument(
      file, filename,
      $("analyze-layout").checked,
      $("run-extraction").checked,
      model,
    );
    currentResult = result;
    addDocument({ id: result.id, name: filename });
    updateDocCount();
    renderCurrentResult();
  } catch (e) {
    $("result-section").innerHTML = `<div class="error-box">Pipeline failed: ${esc(e.detail || e.message)}</div>`;
  } finally {
    $("run-btn").disabled = false;
    $("run-btn").textContent = "Run pipeline";
  }
});

function renderCurrentResult() {
  const outputFormat = $("output-format").value;
  $("result-section").innerHTML = renderResult(currentResult, outputFormat);

  $("download-btn").addEventListener("click", () => {
    const fmt = $("output-format").value;
    const mime = { json: "application/json", txt: "text/plain", csv: "text/csv" }[fmt];
    downloadBlob(exportResult(currentResult, fmt), `ocr_result.${fmt}`, mime);
  });

  $("qa-ask-btn").addEventListener("click", async () => {
    const question = $("qa-question").value.trim();
    if (!question || !llmReady) return;
    $("qa-ask-btn").disabled = true;
    try {
      const model = $("model-select").disabled ? null : $("model-select").value;
      const answer = await askDocument(currentResult.id, question, model);
      $("qa-answer").innerHTML = `<div class="ocr">${esc(answer.answer)}</div>`;
    } catch (e) {
      $("qa-answer").innerHTML = `<div class="error-box">Couldn't answer: ${esc(e.detail || e.message)}</div>`;
    } finally {
      $("qa-ask-btn").disabled = false;
    }
  });
}

$("output-format").addEventListener("change", () => {
  if (currentResult) renderCurrentResult();
});

$("search-query").addEventListener("input", updateDocCount);

$("search-btn").addEventListener("click", async () => {
  const query = $("search-query").value.trim();
  if (!query) return;
  $("search-btn").disabled = true;
  try {
    const model = $("model-select").disabled ? null : $("model-select").value;
    const res = await searchDocuments(query, model);
    let html = `<div class="ocr">${esc(res.answer)}</div>`;
    for (const hit of res.hits || []) {
      html += `<div class="hit"><b>${esc(hit.document_id.slice(0, 8))}</b>
        (score ${hit.score.toFixed(2)}) &mdash; ${esc(hit.snippet.slice(0, 140))}</div>`;
    }
    $("search-result").innerHTML = html;
  } catch (e) {
    $("search-result").innerHTML = `<div class="error-box">Search failed: ${esc(e.detail || e.message)}</div>`;
  } finally {
    $("search-btn").disabled = false;
  }
});

init();
