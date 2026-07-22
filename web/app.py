"""Document OCR + AI Pipeline — Streamlit UI.

A pure HTTP client of the FastAPI backend in api/ (see api_client.py) — this
process never touches OpenCV, Tesseract or an LLM directly. It shows a
document moving through the pipeline stage-by-stage, then offers structured
extraction, grounded Q&A and cross-document search, all delegated to the API.

Run:  streamlit run web/app.py   (with API_BASE_URL pointed at the api/ service)
"""

import base64
import csv
import io
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import api_client  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Document OCR + AI Pipeline", page_icon="·",
                   layout="centered", initial_sidebar_state="expanded")

# ── minimal academic styling (kept from the original single-pipeline UI) ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{ --ink:#1A1B1D; --muted:#646A72; --faint:#9AA0A8; --rule:#E5E3DC;
       --accent:#2B4B7E; --paper:#FBFBF9; --good:#2F7D5B; --warn:#B8860B;
       --sans:'IBM Plex Sans',system-ui,sans-serif;
       --mono:'IBM Plex Mono',monospace; --serif:'Newsreader',Georgia,serif; }

.stApp{ background:var(--paper); color:var(--ink); font-family:var(--sans); }
.block-container{ max-width:820px; padding-top:2.4rem; padding-bottom:4rem; }
#MainMenu, [data-testid="stToolbar"], [data-testid="stHeader"]{ display:none; }
::selection{ background:rgba(43,75,126,.16); }

.kicker{ font-family:var(--mono); font-size:.74rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); margin:0 0 .35rem; }
h1.title{ font-family:var(--serif); font-weight:500; font-size:2.3rem; line-height:1.1;
  letter-spacing:-.01em; margin:0; color:var(--ink); }
.lede{ color:var(--muted); font-size:1rem; line-height:1.55; margin:.7rem 0 0; max-width:62ch; }
.hr{ height:1px; background:var(--rule); border:0; margin:1.4rem 0; }

.seclabel{ font-family:var(--mono); font-size:.72rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--faint); margin:2rem 0 .8rem; }

.stg{ display:flex; align-items:baseline; gap:.7rem; margin:.2rem 0 .15rem; }
.stg .no{ font-family:var(--mono); font-size:.82rem; color:var(--accent); font-weight:500; }
.stg .nm{ font-family:var(--serif); font-size:1.18rem; font-weight:600; color:var(--ink); }
.stg .mt{ font-family:var(--mono); font-size:.74rem; color:var(--faint); margin-left:auto; }
.ds{ color:var(--muted); font-size:.9rem; margin:0 0 .7rem; }

[data-testid="stImage"]{ border:1px solid var(--rule); border-radius:4px; background:#fff; padding:6px; }
[data-testid="stImage"] img{ border-radius:2px; }
[data-testid="stImageCaption"]{ font-family:var(--mono)!important; font-size:.72rem!important; color:var(--faint)!important; }

.kv{ display:flex; gap:1rem; font-family:var(--mono); font-size:.86rem; padding:.45rem 0;
  border-bottom:1px solid var(--rule); }
.kv .k{ color:var(--muted); min-width:120px; text-transform:uppercase; font-size:.74rem; letter-spacing:.05em; }
.kv .v{ color:var(--ink); font-weight:500; }
.ocr{ font-family:var(--mono); font-size:.82rem; line-height:1.6; color:var(--ink); background:#fff;
  border:1px solid var(--rule); border-radius:4px; padding:.9rem 1rem; max-height:300px; overflow:auto; white-space:pre-wrap; }
.summary{ font-family:var(--mono); font-size:.82rem; color:var(--muted); }
.summary b{ color:var(--ink); font-weight:600; }

.badge{ display:inline-block; font-family:var(--mono); font-size:.7rem; letter-spacing:.04em;
  padding:.15rem .5rem; border-radius:3px; border:1px solid var(--rule); margin-bottom:.6rem; }
.badge.llm{ color:var(--good); border-color:var(--good); }
.badge.fallback{ color:var(--warn); border-color:var(--warn); }

.hit{ font-family:var(--mono); font-size:.8rem; color:var(--muted); padding:.4rem 0;
  border-bottom:1px solid var(--rule); }
.hit b{ color:var(--ink); }

.stButton>button, .stDownloadButton>button{ font-family:var(--sans); font-weight:500; border-radius:4px;
  border:1px solid var(--rule); background:#fff; color:var(--ink); }
.stButton>button:hover, .stDownloadButton>button:hover{ border-color:var(--accent); color:var(--accent); }
.stButton>button[kind="primary"]{ background:var(--accent); color:#fff; border:1px solid var(--accent); }
.stButton>button[kind="primary"]:hover{ background:#23416d; color:#fff; }

.stRadio label, .stSelectbox label, .stCheckbox label{ font-family:var(--mono)!important;
  font-size:.74rem!important; color:var(--muted)!important; }
.foot{ font-family:var(--mono); font-size:.72rem; color:var(--faint); margin-top:2.5rem;
  padding-top:1rem; border-top:1px solid var(--rule); }
</style>
""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────
# (stage images arrive already resized/rendered by api/visualize.py)
@st.cache_data(ttl=300)
def cached_models():
    try:
        return api_client.list_models()
    except api_client.APIError:
        return {"models": [], "llm_configured": False}


def model_options(models):
    return {f'{m["label"]} ({m["provider"]})': m["id"] for m in models}


def section(num, name, meta, desc):
    st.markdown(f'<div class="stg"><span class="no">{num}</span>'
                f'<span class="nm">{name}</span><span class="mt">{meta}</span></div>'
                f'<div class="ds">{desc}</div>', unsafe_allow_html=True)


def export_result(res, fmt):
    fmt = fmt.lower()
    if fmt == "txt":
        return res.get("text", "")
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["index", "word"])
        for i, wb in enumerate(res.get("word_boxes", [])):
            w.writerow([i, wb["text"]])
        return buf.getvalue()
    clean = {k: v for k, v in res.items() if k != "stages"}
    return json.dumps(clean, indent=2, ensure_ascii=False)


# ── session state ────────────────────────────────────────────────────────
st.session_state.setdefault("session_id", uuid.uuid4().hex)
st.session_state.setdefault("documents", [])  # [{id, name}]
SESSION_ID = st.session_state["session_id"]


# ── sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:var(--mono);font-size:.75rem;color:#646A72;line-height:1.8">'
                '<b>Document OCR + AI Pipeline</b><br>Classical CV/OCR core + an LLM '
                'extraction / RAG layer over OpenRouter.<br>'
                '<a href="https://github.com/JeganT143/Automated-Document-Analysis-and-OCR-System-for-Multi-Format-Documents" '
                'style="color:#2B4B7E">Source</a></div>', unsafe_allow_html=True)
    st.divider()

    api_up = api_client.is_healthy()
    st.markdown(('<span style="font-family:var(--mono);font-size:.75rem;color:#2F7D5B">● API online</span>'
                if api_up else
                '<span style="font-family:var(--mono);font-size:.75rem;color:#B8860B">● API unreachable</span>'),
               unsafe_allow_html=True)

    models_info = cached_models()
    opts = model_options(models_info["models"])
    llm_ready = models_info["llm_configured"]
    if not llm_ready:
        st.caption("No OPENROUTER_API_KEY on the server — extraction falls back "
                  "to the regex extractor, and Q&A/search are unavailable.")

    model_label = st.selectbox("LLM model", list(opts.keys()) or ["(none configured)"],
                               disabled=not opts)
    selected_model = opts.get(model_label)

    run_extraction = st.checkbox("Run LLM structured extraction", value=llm_ready,
                                 disabled=not llm_ready)
    analyze_layout = st.checkbox("Layout analysis", value=True)
    output_format = st.selectbox("Output format", ["json", "txt", "csv"])


# ── header ───────────────────────────────────────────────────────────────
st.markdown('<p class="kicker">Automated Document Analysis · OCR + LLM</p>', unsafe_allow_html=True)
st.markdown('<h1 class="title">Document OCR + AI Pipeline</h1>', unsafe_allow_html=True)
st.markdown('<p class="lede">Classical computer-vision preprocessing and Tesseract OCR, '
            'with an optional LLM structured-extraction, grounded Q&amp;A and cross-document '
            'search layer on top. Every stage below shows the actual document image as the '
            'API transforms it.</p>', unsafe_allow_html=True)
st.markdown('<hr class="hr">', unsafe_allow_html=True)

if not api_up:
    st.error(f"Cannot reach the API at `{api_client.API_BASE_URL}`. Is it running?")
    st.stop()


# ── input ────────────────────────────────────────────────────────────────
st.markdown('<p class="seclabel">Source document</p>', unsafe_allow_html=True)
src = st.radio("source", ["Upload image", "Generate sample invoice"],
               horizontal=True, label_visibility="collapsed")

file_bytes, file_name = None, None
if src == "Upload image":
    up = st.file_uploader("doc", type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
                          label_visibility="collapsed")
    if up:
        file_bytes, file_name = up.read(), up.name
else:
    lvl = st.select_slider("Degradation", ["clean", "scan", "heavy"], value="scan")
    if st.button("Generate invoice"):
        st.session_state["sample_bytes"] = api_client.sample_document(degrade=lvl)
        st.session_state.pop("result", None)
    file_bytes = st.session_state.get("sample_bytes")
    file_name = "sample_invoice.png"

if file_bytes:
    st.image(file_bytes, caption="loaded source", width=320)

run = st.button("Run pipeline", type="primary", disabled=file_bytes is None)

if run and file_bytes:
    with st.spinner("Processing…"):
        try:
            result = api_client.process_document(
                SESSION_ID, file_bytes, file_name or "document.png",
                analyze_layout=analyze_layout, extract=run_extraction,
                model=selected_model,
            )
        except api_client.APIError as e:
            st.error(f"Pipeline failed: {e.detail}")
            result = None
    if result:
        st.session_state["result"] = result
        st.session_state["documents"].append({"id": result["id"], "name": file_name})

result = st.session_state.get("result")


# ── walkthrough ──────────────────────────────────────────────────────────
if result:
    s = result["mean_confidence"]
    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown(f'<p class="summary"><b>{result["word_count"]}</b> words · '
                f'mean confidence <b>{s:.1f}%</b> · PSM <b>{result["psm_used"]}</b> · '
                f'<b>{len(result["regions"])}</b> regions · '
                f'<b>{len(result["fields"])}</b> regex fields · '
                f'<b>{result["metrics"].get("total_s", 0):.2f}s</b></p>', unsafe_allow_html=True)

    st.markdown('<p class="seclabel">Pipeline walkthrough</p>', unsafe_allow_html=True)
    for i, stage in enumerate(result["stages"], 1):
        section(f"{i:02d}", stage["title"],
               " · ".join(f"{k}: {v}" for k, v in stage["meta"].items()),
               stage["desc"])
        st.image(base64.b64decode(stage["image_png_b64"]))

    n = len(result["stages"]) + 1
    section(f"{n:02d}", "Structured output", output_format.upper(),
           "Regex clean-up, field extraction and serialisation.")
    for k, v in result["fields"].items():
        st.markdown(f'<div class="kv"><span class="k">{k.replace("_", " ")}</span>'
                    f'<span class="v">{v}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="ds" style="margin-top:.8rem">Recognised text</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ocr">{result["text"] or "—"}</div>', unsafe_allow_html=True)

    out = export_result(result, output_format)
    mime = {"json": "application/json", "txt": "text/plain", "csv": "text/csv"}
    st.download_button(f"Download .{output_format}", out,
                       file_name=f"ocr_result.{output_format}", mime=mime[output_format])

    # -- LLM structured extraction --------------------------------------
    extraction = result.get("extraction")
    if extraction:
        st.markdown('<p class="seclabel">LLM structured extraction</p>', unsafe_allow_html=True)
        if extraction["source"] == "llm":
            st.markdown(f'<span class="badge llm">LLM · {extraction["model"]}</span>',
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge fallback">regex fallback · {extraction["warning"]}</span>',
                       unsafe_allow_html=True)
        data = extraction["data"]
        cols = ["vendor", "invoice_no", "date", "currency", "subtotal", "tax", "total"]
        for k in cols:
            if data.get(k) is not None:
                st.markdown(f'<div class="kv"><span class="k">{k.replace("_", " ")}</span>'
                            f'<span class="v">{data[k]}</span></div>', unsafe_allow_html=True)
        if data.get("line_items"):
            st.table(data["line_items"])

    # -- grounded Q&A ------------------------------------------------------
    st.markdown('<p class="seclabel">Ask about this document</p>', unsafe_allow_html=True)
    question = st.text_input("question", placeholder="e.g. What is the tax rate?",
                             label_visibility="collapsed", key="qa_question")
    if st.button("Ask", disabled=not (llm_ready and question)):
        try:
            answer = api_client.ask_document(SESSION_ID, result["id"], question,
                                            model=selected_model)
            st.markdown(f'<div class="ocr">{answer["answer"]}</div>', unsafe_allow_html=True)
        except api_client.APIError as e:
            st.warning(f"Couldn't answer: {e.detail}")
    if not llm_ready:
        st.caption("Requires an OPENROUTER_API_KEY on the server.")


# ── cross-document search ───────────────────────────────────────────────
docs = st.session_state["documents"]
st.markdown('<hr class="hr">', unsafe_allow_html=True)
st.markdown('<p class="seclabel">Search across this session\'s documents</p>', unsafe_allow_html=True)
st.caption(f"{len(docs)} document(s) processed in this session "
          "(in-memory on the server, cleared on restart — see README).")
query = st.text_input("search", placeholder="e.g. Which invoice has the highest total?",
                      label_visibility="collapsed", key="search_query")
if st.button("Search", disabled=not (docs and query)):
    try:
        res = api_client.search_documents(SESSION_ID, query, model=selected_model)
        st.markdown(f'<div class="ocr">{res["answer"]}</div>', unsafe_allow_html=True)
        for hit in res.get("hits", []):
            st.markdown(f'<div class="hit"><b>{hit["document_id"][:8]}</b> '
                        f'(score {hit["score"]:.2f}) — {hit["snippet"][:140]}</div>',
                       unsafe_allow_html=True)
    except api_client.APIError as e:
        st.warning(f"Search failed: {e.detail}")

if not result and not docs:
    st.caption("Load a document and press Run pipeline to trace it through every stage.")

st.markdown('<div class="foot">Document OCR + AI Pipeline · classical CV/OCR core, '
            'LLM extraction + RAG layer via OpenRouter, deployed on Google Cloud Run.</div>',
           unsafe_allow_html=True)
