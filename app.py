"""
Streamlit UI – Automated Document Analysis and OCR System.

A single, end-to-end pipeline:
    preprocess -> layout analysis -> recognise (adaptive Tesseract) -> post-process

Run:  streamlit run app.py
"""

import os
import sys
import time

import numpy as np
import cv2
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from pipeline import DocumentOCRPipeline
from tesseract_setup import tesseract_version

st.set_page_config(page_title="Document OCR System", page_icon="📄",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .main-header {font-size:2rem;font-weight:700;color:#1f4e79;
    border-bottom:3px solid #2e86c1;padding-bottom:0.4rem;margin-bottom:1rem;}
  .stage-badge {display:inline-block;background:#2e86c1;color:white;
    border-radius:12px;padding:2px 10px;font-size:0.78rem;font-weight:600;margin-right:6px;}
  .warn-box {background:#fef9e7;border-left:4px solid #f39c12;
    padding:0.6rem 1rem;border-radius:6px;}
  .ok-box {background:#eafaf1;border-left:4px solid #27ae60;
    padding:0.6rem 1rem;border-radius:6px;}
</style>
""", unsafe_allow_html=True)


# ─── helpers ──────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline(lang):
    return DocumentOCRPipeline(lang=lang)


def gray_to_rgb(g):
    if g.ndim == 2:
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(g, cv2.COLOR_BGR2RGB)


def draw_regions(gray, regions):
    colors = {"header_footer": (255, 140, 0), "table": (0, 160, 255),
              "text": (30, 180, 30), "image": (180, 0, 220), "unknown": (130, 130, 130)}
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for r in regions:
        x, y, w, h = r["bbox"]
        c = colors.get(r["type"], (130, 130, 130))
        cv2.rectangle(vis, (x, y), (x + w, y + h), c, 2)
        cv2.putText(vis, r["type"][:4], (x + 3, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def draw_word_boxes(gray, word_boxes):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for w in word_boxes:
        x, y, ww, hh = w["bbox"]
        c = w["conf"] / 100.0
        color = (0, int(c * 200), int((1 - c) * 220))   # green=high, red=low
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), color, 1)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def make_sample_invoice(level):
    """Render a labelled-style invoice using the dataset generator."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_invoice_dataset as mk
    rng = np.random.default_rng(int(time.time()) % 100000)
    data = mk.build_invoice(rng)
    img, _ = mk.render_invoice(data)
    return mk.degrade(img, rng, level=level)


# ─── sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")
    lang = st.selectbox("OCR language", ["eng"], help="Tesseract language pack")
    analyze_layout = st.checkbox("Run layout analysis", value=True,
                                 help="Detect text / table / header regions")
    output_format = st.selectbox("Output format", ["json", "txt", "csv"])
    st.markdown("---")
    ver = tesseract_version()
    if ver:
        st.markdown(f'<div class="ok-box">Tesseract <b>{ver}</b> ready</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">Tesseract not found.<br>'
                    'Run <code>scripts/install_tesseract_local.sh</code></div>',
                    unsafe_allow_html=True)
    st.markdown("---")
    st.info("EE7204 / EC7205\n\nDept of EIE, University of Ruhuna\n\n"
            "Pipeline: Preprocess → Layout → Recognise → Post-process")


st.markdown('<div class="main-header">📄 Automated Document OCR System</div>',
            unsafe_allow_html=True)
tab_run, tab_about = st.tabs(["🔍 Run Pipeline", "📖 About"])


# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════
with tab_run:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### 📥 Input Document")
        src = st.radio("Image source", ["Upload image", "Generate sample invoice"],
                       horizontal=True)
        image = None   # numpy (BGR or gray) handed to the pipeline

        if src == "Upload image":
            up = st.file_uploader("Choose a document image",
                                  type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"])
            if up:
                arr = np.frombuffer(up.read(), np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                st.image(gray_to_rgb(image), caption="Uploaded document",
                         use_container_width=True)
        else:
            level = st.select_slider("Degradation", ["clean", "scan", "heavy"],
                                     value="scan")
            if st.button("🎲 Generate invoice"):
                st.session_state["sample"] = make_sample_invoice(level)
            if "sample" in st.session_state:
                image = st.session_state["sample"]
                st.image(image, caption="Generated invoice", use_container_width=True)

        run = st.button("🚀 Run OCR Pipeline", type="primary", disabled=image is None)

    with col_r:
        st.markdown("### 📊 Results")
        if run and image is not None:
            if tesseract_version() is None:
                st.error("Tesseract is not available – cannot run OCR.")
                st.stop()
            with st.spinner("Running pipeline…"):
                pipe = get_pipeline(lang)
                result = pipe.run(image, analyze_layout=analyze_layout)
            proc = result["processed_image"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Words", result["word_count"])
            m2.metric("Confidence", f"{result['mean_confidence']:.1f}%")
            m3.metric("Regions", len(result["regions"]))
            m4.metric("Time", f"{result['metrics']['total_s']:.2f}s")

            v1, v2, v3, v4 = st.tabs(["Preprocessing", "Layout", "Recognition", "Output"])

            with v1:
                st.markdown('<span class="stage-badge">Stage 1</span> Preprocessing',
                            unsafe_allow_html=True)
                st.image(gray_to_rgb(proc), caption="OCR-ready image "
                         "(deskewed, upscaled, denoised)", use_container_width=True)
                st.json(result["preprocess_info"])

            with v2:
                st.markdown('<span class="stage-badge">Stage 2</span> Layout Analysis',
                            unsafe_allow_html=True)
                if result["regions"]:
                    st.image(draw_regions(proc, result["regions"]),
                             caption=f"{len(result['regions'])} regions",
                             use_container_width=True)
                    from collections import Counter
                    counts = Counter(r["type"] for r in result["regions"])
                    cols = st.columns(len(counts) or 1)
                    for c, (t, n) in zip(cols, counts.items()):
                        c.metric(t, n)
                else:
                    st.info("Layout analysis disabled or no regions found.")

            with v3:
                st.markdown('<span class="stage-badge">Stage 3</span> Recognition',
                            unsafe_allow_html=True)
                st.image(draw_word_boxes(proc, result["word_boxes"]),
                         caption="Word boxes (green = high confidence, red = low) · "
                                 f"PSM {result['psm_used']}", use_container_width=True)
                st.caption(f"Mean confidence {result['mean_confidence']:.1f}% "
                           f"over {result['word_count']} words")

            with v4:
                st.markdown('<span class="stage-badge">Stage 4</span> Output',
                            unsafe_allow_html=True)
                if result["fields"]:
                    st.markdown("**Extracted fields:**")
                    st.json(result["fields"])
                st.markdown("**Recognised text:**")
                st.text_area("text", result["text"], height=180,
                             label_visibility="collapsed")
                output = pipe.render(result, fmt=output_format)
                st.markdown("**Structured output:**")
                if output_format == "json":
                    st.json(output)
                else:
                    st.code(output, language="text")
                mime = {"json": "application/json", "txt": "text/plain", "csv": "text/csv"}
                st.download_button(f"⬇️ Download .{output_format}", data=output,
                                   file_name=f"ocr_result.{output_format}",
                                   mime=mime[output_format])
        else:
            st.markdown('<div class="warn-box">Upload or generate a document, '
                        'then click <b>Run OCR Pipeline</b>.</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### 📖 Project Overview")
    st.markdown("""
**Automated Document Analysis and OCR System for Multi-Format Documents**

*EE7204 / EC7205 – Image Processing and Computer Vision*
*Department of Electrical and Information Engineering, University of Ruhuna*

| Member | Index |
|--------|-------|
| Arivanan V.  | EG/2021/4414 |
| Arivarasan J.| EG/2021/4415 |
| Bravin K.    | EG/2021/4447 |
| Jegan T.     | EG/2021/4590 |
""")
    st.markdown("### The Pipeline (one path, end to end)")
    for title, items in [
        ("Stage 1 – Preprocessing", [
            "Auto-invert light-on-dark pages",
            "Illumination flattening (rolling-ball) for uneven lighting",
            "Projection-profile deskew",
            "Resolution normalisation (up-scale small text to ~30 px)",
            "Light edge-preserving denoise — grayscale kept (no harsh binarisation)",
        ]),
        ("Stage 2 – Layout Analysis", [
            "Connected-component analysis (character blobs)",
            "Morphological region smearing (fast RLSA equivalent)",
            "Region classification: text / table / image / header-footer",
        ]),
        ("Stage 3 – Recognition", [
            "Tesseract LSTM engine",
            "Adaptive page segmentation: try PSM 4/6/3, keep the most confident",
            "CLAHE retry pass for low-confidence (hard) scans",
        ]),
        ("Stage 4 – Post-Processing & Output", [
            "Safe clean-up (de-hyphenation, spacing, currency) — never rewrites words",
            "Invoice field extraction (number, date, total)",
            "Structured output: JSON / TXT / CSV",
        ]),
    ]:
        with st.expander(title, expanded=False):
            for it in items:
                st.markdown(f"- {it}")
    st.markdown("### Accuracy")
    st.markdown("On labelled invoices (real fonts, scan-like degradation): "
                "**CER ≈ 0.2%**, token-F1 ≈ 98.7%. See `RESULTS.md`.")
