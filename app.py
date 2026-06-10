"""
Automated Document Analysis & OCR System — minimal academic UI.

Shows a document moving through the single pipeline, with the real image at
every stage:

    input → grayscale → polarity → flatten → deskew → up-scale → denoise
          → layout regions → recognition → structured output

Run:  streamlit run app.py
"""

import os
import sys
import time

import io
import csv
import json

import numpy as np
import cv2
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from pipeline import DocumentOCRPipeline
from tesseract_setup import tesseract_version

st.set_page_config(page_title="Document OCR Pipeline", page_icon="·",
                   layout="centered", initial_sidebar_state="collapsed")

# ── minimal academic styling ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{ --ink:#1A1B1D; --muted:#646A72; --faint:#9AA0A8; --rule:#E5E3DC;
       --accent:#2B4B7E; --paper:#FBFBF9;
       --sans:'IBM Plex Sans',system-ui,sans-serif;
       --mono:'IBM Plex Mono',monospace; --serif:'Newsreader',Georgia,serif; }

.stApp{ background:var(--paper); color:var(--ink); font-family:var(--sans); }
.block-container{ max-width:820px; padding-top:3.2rem; padding-bottom:4rem; }
#MainMenu, [data-testid="stToolbar"], [data-testid="stHeader"]{ display:none; }
::selection{ background:rgba(43,75,126,.16); }

/* header */
.kicker{ font-family:var(--mono); font-size:.74rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); margin:0 0 .35rem; }
h1.title{ font-family:var(--serif); font-weight:500; font-size:2.3rem; line-height:1.1;
  letter-spacing:-.01em; margin:0; color:var(--ink); }
.lede{ color:var(--muted); font-size:1rem; line-height:1.55; margin:.7rem 0 0; max-width:62ch; }
.hr{ height:1px; background:var(--rule); border:0; margin:1.4rem 0; }

/* section label */
.seclabel{ font-family:var(--mono); font-size:.72rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--faint); margin:2rem 0 .8rem; }

/* stage heading */
.stg{ display:flex; align-items:baseline; gap:.7rem; margin:.2rem 0 .15rem; }
.stg .no{ font-family:var(--mono); font-size:.82rem; color:var(--accent); font-weight:500; }
.stg .nm{ font-family:var(--serif); font-size:1.18rem; font-weight:600; color:var(--ink); }
.stg .mt{ font-family:var(--mono); font-size:.74rem; color:var(--faint); margin-left:auto; }
.ds{ color:var(--muted); font-size:.9rem; margin:0 0 .7rem; }

/* images: plain, on paper, hairline frame */
[data-testid="stImage"]{ border:1px solid var(--rule); border-radius:4px; background:#fff; padding:6px; }
[data-testid="stImage"] img{ border-radius:2px; }
[data-testid="stImageCaption"]{ font-family:var(--mono)!important; font-size:.72rem!important; color:var(--faint)!important; }

/* output */
.kv{ display:flex; gap:1rem; font-family:var(--mono); font-size:.86rem; padding:.45rem 0;
  border-bottom:1px solid var(--rule); }
.kv .k{ color:var(--muted); min-width:120px; text-transform:uppercase; font-size:.74rem; letter-spacing:.05em; }
.kv .v{ color:var(--ink); font-weight:500; }
.ocr{ font-family:var(--mono); font-size:.82rem; line-height:1.6; color:var(--ink); background:#fff;
  border:1px solid var(--rule); border-radius:4px; padding:.9rem 1rem; max-height:300px; overflow:auto; white-space:pre-wrap; }
.summary{ font-family:var(--mono); font-size:.82rem; color:var(--muted); }
.summary b{ color:var(--ink); font-weight:600; }

/* buttons: quiet, academic */
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
@st.cache_resource
def get_pipeline(lang):
    return DocumentOCRPipeline(lang=lang)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit(img, max_w=760):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    return img


def conf_color(c):  # muted, print-friendly
    return (47, 125, 91) if c >= 75 else (184, 134, 11) if c >= 50 else (176, 74, 74)


def draw_regions(gray, regions):
    colors = {"header_footer": (122, 127, 135), "table": (184, 134, 11),
              "text": (43, 75, 126), "image": (125, 91, 166), "unknown": (150, 150, 150)}
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for r in regions:
        x, y, w, h = r["bbox"]
        c = colors.get(r["type"], (150, 150, 150))[::-1]
        cv2.rectangle(vis, (x, y), (x + w, y + h), c, 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def draw_word_boxes(gray, boxes):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        x, y, w, h = b["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), conf_color(b["conf"])[::-1], 1)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def scale_boxes(items, proc, disp):
    s = disp.shape[1] / proc.shape[1]
    return [{**it, "bbox": tuple(int(v * s) for v in it["bbox"])} for it in items]


def export_result(res, fmt):
    """Serialise the result here in app.py (re-run from disk each time, so it
    never uses a stale cached object). Image arrays are dropped."""
    fmt = fmt.lower()
    if fmt == "txt":
        return res.get("text", "")
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["index", "word"])
        for i, word in enumerate(res.get("words", [])):
            w.writerow([i, word])
        return buf.getvalue()

    def _j(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return None
        return str(o)
    clean = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    return json.dumps(clean, indent=2, ensure_ascii=False, default=_j)


def make_sample(level):
    import make_invoice_dataset as mk
    rng = np.random.default_rng(int(time.time() * 7) % 99999)
    img, _ = mk.render_invoice(mk.build_invoice(rng))
    return mk.degrade(img, rng, level=level)


def stage_meta(stage):
    h, w = stage["image"].shape[:2]
    m, k = stage["meta"], stage["key"]
    if k == "invert":
        return "flip applied" if m.get("applied") else "no flip needed"
    if k == "flatten":
        return "illumination flattened" if m.get("applied") else "lighting even — skipped"
    if k == "deskew":
        return f"angle {m.get('angle', 0):+.2f}°"
    if k == "upscale":
        return f"×{m.get('scale', 1):.2f}  →  {w}×{h}px"
    if k == "denoise":
        return "bilateral filter"
    return f"{w}×{h}px"


def section(num, name, meta, desc):
    st.markdown(f'<div class="stg"><span class="no">{num}</span>'
                f'<span class="nm">{name}</span><span class="mt">{meta}</span></div>'
                f'<div class="ds">{desc}</div>', unsafe_allow_html=True)


# ── header ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:var(--mono);font-size:.75rem;color:#646A72;line-height:1.8">'
                '<b>EE7204 / EC7205</b><br>Image Processing &amp; Computer Vision<br>'
                'Dept. of EIE, University of Ruhuna</div>', unsafe_allow_html=True)
    st.divider()
    lang = st.selectbox("OCR language", ["eng"])
    analyze_layout = st.checkbox("Layout analysis", value=True)
    output_format = st.selectbox("Output format", ["json", "txt", "csv"])

st.markdown('<p class="kicker">Automated Document Analysis · OCR</p>', unsafe_allow_html=True)
st.markdown('<h1 class="title">Document OCR Pipeline</h1>', unsafe_allow_html=True)
st.markdown('<p class="lede">A single classical computer-vision and Tesseract pipeline. '
            'Each numbered stage below shows the actual document image as it is transformed, '
            'so every step can be inspected.</p>', unsafe_allow_html=True)
st.markdown('<hr class="hr">', unsafe_allow_html=True)

ver = tesseract_version()
st.markdown(f'<p class="summary">Engine: <b>Tesseract {ver}</b></p>' if ver else
            '<p class="summary" style="color:#b04a4a">Engine: Tesseract not found — '
            'run scripts/install_tesseract_local.sh</p>', unsafe_allow_html=True)


# ── input ────────────────────────────────────────────────────────────────
st.markdown('<p class="seclabel">Source document</p>', unsafe_allow_html=True)
src = st.radio("source", ["Upload image", "Generate sample invoice"],
               horizontal=True, label_visibility="collapsed")
image = None
if src == "Upload image":
    up = st.file_uploader("doc", type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
                          label_visibility="collapsed")
    if up:
        image = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
else:
    lvl = st.select_slider("Degradation", ["clean", "scan", "heavy"], value="scan")
    if st.button("Generate invoice"):
        st.session_state["sample"] = make_sample(lvl)
        st.session_state.pop("trace", None)
    image = st.session_state.get("sample")

if image is not None:
    st.image(to_rgb(fit(image, 320)), caption="loaded source")

run = st.button("Run pipeline", type="primary", disabled=image is None)

if run and image is not None:
    if tesseract_version() is None:
        st.error("Tesseract engine not available.")
        st.stop()
    with st.spinner("Processing…"):
        st.session_state["trace"] = get_pipeline(lang).trace(image, analyze_layout=analyze_layout)
        st.session_state["src_image"] = image

traced = st.session_state.get("trace")


# ── walkthrough ──────────────────────────────────────────────────────────
if traced:
    res = traced["result"]
    src_img = st.session_state.get("src_image", res["original"])
    s = res["mean_confidence"]

    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown(f'<p class="summary"><b>{res["word_count"]}</b> words · '
                f'mean confidence <b>{s:.1f}%</b> · PSM <b>{res["psm_used"]}</b> · '
                f'<b>{len(res["regions"])}</b> regions · '
                f'<b>{len(res["fields"])}</b> fields · '
                f'<b>{res["metrics"]["total_s"]:.2f}s</b></p>', unsafe_allow_html=True)

    st.markdown('<p class="seclabel">Pipeline walkthrough</p>', unsafe_allow_html=True)

    n = 1
    h, w = src_img.shape[:2]
    section(f"{n:02d}", "Input document",
            f"{w}×{h}px · {1 if src_img.ndim == 2 else src_img.shape[2]} ch",
            "The raw source handed to the pipeline.")
    st.image(to_rgb(fit(src_img)))
    n += 1

    for stage in traced["pre_stages"]:
        section(f"{n:02d}", stage["title"], stage_meta(stage), stage["desc"])
        st.image(to_rgb(fit(stage["image"])))
        n += 1

    proc = res["processed_image"]
    disp = fit(proc)

    if res["regions"]:
        from collections import Counter
        counts = Counter(r["type"] for r in res["regions"])
        meta = " · ".join(f"{c} {t}" for t, c in counts.items())
        section(f"{n:02d}", "Layout analysis", meta,
                "Connected components and morphological smearing classify regions.")
        st.image(draw_regions(disp, scale_boxes(res["regions"], proc, disp)))
        n += 1

    boxes = scale_boxes(res["word_boxes"], proc, disp)
    section(f"{n:02d}", "Recognition",
            f"{res['word_count']} words · mean conf {s:.1f}%",
            "Adaptive-PSM Tesseract. Boxes: green ≥75%, amber ≥50%, red below.")
    st.image(draw_word_boxes(disp, boxes))
    n += 1

    section(f"{n:02d}", "Structured output", output_format.upper(),
            "Safe clean-up, field extraction and serialisation.")
    for k, v in res["fields"].items():
        st.markdown(f'<div class="kv"><span class="k">{k.replace("_", " ")}</span>'
                    f'<span class="v">{v}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="ds" style="margin-top:.8rem">Recognised text</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ocr">{res["text"] or "—"}</div>', unsafe_allow_html=True)
    out = export_result(res, output_format)
    mime = {"json": "application/json", "txt": "text/plain", "csv": "text/csv"}
    st.download_button(f"Download .{output_format}", out,
                       file_name=f"ocr_result.{output_format}", mime=mime[output_format])
else:
    st.caption("Load a document and press Run pipeline to trace it through every stage.")

st.markdown('<div class="foot">Automated Document Analysis &amp; OCR · '
            'University of Ruhuna · EE7204 / EC7205</div>', unsafe_allow_html=True)
