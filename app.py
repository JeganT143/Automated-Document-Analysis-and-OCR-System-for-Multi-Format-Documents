"""
Streamlit UI – Automated Document Analysis and OCR System
Run:  streamlit run app.py
"""

import sys, os, io, time, pickle, tempfile, warnings
import numpy as np
import cv2
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from preprocessing import Preprocessor, NoiseReducer, Binarizer, GeometricCorrector
from layout_analysis import LayoutAnalyzer
from recognition import (CharacterNormalizer, HOGFeatureExtractor,
                          SVMClassifier, KNNClassifier, WordAssembler)
from postprocessing import PostProcessor
from main import OCRPipeline
from tesseract_setup import ensure_tesseract, tesseract_version
from enhanced_ocr import EnhancedOCR
from scripts.test_with_image import generate_test_invoice, TrainedCharRecognizer

# ─── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Document OCR System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1f4e79;
        border-bottom: 3px solid #2e86c1; padding-bottom: 0.4rem; margin-bottom: 1rem;
    }
    .stage-badge {
        display: inline-block; background: #2e86c1; color: white;
        border-radius: 12px; padding: 2px 10px; font-size: 0.78rem;
        font-weight: 600; margin-right: 6px;
    }
    .metric-card {
        background: #eaf4fc; border-left: 4px solid #2e86c1;
        padding: 0.6rem 1rem; border-radius: 6px; margin: 0.3rem 0;
    }
    .success-box {
        background: #eafaf1; border-left: 4px solid #27ae60;
        padding: 0.6rem 1rem; border-radius: 6px;
    }
    .warn-box {
        background: #fef9e7; border-left: 4px solid #f39c12;
        padding: 0.6rem 1rem; border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────

def bgr_to_rgb(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def np_to_pil(img):
    if len(img.shape) == 2:
        return Image.fromarray(img)
    return Image.fromarray(bgr_to_rgb(img))


@st.cache_resource
def load_model(path):
    if not path or not os.path.isfile(path):
        return None
    scaler_path = os.path.join(os.path.dirname(path), 'scaler.pkl')
    return TrainedCharRecognizer(path, scaler_path)


def available_models():
    models_dir = os.path.join(ROOT, 'data', 'models')
    if not os.path.isdir(models_dir):
        return {}
    return {
        os.path.basename(f): os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith('.pkl') and f != 'scaler.pkl'
    }


def draw_components(binary, components, max_draw=400):
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for c in components[:max_draw]:
        x, y, w, h = c['bbox']
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 0), 1)
    return vis


def draw_regions(binary, regions):
    color_map = {
        'header_footer': (255, 140, 0),
        'table':         (0, 160, 255),
        'text':          (30, 180, 30),
        'image':         (180, 0, 220),
        'unknown':       (120, 120, 120),
    }
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for r in regions:
        x, y, w, h = r['bbox']
        c = color_map.get(r['type'], (128, 128, 128))
        cv2.rectangle(vis, (x, y), (x+w, y+h), c[::-1], 2)
        cv2.putText(vis, r['type'][:3], (x+3, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, c[::-1], 1)
    return vis


def draw_recognition(binary, components, char_labels, char_confs):
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for comp, lbl, conf in zip(components, char_labels, char_confs):
        x, y, w, h = comp['bbox']
        green = int(conf * 255)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, green, 255 - green), 1)
        if w > 6 and h > 6:
            cv2.putText(vis, str(lbl), (x, max(y-1, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 0), 1)
    return vis


# ─── Sidebar ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")
    st.markdown("---")

    st.markdown("**Stage 1 – Preprocessing**")
    binarization = st.selectbox(
        "Binarization method",
        ["otsu", "adaptive", "sauvola"],
        help="Otsu: global threshold. Adaptive: local. Sauvola: document-optimised."
    )
    correct_skew = st.checkbox("Correct skew (Hough)", value=True)

    st.markdown("---")
    st.markdown("**Stage 3 – Recognition**")

    _tess_ver = tesseract_version()
    engine = st.selectbox(
        "Recognition engine",
        ["Enhanced (Tesseract)", "Trained model (HOG+SVM/k-NN)", "RLSA word-blocks"],
        help="Enhanced = OCR-tuned preprocessing + adaptive PSM + Tesseract "
             "(best accuracy). Trained model = the custom HOG classifier. "
             "RLSA = layout-only word blocks (no text)."
    )
    if engine == "Enhanced (Tesseract)":
        if _tess_ver:
            st.caption(f"✅ Tesseract {_tess_ver} detected")
        else:
            st.caption("⚠️ Tesseract not found — run scripts/install_tesseract_local.sh")

    models = available_models()
    model_choice = st.selectbox(
        "Classifier model (for trained-model engine)",
        ["None (RLSA word-blocks)"] + list(models.keys()),
        help="Train a model first: python scripts/train_classifier.py --dataset synthetic"
    )
    model_path = models.get(model_choice) if model_choice != "None (RLSA word-blocks)" else None

    st.markdown("---")
    st.markdown("**Stage 4 – Output**")
    output_format = st.selectbox("Output format", ["json", "txt", "csv"])

    st.markdown("---")
    st.markdown("**About**")
    st.info(
        "EE7204 / EC7205\n\n"
        "Dept of EIE, University of Ruhuna\n\n"
        "Pipeline: Preprocess → Layout → Recognise → Post-process"
    )


# ─── Main area ────────────────────────────────────────────────────

st.markdown('<div class="main-header">📄 Automated Document OCR System</div>',
            unsafe_allow_html=True)

tab_run, tab_train, tab_about = st.tabs(["🔍 Run Pipeline", "🏋️ Train Classifier", "📖 About"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 – RUN PIPELINE
# ══════════════════════════════════════════════════════════════════
with tab_run:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📥 Input Document")

        src = st.radio("Image source", ["Upload image", "Generate synthetic invoice"],
                       horizontal=True)

        image_path = None

        if src == "Upload image":
            uploaded = st.file_uploader(
                "Choose a document image",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
            )
            if uploaded:
                suffix = os.path.splitext(uploaded.name)[1] or '.png'
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                tmp.write(uploaded.read())
                tmp.close()
                image_path = tmp.name
                st.image(Image.open(image_path), caption="Uploaded document", use_container_width=True)

        else:
            skew_deg    = st.slider("Skew angle (°)", 0.0, 15.0, 3.0, 0.5)
            noise_sigma = st.slider("Noise sigma",    0,   30,   10,  1)
            if st.button("🎲 Generate invoice"):
                with st.spinner("Generating…"):
                    gen_path = os.path.join(ROOT, 'data', 'test', 'ui_invoice.png')
                    os.makedirs(os.path.dirname(gen_path), exist_ok=True)
                    generate_test_invoice(gen_path, skew_deg=skew_deg,
                                          noise_sigma=noise_sigma)
                    image_path = gen_path
                    st.session_state['gen_image_path'] = gen_path
                    st.image(Image.open(gen_path), caption="Generated invoice",
                             use_container_width=True)

            elif 'gen_image_path' in st.session_state:
                image_path = st.session_state['gen_image_path']
                if os.path.isfile(image_path):
                    st.image(Image.open(image_path), caption="Generated invoice",
                             use_container_width=True)

        run_btn = st.button("🚀 Run OCR Pipeline", type="primary",
                            disabled=(image_path is None))

    # ── Results column ────────────────────────────────────────────
    with col_right:
        st.markdown("### 📊 Results")

        if run_btn and image_path:
            t_total = time.time()
            progress = st.progress(0, text="Starting…")

            try:
                # Stage 1 ─ Preprocessing
                progress.progress(10, "Stage 1 – Preprocessing…")
                prep   = Preprocessor()
                result = prep.process(image_path,
                                      binarization=binarization,
                                      correct_skew=correct_skew)
                binary   = result['binary']
                gray     = result['gray']
                original = result['original']
                t_prep   = time.time()

                # Stage 2 ─ Layout Analysis
                progress.progress(35, "Stage 2 – Layout Analysis…")
                la      = LayoutAnalyzer()
                layout  = la.analyze(binary)
                comps   = layout['components']
                regions = layout['regions']
                t_layout = time.time()

                # Stage 3 ─ Recognition
                progress.progress(60, "Stage 3 – Recognition…")
                words, char_labels, char_confs = [], [], []
                enhanced_rec = None

                if engine == "Enhanced (Tesseract)":
                    if ensure_tesseract() is not None:
                        enhanced_rec = EnhancedOCR().run(image_path)
                        words = enhanced_rec['words']
                    else:
                        st.warning("Tesseract not found — run "
                                   "scripts/install_tesseract_local.sh. "
                                   "Falling back to RLSA word-blocks.")
                        with warnings.catch_warnings(record=True):
                            warnings.simplefilter('always')
                            pl = OCRPipeline(use_tesseract=False)
                        words, _ = pl.recognize_text(binary, gray, layout)
                elif engine == "Trained model (HOG+SVM/k-NN)" and model_path:
                    recognizer   = load_model(model_path)
                    if recognizer and comps:
                        char_results = recognizer.recognize_all(binary, comps)
                        char_labels  = [r[0] for r in char_results]
                        char_confs   = [r[1] for r in char_results]
                        assembler    = WordAssembler()
                        words        = assembler.assemble(comps, char_labels)
                    elif not comps:
                        words = ['(no components found)']
                else:
                    # RLSA word-block fallback
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter('always')
                        pl = OCRPipeline(use_tesseract=False)
                    words, _ = pl.recognize_text(binary, gray, layout)
                t_recog = time.time()

                # Stage 4 ─ Post-processing
                progress.progress(85, "Stage 4 – Post-processing…")
                post   = PostProcessor()
                output = post.format_output(words, regions=regions, fmt=output_format)
                t_post = time.time()

                progress.progress(100, "Done!")
                time.sleep(0.3)
                progress.empty()

                elapsed = time.time() - t_total

                # ── Metrics ───────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Components", len(comps))
                m2.metric("Regions",    len(regions))
                m3.metric("Words",      len(words))
                m4.metric("Time (s)",   f"{elapsed:.2f}")

                # ── Visualisation tabs ────────────────────────────
                v1, v2, v3, v4 = st.tabs(
                    ["Preprocessing", "Layout", "Recognition", "Output"]
                )

                with v1:
                    st.markdown('<span class="stage-badge">Stage 1</span> Preprocessing', unsafe_allow_html=True)
                    nr = NoiseReducer()
                    bz = Binarizer()
                    denoised = nr.bilateral_filter(gray)

                    c1, c2, c3 = st.columns(3)
                    c1.image(np_to_pil(original), caption="Original", use_container_width=True)
                    c2.image(np_to_pil(denoised),  caption="Bilateral denoised", use_container_width=True)
                    c3.image(np_to_pil(binary),    caption=f"Binarized ({binarization})", use_container_width=True)

                    if correct_skew:
                        gc    = GeometricCorrector()
                        angle = gc.detect_skew_angle(binary)
                        st.info(f"Detected skew angle: **{angle:.2f}°**")

                with v2:
                    st.markdown('<span class="stage-badge">Stage 2</span> Layout Analysis', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.image(np_to_pil(layout['smoothed']),
                             caption="RLSA smoothed", use_container_width=True)
                    c2.image(np_to_pil(draw_components(binary, comps)),
                             caption=f"Connected components ({len(comps)})", use_container_width=True)
                    c3.image(np_to_pil(draw_regions(binary, regions)),
                             caption=f"Region classification ({len(regions)})", use_container_width=True)

                    from collections import Counter
                    region_counts = Counter(r['type'] for r in regions)
                    st.markdown("**Region types detected:**")
                    cols = st.columns(len(region_counts) or 1)
                    for col, (rtype, cnt) in zip(cols, region_counts.items()):
                        col.metric(rtype, cnt)

                with v3:
                    st.markdown('<span class="stage-badge">Stage 3</span> Recognition', unsafe_allow_html=True)
                    if enhanced_rec is not None:
                        meta = enhanced_rec
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Words", len(meta['words']))
                        c2.metric("Mean confidence", f"{meta['mean_confidence']:.1f}")
                        c3.metric("PSM used", meta['psm_used'])
                        st.caption(f"Preprocessing: {meta['preprocess_info']}")
                        st.image(np_to_pil(meta['processed_image']),
                                 caption="OCR-tuned image fed to Tesseract "
                                         "(deskewed, resolution-normalised)",
                                 use_container_width=True)
                    elif char_labels:
                        vis_rec = draw_recognition(binary, comps, char_labels, char_confs)
                        st.image(np_to_pil(vis_rec),
                                 caption="Recognised characters (green=high conf, red=low conf)",
                                 use_container_width=True)
                        avg_conf = float(np.mean(char_confs)) if char_confs else 0
                        st.info(f"Average confidence: **{avg_conf:.3f}** over **{len(char_labels)}** characters")

                        # Confidence histogram
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(6, 2))
                        ax.hist(char_confs, bins=20, color='steelblue', edgecolor='white')
                        ax.axvline(0.5, color='red', linestyle='--', label='0.5 threshold')
                        ax.set_xlabel('Confidence'); ax.set_ylabel('Count')
                        ax.set_title('Confidence distribution')
                        ax.legend()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                    else:
                        st.markdown(
                            '<div class="warn-box">No model loaded — '
                            'showing RLSA word-block detection. '
                            'Train a model to see per-character recognition.</div>',
                            unsafe_allow_html=True
                        )
                        from layout_analysis import RLSAProcessor, ConnectedComponentAnalyzer
                        rlsa_p = RLSAProcessor()
                        h_smooth = rlsa_p.horizontal_rlsa(binary, threshold=max(15, binary.shape[1]//20))
                        inv_s = cv2.bitwise_not(h_smooth)
                        cca = ConnectedComponentAnalyzer()
                        wc, _ = cca.find_components(inv_s)
                        wc = cca.filter_components(wc, min_area=50, max_area=binary.shape[0]*binary.shape[1]//4,
                                                    min_aspect=0.2, max_aspect=40)
                        vis_w = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                        for c in wc:
                            x, y, w, h = c['bbox']
                            cv2.rectangle(vis_w, (x,y), (x+w,y+h), (0,180,255)[::-1], 2)
                        st.image(np_to_pil(vis_w),
                                 caption=f"RLSA word blocks ({len(wc)} detected)",
                                 use_container_width=True)

                with v4:
                    st.markdown('<span class="stage-badge">Stage 4</span> Output', unsafe_allow_html=True)

                    st.markdown("**Recognised text:**")
                    st.text_area("", ' '.join(str(w) for w in words), height=120)

                    st.markdown("**Structured output:**")
                    if output_format == 'json':
                        st.json(output)
                    else:
                        st.code(output, language='text')

                    # Download button
                    ext_map  = {'json': 'application/json', 'txt': 'text/plain', 'csv': 'text/csv'}
                    st.download_button(
                        label=f"⬇️ Download .{output_format}",
                        data=output,
                        file_name=f"ocr_result.{output_format}",
                        mime=ext_map[output_format],
                    )

                    # Timing breakdown
                    st.markdown("**Timing breakdown:**")
                    timing = {
                        "Preprocessing":   round(t_prep   - t_total,  3),
                        "Layout Analysis": round(t_layout - t_prep,   3),
                        "Recognition":     round(t_recog  - t_layout, 3),
                        "Post-processing": round(t_post   - t_recog,  3),
                    }
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    ax.barh(list(timing.keys()), list(timing.values()), color='steelblue')
                    ax.set_xlabel('Seconds')
                    ax.set_title('Stage timing')
                    for i, (stage, t) in enumerate(timing.items()):
                        ax.text(t + 0.001, i, f'{t:.3f}s', va='center', fontsize=9)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

            except Exception as e:
                progress.empty()
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())

        else:
            st.markdown(
                '<div class="warn-box">Upload or generate a document image, '
                'then click <b>Run OCR Pipeline</b>.</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════
# TAB 2 – TRAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown("### 🏋️ Train Character Classifier")
    st.markdown(
        "Train an SVM or k-NN classifier on character image data. "
        "The trained model is saved to `data/models/` and auto-detected by the pipeline."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        dataset = st.selectbox(
            "Dataset",
            ["synthetic (offline)", "mnist (online)", "emnist (online)"],
            help="Synthetic: works offline. MNIST/EMNIST: downloads ~10-60 MB."
        )
        classifier_type = st.selectbox("Classifier", ["svm", "knn", "both"])
        n_per_class     = st.slider("Samples per class (synthetic)",
                                    20, 500, 100, 20,
                                    help="More samples = better accuracy but longer training")
        max_samples     = st.number_input("Max total samples (0 = unlimited)",
                                          min_value=0, max_value=200000, value=0, step=1000)

    with col_b:
        st.markdown("**Expected training time:**")
        est_samples = n_per_class * 62
        est_time    = 0.006 * est_samples
        st.markdown(f"""
        | Setting | Value |
        |---------|-------|
        | Dataset | {dataset} |
        | Classes | {'62' if 'synthetic' in dataset else '10 (MNIST)' if 'mnist' in dataset else '47 (EMNIST)'} |
        | Est. samples | {est_samples:,} |
        | Est. SVM train time | ~{est_time:.0f}s |
        """)
        st.info(
            "After training, the model appears in the sidebar dropdown "
            "and is ready to use immediately."
        )

    if st.button("▶ Start Training", type="primary"):
        status_box = st.empty()
        log_box    = st.empty()

        dataset_arg = dataset.split()[0]
        samples_arg = str(max_samples) if max_samples > 0 else None

        cmd_parts = [
            sys.executable,
            os.path.join(ROOT, 'scripts', 'train_classifier.py'),
            '--dataset', dataset_arg,
            '--classifier', classifier_type,
            '--samples-per-class', str(n_per_class),
        ]
        if samples_arg:
            cmd_parts += ['--samples', samples_arg]

        import subprocess
        status_box.info("Training started…")
        with st.spinner("Training in progress (this may take a few minutes)…"):
            proc = subprocess.run(
                cmd_parts,
                capture_output=True, text=True, timeout=600
            )

        if proc.returncode == 0:
            status_box.success("Training complete!")
            log_box.code(proc.stdout[-3000:] if len(proc.stdout) > 3000
                         else proc.stdout)
            st.balloons()
        else:
            status_box.error("Training failed!")
            log_box.code(proc.stderr[-2000:])

    st.markdown("---")
    st.markdown("**Existing models:**")
    models = available_models()
    if models:
        for name, path in models.items():
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                acc  = data.get('accuracy', None)
                acc_str = f"{acc*100:.2f}%" if acc else "unknown"
                size_kb = os.path.getsize(path) // 1024
                st.markdown(
                    f'<div class="metric-card">'
                    f'<b>{name}</b> &nbsp;|&nbsp; accuracy: <b>{acc_str}</b> '
                    f'&nbsp;|&nbsp; type: {data.get("type","?")} '
                    f'&nbsp;|&nbsp; {size_kb} KB'
                    f'</div>',
                    unsafe_allow_html=True
                )
            except Exception:
                st.markdown(f'<div class="metric-card">{name}</div>',
                            unsafe_allow_html=True)
    else:
        st.warning("No models found. Train one above.")


# ══════════════════════════════════════════════════════════════════
# TAB 3 – ABOUT
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

    st.markdown("---")
    st.markdown("### Pipeline Architecture")

    stages = [
        ("Stage 1 – Preprocessing", [
            "Gaussian / Bilateral / Median noise reduction",
            "Otsu / Adaptive Gaussian / Sauvola binarization",
            "Morphological operations (open, close, erode, dilate)",
            "Hough-transform deskewing",
            "Harris corner detection + perspective correction",
        ]),
        ("Stage 2 – Layout Analysis", [
            "Connected Component Analysis (area, centroid, aspect ratio)",
            "Contour detection and filtering",
            "Run-Length Smoothing Algorithm (horizontal + vertical)",
            "Recursive X-Y Cut segmentation",
            "Region classification (text / table / image / header-footer)",
        ]),
        ("Stage 3 – Character Recognition", [
            "Character patch extraction and 32×32 normalisation",
            "HOG feature extraction (1764-dim descriptor)",
            "SVM classifier (RBF kernel, C=10) – 98.78% accuracy",
            "k-NN classifier (k=5) – alternative",
            "Template matching fallback (normalised cross-correlation)",
            "Character-to-word assembly via spacing analysis",
        ]),
        ("Stage 4 – Post-Processing", [
            "Levenshtein-distance spell correction",
            "Rule-based OCR confusion correction (0↔O, 1↔l, rn↔m…)",
            "Output formatting: JSON, plain text, CSV",
        ]),
    ]

    for title, items in stages:
        with st.expander(title, expanded=False):
            for item in items:
                st.markdown(f"- {item}")

    st.markdown("---")
    st.markdown("### Training Data Sources")
    st.markdown("""
| Dataset | Classes | Samples | Source |
|---------|---------|---------|--------|
| **Synthetic** (offline) | 62 | configurable | OpenCV text rendering + augmentation |
| **MNIST** | 10 | 70,000 | `fetch_openml('mnist_784')` |
| **EMNIST Balanced** | 47 | 131,600 | `fetch_openml('EMNIST_Balanced')` |
""")

    st.markdown("---")
    st.markdown("### How to get full text recognition")
    st.code("""# Option 1 – Install Tesseract (best accuracy)
sudo apt install tesseract-ocr
# Then reload this page – pipeline auto-detects it

# Option 2 – Train on EMNIST (when internet available)
python scripts/train_classifier.py --dataset emnist

# Option 3 – Train on synthetic data (always works offline)
python scripts/train_classifier.py --dataset synthetic --samples-per-class 300
""", language="bash")
