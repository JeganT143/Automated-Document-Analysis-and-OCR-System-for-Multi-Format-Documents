# Automated Document Analysis and OCR System for Multi-Format Documents

A single, end-to-end pipeline that turns a document image (invoice, receipt,
form…) into clean, structured, machine-readable text.

*EE7204 / EC7205 – Image Processing and Computer Vision*
*Department of Electrical & Information Engineering, University of Ruhuna*

```
image ─▶ preprocess ─▶ layout analysis ─▶ recognise ─▶ post-process ─▶ JSON / TXT / CSV
        (classical CV)   (classical CV)   (Tesseract,    (safe clean-up
                                          adaptive PSM)   + field extraction)
```

There is **one** recognition pipeline (`src/pipeline.py`). The earlier toy
paths (HOG+SVM / k-NN / template-matching / `[word]` placeholder fallback) have
been removed.

---

## The pipeline

| Stage | Module | What it does |
|-------|--------|--------------|
| 1. Preprocess | `src/pipeline.py` (`OCRPreprocessor`) | auto-invert dark pages · illumination flattening · projection-profile deskew · **resolution normalisation** (up-scale small text) · light denoise. Keeps a clean **grayscale** image — no destructive binarisation. |
| 2. Layout analysis | `src/layout_analysis.py` | connected components · fast morphological region smearing · region classification (text / table / image / header-footer). |
| 3. Recognise | `src/pipeline.py` (`DocumentOCRPipeline`) | Tesseract LSTM with **adaptive page segmentation** (tries PSM 4/6/3, keeps the most confident) and a CLAHE retry for hard scans. |
| 4. Post-process | `src/postprocessing.py` | **safe** clean-up (de-hyphenation, spacing, currency) that never rewrites words · invoice field extraction · JSON / TXT / CSV. |

Supporting modules: `src/tesseract_setup.py` (finds a system *or* user-local
Tesseract) and `src/evaluation.py` (CER / WER / token-F1 / field metrics).

## Why accuracy was low before — and what fixed it

A measured root-cause analysis (`scripts/evaluate.py --baseline`) found:

1. **No text at all.** Without Tesseract installed, the old default emitted
   `[word]` placeholders → **0 %** of the text recovered (CER ≈ 90 %).
2. **Preprocessing hurt OCR.** Bilateral + Otsu + morphology was *worse* than
   raw grayscale — Tesseract has its own, better binariser.
3. **Post-processing corrupted text.** A ~100-word spell-corrector mapped valid
   words to nonsense (`Kandy → And`). It is now removed; clean-up is safe-only.

The single pipeline keeps only the preprocessing that *helps* OCR, adds
resolution normalisation + adaptive PSM, and structures the output.

## Measured results

Labelled synthetic invoices (rendered with real fonts, scan-like noise + skew —
exact ground truth). `python scripts/evaluate.py --baseline`:

| Configuration | CER ↓ | Char acc. | WER ↓ | Token-F1 ↑ |
|---|--:|--:|--:|--:|
| Raw Tesseract (`--psm 6`) | 6.3 % | 93.7 % | 13.8 % | 92.8 % |
| **This pipeline** | **0.2 %** | **99.8 %** | **1.7 %** | **98.8 %** |

It was also run on **30 real document images** downloaded from Wikimedia
Commons — see [`RESULTS.md`](RESULTS.md) for the full per-image table.

## Quick start

```bash
# 1. environment
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt

# 2. Tesseract engine (pick one)
sudo apt install tesseract-ocr            # normal machines
bash scripts/install_tesseract_local.sh   # NO root: unpacks to ~/.local/opt/tesseract

# 3. run on an image
python src/main.py path/to/invoice.png --format txt

# 4. web UI
streamlit run app.py
```

## Data & evaluation

```bash
python scripts/make_invoice_dataset.py --n 12 --degrade scan   # labelled set (exact GT)
python scripts/download_invoices.py --n 30                     # 30 real images (no GT)
python scripts/evaluate.py --baseline --report RESULTS.md      # full evaluation + report
```

Datasets are git-ignored (regenerable with the scripts above).
