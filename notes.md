# Project Notes — Automated Document Analysis & OCR System

> **Your complete guide to understanding and presenting this project.**
> Read top to bottom once, then use the [Cheat Sheet](#16-cheat-sheet--memorize-these) and
> [Q&A prep](#14-qa-preparation--likely-questions-and-strong-answers) right before the presentation.
>
> *EE7204 / EC7205 — Image Processing and Computer Vision, Dept. of EIE, University of Ruhuna*

---

## Table of Contents

1. [The 30-Second Elevator Pitch](#1-the-30-second-elevator-pitch)
2. [Explain It Like I'm 5 — The Story](#2-explain-it-like-im-5--the-story)
3. [The Problem We Solve](#3-the-problem-we-solve)
4. [The Big Picture — One Pipeline](#4-the-big-picture--one-pipeline)
5. [Stage 0: Finding the Engine](#5-stage-0-finding-the-engine-srctesseract_setuppy)
6. [Stage 1: Preprocessing](#6-stage-1-preprocessing--cleaning-the-page-srcpipelinepy--ocrpreprocessor)
7. [Stage 2: Layout Analysis](#7-stage-2-layout-analysis--understanding-the-page-srclayout_analysispy)
8. [Stage 3: Recognition](#8-stage-3-recognition--reading-the-words-srcpipelinepy--documentocrpipeline)
9. [Stage 4: Post-Processing & Output](#9-stage-4-post-processing--structured-output-srcpostprocessingpy)
10. [The Detective Story — Why Accuracy Was Terrible Before](#10-the-detective-story--why-accuracy-was-terrible-before)
11. [How We Prove It Works — Evaluation](#11-how-we-prove-it-works--evaluation)
12. [The Results — and How to Say Them Out Loud](#12-the-results--and-how-to-say-them-out-loud)
13. [The Demo — App, CLI, and a Demo Script](#13-the-demo--app-cli-and-a-demo-script)
14. [Q&A Preparation — Likely Questions and Strong Answers](#14-qa-preparation--likely-questions-and-strong-answers)
15. [Repository Map & How to Run Everything](#15-repository-map--how-to-run-everything)
16. [Cheat Sheet — Memorize These](#16-cheat-sheet--memorize-these)
17. [Glossary](#17-glossary)

---

## 1. The 30-Second Elevator Pitch

> "We built a system that takes a **photo or scan of a document** — an invoice, a receipt, a form —
> and turns it into **clean, structured, machine-readable text**: the full text, plus key fields like
> the invoice number, date, and total, exported as JSON, CSV, or plain text.
>
> It is **one end-to-end pipeline**: classical computer-vision preprocessing → layout analysis →
> the Tesseract OCR engine driven adaptively → safe post-processing.
>
> On labelled test invoices, plain Tesseract makes a **6.3 % character error rate**. Our pipeline
> brings that down to **0.19 %** — that's **99.8 % character accuracy**, roughly **33× fewer errors**.
> And every claim is backed by a measured, reproducible evaluation."

---

## 2. Explain It Like I'm 5 — The Story

Imagine your grandmother hands you a **crumpled, slightly tilted, badly photocopied shopping bill**
and asks: *"How much did I spend, and when?"*

What do you do? You naturally do four things, without thinking:

1. **You smooth out the paper and tilt it straight** so you can see it properly.
   *(That's our **preprocessing** stage.)*
2. **You glance at the page to see where things are** — "ah, the shop name is at the top, the
   item list is in this table, the total is at the bottom."
   *(That's **layout analysis**.)*
3. **You actually read the letters and numbers.**
   *(That's **recognition** — the OCR engine.)*
4. **You write the answer down neatly**: "Total: $113.05, dated 12 Mar 2026."
   *(That's **post-processing and field extraction**.)*

Our project teaches a computer to do those exact four steps, in that exact order — and to do it
in about a second per page, thousands of times in a row, without getting tired or making typos.

**The single most important idea of the whole project:**
A computer reading a document is like a person reading one — **the cleaner and straighter the page,
the better the reading.** Almost all of our accuracy gain comes not from a fancier "reader," but
from *handing the reader a better page*. We use the same reading engine (Tesseract) as the
baseline; we just prepare its input intelligently and check its work — and errors drop 33-fold.

---

## 3. The Problem We Solve

### Real-life motivation

- An accounting office receives **thousands of invoices** per month. A human typing each one into
  the system takes minutes per document and makes mistakes ("$1,150.00" typed as "$1,510.00" is a
  real, expensive error).
- Hospitals, banks, government offices, logistics companies — all drown in scanned paper.
- This problem domain is called **document digitization / automated data entry**, and OCR
  (Optical Character Recognition) is its core technology.

### Why it's genuinely hard (this is the engineering motivation)

Real scans and phone photos are messy in ways humans barely notice but machines stumble on:

| Real-world mess | Everyday analogy | What it does to OCR |
|---|---|---|
| **Skew** (page tilted 1–3°) | A picture frame hung crooked | Text lines cut across pixel rows; characters get misshaped |
| **Uneven lighting / shadows** | A lamp shining on one side of the page | Dark regions get read as ink, or faint text disappears |
| **Low resolution** (small text) | Reading a sign from too far away | Letters are only a few pixels tall — too little detail to recognise |
| **Noise / grain** | Dust and specks on a photocopier glass | Specks get misread as punctuation; broken strokes split letters |
| **Inverted polarity** | A film negative | Engines expect dark ink on light paper, not the reverse |
| **Complex layout** (columns, tables, headers) | A newspaper page vs. a novel page | Reading order goes wrong; table cells get jumbled together |

**Goal:** automatically produce *structured* text (not just a blob of characters) from such images,
robustly, and **prove with measurements** that it works.

---

## 4. The Big Picture — One Pipeline

```
                ┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐   ┌─────────────────┐
  document      │ 1 PREPROCESS │   │ 2 LAYOUT ANALYSIS│   │ 3 RECOGNISE     │   │ 4 POST-PROCESS  │    JSON
  image  ───────▶ clean &      ├───▶ find text/table/ ├───▶ Tesseract LSTM, ├───▶ safe clean-up + ├──▶ CSV
  (any format)  │ straighten   │   │ header regions   │   │ adaptive PSM    │   │ field extraction│    TXT
                └──────────────┘   └──────────────────┘   └─────────────────┘   └─────────────────┘
                 classical CV        classical CV           OCR engine           rules + regex
```

**Assembly-line analogy:** think of a car factory. One conveyor belt, four stations, each station
does one job well and passes the result on. There is **exactly one belt** — early in the project
there were several competing experimental "readers" (HOG+SVM, k-NN, template matching); they were
**deleted**, because a factory with five half-working assembly lines produces nothing reliable.
Everything now lives in one pipeline: [src/pipeline.py](src/pipeline.py).

### The public API (engineering view)

The whole system is one class, `DocumentOCRPipeline`, with three entry points:

| Method | Returns | Used by |
|---|---|---|
| `image_to_text(img)` | just the text string (fast path) | evaluation harness |
| `run(img)` | full result dict: text, word boxes + confidences, fields, regions, timings | CLI ([src/main.py](src/main.py)) |
| `trace(img)` | everything `run` gives **plus the actual image at every intermediate stage** | the Streamlit demo app ([app.py](app.py)) |

`trace()` is what makes the demo special: the audience sees the *real document* transform step by
step — not a diagram, the actual pixels.

---

## 5. Stage 0: Finding the Engine ([src/tesseract_setup.py](src/tesseract_setup.py))

**ELI5:** our system needs a "reading brain" called **Tesseract** (a famous open-source OCR engine,
originally from HP, later maintained by Google). Tesseract is not a Python library — it's a
*program installed on the computer*, like a printer driver. Before reading anything, we must find it.

**The problem we solved:** on university lab machines you often **don't have admin (root) rights**,
so you can't run `sudo apt install tesseract-ocr`. Our solution:

1. First look for a normal system install on the `PATH`.
2. If absent, fall back to a **user-local copy** unpacked under `~/.local/opt/tesseract` by
   [scripts/install_tesseract_local.sh](scripts/install_tesseract_local.sh) — it downloads the
   Debian packages and extracts them into your home folder, **no root needed**.

**Engineering details worth mentioning:** for the local copy, the module wires up three things —
`pytesseract.tesseract_cmd` (where the binary is), `TESSDATA_PREFIX` (where the trained language
models live), and `LD_LIBRARY_PATH` (so the binary can find its bundled `libtesseract` /
`libleptonica` shared libraries at run time). `ensure_tesseract()` is idempotent — safe to call
repeatedly.

**Analogy:** the app "brings its own engine" if the building doesn't have one installed —
like carrying a portable generator in case the venue has no power socket.

---

## 6. Stage 1: Preprocessing — Cleaning the Page ([src/pipeline.py](src/pipeline.py) → `OCRPreprocessor`)

This is the heart of the accuracy gain. Six steps, each **conditional** — applied only when the
image actually needs it (the code *measures* first, then acts).

### 6.1 Grayscale

- **ELI5:** make a black-and-white photocopy. Colour doesn't help reading; only "how dark is each
  spot" matters.
- **Engineering:** `cv2.cvtColor(BGR2GRAY)` — collapse 3 colour channels into 1 luminance channel.
  Every later step works on this single channel (3× less data, simpler math).

### 6.2 Polarity normalisation (auto-invert)

- **ELI5:** some receipts are like film negatives — white letters on a dark background. A human
  flips a negative before reading; so do we. The engine expects **dark ink on light paper**.
- **Engineering:** sample an 8-pixel border strip around the image; if the **median** border
  brightness is below 110 (out of 255), the background is dark → invert with `bitwise_not`.
  Why the *border*? The page background dominates the edges; the median makes it robust to a few
  dark decorations.

### 6.3 Illumination flattening (the "rolling ball")

- **ELI5:** imagine a lamp shining on one corner of the page, leaving the other corner in shadow.
  We estimate "what would this page look like with **no text on it**, just the lighting" — then
  divide the image by that estimate. Shadow cancels out; only ink remains. Like ironing the
  lighting flat.
- **Engineering:**
  - *Detect first:* shrink the page to 64×64, blur heavily (Gaussian σ=8) so only the lighting
    pattern survives; if `max − min > 55` grey levels, lighting is uneven → fix it. Otherwise skip
    (don't touch what isn't broken).
  - *Fix:* morphological **closing** with a large ellipse kernel (size ≈ shorter side ÷ 20 — big
    enough that text "falls into" the kernel and disappears, leaving only background), then a
    Gaussian blur, then `cv2.divide(gray, background, scale=255)`. This is the classic
    **rolling-ball background subtraction**, done multiplicatively.

### 6.4 Deskew (projection-profile method) ⭐

- **ELI5:** the page is hung crooked; straighten the frame. How do we *know* the angle? Imagine
  squashing all the ink sideways into a bar chart, one bar per pixel row. If the text lines are
  perfectly horizontal, the chart looks like a comb: tall spikes (text rows) and deep gaps (white
  space between lines). If the page is tilted, everything smears together and the comb flattens.
  So: **rotate by trial, and keep the angle that makes the comb spikiest.**
- **Engineering:**
  - Otsu-threshold the page (text = white), downscale to ≤600×800 for speed.
  - Score for an angle = `Σ (diff of row-projection)²` — a standard sharpness measure of the
    horizontal projection profile.
  - **Coarse-to-fine search:** try −8°…+8° in 1° steps, then refine ±1° around the winner in 0.2°
    steps. (17 + 11 trials instead of hundreds — cheap and accurate to ~0.2°.)
  - Apply rotation with `INTER_CUBIC` and `BORDER_REPLICATE`; skip entirely if |angle| < 0.2°.
  - **Why not Hough lines?** Projection profiles use *all* the text ink, so they are more robust on
    documents than edge-based Hough, which gets distracted by table borders and images.

### 6.5 Resolution normalisation (upscaling small text) ⭐⭐ — the single biggest win

- **ELI5:** Goldilocks. Tesseract reads best when letters are a certain size — roughly **30 pixels
  tall** for a lowercase letter. Tiny text is like reading a street sign from too far away. So we
  *measure* how big the letters actually are, and zoom the whole page so they land in the
  sweet spot.
- **Engineering:**
  - Measure letter height: Otsu-threshold, find **connected components** (ink blobs), keep
    plausibly-letter-sized ones (height 6 px … 10 % of page, width 2 px … 10 % of page, area ≥ 8),
    take the **median** height. Require ≥ 10 blobs, else skip (not enough evidence).
  - Scale factor = `clip(30 / median_height, 1.0, 4.0)` — never downscale, never blow up more than
    4×; also cap the longest side at 3500 px (don't create monster images).
  - Upscale with `INTER_CUBIC` (smooth interpolation that keeps stroke edges clean).
- **Why this matters so much:** Tesseract's LSTM was trained on text of a particular x-height range.
  Below ~20 px it falls off a cliff. Most "bad scans" are really just *small-text scans*.

### 6.6 Light denoising

- **ELI5:** wipe the dust specks off the photocopy — but with a careful eraser that never smudges
  the ink itself.
- **Engineering:** `cv2.bilateralFilter(gray, 5, 35, 35)` — an **edge-preserving** smoother. It
  averages each pixel with neighbours that are *similar in brightness*, so flat areas get cleaned
  while sharp ink-paper edges stay sharp. (A plain Gaussian blur would soften letter edges and
  *hurt* recognition.)

### 6.7 Optional: CLAHE contrast boost (only as a retry)

- **ELI5:** if the engine later says "I'm not confident" (the page is faint), we turn on a brighter
  reading lamp and try once more.
- **Engineering:** CLAHE = Contrast-Limited Adaptive Histogram Equalisation (`clipLimit=2.0`,
  8×8 tiles) — boosts *local* contrast without blowing out the page. Deliberately **not** applied
  by default, because on clean pages it amplifies paper texture into noise.

### ⚠️ The most important *negative* design decision

> **We do NOT binarise (threshold to pure black/white) before OCR. We hand Tesseract a clean
> *grayscale* image.**

We measured it: bilateral + Otsu + morphology *before* OCR was **worse** than raw grayscale.
Reason: Tesseract has its own, very good internal binariser, and it benefits from the grayscale
anti-aliasing information on stroke edges. Pre-binarising destroys that information irreversibly —
like photocopying a photocopy. **Good engineering is also knowing what NOT to do** — this is a
great line for the presentation, because it shows the design was driven by measurement, not habit.

---

## 7. Stage 2: Layout Analysis — Understanding the Page ([src/layout_analysis.py](src/layout_analysis.py))

**ELI5:** before reading word-by-word, glance at the page like a human: "this strip at the top is a
header, this grid in the middle is a table, this block is normal text, that rectangle is a photo."
We draw a labelled box around each zone.

### How it works, step by step

1. **Binarise (Otsu)** — for layout geometry (not for OCR!), a clean ink/paper mask is what we need.
2. **Connected components** — find every separate "island of ink" (roughly = characters). Filter to
   plausibly-character-sized blobs for the visualisation.
3. **Morphological smearing** — the clever bit:
   - **ELI5:** imagine the ink is still wet, and you smear it **sideways** — letters in the same
     line bleed into one solid stripe. Then dab **downwards** a little — neighbouring stripes merge
     into paragraph blocks. Now each blob *is* a region.
   - **Engineering:** horizontal **closing** with a `(kx, 1)` kernel where `kx = max(10, width/40)`
     merges characters into lines; vertical **dilation** with `(1, ky)`, `ky = max(3, height/250)`
     merges lines into blocks. This is the classical **RLSA** (Run-Length Smoothing Algorithm) —
     but implemented with OpenCV morphology instead of pure-Python loops, which took it from
     *seconds* per page to *milliseconds*. (A real optimisation story: same algorithmic idea,
     1000× faster implementation.)
4. **Region classification** — simple, explainable rules:

   | Rule (in priority order) | Label | Intuition |
   |---|---|---|
   | top/bottom 12 % of the page | `header_footer` | where letterheads & page numbers live |
   | ≥ 3 horizontal ruled lines AND wide (aspect > 1.5) | `table` | tables have row separators |
   | ink density < 2 % | `image` | photos/diagrams binarise to sparse speckle |
   | otherwise | `text` | default |

   ("Horizontal ruled line" = a pixel row where > 60 % of the width is ink.)
5. Regions are sorted top-to-bottom, left-to-right (reading order).

**Honest scoping (good to say if asked):** layout analysis here is *descriptive metadata* — it
feeds the structured output and the demo visualisation. Recognition itself is done by Tesseract,
whose internal page segmentation we steer in Stage 3. We don't OCR region-by-region because
Tesseract's own segmentation, properly driven, was measurably sufficient.

---

## 8. Stage 3: Recognition — Reading the Words ([src/pipeline.py](src/pipeline.py) → `DocumentOCRPipeline`)

### The engine

**Tesseract 5** in **LSTM mode** (`--oem 3`): a recurrent neural network that reads a text line as
a sequence — like how you read, character by character, with context. We also set
`preserve_interword_spaces=1` so column spacing survives into the text.

### The smart part: adaptive page segmentation (PSM) ⭐

- **ELI5:** Tesseract needs to be told *what kind of page to expect* — "one clean paragraph"? "a
  page with columns"? "scattered bits of text"? That hint is called the **PSM** (Page Segmentation
  Mode), and the right answer differs per document. Instead of guessing once, we let the engine
  try **three pairs of glasses** and keep the pair it sees best with:
  - **PSM 4** — single column, variable text sizes (great for invoices)
  - **PSM 6** — one uniform block of text
  - **PSM 3** — fully automatic segmentation
- **Engineering — how "best" is judged:**
  - For each PSM we call `image_to_data`, which returns every word with a **confidence (0–100)**
    and bounding box.
  - Score = `mean_confidence × log(1 + word_count)`.
    **Why the log term?** A result that says just "INVOICE" with 99 % confidence must not beat a
    result that read all 120 words at 92 %. The log rewards *amount read* without letting word
    count dominate. *(Analogy: a student who answers one question perfectly shouldn't outrank a
    student who answered the whole exam well.)*
  - **Fast path:** if the first PSM already scores ≥ 85 mean confidence, stop — no need to try the
    rest. Keeps easy pages fast.
- **Confidence-driven retry:** if the best result is still < 60 mean confidence, re-run the whole
  preprocess with **CLAHE** contrast boost and keep whichever result scores higher. The system
  notices it's struggling and adapts — a small feedback loop.

### What recognition outputs

Per word: text, confidence, bounding box `(x, y, w, h)`. Lines are reconstructed from Tesseract's
`(block, paragraph, line)` indices. The word boxes + confidences power the demo's colour-coded
overlay (green ≥ 75 %, amber ≥ 50 %, red below).

---

## 9. Stage 4: Post-Processing & Structured Output ([src/postprocessing.py](src/postprocessing.py))

### Safe clean-up — `SafePostProcessor`

**ELI5:** a proofreader who is only allowed to fix *formatting*, never to "correct" your words.

Only three fixes, each 100 % safe:
1. **De-hyphenation:** `"inv-\noice"` → `"invoice"` (words split across a line break).
2. **Whitespace:** collapse runs of spaces/tabs, drop empty lines, keep line structure.
3. **Currency spacing:** `"$ 12.50"` → `"$12.50"`.

> **The cautionary tale (tell this in the presentation):** an earlier version had a ~100-word
> dictionary spell-corrector. It turned the *correct* Sri Lankan city name **"Kandy" into "And"**
> and mangled currency codes like "LKR". It actively **lowered** accuracy, so it was **removed
> entirely** — not just disabled. Lesson: a post-processor must never rewrite numbers or proper
> nouns it doesn't understand. *(Analogy: an overzealous autocorrect that "fixes" your friend's
> name into a dictionary word.)*

### Field extraction — `extract_fields`

Best-effort regex extraction of the three fields that matter on an invoice:

| Field | Pattern (intuition) | Subtlety handled |
|---|---|---|
| `invoice_no` | `INV-2026-1234` style | tolerates lost/extra spaces & hyphens from OCR |
| `total` | an amount following the word "total" | **must not match "subtotal"** (negative lookbehind on letters); takes the **last** match because the grand total appears last on an invoice |
| `date` | `12 Mar 2026` style | normalises to zero-padded day + title-case month |

### Output formats — `OutputFormatter`

- **JSON** — the full structured result (text, words, confidences, fields, regions, timings);
  numpy types converted safely, image arrays stripped.
- **TXT** — just the recognised text.
- **CSV** — one row per word.

**ELI5:** the same answer written three ways — a full report (JSON), a quick note (TXT), and a
spreadsheet (CSV) — so any downstream system can consume it.

---

## 10. The Detective Story — Why Accuracy Was Terrible Before

This is your strongest narrative. The project didn't start great — it was **debugged with
measurements**, like a detective following evidence. Three culprits were found
(via `scripts/evaluate.py --baseline`):

| # | Culprit | What was happening | The fix |
|---|---|---|---|
| 1 | **No text at all** | Without Tesseract installed, an old fallback path emitted literal `[word]` placeholders — the system *looked* like it ran but recovered **0 %** of the text (CER ≈ 90 %) | One pipeline, fails loudly if the engine is missing, plus the **no-root installer** so the engine is always available |
| 2 | **Preprocessing was hurting** | Bilateral + Otsu binarisation + morphology before OCR was *worse* than feeding the raw grayscale | Keep only steps that measurably help; **never binarise before OCR** |
| 3 | **Post-processing corrupted text** | The toy spell-corrector mapped valid words to nonsense (Kandy → And) | Deleted; replaced with **safe-only** clean-up |

**The meta-lesson (say this!):** every fix came from the **evaluation harness**, not from intuition.
We could only fix what we could measure. That is the engineering method: *measure → diagnose →
fix → re-measure*.

---

## 11. How We Prove It Works — Evaluation

### The grading problem

**ELI5:** to grade a student's dictation, you need the **answer key**. For OCR, the answer key is
called **ground truth** — the exact text that *is* on the page. For random internet images, nobody
knows the exact ground truth. So we use two complementary test sets:

### Test set A — labelled synthetic invoices (rigorous numbers)

Built by [scripts/make_invoice_dataset.py](scripts/make_invoice_dataset.py):

- Invoices are **rendered from structured data** with **real TrueType fonts** (Liberation Sans,
  DejaVu Sans/Serif, Ubuntu) at A4-like size (794×1123 px) — realistic companies, line items,
  taxes, currencies. Because *we* wrote the text, we know the ground truth **exactly**.
- Then we **degrade them like real scans**: Gaussian blur (σ=0.6), additive Gaussian noise
  (σ=6 for "scan", σ=14 for "heavy"), random skew (±2° / ±4°), and for "heavy" an illumination
  gradient across the page. So the test is honest — the pipeline must undo realistic damage.
- Each sample = three files: the PNG image, the ground-truth text (`.gt.txt`, in visual
  **reading order** so multi-column headers compare fairly), and the structured fields (`.json`).

**Why synthetic ground truth is legitimate (defend this in Q&A):** the *degradations* are what
make OCR hard, and those are applied realistically. Hand-labelling real scans at character level
is error-prone — your "ground truth" itself would have typos. Synthetic rendering gives an exact,
reproducible answer key. The real-image batch (set B) then guards against the synthetic set being
too easy.

### Test set B — 30 real documents (robustness check)

Downloaded from **Wikimedia Commons** by [scripts/download_invoices.py](scripts/download_invoices.py)
(with licences recorded in `ATTRIBUTION.txt`). Deliberately **uncurated** — they include a 1902
German letterhead, Japanese postal forms, Arabic and Cyrillic invoices, even **3,500-year-old
Egyptian coffin-making receipts**. No ground truth exists, so we report Tesseract's own
**confidence** as a quality proxy, plus word counts, fields found, and speed.

### The metrics ([src/evaluation.py](src/evaluation.py))

| Metric | ELI5 | Engineering definition |
|---|---|---|
| **CER** (character error rate) | "Out of 100 letters, how many did you get wrong?" — like a typing-test score | Levenshtein edit distance between reference and hypothesis at character level (whitespace-normalised, case-folded, spaces removed) ÷ reference length |
| **WER** (word error rate) | Same, but counting whole words | Levenshtein over word tokens |
| **Token-F1** | "Did you find the right words *somewhere*, even if in a different order?" | Multiset precision/recall/F1 over normalised tokens — **order-free**, which matters because multi-column pages have ambiguous reading order |
| **Field accuracy** | "Did you get the 3 facts that matter — invoice no., total, date?" | Exact match of extracted fields vs. structured ground truth |

The Levenshtein distance is implemented from scratch (dynamic programming, two-row memory
optimisation) — worth mentioning that no metric library was used.

### The baseline

`raw Tesseract (--psm 6)` on the unmodified grayscale image = "what you get out of the box."
Same engine as ours — so the comparison isolates exactly **the value our pipeline adds**.

---

## 12. The Results — and How to Say Them Out Loud

### Labelled set (12 synthetic invoices, "scan" degradation) — from [RESULTS.md](RESULTS.md)

| Configuration | CER ↓ | Char acc. ↑ | WER ↓ | Token-F1 ↑ | inv_no | total | date | s/doc |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Raw Tesseract (`--psm 6`) | 6.33 % | 93.67 % | 13.79 % | 92.75 % | 100 % | 92 % | 100 % | 0.57 |
| **Our pipeline** | **0.19 %** | **99.81 %** | **1.67 %** | **98.79 %** | 100 % | 92 % | 100 % | 1.42 |

**How to say it:** *"Plain Tesseract gets about six errors per hundred characters. Our pipeline
gets about one error per five hundred characters — a thirty-three-fold reduction, achieved with
the same recognition engine, purely by preparing its input well and steering it adaptively."*

**Be ready for the speed question:** yes, 1.42 s vs 0.57 s per document — about 2.5× slower.
That is the cost of deskew, upscaling (more pixels to OCR), and trying multiple PSMs. For document
archiving, trading ~0.9 s for 33× fewer errors is overwhelmingly worth it; the early-accept fast
path already keeps easy pages quick.

### Real-image batch (30 uncurated Wikimedia documents)

- **30/30 processed without crashing** (robustness), mean confidence 54.3 %.
- **Confidence tracks document type** — and that's the insight, not a weakness:
  - Machine-printed, Latin-script documents (the target domain): **75–93 %** confidence
    (`Invoice1633` 92.7 %, `Facturas` 89.3 %, `Toonimo-bill-pay` 85.1 %).
  - Handwritten manuscripts (the Egyptian coffin receipts, a Victorian letter) and non-Latin
    scripts (Japanese, Arabic, Greek, Cyrillic): low — **expected**, because we run the English
    *print* model. Out of scope, declared honestly.

**How to frame it:** *"The rigorous accuracy number is the labelled 0.19 % CER. The batch test
shows the system never crashes on arbitrary real documents and tells us — through its own
confidence — exactly where its domain boundary lies."* Knowing your system's limits **and being
able to detect them automatically** (low confidence ⇒ flag for human review) is itself an
engineering feature.

---

## 13. The Demo — App, CLI, and a Demo Script

### The Streamlit app ([app.py](app.py))

`streamlit run app.py` — a clean academic UI that runs `pipeline.trace()` and shows **the actual
document image at every stage**:

input → grayscale → polarity → flatten → deskew → upscale → denoise → layout regions
→ word-level recognition overlay → structured output.

Features to show off:
- **Built-in sample generator** — no test file needed; it renders a fresh synthetic invoice on the
  spot with a chosen degradation level (clean / scan / heavy).
- Each stage shows its **metadata**: deskew shows the detected angle, upscale shows the zoom
  factor and new size, flatten says whether it was needed or skipped.
- Recognition overlay: word boxes coloured by confidence (green ≥ 75 %, amber ≥ 50 %, red below).
- Extracted fields (invoice no / total / date) + downloadable JSON / CSV / TXT.

### Suggested 90-second live-demo script

1. *"Let me generate a degraded sample invoice — note it's blurry, noisy, and tilted."*
   (Pick **scan** or **heavy**, click Generate, point at the tilt.)
2. Click **Run pipeline**. *"Watch the document clean itself up stage by stage."*
3. Pause on **Deskew** — *"it measured a tilt of X degrees and corrected it"* — and on
   **Resolution normalise** — *"it measured the letter size and zoomed to the engine's sweet spot."*
4. Pause on **Recognition** — *"every word box is green: the engine is above 75 % confidence
   nearly everywhere."*
5. Finish on **Structured output** — *"and here are the extracted invoice number, date and total,
   downloadable as JSON."*

**Demo safety net:** if live demos scare you, pre-record a screen capture, or keep screenshots of
each stage as backup slides. Also run the app once *before* the talk so the Streamlit cache and
Tesseract are warm.

### The CLI ([src/main.py](src/main.py))

```bash
python src/main.py path/to/invoice.png --format txt          # to stdout
python src/main.py invoice.jpg --format json --output r.json # to file
python src/main.py invoice.jpg --no-layout                   # skip region analysis (faster)
```

Prints a summary (word count, mean confidence, PSM used, regions, fields, time) and then the
rendered output. This shows the system is a usable *tool*, not just a notebook.

---

## 14. Q&A Preparation — Likely Questions and Strong Answers

**Q: Why classical CV + Tesseract instead of deep learning end-to-end (e.g., TrOCR, EasyOCR, donut)?**
A: Three reasons. (1) This is an image-processing course project — the point is mastering the
classical toolbox: morphology, Otsu, projection profiles, connected components. (2) Tesseract's
LSTM *is* a neural recogniser — we use deep learning where it's strongest and classical CV where
it's transparent and fast. (3) Our approach runs on CPU in ~1.4 s/doc with no GPU, no training
data, and every stage is inspectable and debuggable. And the result — 0.19 % CER on the target
domain — leaves little headroom for a heavier model to gain.

**Q: Isn't testing on synthetic data cheating?**
A: We render with real TrueType fonts and apply realistic scan degradations (blur, noise, skew,
illumination), and the *same engine* sees the same images in baseline and pipeline conditions — so
the comparison is fair and the ground truth is exact. We additionally ran 30 uncurated real
documents as a robustness check. Hand-labelled real scans would introduce label noise; exact GT is
*more* rigorous for measuring character error.

**Q: Why does your pipeline help if Tesseract already preprocesses internally?**
A: Tesseract binarises internally but does **not** fix skew beyond small angles, does not upscale
small text, and does not flatten illumination. We supply exactly the corrections it lacks, and
deliberately *don't* duplicate the one it does well (binarisation).

**Q: Why is the pipeline slower than the baseline?**
A: Deskew search, upscaling (more pixels), and up to three PSM attempts. The early-accept rule
(stop if confidence ≥ 85) bounds it. 1.4 s/doc is well within batch-archiving budgets; 33× fewer
errors is the payoff.

**Q: What is PSM and why try several?**
A: Page Segmentation Mode — Tesseract's prior about page structure (single block? column? auto?).
The wrong prior scrambles reading order or merges columns. We try PSM 4, 6, 3 and keep the result
with the best confidence×log(words) score — letting the engine itself judge, with a fast path.

**Q: Confidence isn't accuracy — why report it for the real batch?**
A: Correct, and we say so explicitly: confidence is a *proxy* used only where ground truth cannot
exist. The rigorous claim (0.19 % CER) comes solely from the labelled set. On that set, confidence
and accuracy correlate well, which justifies the proxy — and low confidence usefully flags
documents needing human review.

**Q: Why did the "total" field hit only 92 %?**
A: One invoice in twelve where the amount glyphs degraded enough that the regex found a corrupted
number. Invoice number and date were 100 %. Field extraction is regex-based best-effort by design;
a learned key-value extractor is listed future work.

**Q: What about handwriting / other languages?**
A: Out of scope by declaration — we use Tesseract's English print model. The architecture is ready
for it: `lang` is a parameter (swap in other traineddata), and handwriting would mean swapping the
recognition stage only, since the pipeline is modular.

**Q: What was the hardest bug?**
A: Tell the detective story (§10): the system *looked* functional while emitting `[word]`
placeholders — 0 % real text. Found only because we built measurement first. Second favourite:
discovering our own preprocessing was *hurting* accuracy, and deleting it improved results.

**Q: What's novel here?**
A: No single component is novel — the *engineering* is: measurement-driven selection of which
classical steps actually help, an adaptive confidence-scored PSM strategy with a CLAHE retry
feedback loop, a safe-by-construction post-processor, a no-root deployment story, and a
reproducible evaluation harness with exact ground truth. The 33× error reduction over the same
engine is the evidence the engineering worked.

---

## 15. Repository Map & How to Run Everything

### Map

| Path | Role |
|---|---|
| [src/pipeline.py](src/pipeline.py) | **The** pipeline: `OCRPreprocessor` + `DocumentOCRPipeline` (preprocess, adaptive OCR, trace) |
| [src/layout_analysis.py](src/layout_analysis.py) | Connected components, morphological smearing, region classification |
| [src/postprocessing.py](src/postprocessing.py) | `SafePostProcessor`, `extract_fields`, JSON/TXT/CSV `OutputFormatter` |
| [src/evaluation.py](src/evaluation.py) | CER / WER / token-F1 / field-accuracy metrics (Levenshtein from scratch) |
| [src/tesseract_setup.py](src/tesseract_setup.py) | Finds system *or* user-local Tesseract; wires env vars |
| [src/main.py](src/main.py) | CLI wrapper |
| [app.py](app.py) | Streamlit demo UI (stage-by-stage walkthrough via `trace()`) |
| [scripts/make_invoice_dataset.py](scripts/make_invoice_dataset.py) | Renders labelled synthetic invoices + scan-like degradation |
| [scripts/download_invoices.py](scripts/download_invoices.py) | Fetches real documents from Wikimedia Commons (with attribution) |
| [scripts/evaluate.py](scripts/evaluate.py) | Full evaluation harness; `--baseline` adds raw-Tesseract; `--report` writes RESULTS.md |
| [scripts/install_tesseract_local.sh](scripts/install_tesseract_local.sh) | No-root Tesseract install into `~/.local/opt/tesseract` |
| [RESULTS.md](RESULTS.md) | Generated evaluation report (the numbers in §12) |
| [presentation.md](presentation.md) | Ready-to-build 10-slide deck: content, image prompts, speaker notes |

### Commands

```bash
# 1. environment (one-time)
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt          # opencv, numpy, pytesseract, pillow, streamlit, requests

# 2. the OCR engine (one-time, pick one)
sudo apt install tesseract-ocr                 # with root
bash scripts/install_tesseract_local.sh        # WITHOUT root → ~/.local/opt/tesseract

# 3. build the test data (git-ignored, regenerable)
python scripts/make_invoice_dataset.py --n 12 --degrade scan   # labelled set, exact GT
python scripts/download_invoices.py --n 30                     # real images, no GT

# 4. reproduce every number in this document
python scripts/evaluate.py --baseline --report RESULTS.md

# 5. use it
python src/main.py data/invoices/synthetic/inv_00.png --format json
streamlit run app.py
```

*(On this machine: use `./env/bin/python` directly; Tesseract is the local install in `~/.local`.)*

---

## 16. Cheat Sheet — Memorize These

**The five numbers:**
- CER: **6.33 % → 0.19 %** (≈ **33× fewer** character errors)
- Character accuracy: **99.8 %**
- WER: 13.79 % → **1.67 %**; Token-F1: 92.75 % → **98.79 %**
- Speed: **1.42 s/doc** (baseline 0.57) — fast path: accept at confidence ≥ 85
- Real batch: **30/30 processed**; printed Latin docs score **75–93 %** confidence

**The four stages (one breath):**
*Preprocess (clean & straighten) → Layout (find regions) → Recognise (adaptive Tesseract) →
Post-process (safe clean-up + fields) → JSON/CSV/TXT.*

**The three counterintuitive lessons (your best moments):**
1. **Don't binarise before OCR** — the engine's own binariser is better; grayscale wins.
2. **Delete the spell-corrector** — "Kandy → And"; safe-only post-processing.
3. **Upscaling small text** (to ~30 px x-height) is the single biggest accuracy win.

**The thesis sentence:** *"Same engine, 33× fewer errors — the gain is entirely from
measurement-driven preparation of the input and adaptive control of the engine."*

**Key constants if pressed:** target x-height 30 px (scale clipped 1–4×, max side 3500 px); deskew
search ±8° coarse 1° / fine 0.2°; invert if border median < 110; flatten if blurred range > 55;
PSM order 4→6→3; accept ≥ 85; CLAHE retry < 60; score = conf × log(1+words).

---

## 17. Glossary

| Term | Plain-English meaning |
|---|---|
| **OCR** | Optical Character Recognition — software reading text out of images |
| **Tesseract** | The leading open-source OCR engine (HP → Google); v5 uses an LSTM neural net |
| **LSTM** | A neural network that reads sequences with memory — like reading with context |
| **PSM** | Page Segmentation Mode — Tesseract's assumption about page structure |
| **OEM** | OCR Engine Mode — which engine generation to use (3 = default/LSTM) |
| **Grayscale** | Image with only brightness, no colour |
| **Binarisation** | Forcing every pixel to pure black or white |
| **Otsu's method** | Automatic way to pick the best black/white threshold from the histogram |
| **CLAHE** | Adaptive local contrast enhancement, limited to avoid amplifying noise |
| **Bilateral filter** | Blur that smooths flat areas but preserves edges |
| **Morphology (close/dilate)** | Shape-based operations that grow/merge ink blobs |
| **Connected components** | Finding each separate island of ink pixels |
| **RLSA** | Run-Length Smoothing — smearing ink so characters merge into lines/blocks |
| **Projection profile** | Squashing ink counts onto an axis; spiky = straight text lines |
| **Deskew** | Detecting and undoing page rotation |
| **x-height** | Height of a lowercase letter like "x" — our upscaling target is ~30 px |
| **Ground truth** | The exact correct answer used for grading |
| **CER / WER** | Character / word error rate — edit distance ÷ reference length |
| **Token-F1** | Order-free overlap score between recognised and true words |
| **Levenshtein distance** | Minimum insertions+deletions+substitutions to turn one string into another |
| **Confidence** | The engine's own 0–100 certainty per word (a proxy, not accuracy) |
