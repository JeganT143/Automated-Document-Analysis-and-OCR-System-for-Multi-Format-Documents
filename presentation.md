# Presentation — Automated Document Analysis and OCR System

A ready-to-build deck (**10 slides max**) for the project
*Automated Document Analysis and OCR System for Multi-Format Documents*
(EE7204 / EC7205 — Image Processing and Computer Vision, Dept. of EIE,
University of Ruhuna).

For every slide you get: **on-slide content**, an **AI image-generation prompt**,
and **speaker notes** (what to say). The deck describes the project's **single,
end-to-end pipeline** (see `RESULTS.md` for the measured numbers).

---

## How to use this file

- **Tools:** Build in PowerPoint / Google Slides / Canva. Generate images with
  DALL·E 3, Midjourney, Ideogram, or Adobe Firefly using the prompts below.
- **Important — text in images:** AI image generators produce *garbled text*.
  Every prompt ends with **"no text"**. Add all titles, numbers, the
  architecture diagram, and the results table **natively in PowerPoint** so they
  are readable. The generated images are visuals/backgrounds only.
- **Design system (keep consistent):** palette deep blue `#1F4E79`, primary
  blue `#2E86C1`, light blue `#EAF4FC`, success green `#27AE60`, accent orange
  `#F39C12`, white background. Headings Montserrat/Calibri ~32–40 pt, body
  Lato/Calibri ~20–24 pt. One idea per slide, ≤5 bullets.
- **Style suffix** reused in prompts: *"modern flat vector illustration,
  professional, clean, corporate blue and teal palette with subtle orange
  accents, white background, soft shadows, 16:9, no text, no words, no letters."*

---

## Slide 1 — Title

**On-slide content**
- **Title:** Automated Document Analysis & OCR System for Multi-Format Documents
- **Subtitle:** A classical-CV preprocessing + adaptive-Tesseract pipeline
- Course: EE7204 / EC7205 — Image Processing and Computer Vision
- Department of Electrical & Information Engineering, University of Ruhuna
- **Team:** Arivanan V. (EG/2021/4414) · Arivarasan J. (EG/2021/4415) ·
  Bravin K. (EG/2021/4447) · Jegan T. (EG/2021/4590)

**🎨 Image prompt**
> A sheet of printed paper (an invoice) being transformed into glowing digital
> data, a bright horizontal scan-line crossing it, structured data blocks lifting
> off the page on the right. Hero composition, "paper to data". Modern flat
> vector illustration, professional, clean, corporate blue and teal palette with
> subtle orange accents, white background, soft shadows, 16:9, no text, no words,
> no letters.

**🎤 Speaker notes**
> "Good morning. We're presenting our Image Processing and Computer Vision
> project: an Automated Document Analysis and OCR System. It takes a photo or
> scan of a document — like an invoice — and turns it into clean, structured,
> machine-readable text. I'm [name], with my teammates. We'll cover the problem,
> our single end-to-end pipeline, and the accuracy we achieved."

---

## Slide 2 — Problem & Motivation

**On-slide content**
- Manual data entry from documents is **slow, costly, and error-prone**
- Documents vary: fonts, layouts, tables, multiple columns
- Real scans are **noisy, skewed, low-resolution, unevenly lit**
- **Goal:** automatically extract *structured* text from document images
- Use case: invoices, receipts, forms → databases / accounting

**🎨 Image prompt**
> Split concept: left, a messy pile of diverse paper invoices and receipts,
> slightly skewed and crumpled; right, the same information as a neat organized
> digital table on a laptop. Visual story of "chaos to order". Modern flat vector
> illustration, professional, clean, corporate blue and teal palette with subtle
> orange accents, white background, soft shadows, 16:9, no text, no words, no
> letters.

**🎤 Speaker notes**
> "Businesses process thousands of invoices, receipts and forms, and typing that
> data in by hand is slow and error-prone. The hard part is that documents are
> never uniform — different fonts, multi-column layouts, tables — and scans are
> noisy, tilted or poorly lit. Our goal was a system that reads them
> automatically and outputs structured data a computer can use directly."

---

## Slide 3 — System Architecture (one pipeline)

**On-slide content** — *(build as a real diagram in PowerPoint)*
- Left-to-right flow with 4 boxes + arrows ending in an output cylinder:
  1. **Preprocess** (classical CV) — clean & straighten
  2. **Layout Analysis** (classical CV) — find text/table/header regions
  3. **Recognise** — Tesseract with adaptive page segmentation
  4. **Post-Process** — safe clean-up + field extraction
  - → **Output:** JSON · CSV · TXT
- Caption: *One path, end to end — no competing recognition methods*

**🎨 Image prompt** *(decorative background; draw labelled boxes in PPT)*
> Four glowing connected nodes left to right with smooth arrows, like a clean
> processing pipeline, isometric technology style, a document entering on the
> left and a database icon on the right. Modern flat vector illustration,
> professional, clean, corporate blue and teal palette with subtle orange
> accents, white background, soft shadows, 16:9, no text, no words, no letters.

**🎤 Speaker notes**
> "Our system is a single pipeline of four stages. Preprocessing cleans and
> straightens the image. Layout analysis finds the text, tables and headers.
> Recognition reads the characters with Tesseract. Post-processing cleans the
> text and structures it into JSON, CSV or text. Importantly, there is exactly
> *one* recognition path — we removed the earlier experimental classifiers so
> the system is clear and reliable. I'll take each stage in turn."

---

## Slide 4 — Stage 1: Preprocessing (classical CV)

**On-slide content**
- **Auto-invert** light-on-dark pages (dark-mode receipts, negatives)
- **Illumination flattening** (rolling-ball) for uneven lighting/shadows
- **Projection-profile deskew** — robust for text, better than Hough-on-edges
- ⭐ **Resolution normalization** — up-scale small text to ~30 px x-height
- Light edge-preserving denoise — hand the engine a **clean grayscale**
- *Insight:* clean grayscale beats heavy binarization for the OCR engine

**🎨 Image prompt**
> Side-by-side comparison of the same document scan: left is dark, noisy, grainy
> and tilted; right is bright, sharp, high-contrast and perfectly straightened,
> with a small rotation arrow between them. Modern flat vector illustration,
> professional, clean, corporate blue and teal palette with subtle orange
> accents, white background, soft shadows, 16:9, no text, no words, no letters.

**🎤 Speaker notes**
> "Stage one prepares the image. We flip pages that are light-on-dark, flatten
> uneven lighting, and straighten skew using a projection-profile method that's
> robust for text. The single biggest win is resolution normalization: we
> up-scale small text so characters land in the size the engine reads best.
> Crucially, we *don't* binarize — we learned that handing Tesseract a clean
> grayscale image beats aggressive thresholding, because it has its own better
> binarizer. That one decision moved the needle a lot, as the results show."

---

## Slide 5 — Stage 2: Layout Analysis (classical CV)

**On-slide content**
- **Connected-component analysis** — find character/word blobs
- **Morphological region smearing** — fast RLSA-equivalent that merges
  characters into lines and lines into blocks
- **Region classification:** text · table · image · header/footer
- Provides structure for the output and the visualization

**🎨 Image prompt**
> A clean document page top-down with colored translucent rectangles over
> different zones: blue over paragraph text, orange over a table grid, green over
> the header strip. Document segmentation visualization, neat and schematic.
> Modern flat vector illustration, professional, clean, corporate blue and teal
> palette with subtle orange accents, white background, soft shadows, 16:9, no
> text, no words, no letters.

**🎤 Speaker notes**
> "Stage two understands structure. Connected-component analysis finds blobs of
> ink; fast morphological smearing merges them into lines and blocks — this
> replaced an older, slow pure-Python algorithm and runs in milliseconds. We
> then classify each region as text, table, image or header/footer. This is the
> 'document analysis' half of the project and drives both the structured output
> and the region overlay you'll see in the demo."

---

## Slide 6 — Stage 3: Recognition (adaptive Tesseract)

**On-slide content**
- **Tesseract LSTM** engine for character/word recognition
- ⭐ **Adaptive page segmentation** — try PSM 4 / 6 / 3, keep the result the
  engine is **most confident** about (fast path when the first is already good)
- **CLAHE retry** pass for low-confidence (hard / low-contrast) scans
- Per-word confidences + bounding boxes drive the visualization

**🎨 Image prompt**
> A magnifying glass over a row of printed characters, each inside its own
> highlighted bounding box, with faint connecting lines suggesting a recognition
> engine identifying them. Sense of precise character recognition. Modern flat
> vector illustration, professional, clean, corporate blue and teal palette with
> subtle orange accents, white background, soft shadows, 16:9, no text, no words,
> no letters.

**🎤 Speaker notes**
> "Stage three is recognition. We use the Tesseract LSTM engine, but the smart
> part is how we drive it. Rather than guessing one page-segmentation mode, we
> try a few and keep the result Tesseract itself is most confident about — with
> a fast path so it stays quick on easy pages. For hard, low-contrast scans we
> add a contrast-enhancement retry. The engine returns a confidence and a box
> for every word, which we use both for quality control and for the visualization
> in the app."

---

## Slide 7 — Stage 4: Post-Processing & Output

**On-slide content**
- **Safe clean-up:** de-hyphenation, whitespace, currency spacing
- **No naïve spell-correction** — a small dictionary corrupts valid words
  (e.g. *Kandy → And*), so it is **removed**, not just disabled
- **Field extraction:** invoice no., date, total amount
- **Structured output:** JSON · CSV · plain text (downloadable)

**🎨 Image prompt**
> A printed document on the left with a glowing arrow flowing into a clean
> structured data card on the right showing rows of organized fields and values
> (abstract colored bars, not readable text), plus small JSON-bracket and
> spreadsheet-grid icons. "Unstructured to structured". Modern flat vector
> illustration, professional, clean, corporate blue and teal palette with subtle
> orange accents, white background, soft shadows, 16:9, no text, no words, no
> letters.

**🎤 Speaker notes**
> "The final stage turns raw text into a usable result. We apply only *safe*
> corrections — joining hyphenated words, fixing spacing and currency. We
> deliberately removed dictionary spell-correction, because in testing it
> corrupted correct words, like turning 'Kandy' into 'And'. We then extract key
> fields — invoice number, date, total — and export JSON, CSV or text that other
> systems can consume."

---

## Slide 8 — Evaluation & Results ⭐ (the highlight)

**On-slide content** — *(build the table/chart natively in PowerPoint)*
- **How we measured:** invoices rendered with **real fonts** (exact ground
  truth) + scan-like noise/skew. Metrics = **CER, WER, token-F1, field accuracy**
- **Results (labelled invoices):**

  | Configuration | CER ↓ | Char acc. ↑ | Token-F1 ↑ |
  |---|--:|--:|--:|
  | Raw Tesseract (`--psm 6`) | 6.3 % | 93.7 % | 92.8 % |
  | **Our pipeline** | **0.2 %** | **99.8 %** | **98.8 %** |

- Also tested on **30 real downloaded documents** (Wikimedia Commons): clean
  printed invoices score **80–93 %** confidence; handwritten/non-Latin artifacts
  are out of scope for the English engine. *(see `RESULTS.md`)*

**🎨 Image prompt** *(decorative — put the real numbers in PPT)*
> A bold upward-trending bar chart with a glowing green arrow rising steeply
> left to right, conveying a dramatic jump in performance. Small checkmark and
> target icons. Modern flat vector illustration, professional, clean, corporate
> blue and teal palette with a strong success-green accent, white background,
> soft shadows, 16:9, no text, no words, no letters.

**🎤 Speaker notes**
> "Now the results, which we're proud of. To measure honestly we generated
> invoices with real fonts so we know the exact correct text, then compared
> outputs with standard OCR metrics. Plain Tesseract on the raw image gets about
> a six percent character error rate. Our full pipeline brings that down to
> **two-tenths of one percent** — character accuracy near ninety-nine-point-eight
> percent, a thirty-times reduction in errors. We also ran it on thirty real
> documents from Wikimedia: clean printed invoices score eighty to ninety-three
> percent confidence, while handwritten manuscripts and non-Latin scripts score
> lower — expected, since we use the English print model."

---

## Slide 9 — Live Demo (Interactive App)

**On-slide content**
- Interactive **Streamlit web app**
- Upload or generate a document → run the full pipeline
- **Visualizes every stage:** preprocessing, layout regions, word-level
  recognition confidence, final structured output
- One-click **download** of JSON / CSV / TXT
- *(Insert a real screenshot of the running app here)*

**🎨 Image prompt**
> A sleek modern web dashboard mockup on a laptop: left panel a document
> thumbnail, right panel analysis results as colored stat cards, progress bars
> and a small results panel. Clean SaaS interface. Modern flat vector
> illustration, corporate blue and teal palette with subtle orange accents,
> white background, soft shadows, 16:9, no text, no words, no letters.

**🎤 Speaker notes**
> "To make it usable we built an interactive Streamlit app. You upload or
> generate a document, click run, and it walks through the whole pipeline live —
> the cleaned image, the detected regions, the per-word recognition confidence,
> and the final structured output, downloadable as JSON, CSV or text. [If live:]
> Let me show a quick run on a sample invoice."

---

## Slide 10 — Conclusion & Future Work

**On-slide content**
- **Built** one clean, end-to-end classical-CV + adaptive-OCR pipeline
- **Achieved** ~6 % → ~0.2 % CER (≈ 99.8 % character accuracy) on labelled invoices
- **Contributions:** modular pipeline · no-root Tesseract deployment ·
  evaluation harness (CER/WER/F1/fields) · labelled invoice dataset
- **Future work:** handwriting recognition · multi-language · table-structure
  extraction · mobile / real-time capture
- **Thank you — Questions?**

**🎨 Image prompt**
> An open digital document at the center radiating outward into future
> technologies: a smartphone capture icon, a globe for multi-language, a
> spreadsheet grid and a neural-network motif, connected by glowing lines,
> conveying a forward-looking roadmap. Modern flat vector illustration,
> professional, clean, corporate blue and teal palette with subtle orange
> accents, white background, soft shadows, 16:9, no text, no words, no letters.

**🎤 Speaker notes**
> "To wrap up: we built a single, clean document-OCR pipeline grounded in
> classical computer vision, and improved character accuracy from around
> ninety-four percent to nearly ninety-nine-point-eight percent on labelled
> invoices. We also contributed an evaluation harness and a labelled dataset, and
> made it deployable without admin rights. Next we'd add handwriting recognition,
> more languages, full table extraction, and mobile capture. Thank you — we're
> happy to take questions."

---

## Appendix — Quick build checklist
- [ ] Apply the color palette + fonts to a master slide
- [ ] Generate 10 images from the prompts; place as side visuals/backgrounds
- [ ] Draw the **architecture diagram** (Slide 3) natively with readable labels
- [ ] Build the **results table** (Slide 8) natively with the real numbers
- [ ] Insert a real **app screenshot** on Slide 9
- [ ] Add slide numbers + a small course/team footer
- [ ] Rehearse to ~6–8 minutes (~40–50 s per slide)
