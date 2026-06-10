"""
The Document OCR pipeline – the single, end-to-end recognition pipeline.

    image  ->  preprocess  ->  layout analysis  ->  recognise  ->  post-process
                                                                       |
                                                              structured output

Design decisions (validated by the evaluation harness, scripts/evaluate.py):

  * Heavy binarisation + morphology BEFORE OCR *hurts* accuracy – Tesseract has
    its own, better internal binariser and prefers a clean grayscale image.
    So preprocessing keeps only the steps that genuinely help OCR.
  * Resolution normalisation (up-scaling small text to ~30 px x-height) is the
    single biggest win for low-DPI scans.
  * Adaptive page segmentation: try a few PSM modes and keep the result
    Tesseract itself is most confident about.
  * Post-processing is *safe* – it never rewrites numbers or proper nouns.

Public API:
    DocumentOCRPipeline().image_to_text(img)   -> str           (fast)
    DocumentOCRPipeline().run(img)             -> result dict    (full, for UI/CLI)
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from tesseract_setup import ensure_tesseract
from postprocessing import SafePostProcessor, OutputFormatter, extract_fields
from layout_analysis import LayoutAnalyzer


# ---------------------------------------------------------------------------
# OCR-tuned preprocessing (classical computer vision)
# ---------------------------------------------------------------------------
class OCRPreprocessor:
    def __init__(self, target_text_height=30, max_side=3500,
                 deskew=True, flatten_illumination=True, denoise=True):
        self.target_text_height = target_text_height
        self.max_side = max_side
        self.do_deskew = deskew
        self.do_flatten = flatten_illumination
        self.do_denoise = denoise

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    @staticmethod
    def _maybe_invert(gray):
        """Flip light-on-dark pages (dark-mode receipts, scanned negatives) so
        text is always dark on a light background, the way Tesseract expects."""
        b = 8
        border = np.concatenate([
            gray[:b, :].ravel(), gray[-b:, :].ravel(),
            gray[:, :b].ravel(), gray[:, -b:].ravel(),
        ])
        if np.median(border) < 110:          # background is dark
            return cv2.bitwise_not(gray), True
        return gray, False

    @staticmethod
    def _needs_illumination_fix(gray):
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (0, 0), 8)
        return float(blur.max() - blur.min()) > 55

    def flatten(self, gray):
        """Divide out a smooth background estimate (rolling-ball style)."""
        k = max(15, (min(gray.shape) // 20) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        background = cv2.GaussianBlur(background, (0, 0), k / 3.0)
        return cv2.divide(gray, background, scale=255)

    @staticmethod
    def _estimate_text_height(gray):
        thr = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
        h_img, w_img = gray.shape
        heights = []
        for i in range(1, n):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            if 6 <= h <= h_img * 0.1 and 2 <= w <= w_img * 0.1 and a >= 8:
                heights.append(h)
        return float(np.median(heights)) if len(heights) >= 10 else None

    def normalize_resolution(self, gray):
        th = self._estimate_text_height(gray)
        if th is None or th <= 0:
            return gray, 1.0
        scale = float(np.clip(self.target_text_height / th, 1.0, 4.0))
        long_side = max(gray.shape)
        if long_side * scale > self.max_side:
            scale = self.max_side / long_side
        if scale <= 1.01:
            return gray, 1.0
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp), scale

    def deskew_angle(self, gray):
        """Projection-profile skew estimation (robust for text)."""
        thr = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        small = cv2.resize(thr, (min(600, thr.shape[1]), min(800, thr.shape[0])),
                           interpolation=cv2.INTER_AREA)

        def score(angle):
            h, w = small.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rot = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_NEAREST)
            proj = rot.sum(axis=1, dtype=np.float64)
            return np.sum(np.diff(proj) ** 2)

        best = max(np.arange(-8, 8.1, 1.0), key=score)
        best = max(np.arange(best - 1.0, best + 1.01, 0.2), key=score)
        return float(best)

    def deskew(self, gray):
        angle = self.deskew_angle(gray)
        if abs(angle) < 0.2:
            return gray, 0.0
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        return rot, angle

    @staticmethod
    def _rec(steps, key, title, desc, image, meta):
        """Record an intermediate stage image (only when tracing)."""
        if steps is not None:
            steps.append({"key": key, "title": title, "desc": desc,
                          "image": image.copy(), "meta": meta})

    # -- main --------------------------------------------------------------
    def process(self, image, clahe=False, steps=None):
        gray = self.to_gray(image)
        info = {}
        self._rec(steps, "grayscale", "Grayscale",
                  "Reduce to a single luminance channel.", gray, {})

        gray, inverted = self._maybe_invert(gray)
        if inverted:
            info["inverted"] = True
        self._rec(steps, "invert", "Polarity normalise",
                  "Flip light-on-dark pages to dark-on-light.", gray,
                  {"applied": bool(inverted)})

        need_flat = self.do_flatten and self._needs_illumination_fix(gray)
        if need_flat:
            gray = self.flatten(gray)
            info["illumination_fixed"] = True
        self._rec(steps, "flatten", "Illumination flatten",
                  "Divide out an uneven-lighting background estimate.", gray,
                  {"applied": bool(need_flat)})

        if clahe:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            info["clahe"] = True
            self._rec(steps, "clahe", "Contrast (CLAHE)",
                      "Adaptive local contrast for faint scans.", gray,
                      {"applied": True})

        angle = 0.0
        if self.do_deskew:
            gray, angle = self.deskew(gray)
            info["skew_angle"] = round(angle, 2)
        self._rec(steps, "deskew", "Deskew",
                  "Estimate and correct page rotation (projection profile).",
                  gray, {"angle": round(angle, 2)})

        gray, scale = self.normalize_resolution(gray)
        info["upscale"] = round(scale, 2)
        self._rec(steps, "upscale", "Resolution normalise",
                  "Up-scale so text x-height hits the OCR sweet spot (~30 px).",
                  gray, {"scale": round(scale, 2)})

        if self.do_denoise:
            gray = cv2.bilateralFilter(gray, 5, 35, 35)
        self._rec(steps, "denoise", "Denoise",
                  "Light edge-preserving smoothing; the OCR-ready image.",
                  gray, {"applied": bool(self.do_denoise)})

        return gray, info


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
class DocumentOCRPipeline:
    def __init__(self, lang="eng", psm_candidates=(4, 6, 3), oem=3,
                 preprocess=True, postprocess=True, preprocessor=None,
                 accept_conf=85.0, clahe_retry=True):
        self.lang = lang
        self.psm_candidates = psm_candidates
        self.oem = oem
        self.accept_conf = accept_conf      # first PSM this confident -> stop early
        self.clahe_retry = clahe_retry      # low-confidence pages get a CLAHE pass
        self.do_preprocess = preprocess
        self.do_postprocess = postprocess
        self.pre = preprocessor or OCRPreprocessor()
        self.post = SafePostProcessor()
        self.layout = LayoutAnalyzer()
        self.formatter = OutputFormatter()
        self._ready = ensure_tesseract() is not None

    # -- low level ---------------------------------------------------------
    def _config(self, psm):
        return (f"--oem {self.oem} --psm {psm} -l {self.lang} "
                f"-c preserve_interword_spaces=1")

    def _ocr_with_conf(self, img, psm):
        """OCR one PSM; return (text, mean_conf, score, words)."""
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(img, config=self._config(psm),
                                         output_type=Output.DICT)
        confs, words, lines = [], [], {}
        for i in range(len(data["text"])):
            txt = data["text"][i].strip()
            try:
                c = float(data["conf"][i])
            except (ValueError, TypeError):
                c = -1
            if not txt or c < 0:
                continue
            confs.append(c)
            words.append({
                "text": txt, "conf": round(c, 1),
                "bbox": (int(data["left"][i]), int(data["top"][i]),
                         int(data["width"][i]), int(data["height"][i])),
            })
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(txt)
        text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
        mean_conf = float(np.mean(confs)) if confs else 0.0
        # weight confidence by amount of text so a confident-but-empty result loses
        score = mean_conf * np.log1p(len(confs))
        return text, mean_conf, score, words

    def _best_ocr(self, img):
        best = ("", 0.0, -1.0, [], None)   # text, conf, score, words, psm
        for psm in self.psm_candidates:
            try:
                text, conf, score, words = self._ocr_with_conf(img, psm)
            except Exception:
                continue
            if score > best[2]:
                best = (text, conf, score, words, psm)
            if conf >= self.accept_conf:     # fast path
                break
        return best

    def _recognise(self, image):
        """Preprocess + adaptive-PSM OCR, with a CLAHE retry for hard pages."""
        proc, info = (self.pre.process(image) if self.do_preprocess
                      else (self.pre.to_gray(image), {}))
        text, conf, score, words, psm = self._best_ocr(proc)

        if self.clahe_retry and self.do_preprocess and conf < 60:
            proc2, info2 = self.pre.process(image, clahe=True)
            t2, c2, s2, w2, p2 = self._best_ocr(proc2)
            if s2 > score:
                proc, info = proc2, info2
                text, conf, score, words, psm = t2, c2, s2, w2, p2

        if self.do_postprocess:
            text = self.post.process(text)
        return proc, info, text, conf, words, psm

    # -- public ------------------------------------------------------------
    def image_to_text(self, image_or_path):
        self._require_tesseract()
        image = self._load(image_or_path)
        _, _, text, _, _, _ = self._recognise(image)
        return text

    def run(self, image_or_path, analyze_layout=True):
        """Full structured result with metadata (for the UI / CLI)."""
        self._require_tesseract()
        t0 = time.time()
        image = self._load(image_or_path)

        proc, info, text, conf, words, psm = self._recognise(image)
        t_rec = time.time()

        regions = []
        if analyze_layout:
            try:
                regions = self.layout.analyze(proc)["regions"]
            except Exception:
                regions = []
        t_lay = time.time()

        return {
            "text": text,
            "words": [w["text"] for w in words],
            "word_boxes": words,
            "word_count": len(words),
            "mean_confidence": round(conf, 1),
            "psm_used": psm,
            "fields": extract_fields(text),
            "regions": regions,
            "preprocess_info": info,
            "processed_image": proc,
            "metrics": {
                "recognition_s": round(t_rec - t0, 3),
                "layout_s": round(t_lay - t_rec, 3),
                "total_s": round(t_lay - t0, 3),
            },
        }

    def trace(self, image_or_path, analyze_layout=True):
        """Run the pipeline while capturing the image at every stage.

        Returns {"pre_stages": [...], "result": {...}} where each pre-stage is a
        dict with the intermediate image and metadata. Used by the UI to show
        the document at every point in the pipeline.
        """
        self._require_tesseract()
        t0 = time.time()
        image = self._load(image_or_path)

        steps = []
        proc, info = self.pre.process(image, steps=steps)
        text, conf, score, words, psm = self._best_ocr(proc)
        if self.do_postprocess:
            text = self.post.process(text)
        regions = self.layout.analyze(proc)["regions"] if analyze_layout else []

        result = {
            "text": text,
            "words": [w["text"] for w in words],
            "word_boxes": words,
            "word_count": len(words),
            "mean_confidence": round(conf, 1),
            "psm_used": psm,
            "fields": extract_fields(text),
            "regions": regions,
            "preprocess_info": info,
            "original": image,
            "processed_image": proc,
            "metrics": {"total_s": round(time.time() - t0, 3)},
        }
        return {"pre_stages": steps, "result": result}

    def render(self, result, fmt="json"):
        return self.formatter.render(result, fmt=fmt)

    # -- internals ---------------------------------------------------------
    def _require_tesseract(self):
        if not self._ready:
            raise RuntimeError(
                "Tesseract not available. Run scripts/install_tesseract_local.sh "
                "(no root needed) or `sudo apt install tesseract-ocr`.")

    @staticmethod
    def _load(image_or_path):
        if isinstance(image_or_path, str):
            img = cv2.imread(image_or_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot load image: {image_or_path}")
            return img
        return image_or_path
