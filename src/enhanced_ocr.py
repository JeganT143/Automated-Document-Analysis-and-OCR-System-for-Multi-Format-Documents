"""
Enhanced OCR recognition pipeline.

Lessons from the baseline evaluation:
  * The default "[word]"-placeholder fallback recovers 0% of the text.
  * Heavy binarisation + morphology BEFORE Tesseract *hurts* accuracy –
    Tesseract has its own (better) internal binariser and prefers a clean
    grayscale image.

So this module keeps the classical pre-processing that genuinely helps OCR
and drops what hurts it:

  1. Illumination flattening (only when lighting is uneven)
  2. Robust projection-profile deskew (better than Hough-on-edges for text)
  3. Resolution normalisation – up-scale so the text x-height lands in
     Tesseract's sweet spot (~30 px). This is the single biggest win for
     small/low-DPI scans.
  4. Light, *conditional* denoising
  5. Adaptive page-segmentation: try a few PSM modes and keep the result
     Tesseract itself is most confident about.
  6. Safe post-processing (never "corrects" numbers or proper nouns).

The grayscale image is handed to Tesseract un-binarised on purpose.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from tesseract_setup import ensure_tesseract
from postprocessing import OutputFormatter


# ---------------------------------------------------------------------------
# OCR-tuned preprocessing
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
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def _needs_illumination_fix(gray):
        # Large-scale brightness spread => uneven lighting / shadows.
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (0, 0), 8)
        return float(blur.max() - blur.min()) > 55

    def flatten(self, gray):
        """Divide out a smooth background estimate (rolling-ball style)."""
        k = max(15, (min(gray.shape) // 20) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        background = cv2.GaussianBlur(background, (0, 0), k / 3.0)
        norm = cv2.divide(gray, background, scale=255)
        return norm

    @staticmethod
    def _estimate_text_height(gray):
        """Median height of plausible character components."""
        thr = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
        h_img, w_img = gray.shape
        heights = []
        for i in range(1, n):
            w, h, a = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
            if 6 <= h <= h_img * 0.1 and 2 <= w <= w_img * 0.1 and a >= 8:
                heights.append(h)
        if len(heights) < 10:
            return None
        return float(np.median(heights))

    def normalize_resolution(self, gray):
        th = self._estimate_text_height(gray)
        if th is None or th <= 0:
            return gray, 1.0
        scale = self.target_text_height / th
        scale = float(np.clip(scale, 1.0, 4.0))
        long_side = max(gray.shape)
        if long_side * scale > self.max_side:
            scale = self.max_side / long_side
        if scale <= 1.01:
            return gray, 1.0
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        out = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
        return out, scale

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
            return np.sum(np.diff(proj) ** 2)  # sharp line gaps => high score

        coarse = np.arange(-8, 8.1, 1.0)
        best = max(coarse, key=score)
        fine = np.arange(best - 1.0, best + 1.01, 0.2)
        best = max(fine, key=score)
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

    # -- main --------------------------------------------------------------
    def process(self, image):
        gray = self.to_gray(image)
        info = {}

        if self.do_flatten and self._needs_illumination_fix(gray):
            gray = self.flatten(gray)
            info["illumination_fixed"] = True

        if self.do_deskew:
            gray, angle = self.deskew(gray)
            info["skew_angle"] = round(angle, 2)

        gray, scale = self.normalize_resolution(gray)
        info["upscale"] = round(scale, 2)

        if self.do_denoise:
            # Edge-preserving, only mild.
            gray = cv2.bilateralFilter(gray, 5, 35, 35)

        return gray, info


# ---------------------------------------------------------------------------
# Safe post-processing
# ---------------------------------------------------------------------------
class SafePostProcessor:
    """Conservative clean-up that never rewrites numbers or unknown words."""

    def process(self, text):
        import re
        # de-hyphenate words split across line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # normalise newlines/spaces but keep line structure
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        text = "\n".join(lines)
        # common glyph spacing artefacts inside money amounts: "$ 12.50"->"$12.50"
        text = re.sub(r"([$£€])\s+(\d)", r"\1\2", text)
        return text


# ---------------------------------------------------------------------------
# Enhanced OCR engine
# ---------------------------------------------------------------------------
class EnhancedOCR:
    def __init__(self, lang="eng", psm_candidates=(4, 6, 3),
                 oem=3, preprocess=True, postprocess=True,
                 preprocessor=None, accept_conf=85.0):
        self.lang = lang
        self.psm_candidates = psm_candidates
        self.oem = oem
        # If the first PSM is already this confident, skip the others (speed).
        self.accept_conf = accept_conf
        self.do_preprocess = preprocess
        self.do_postprocess = postprocess
        self.pre = preprocessor or OCRPreprocessor()
        self.post = SafePostProcessor()
        self.formatter = OutputFormatter()
        self._ready = ensure_tesseract() is not None

    # -- low level ---------------------------------------------------------
    def _config(self, psm):
        return f"--oem {self.oem} --psm {psm} -l {self.lang}"

    def _ocr_with_conf(self, img, psm):
        """Return (text, mean_confidence) for one PSM using image_to_data."""
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(img, config=self._config(psm),
                                         output_type=Output.DICT)
        words, confs = [], []
        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i].strip()
            try:
                c = float(data["conf"][i])
            except (ValueError, TypeError):
                c = -1
            if txt and c >= 0:
                words.append((data["line_num"][i], data["block_num"][i],
                              data["par_num"][i], txt))
                confs.append(c)
        # reconstruct text with line breaks
        text = self._reassemble(data)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        # weight by amount of text recovered so a confident-but-empty result loses
        score = mean_conf * np.log1p(len(confs))
        return text, mean_conf, score

    @staticmethod
    def _reassemble(data):
        lines = {}
        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i]
            if not txt.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(txt)
        return "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))

    def _best_ocr(self, img):
        best = ("", 0.0, -1.0, None)
        for psm in self.psm_candidates:
            try:
                text, conf, score = self._ocr_with_conf(img, psm)
            except Exception:
                continue
            if score > best[2]:
                best = (text, conf, score, psm)
            # Fast path: the first PSM is already confident enough.
            if conf >= self.accept_conf:
                break
        return best  # (text, mean_conf, score, psm)

    # -- public ------------------------------------------------------------
    def image_to_text(self, image_or_path):
        if not self._ready:
            raise RuntimeError(
                "Tesseract not available. Run scripts/install_tesseract_local.sh "
                "or `sudo apt install tesseract-ocr`."
            )
        image = self._load(image_or_path)
        proc, info = (self.pre.process(image) if self.do_preprocess
                      else (self.pre.to_gray(image), {}))
        text, conf, score, psm = self._best_ocr(proc)
        if self.do_postprocess:
            text = self.post.process(text)
        return text

    def run(self, image_or_path):
        """Full result with metadata (for the UI / CLI)."""
        if not self._ready:
            raise RuntimeError("Tesseract not available.")
        image = self._load(image_or_path)
        proc, info = (self.pre.process(image) if self.do_preprocess
                      else (self.pre.to_gray(image), {}))
        text, conf, score, psm = self._best_ocr(proc)
        if self.do_postprocess:
            text = self.post.process(text)
        words = text.split()
        return {
            "text": text,
            "words": words,
            "mean_confidence": round(conf, 1),
            "psm_used": psm,
            "preprocess_info": info,
            "processed_image": proc,
        }

    @staticmethod
    def _load(image_or_path):
        if isinstance(image_or_path, str):
            img = cv2.imread(image_or_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot load image: {image_or_path}")
            return img
        return image_or_path
