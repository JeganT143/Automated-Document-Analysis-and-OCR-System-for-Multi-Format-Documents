"""
Test the OCR pipeline on a real document image.

Usage (with your own image):
    python scripts/test_with_image.py --image /path/to/invoice.jpg

Usage (generate a high-quality synthetic test image):
    python scripts/test_with_image.py --generate

Usage (with a trained model):
    python scripts/test_with_image.py --image doc.jpg --model data/models/svm_emnist.pkl
"""

import sys
import os
import argparse
import time
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from preprocessing import Preprocessor, NoiseReducer, Binarizer, GeometricCorrector
from layout_analysis import LayoutAnalyzer
from recognition import (CharacterNormalizer, HOGFeatureExtractor,
                          SVMClassifier, KNNClassifier,
                          CharacterRecognizer, WordAssembler)
from postprocessing import PostProcessor
from main import OCRPipeline


OUTPUT_DIR = os.path.join(ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# High-quality synthetic invoice (used when no real image supplied)
# ──────────────────────────────────────────────────────────────────
def generate_test_invoice(path, skew_deg=2.5, noise_sigma=12):
    """Render a realistic invoice image and save to path."""
    W, H = 794, 1123   # A4 at 96 dpi
    img = np.ones((H, W), np.uint8) * 252

    BOLD  = cv2.FONT_HERSHEY_DUPLEX
    PLAIN = cv2.FONT_HERSHEY_SIMPLEX

    def put(text, x, y, font=PLAIN, scale=0.55, thickness=1, color=0):
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def hline(y, x1=50, x2=744, t=1):
        cv2.line(img, (x1, y), (x2, y), 0, t)

    def rect(x, y, w, h, t=1):
        cv2.rectangle(img, (x, y), (x+w, y+h), 0, t)

    # ── Header ────────────────────────────────────────────────────
    put("INVOICE",              50,  70, BOLD,  1.6, 3)
    put("University of Ruhuna", 50, 105, PLAIN, 0.6, 1, 80)
    put("Dept of Electrical & Information Engineering", 50, 128, PLAIN, 0.48, 1, 80)
    put("#INV-2024-0042",      530,  70, BOLD,  0.8, 2)
    put("Date:  23 Jan 2026",  530,  98, PLAIN, 0.5, 1)
    put("Due:   07 Feb 2026",  530, 120, PLAIN, 0.5, 1)
    hline(145, t=2)

    # ── Bill To ───────────────────────────────────────────────────
    put("Bill To:",             50, 175, BOLD,  0.6, 1)
    put("Department of EIE",    50, 200, PLAIN, 0.55, 1)
    put("Faculty of Engineering", 50, 222, PLAIN, 0.5, 1)
    put("Matara, Sri Lanka",    50, 244, PLAIN, 0.5, 1)

    put("Payment Method:",     430, 175, BOLD,  0.6, 1)
    put("Bank Transfer",       430, 200, PLAIN, 0.55, 1)
    put("Account: 001234567890", 430, 222, PLAIN, 0.5, 1)
    put("Bank: Bank of Ceylon", 430, 244, PLAIN, 0.5, 1)

    hline(265)

    # ── Table header ──────────────────────────────────────────────
    COL = [50, 100, 360, 490, 600, 744]
    TH  = 290
    rect(50, TH, 694, 30, -1)   # filled header bar
    img[TH:TH+30, 50:744] = 210  # light gray fill

    headers = ["#", "Description", "Qty", "Unit Price", "Total"]
    for i, (h_txt, x) in enumerate(zip(headers, COL)):
        put(h_txt, x+5, TH+21, PLAIN, 0.48, 1)
    hline(TH+30)

    # ── Table rows ────────────────────────────────────────────────
    rows = [
        ("1", "HOG Feature Extraction Module",     "2", "LKR  12,500", "LKR  25,000"),
        ("2", "SVM Classifier (RBF kernel)",        "1", "LKR  45,000", "LKR  45,000"),
        ("3", "k-NN Classifier",                    "3", "LKR   8,000", "LKR  24,000"),
        ("4", "Tesseract OCR Integration",          "1", "LKR       0", "LKR       0"),
        ("5", "Document Layout Analysis Engine",    "1", "LKR  35,000", "LKR  35,000"),
    ]
    for ri, row_data in enumerate(rows):
        ry = TH + 30 + ri * 35
        if ri % 2 == 1:
            img[ry:ry+35, 50:744] = 245  # zebra stripe
        for ci, (cell, cx) in enumerate(zip(row_data, COL)):
            put(cell, cx+5, ry+23, PLAIN, 0.46, 1)
        hline(ry+35)

    # Vertical table lines
    for cx in COL:
        cv2.line(img, (cx, TH), (cx, TH + 30 + len(rows)*35), 0, 1)

    # ── Totals ────────────────────────────────────────────────────
    ty = TH + 30 + len(rows)*35 + 20
    put("Subtotal:",    530, ty+0,  PLAIN, 0.52, 1)
    put("LKR 1,29,000", 630, ty+0,  PLAIN, 0.52, 1)
    put("Tax (0%):",    530, ty+25, PLAIN, 0.52, 1)
    put("LKR       0", 630, ty+25, PLAIN, 0.52, 1)
    hline(ty+35, x1=520)
    put("TOTAL DUE:", 530, ty+55, BOLD,  0.72, 2)
    put("LKR 1,29,000", 630, ty+55, BOLD,  0.72, 2)

    # ── Footer ────────────────────────────────────────────────────
    hline(H-80)
    put("Thank you for your business!  |  ocr-project@eie.ruh.ac.lk",
        50, H-55, PLAIN, 0.45, 1, 120)
    put("Terms: Payment due within 15 days of invoice date.",
        50, H-32, PLAIN, 0.42, 1, 150)

    # ── Add realistic degradation ─────────────────────────────────
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if skew_deg:
        cx, cy = W//2, H//2
        M      = cv2.getRotationMatrix2D((cx, cy), skew_deg, 1.0)
        img    = cv2.warpAffine(img, M, (W, H), borderValue=252)

    cv2.imwrite(path, img)
    return path


# ──────────────────────────────────────────────────────────────────
# Character-level recognition using saved model
# ──────────────────────────────────────────────────────────────────
class TrainedCharRecognizer:
    """Wraps a saved EMNIST model for character-level inference."""

    def __init__(self, model_path, scaler_path=None):
        self.hog     = HOGFeatureExtractor()
        self.norm    = CharacterNormalizer()
        self.scaler  = None
        self.clf     = None
        self.classes = None
        self._load(model_path, scaler_path)

    def _load(self, model_path, scaler_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        clf_type = data.get('type', 'svm')
        if clf_type == 'svm':
            self.clf = SVMClassifier()
        else:
            self.clf = KNNClassifier()
        self.clf.clf           = data['clf']
        self.clf.label_encoder = data['encoder']
        self.clf.is_trained    = True
        self.classes           = list(data['encoder'].classes_)

        if scaler_path and os.path.isfile(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        elif 'scaler' in data:
            self.scaler = data['scaler']

    def recognize_patch(self, patch):
        norm = self.norm.normalize(patch)
        feat = self.hog.extract(norm).reshape(1, -1)
        if self.scaler is not None:
            feat = self.scaler.transform(feat)
        proba = self.clf.predict_proba(feat)[0]
        label = self.clf.predict(feat)[0]
        return label, float(proba.max())

    def recognize_all(self, binary, components):
        patches = self.norm.extract_char_patches(binary, components)
        results = [self.recognize_patch(p) for p in patches]
        return results


# ──────────────────────────────────────────────────────────────────
# Full test run
# ──────────────────────────────────────────────────────────────────
def run_pipeline(image_path, model_path=None, output_format='json',
                 binarization='otsu', show_plots=True):

    print(f"\nProcessing: {image_path}")
    print("=" * 60)

    t_start = time.time()

    # ── Stage 1: Preprocessing ─────────────────────────────────────
    print("[1/4] Preprocessing…")
    prep     = Preprocessor()
    result   = prep.process(image_path, binarization=binarization, correct_skew=True)
    binary   = result['binary']
    gray     = result['gray']
    original = result['original']
    print(f"      image size: {binary.shape[1]}×{binary.shape[0]}  "
          f"| black pixel ratio: {(binary==0).mean():.3f}")

    # ── Stage 2: Layout Analysis ───────────────────────────────────
    print("[2/4] Layout analysis…")
    la      = LayoutAnalyzer()
    layout  = la.analyze(binary)
    comps   = layout['components']
    regions = layout['regions']
    print(f"      components: {len(comps)}  |  regions: {len(regions)}")

    # ── Stage 3: Character Recognition ────────────────────────────
    print("[3/4] Character recognition…")
    words     = []
    char_info = []

    if model_path and os.path.isfile(model_path):
        scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
        recognizer  = TrainedCharRecognizer(model_path, scaler_path)
        if comps:
            char_results = recognizer.recognize_all(binary, comps)
            char_labels  = [r[0] for r in char_results]
            char_confs   = [r[1] for r in char_results]
            assembler    = WordAssembler()
            words        = assembler.assemble(comps, char_labels)
            char_info    = list(zip(comps, char_labels, char_confs))
            print(f"      chars recognised: {len(char_labels)}  "
                  f"| avg confidence: {np.mean(char_confs):.3f}")
        else:
            print("      no character components found")
    else:
        # Default: enhanced Tesseract pipeline (best accuracy) on the original.
        from tesseract_setup import ensure_tesseract
        if ensure_tesseract() is not None:
            from enhanced_ocr import EnhancedOCR
            rec   = EnhancedOCR().run(image_path)
            words = rec['words']
            print(f"      enhanced OCR: {len(words)} words  "
                  f"| mean confidence {rec['mean_confidence']}  "
                  f"| psm {rec['psm_used']}  | {rec['preprocess_info']}")
        else:
            # No Tesseract available – RLSA word-block detection only.
            from main import OCRPipeline
            import warnings
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                pl = OCRPipeline(use_tesseract=False, output_format=output_format)
            words, _ = pl.recognize_text(binary, gray, layout)
            print(f"      word blocks detected: {len(words)}  "
                  f"(no Tesseract – RLSA fallback; run "
                  f"scripts/install_tesseract_local.sh)")

    # ── Stage 4: Post-Processing ───────────────────────────────────
    print("[4/4] Post-processing…")
    post   = PostProcessor()
    output = post.format_output(words, regions=regions, fmt=output_format)
    total  = time.time() - t_start
    print(f"      words: {len(words)}  |  total time: {total:.3f}s")

    # ── Save output ────────────────────────────────────────────────
    stem     = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f'{stem}_result.{output_format}')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f"\nOutput saved → {out_path}")

    # ── Visualise ──────────────────────────────────────────────────
    if show_plots:
        _visualise(original, binary, layout, comps, char_info, regions, stem)

    return words, output


def _visualise(original, binary, layout, comps, char_info, regions, stem):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original
    axes[0, 0].imshow(original if len(original.shape) == 2
                      else cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                      cmap='gray')
    axes[0, 0].set_title('1. Original Image', fontsize=11)
    axes[0, 0].axis('off')

    # Binary
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('2. Binarized + Deskewed', fontsize=11)
    axes[0, 1].axis('off')

    # RLSA smoothed
    axes[0, 2].imshow(layout['smoothed'], cmap='gray')
    axes[0, 2].set_title('3. RLSA Smoothed', fontsize=11)
    axes[0, 2].axis('off')

    # Connected components
    vis_cc = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for c in comps:
        x, y, w, h = c['bbox']
        cv2.rectangle(vis_cc, (x, y), (x+w, y+h), (0, 200, 0), 1)
    axes[1, 0].imshow(cv2.cvtColor(vis_cc, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'4. Connected Components (n={len(comps)})', fontsize=11)
    axes[1, 0].axis('off')

    # Character recognition results
    vis_rec = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if char_info:
        for comp, label, conf in char_info[:300]:
            x, y, w, h = comp['bbox']
            color = (0, int(conf*200), 0)
            cv2.rectangle(vis_rec, (x, y), (x+w, y+h), color, 1)
            if w > 8 and h > 8:
                cv2.putText(vis_rec, str(label), (x, y-1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 0, 0), 1)
    axes[1, 1].imshow(cv2.cvtColor(vis_rec, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('5. Character Recognition', fontsize=11)
    axes[1, 1].axis('off')

    # Region classification
    type_colors = {'header_footer': (255, 140, 0), 'table': (0, 180, 255),
                   'text': (0, 200, 0), 'image': (180, 0, 255), 'unknown': (128, 128, 128)}
    vis_reg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for r in regions:
        x, y, w, h = r['bbox']
        c = type_colors.get(r['type'], (128, 128, 128))
        cv2.rectangle(vis_reg, (x, y), (x+w, y+h), c[::-1], 2)
        cv2.putText(vis_reg, r['type'][:3], (x+3, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c[::-1], 1)
    axes[1, 2].imshow(cv2.cvtColor(vis_reg, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'6. Region Classification (n={len(regions)})', fontsize=11)
    axes[1, 2].axis('off')

    plt.suptitle(f'OCR Pipeline Results – {stem}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    vis_path = os.path.join(OUTPUT_DIR, f'{stem}_visualisation.png')
    plt.savefig(vis_path, dpi=120, bbox_inches='tight')
    print(f"Visualisation saved → {vis_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Test OCR pipeline on a document image')
    parser.add_argument('--image',    default=None,
                        help='Path to input image (JPG, PNG, BMP, TIFF)')
    parser.add_argument('--generate', action='store_true',
                        help='Generate a high-quality synthetic test invoice')
    parser.add_argument('--model',    default=None,
                        help='Path to trained classifier model (data/models/svm_emnist.pkl)')
    parser.add_argument('--format',   choices=['json', 'txt', 'csv'], default='json',
                        help='Output format (default: json)')
    parser.add_argument('--binarization', choices=['otsu', 'adaptive', 'sauvola'],
                        default='otsu', help='Binarization method (default: otsu)')
    parser.add_argument('--no-plot',  action='store_true',
                        help='Skip matplotlib visualisation')
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if model_path is None:
        default_svm = os.path.join(ROOT, 'data', 'models', 'svm_emnist.pkl')
        if os.path.isfile(default_svm):
            model_path = default_svm
            print(f"[INFO] Auto-loaded model: {model_path}")

    # Resolve image
    if args.generate or args.image is None:
        test_img_path = os.path.join(ROOT, 'data', 'test', 'synthetic_invoice.png')
        os.makedirs(os.path.dirname(test_img_path), exist_ok=True)
        print("Generating high-quality synthetic invoice…")
        generate_test_invoice(test_img_path, skew_deg=2.5, noise_sigma=10)
        print(f"  Saved → {test_img_path}")
        image_path = test_img_path
    else:
        image_path = args.image

    if not os.path.isfile(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    words, output = run_pipeline(
        image_path,
        model_path=model_path,
        output_format=args.format,
        binarization=args.binarization,
        show_plots=not args.no_plot,
    )

    print("\n── Recognised words ────────────────────────────────────────")
    print(' '.join(words[:60]))
    if len(words) > 60:
        print(f"  … ({len(words) - 60} more)")


if __name__ == '__main__':
    main()
