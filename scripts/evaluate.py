"""
Evaluate OCR accuracy over the labelled invoice dataset.

Modes (recognition strategies) are pluggable so we can quantify the gain of
each enhancement:

    wordblock    – current default fallback (RLSA word blobs -> "[word]")
    tess_naive   – Tesseract on the existing Otsu-binarised, deskewed image
    tess_raw     – Tesseract on raw grayscale, --psm 6
    enhanced     – the enhanced pipeline (src/enhanced_ocr.py)

Usage:
    python scripts/evaluate.py --mode enhanced
    python scripts/evaluate.py --mode tess_naive --dataset data/invoices/synthetic
    python scripts/evaluate.py --compare wordblock tess_naive tess_raw enhanced
"""

import os
import sys
import glob
import json
import argparse
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2
import numpy as np

import evaluation as ev

DATASET = os.path.join(ROOT, "data", "invoices", "synthetic")


# ---------------------------------------------------------------------------
# Recognition modes -> each returns plain text for an image path
# ---------------------------------------------------------------------------
def rec_wordblock(path):
    from main import OCRPipeline
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pl = OCRPipeline(use_tesseract=False)
        res = pl.run(path)
    return " ".join(str(w) for w in res["words"])


def rec_tess_naive(path):
    from tesseract_setup import ensure_tesseract
    from preprocessing import Preprocessor
    import pytesseract
    ensure_tesseract()
    prep = Preprocessor().process(path, binarization="otsu", correct_skew=True)
    return pytesseract.image_to_string(prep["binary"], config="--oem 3 --psm 6")


def rec_tess_raw(path):
    from tesseract_setup import ensure_tesseract
    import pytesseract
    ensure_tesseract()
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return pytesseract.image_to_string(gray, config="--oem 3 --psm 6")


def rec_enhanced(path):
    from enhanced_ocr import EnhancedOCR
    return EnhancedOCR().image_to_text(path)


MODES = {
    "wordblock": rec_wordblock,
    "tess_naive": rec_tess_naive,
    "tess_raw": rec_tess_raw,
    "enhanced": rec_enhanced,
}


# ---------------------------------------------------------------------------
def load_dataset(dataset):
    pairs = []
    for img_path in sorted(glob.glob(os.path.join(dataset, "*.png"))):
        stem = os.path.splitext(img_path)[0]
        gt_path = stem + ".gt.txt"
        if not os.path.isfile(gt_path):
            continue
        with open(gt_path, encoding="utf-8") as f:
            gt = f.read()
        struct = None
        if os.path.isfile(stem + ".json"):
            with open(stem + ".json", encoding="utf-8") as f:
                struct = json.load(f)
        pairs.append((img_path, gt, struct))
    return pairs


def run_mode(mode, pairs, verbose=False):
    fn = MODES[mode]
    agg = {"cer": [], "wer": [], "token_recall": [],
           "token_precision": [], "token_f1": []}
    field_hits = {"invoice_no": 0, "total": 0, "date": 0}
    n_fields = 0
    t0 = time.time()

    for img_path, gt, struct in pairs:
        try:
            hyp = fn(img_path)
        except Exception as e:
            hyp = ""
            if verbose:
                print(f"  [{os.path.basename(img_path)}] ERROR: {e}")
        m = ev.evaluate_pair(gt, hyp)
        for k in agg:
            agg[k].append(m[k])
        if struct:
            fa = ev.field_accuracy(struct, hyp)
            for k in field_hits:
                field_hits[k] += int(fa[k])
            n_fields += 1
        if verbose:
            print(f"  {os.path.basename(img_path):14s} "
                  f"CER={m['cer']:.3f} WER={m['wer']:.3f} "
                  f"recall={m['token_recall']:.3f}")

    elapsed = time.time() - t0
    summary = {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}
    summary["fields"] = {k: (field_hits[k] / n_fields if n_fields else 0.0)
                         for k in field_hits}
    summary["sec_per_doc"] = elapsed / max(1, len(pairs))
    return summary


def print_summary(mode, s):
    print(f"\n=== {mode} ===")
    print(f"  CER (char err)   : {s['cer']*100:6.2f}%   (lower better)")
    print(f"  WER (word err)   : {s['wer']*100:6.2f}%   (lower better)")
    print(f"  Token recall     : {s['token_recall']*100:6.2f}%   (higher better)")
    print(f"  Token precision  : {s['token_precision']*100:6.2f}%")
    print(f"  Token F1         : {s['token_f1']*100:6.2f}%")
    print(f"  Field accuracy   : inv_no={s['fields']['invoice_no']*100:.0f}%  "
          f"total={s['fields']['total']*100:.0f}%  date={s['fields']['date']*100:.0f}%")
    print(f"  Speed            : {s['sec_per_doc']:.2f} s/doc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), default="enhanced")
    ap.add_argument("--compare", nargs="+", choices=list(MODES))
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pairs = load_dataset(args.dataset)
    if not pairs:
        print(f"No labelled data found in {args.dataset}. "
              f"Run scripts/make_invoice_dataset.py first.")
        sys.exit(1)
    print(f"Loaded {len(pairs)} labelled invoices from {args.dataset}")

    modes = args.compare if args.compare else [args.mode]
    results = {}
    for mode in modes:
        results[mode] = run_mode(mode, pairs, verbose=args.verbose)
        print_summary(mode, results[mode])

    if len(modes) > 1:
        print("\n" + "=" * 64)
        print(f"{'mode':<12} {'CER%':>8} {'WER%':>8} {'tok-recall%':>12} {'tok-F1%':>9} {'s/doc':>7}")
        print("-" * 64)
        for mode in modes:
            s = results[mode]
            print(f"{mode:<12} {s['cer']*100:8.2f} {s['wer']*100:8.2f} "
                  f"{s['token_recall']*100:12.2f} {s['token_f1']*100:9.2f} "
                  f"{s['sec_per_doc']:7.2f}")


if __name__ == "__main__":
    main()
