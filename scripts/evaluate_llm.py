"""
Compare OpenRouter models on structured invoice-field extraction.

Runs each labelled invoice's OCR text through src.llm.extract.extract() for
every model in --models, measuring field-level accuracy against the exact
synthetic ground truth, wall-clock latency, and an *estimated* token cost
(chars/4 heuristic x OpenRouter's published per-model pricing — the quick
harness here doesn't wire through exact token usage, so this is a relative
comparison, not a billing figure).

Requires OPENROUTER_API_KEY — this script makes real, small, cheap API calls
(a handful of short invoices x a few models = a few cents total).

Usage:
    OPENROUTER_API_KEY=... python scripts/evaluate_llm.py
    OPENROUTER_API_KEY=... python scripts/evaluate_llm.py --n 6 --report RESULTS.md
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.llm import client as llm_client
from src.llm import extract as llm_extract
from src.pipeline import DocumentOCRPipeline

LABELLED = os.path.join(ROOT, "data", "invoices", "synthetic")


def load_labelled(d, n=None):
    pairs = []
    for img_path in sorted(glob.glob(os.path.join(d, "*.png"))):
        stem = os.path.splitext(img_path)[0]
        struct_path = stem + ".json"
        if not os.path.isfile(struct_path):
            continue
        with open(struct_path, encoding="utf-8") as f:
            struct = json.load(f)
        pairs.append((img_path, struct))
    return pairs[:n] if n else pairs


def _estimate_tokens(text):
    return max(1, len(text) // 4)


def field_accuracy(struct, data):
    """data: the `.data` (ExtractedInvoice) of an ExtractionResult, as a dict."""
    res = {}
    vendor = (data.get("vendor") or "").lower()
    res["vendor"] = bool(vendor) and struct["company"][0].lower() in vendor
    res["invoice_no"] = (data.get("invoice_no") or "").upper() == struct["invoice_no"].upper()
    res["date"] = (data.get("date") or "").strip() == struct["date"].strip()
    total = data.get("total")
    res["total"] = total is not None and abs(float(total) - round(struct["total"], 2)) < 0.01
    return res


def evaluate_model(model_id, ocr_texts_and_structs, cost_in, cost_out, verbose=False):
    fields = {"vendor": 0, "invoice_no": 0, "date": 0, "total": 0}
    latencies, est_costs, fallbacks = [], [], 0
    n = len(ocr_texts_and_structs)

    for ocr_text, struct in ocr_texts_and_structs:
        t0 = time.time()
        result = llm_extract.extract(ocr_text, model=model_id)
        dt = time.time() - t0
        latencies.append(dt)

        if result.source != "llm":
            fallbacks += 1

        data = result.data.model_dump()
        fa = field_accuracy(struct, data)
        for k in fields:
            fields[k] += int(fa[k])

        in_tok = _estimate_tokens(ocr_text) + 150  # + system-prompt overhead
        out_tok = _estimate_tokens(json.dumps(data))
        est_costs.append((in_tok / 1e6) * cost_in + (out_tok / 1e6) * cost_out)

        if verbose:
            print(f"    source={result.source:14s} {dt:5.2f}s  "
                  f"fields_ok={sum(fa.values())}/4")

    return {
        "model": model_id,
        "n": n,
        "fallback_rate": fallbacks / n if n else 0.0,
        "field_accuracy": {k: fields[k] / n if n else 0.0 for k in fields},
        "overall_field_accuracy": sum(fields.values()) / (n * len(fields)) if n else 0.0,
        "mean_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "est_cost_per_doc_usd": float(np.mean(est_costs)) if est_costs else 0.0,
    }


def print_row(s):
    fa = s["field_accuracy"]
    print(f"  {s['model']:32s} overall={s['overall_field_accuracy']*100:5.1f}%  "
          f"vendor={fa['vendor']*100:5.1f}% inv_no={fa['invoice_no']*100:5.1f}% "
          f"date={fa['date']*100:5.1f}% total={fa['total']*100:5.1f}%  "
          f"latency={s['mean_latency_s']:.2f}s  "
          f"~${s['est_cost_per_doc_usd']*1000:.3f}/1000 docs  "
          f"fallback={s['fallback_rate']*100:.0f}%")


def append_report(path, rows):
    L = ["\n## LLM structured-extraction comparison (OpenRouter)\n"]
    L.append(f"Field-level accuracy of `src/llm/extract.py` against the same exact "
             f"synthetic ground truth as above, across {len(rows)} models, on "
             f"{rows[0]['n'] if rows else 0} labelled invoices. Cost is an "
             "*estimate* (chars/4 token heuristic x OpenRouter's published "
             "per-model pricing), not a billed figure.\n")
    L.append("| Model | Overall | Vendor | Invoice No | Date | Total | Mean latency | Est. cost/1000 docs |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for s in rows:
        fa = s["field_accuracy"]
        L.append(f"| {s['model']} | {s['overall_field_accuracy']*100:.1f}% | "
                 f"{fa['vendor']*100:.0f}% | {fa['invoice_no']*100:.0f}% | "
                 f"{fa['date']*100:.0f}% | {fa['total']*100:.0f}% | "
                 f"{s['mean_latency_s']:.2f}s | ${s['est_cost_per_doc_usd']*1000:.3f} |")
    L.append("")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\nAppended LLM comparison to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default=LABELLED)
    ap.add_argument("--n", type=int, default=8, help="number of invoices to test per model")
    ap.add_argument("--models", nargs="*", default=None,
                    help="OpenRouter model ids (default: src.llm.client.AVAILABLE_MODELS)")
    ap.add_argument("--report", default=None, help="markdown file to APPEND results to")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not llm_client.is_configured():
        print("ERROR: OPENROUTER_API_KEY not set. This script makes real "
              "(small, cheap) API calls and needs a key.")
        sys.exit(1)

    pairs = load_labelled(args.labelled, n=args.n)
    if not pairs:
        print(f"No labelled data in {args.labelled} — generate it with:\n"
              "  python scripts/make_invoice_dataset.py --n 12 --degrade scan")
        sys.exit(1)

    print(f"OCR-ing {len(pairs)} labelled invoices once, then re-using the text "
          "for every model...")
    pipeline = DocumentOCRPipeline()
    ocr_texts_and_structs = []
    for img_path, struct in pairs:
        text = pipeline.image_to_text(img_path)
        ocr_texts_and_structs.append((text, struct))

    model_meta = {m["id"]: m for m in llm_client.AVAILABLE_MODELS}
    model_ids = args.models or list(model_meta.keys())

    rows = []
    for model_id in model_ids:
        meta = model_meta.get(model_id, {"cost_per_1m_in": 0.0, "cost_per_1m_out": 0.0})
        print(f"\nEvaluating {model_id} ...")
        s = evaluate_model(model_id, ocr_texts_and_structs,
                           meta["cost_per_1m_in"], meta["cost_per_1m_out"],
                           verbose=args.verbose)
        print_row(s)
        rows.append(s)

    if args.report:
        append_report(args.report, rows)


if __name__ == "__main__":
    main()
