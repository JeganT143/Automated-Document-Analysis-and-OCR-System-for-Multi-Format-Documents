"""
OCR evaluation metrics.

Provides the standard text-recognition metrics plus a couple that are robust
to reading-order differences (which matter for multi-column invoices):

    cer / wer        – character / word error rate (lower is better)
    token_recall     – fraction of ground-truth tokens recovered anywhere
    token_f1         – precision/recall/F1 over normalised tokens
    field_accuracy   – did we recover key invoice fields (no., total, date)?
"""

import re
from collections import Counter

from .postprocessing import extract_fields  # field extraction lives with post-processing


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------
def levenshtein(a, b):
    """Token- or character-level Levenshtein distance over any sequence."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_text(text, keep_case=False):
    """Collapse whitespace; optionally lowercase. Used before CER/WER."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text if keep_case else text.lower()


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,:/@%&-]*[A-Za-z0-9]|[A-Za-z0-9]")


def tokens(text):
    """Lower-cased content tokens with surrounding punctuation stripped."""
    out = []
    for t in _TOKEN_RE.findall(text.lower()):
        t = t.strip(".,:/@%&-")
        if t:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def cer(reference, hypothesis):
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(reference, hypothesis):
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def token_prf(reference, hypothesis):
    """Multiset precision / recall / F1 over normalised tokens (order-free)."""
    ref = Counter(tokens(reference))
    hyp = Counter(tokens(hypothesis))
    if not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "matched": 0, "n_ref": 0, "n_hyp": sum(hyp.values())}
    matched = sum((ref & hyp).values())
    n_ref = sum(ref.values())
    n_hyp = sum(hyp.values())
    recall = matched / n_ref if n_ref else 0.0
    precision = matched / n_hyp if n_hyp else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "matched": matched, "n_ref": n_ref, "n_hyp": n_hyp}


def token_recall(reference, hypothesis):
    return token_prf(reference, hypothesis)["recall"]


# ---------------------------------------------------------------------------
# Invoice field accuracy (uses extract_fields from postprocessing)
# ---------------------------------------------------------------------------
def field_accuracy(gt_struct, ocr_text):
    """Compare extracted fields against the structured ground truth."""
    got = extract_fields(ocr_text)
    res = {}

    res["invoice_no"] = (got.get("invoice_no", "").upper()
                         == gt_struct["invoice_no"].upper())

    gt_total = f"{gt_struct['total']:.2f}"
    res["total"] = got.get("total", "") == gt_total

    res["date"] = got.get("date", "") == gt_struct["date"]
    return res


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def evaluate_pair(reference, hypothesis):
    prf = token_prf(reference, hypothesis)
    return {
        "cer": cer(reference, hypothesis),
        "wer": wer(reference, hypothesis),
        "token_recall": prf["recall"],
        "token_precision": prf["precision"],
        "token_f1": prf["f1"],
    }
