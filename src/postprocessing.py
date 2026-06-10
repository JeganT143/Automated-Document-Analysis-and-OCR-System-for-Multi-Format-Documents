"""
Post-processing for the document OCR pipeline.

Two responsibilities:

  1. SafePostProcessor – conservative clean-up of raw OCR text. It only fixes
     things that are unambiguously formatting noise (words split across a line
     break, repeated spaces, "$ 12.50" -> "$12.50"). It deliberately NEVER
     rewrites words against a dictionary: a naive spell-corrector turns valid
     tokens such as "Kandy" or "LKR" into garbage and *lowers* accuracy.

  2. OutputFormatter – render the structured pipeline result as JSON / TXT /
     CSV, plus best-effort extraction of key invoice fields.
"""

import re
import io
import csv
import json


# ---------------------------------------------------------------------------
# Safe text clean-up
# ---------------------------------------------------------------------------
class SafePostProcessor:
    """Conservative clean-up that never rewrites numbers or unknown words."""

    def process(self, text):
        # join words split across a line break:  "inv-\noice" -> "invoice"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # collapse runs of spaces/tabs but keep the line structure
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        text = "\n".join(lines)
        # currency glyph spacing artefacts:  "$ 12.50" -> "$12.50"
        text = re.sub(r"([$£€])\s+(\d)", r"\1\2", text)
        return text


# ---------------------------------------------------------------------------
# Invoice field extraction (content-level structuring)
# ---------------------------------------------------------------------------
def extract_fields(text):
    """Best-effort extraction of key invoice fields from OCR text.

    Returns a dict that may contain ``invoice_no``, ``total`` and ``date``.
    Used both for the structured output and for field-level evaluation.
    """
    t = text.replace("\n", " ")
    fields = {}

    m = re.search(r"INV[-\s]?(\d{4})[-\s]?(\d{3,4})", t, re.I)
    if m:
        fields["invoice_no"] = f"INV-{m.group(1)}-{m.group(2)}"

    # grand total: amount after a standalone "TOTAL" (not "subtotal").
    # Take the last such match – the grand total comes last on an invoice.
    totals = re.findall(r"(?<![A-Za-z])total\b[^0-9]{0,12}([0-9][0-9,]*\.\d{2})",
                        t, re.I)
    if totals:
        fields["total"] = totals[-1].replace(",", "")

    m = re.search(r"\b(\d{1,2})\s*([A-Za-z]{3})\s*(20\d{2})\b", t)
    if m:
        fields["date"] = f"{int(m.group(1)):02d} {m.group(2).title()} {m.group(3)}"

    return fields


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------
class OutputFormatter:
    """Render the pipeline result dict to JSON / plain-text / CSV."""

    def to_json(self, result):
        return json.dumps(self._clean(result), indent=2, ensure_ascii=False)

    def to_txt(self, result):
        return result.get("text", "")

    def to_csv(self, result):
        """One row per recognised word (index, word)."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["index", "word"])
        for i, w in enumerate(result.get("words", [])):
            writer.writerow([i, w])
        return buf.getvalue()

    def render(self, result, fmt="json"):
        fmt = fmt.lower()
        if fmt == "json":
            return self.to_json(result)
        if fmt == "txt":
            return self.to_txt(result)
        if fmt == "csv":
            return self.to_csv(result)
        raise ValueError(f"Unknown output format: {fmt}")

    @staticmethod
    def _clean(result):
        """Drop non-serialisable entries (e.g. the processed image array)."""
        return {k: v for k, v in result.items()
                if k not in ("processed_image",)}

    def save(self, content, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
