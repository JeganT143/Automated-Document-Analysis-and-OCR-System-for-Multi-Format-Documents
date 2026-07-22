"""
Generate a labelled invoice dataset for OCR evaluation.

Invoices are rendered from structured data with REAL TrueType fonts (via PIL),
so they look like genuine printed invoices and we still know the EXACT
ground-truth text. For each sample we write:

    data/invoices/synthetic/inv_XX.png      – the (optionally degraded) image
    data/invoices/synthetic/inv_XX.gt.txt   – ground-truth text, reading order
    data/invoices/synthetic/inv_XX.json     – structured fields (for field eval)

Usage:
    python scripts/make_invoice_dataset.py --n 10 --degrade scan
    python scripts/make_invoice_dataset.py --n 6  --degrade clean
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "invoices", "synthetic")

FONT_DIR = "/usr/share/fonts/truetype"
# (regular, bold) families that look like real invoice typefaces
FONT_FAMILIES = [
    (f"{FONT_DIR}/liberation/LiberationSans-Regular.ttf",
     f"{FONT_DIR}/liberation/LiberationSans-Bold.ttf"),
    (f"{FONT_DIR}/dejavu/DejaVuSans.ttf",
     f"{FONT_DIR}/dejavu/DejaVuSans-Bold.ttf"),
    (f"{FONT_DIR}/ubuntu/Ubuntu-R.ttf",
     f"{FONT_DIR}/ubuntu/Ubuntu-B.ttf"),
    (f"{FONT_DIR}/dejavu/DejaVuSerif.ttf",
     f"{FONT_DIR}/dejavu/DejaVuSerif-Bold.ttf"),
]
FONT_FAMILIES = [(r, b) for r, b in FONT_FAMILIES
                 if os.path.isfile(r) and os.path.isfile(b)]

COMPANIES = [
    ("Northwind Traders", "1420 Commerce Street", "Seattle, WA 98101"),
    ("Globex Corporation", "88 Industrial Avenue", "Springfield, IL 62704"),
    ("Acme Supplies Ltd", "27 Riverside Walk", "Manchester, M3 4LZ"),
    ("Ceylon Tea Exports", "14 Galle Road", "Colombo 03, Sri Lanka"),
    ("Initech Systems", "500 Technology Park", "Austin, TX 73301"),
    ("Umbrella Logistics", "9 Harbour Quay", "Liverpool, L3 1BW"),
    ("Stark Industrial", "200 Park Avenue", "New York, NY 10166"),
    ("Wayne Enterprises", "1007 Mountain Drive", "Gotham, NJ 07001"),
    ("Hooli Analytics", "1 Innovation Loop", "Palo Alto, CA 94301"),
    ("Soylent Foods Co", "73 Greenfield Road", "Portland, OR 97201"),
]

CUSTOMERS = [
    ("Blue Ocean Retail", "45 Market Square", "Boston, MA 02108"),
    ("Sunrise Hardware", "12 Kandy Road", "Kurunegala, Sri Lanka"),
    ("Pioneer Electricals", "301 Edison Way", "Detroit, MI 48201"),
    ("Greenleaf Grocers", "78 Orchard Lane", "Bristol, BS1 5TR"),
    ("Summit Outfitters", "640 Alpine Road", "Denver, CO 80202"),
    ("Harbor Freight LLC", "55 Dockside Blvd", "Tampa, FL 33602"),
]

ITEMS = [
    ("Stainless steel bolts M8", 4.50), ("Copper wire spool 10m", 12.00),
    ("LED panel light 40W", 28.75), ("PVC pipe 2 inch", 6.40),
    ("Circuit breaker 32A", 18.90), ("Ethernet cable Cat6", 9.25),
    ("Cordless drill 18V", 89.99), ("Safety helmet yellow", 14.50),
    ("Paint roller set", 7.80), ("Aluminium ladder 6ft", 64.00),
    ("Digital multimeter", 32.40), ("Welding gloves pair", 11.20),
    ("Hydraulic jack 2 ton", 45.60), ("Insulation tape roll", 2.95),
    ("Solar charge controller", 54.30), ("Brass hinges box", 8.65),
]

CURRENCIES = ["$", "USD ", "LKR ", "GBP ", "EUR "]


def _fmt_money(sym, amount):
    return f"{sym}{amount:,.2f}"


def build_invoice(rng):
    """Sample structured invoice content."""
    comp = COMPANIES[rng.integers(len(COMPANIES))]
    cust = CUSTOMERS[rng.integers(len(CUSTOMERS))]
    sym = CURRENCIES[rng.integers(len(CURRENCIES))]
    inv_no = f"INV-{rng.integers(2023, 2027)}-{rng.integers(1000, 9999)}"
    day = int(rng.integers(1, 28))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mi = int(rng.integers(12))
    year = 2026
    date = f"{day:02d} {months[mi]} {year}"
    due = f"{day:02d} {months[(mi + 1) % 12]} {year}"

    n_items = int(rng.integers(3, 7))
    chosen = rng.choice(len(ITEMS), size=n_items, replace=False)
    rows = []
    subtotal = 0.0
    for i, idx in enumerate(chosen, 1):
        desc, unit = ITEMS[int(idx)]
        qty = int(rng.integers(1, 12))
        amount = round(qty * unit, 2)
        subtotal += amount
        rows.append({"n": i, "desc": desc, "qty": qty,
                     "unit": round(unit, 2), "amount": amount})
    subtotal = round(subtotal, 2)
    tax_rate = int(rng.choice([0, 5, 8, 12, 15]))
    tax = round(subtotal * tax_rate / 100.0, 2)
    total = round(subtotal + tax, 2)

    return {
        "company": comp, "customer": cust, "currency": sym,
        "invoice_no": inv_no, "date": date, "due": due,
        "rows": rows, "subtotal": subtotal,
        "tax_rate": tax_rate, "tax": tax, "total": total,
        "font_family": int(rng.integers(len(FONT_FAMILIES))),
    }


def _reading_order(items, row_tol=14):
    """Turn (y, x, text) items into reading-order text lines.

    Groups items whose top-y is within row_tol into one visual line and sorts
    left-to-right, so the ground truth matches how OCR scans the page (vital
    for fair CER/WER on multi-column headers)."""
    items = sorted(items, key=lambda it: (it[0], it[1]))
    lines, row, row_y = [], [], None
    for y, x, text in items:
        if row_y is None or abs(y - row_y) <= row_tol:
            row.append((x, text))
            row_y = y if row_y is None else row_y
        else:
            lines.append(" ".join(t for _, t in sorted(row)))
            row, row_y = [(x, text)], y
    if row:
        lines.append(" ".join(t for _, t in sorted(row)))
    return lines


def render_invoice(data):
    """Render the invoice with real fonts; return (gray_uint8, gt_lines)."""
    W, H = 794, 1123
    img = Image.new("L", (W, H), 253)
    draw = ImageDraw.Draw(img)
    gt_items = []   # (y, x, text) collected in reading order at the end

    reg_path, bold_path = FONT_FAMILIES[data["font_family"]]

    def font(size, bold=False):
        return ImageFont.truetype(bold_path if bold else reg_path, size)

    def put(text, x, y, size=15, bold=False, fill=0, anchor="la", record=True):
        draw.text((x, y), text, font=font(size, bold), fill=fill, anchor=anchor)
        if record and text.strip():
            gt_items.append((y, x, text))

    def hline(y, x1=50, x2=744, t=1):
        draw.line([(x1, y), (x2, y)], fill=0, width=t)

    comp, cust, sym = data["company"], data["customer"], data["currency"]

    # Header -----------------------------------------------------------------
    put("INVOICE", 50, 40, size=40, bold=True)
    put(comp[0], 50, 95, size=18, bold=True)
    put(comp[1], 50, 120, size=14)
    put(comp[2], 50, 140, size=14)

    put(f"Invoice No: {data['invoice_no']}", 470, 95, size=14)
    put(f"Date: {data['date']}", 470, 118, size=14)
    put(f"Due Date: {data['due']}", 470, 141, size=14)
    hline(175, t=2)

    # Bill-to ----------------------------------------------------------------
    put("Bill To:", 50, 192, size=15, bold=True)
    put(cust[0], 50, 216, size=14)
    put(cust[1], 50, 236, size=14)
    put(cust[2], 50, 256, size=14)
    hline(290)

    # Table header (well-separated columns, like a real invoice) -------------
    COLX = [60, 105, 440, 520, 700]   # Unit left-aligned, Amount right-aligned
    TH = 300
    draw.rectangle([50, TH - 4, 744, TH + 22], fill=228)
    for txt, x, anc in zip(["No.", "Description", "Qty", "Unit", "Amount"],
                           COLX, ["la", "la", "ra", "la", "ra"], strict=True):
        put(txt, x, TH, size=14, bold=True, anchor=anc)
    hline(TH + 24)

    # Table rows -------------------------------------------------------------
    ry = TH + 24
    for r in data["rows"]:
        ry += 30
        put(str(r["n"]), COLX[0], ry, size=14)
        put(r["desc"], COLX[1], ry, size=14)
        put(str(r["qty"]), COLX[2], ry, size=14, anchor="ra")
        put(_fmt_money(sym, r["unit"]), COLX[3], ry, size=14)
        put(_fmt_money(sym, r["amount"]), COLX[4], ry, size=14, anchor="ra")
    hline(ry + 28)

    # Totals -----------------------------------------------------------------
    ty = ry + 56
    put("Subtotal:", 470, ty, size=14)
    put(_fmt_money(sym, data["subtotal"]), 690, ty, size=14, anchor="ra")
    put(f"Tax ({data['tax_rate']}%):", 470, ty + 24, size=14)
    put(_fmt_money(sym, data["tax"]), 690, ty + 24, size=14, anchor="ra")
    hline(ty + 46, x1=460)
    put("TOTAL:", 470, ty + 56, size=20, bold=True)
    put(_fmt_money(sym, data["total"]), 690, ty + 56, size=20, bold=True, anchor="ra")

    # Footer -----------------------------------------------------------------
    put("Thank you for your business", 50, H - 95, size=14)
    put("Payment due within 30 days of invoice date.", 50, H - 70, size=13)
    put("Questions? accounts@example.com", 50, H - 48, size=13)

    return np.array(img), _reading_order(gt_items)


def degrade(img, rng, level="scan"):
    """Apply realistic scan-like degradation."""
    if level == "clean":
        return img
    out = img.copy()
    if level in ("scan", "heavy"):
        out = cv2.GaussianBlur(out, (3, 3), 0.6)
        sigma = 6 if level == "scan" else 14
        noise = rng.normal(0, sigma, out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        max_skew = 2.0 if level == "scan" else 4.0
        angle = float(rng.uniform(-max_skew, max_skew))
        h, w = out.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderValue=253,
                             flags=cv2.INTER_CUBIC)
    if level == "heavy":
        h, w = out.shape
        grad = np.tile(np.linspace(0.82, 1.06, w), (h, 1))
        out = np.clip(out.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of invoices")
    ap.add_argument("--degrade", choices=["clean", "scan", "heavy"], default="scan")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    if not FONT_FAMILIES:
        print("ERROR: no TrueType fonts found under", FONT_DIR)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.n):
        data = build_invoice(rng)
        img, gt = render_invoice(data)
        img = degrade(img, rng, level=args.degrade)
        stem = f"inv_{i:02d}"
        cv2.imwrite(os.path.join(args.out, stem + ".png"), img)
        with open(os.path.join(args.out, stem + ".gt.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(gt))
        with open(os.path.join(args.out, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    print(f"Wrote {args.n} invoices ({args.degrade}, {len(FONT_FAMILIES)} fonts) to {args.out}")


if __name__ == "__main__":
    main()
