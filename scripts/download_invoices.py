"""
Download a few real, freely-licensed invoice/receipt images for qualitative
testing. Sources are Wikimedia Commons (public domain / CC). These have NO
ground truth, so they are for eyeballing only – use the synthetic dataset
(scripts/make_invoice_dataset.py) for quantitative CER/WER.

Usage:
    python scripts/download_invoices.py
"""

import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "data", "invoices", "real")

# name -> (url, attribution)
SAMPLES = {
    "performa_invoice.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/d/d8/Performa_invoice_merchandising.jpg",
        "Wikimedia Commons, 'Performa invoice merchandising'"),
    "uniform_invoice_taiwan.png": (
        "https://upload.wikimedia.org/wikipedia/commons/f/f7/Uniform-Invoice_Taiwan.png",
        "Wikimedia Commons, 'Uniform-Invoice Taiwan'"),
    "bandcamp_receipt.png": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/"
        "Example_of_digital_receipt_at_Bandcamp.webp/960px-"
        "Example_of_digital_receipt_at_Bandcamp.webp.png",
        "Wikimedia Commons, 'Example of digital receipt at Bandcamp'"),
}


def main():
    os.makedirs(DST, exist_ok=True)
    attrib = []
    for name, (url, credit) in SAMPLES.items():
        dst = os.path.join(DST, name)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "OCR-edu-project/1.0 (academic)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(dst, "wb") as f:
                f.write(data)
            attrib.append(f"{name}: {credit} ({url})")
            print(f"OK  {name:30s} {len(data)//1024:5d} KB")
        except Exception as e:
            print(f"ERR {name:30s} {e}")

    with open(os.path.join(DST, "ATTRIBUTION.txt"), "w", encoding="utf-8") as f:
        f.write("Real invoice samples – sources and licences\n")
        f.write("=" * 44 + "\n\n")
        f.write("\n".join(attrib) + "\n")
    print(f"\nSaved to {DST} (see ATTRIBUTION.txt)")


if __name__ == "__main__":
    main()
