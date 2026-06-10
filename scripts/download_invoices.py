"""
Download real, freely-licensed document images (invoices / receipts / forms /
letters) from Wikimedia Commons for qualitative OCR testing.

It queries the Commons API for files in document-related categories, downloads
a web-sized rendering of each, validates it with OpenCV and records source +
licence in ATTRIBUTION.txt. These images have NO ground truth – use them with
scripts/evaluate.py's batch mode (confidence-based), and the synthetic labelled
set for rigorous CER/WER.

Usage:
    python scripts/download_invoices.py            # 30 images
    python scripts/download_invoices.py --n 40
"""

import os
import sys
import time
import argparse

import numpy as np
import cv2
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "data", "invoices", "real")

API = "https://commons.wikimedia.org/w/api.php"
UA = "OCR-edu-project/1.0 (academic coursework; contact: student@example.edu)"

# Document-heavy categories, tried in order until we have enough images.
CATEGORIES = [
    "Invoices",
    "Receipts",
    "Commercial documents",
    "Forms (documents)",
    "Letters (correspondence)",
    "Bills of sale",
]
ALLOWED_MIME = {"image/jpeg", "image/png", "image/tiff"}


def fetch_category(category, limit=80, thumbwidth=1600):
    """Yield (title, thumburl, page_url, licence) for files in a category."""
    params = {
        "action": "query", "format": "json",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}", "gcmtype": "file", "gcmlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata", "iiurlwidth": thumbwidth,
    }
    try:
        r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
    except Exception as e:
        print(f"  [api] {category}: {e}")
        return
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        mime = info.get("mime", "")
        thumb = info.get("thumburl")
        if not thumb or mime not in ALLOWED_MIME:
            continue
        meta = info.get("extmetadata", {})
        licence = meta.get("LicenseShortName", {}).get("value", "see source")
        yield page.get("title", "File:unknown"), thumb, info.get("descriptionurl", ""), licence


def safe_name(title, idx):
    base = title.split(":", 1)[-1]
    base = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
    base = base[:48].strip("_") or "doc"
    if not base.lower().endswith((".jpg", ".jpeg", ".png")):
        base += ".jpg"
    return f"{idx:02d}_{base}"


def download_valid(url):
    """Download and validate as a decodable image; return bytes or None."""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=90)
        r.raise_for_status()
        data = r.content
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 80:
            return None
        return data
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of images (default 30)")
    args = ap.parse_args()

    os.makedirs(DST, exist_ok=True)
    saved, attribution, seen = 0, [], set()

    for category in CATEGORIES:
        if saved >= args.n:
            break
        print(f"Category: {category}")
        for title, thumb, page_url, licence in fetch_category(category):
            if saved >= args.n:
                break
            if title in seen:
                continue
            seen.add(title)
            data = download_valid(thumb)
            if data is None:
                continue
            name = safe_name(title, saved)
            with open(os.path.join(DST, name), "wb") as f:
                f.write(data)
            attribution.append(f"{name}\t{title}\t{licence}\t{page_url}")
            saved += 1
            print(f"  OK  {name:34s} {len(data)//1024:5d} KB  [{licence}]")
            time.sleep(0.3)  # be polite to the API

    with open(os.path.join(DST, "ATTRIBUTION.txt"), "w", encoding="utf-8") as f:
        f.write("Real document images from Wikimedia Commons\n")
        f.write("file\tsource_title\tlicence\tpage_url\n")
        f.write("\n".join(attribution) + "\n")

    print(f"\nSaved {saved}/{args.n} images to {DST}")
    if saved < args.n:
        print("  (fewer than requested – some categories were exhausted or blocked)")


if __name__ == "__main__":
    main()
