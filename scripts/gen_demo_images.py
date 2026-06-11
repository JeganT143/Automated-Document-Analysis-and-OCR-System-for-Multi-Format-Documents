"""
Generate 10 realistic demo document images for OCR pipeline demonstration.

Each image is a different document type with different scan defects so that
every preprocessing stage (invert, flatten, deskew, upscale, denoise, CLAHE)
is visibly triggered on at least one image.

Output: ~/Desktop/ocr_demo/img_01.png … img_10.png
"""

import os
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ── fonts ─────────────────────────────────────────────────────────────────
F = "/usr/share/fonts/truetype"
SANS_R = f"{F}/liberation/LiberationSans-Regular.ttf"
SANS_B = f"{F}/liberation/LiberationSans-Bold.ttf"
SANS_N = f"{F}/liberation/LiberationSansNarrow-Regular.ttf"
MONO_R = f"{F}/liberation/LiberationMono-Regular.ttf"
MONO_B = f"{F}/liberation/LiberationMono-Bold.ttf"
SERIF_R = f"{F}/dejavu/DejaVuSerif.ttf"
SERIF_B = f"{F}/dejavu/DejaVuSerif-Bold.ttf"
UBUNTU_R = f"{F}/ubuntu/Ubuntu-R.ttf"
UBUNTU_B = f"{F}/ubuntu/Ubuntu-B.ttf"

OUT = os.path.expanduser("~/Desktop/ocr_demo")
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(42)


# ── helpers ────────────────────────────────────────────────────────────────
def font(path, size):
    return ImageFont.truetype(path, size)


def hline(draw, y, x0, x1, fill=0, width=1):
    draw.line([(x0, y), (x1, y)], fill=fill, width=width)


def vline(draw, x, y0, y1, fill=0, width=1):
    draw.line([(x, y0), (x, y1)], fill=fill, width=width)


def text(draw, xy, msg, fnt, fill=0, anchor="la"):
    draw.text(xy, msg, font=fnt, fill=fill, anchor=anchor)


def add_noise(arr, sigma):
    noise = RNG.normal(0, sigma, arr.shape).astype(np.int16)
    return np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def rotate_img(arr, angle, bg=248):
    h, w = arr.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=bg)


def to_gray_arr(img):
    return np.array(img.convert("L"))


def save(arr, name):
    path = os.path.join(OUT, name)
    cv2.imwrite(path, arr)
    print(f"  saved  {path}")


def illumination_gradient(arr, left=0.72, right=1.08):
    h, w = arr.shape
    grad = np.tile(np.linspace(left, right, w), (h, 1)).astype(np.float32)
    return np.clip(arr.astype(np.float32) * grad, 0, 255).astype(np.uint8)


def gaussian_blur(arr, ksize=3, sigma=0.8):
    return cv2.GaussianBlur(arr, (ksize, ksize), sigma)


def shrink(arr, factor):
    h, w = arr.shape
    return cv2.resize(arr, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)


def dark_background(arr):
    """Invert so text is light on dark (triggers polarity fix)."""
    return cv2.bitwise_not(arr)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Restaurant receipt  — skewed + noisy
# ─────────────────────────────────────────────────────────────────────────────
def make_restaurant_receipt():
    W, H = 380, 600
    img = Image.new("L", (W, H), 252)
    d = ImageDraw.Draw(img)

    fB = font(SANS_B, 18)
    fR = font(SANS_R, 14)
    fS = font(SANS_R, 12)
    fM = font(MONO_R, 13)

    text(d, (W//2, 28), "THE CORNER BISTRO", font(SANS_B, 20), anchor="mm")
    text(d, (W//2, 50), "42 Riverside Ave   Tel: 555-0182", fS, anchor="mm")
    text(d, (W//2, 65), "Open daily 07:00 – 23:00", fS, anchor="mm")
    hline(d, 80, 20, W - 20, width=2)

    text(d, (20, 92), "Table: 7        Server: Maria", fS)
    text(d, (20, 108), "Date: 14 May 2025     18:43", fS)
    hline(d, 122, 20, W - 20)

    items = [
        ("Grilled salmon", "24.50"),
        ("Caesar salad", "9.75"),
        ("Mushroom risotto", "17.00"),
        ("House wine (glass)", "7.50"),
        ("Sparkling water", "3.20"),
        ("Tiramisu", "8.00"),
    ]
    y = 132
    text(d, (20, y), "Item", fR)
    text(d, (W - 20, y), "Amount", fR, anchor="ra")
    hline(d, y + 18, 20, W - 20)
    y += 24
    for item, price in items:
        text(d, (20, y), item, fR)
        text(d, (W - 20, y), f"${price}", fM, anchor="ra")
        y += 22

    hline(d, y + 4, 20, W - 20)
    y += 16
    text(d, (20, y), "Subtotal", fR);      text(d, (W - 20, y), "$69.95", fM, anchor="ra"); y += 22
    text(d, (20, y), "Service (12%)", fR); text(d, (W - 20, y), "$ 8.39", fM, anchor="ra"); y += 22
    text(d, (20, y), "Tax (8%)", fR);      text(d, (W - 20, y), "$ 5.60", fM, anchor="ra"); y += 22
    hline(d, y + 2, 20, W - 20, width=2)
    y += 10
    text(d, (20, y), "TOTAL", fB);         text(d, (W - 20, y), "$83.94", font(MONO_B, 15), anchor="ra")
    y += 36
    hline(d, y, 20, W - 20)
    y += 12
    text(d, (W//2, y), "Thank you for dining with us!", fS, anchor="mm")
    text(d, (W//2, y + 18), "Please come again :)", fS, anchor="mm")

    arr = to_gray_arr(img)
    arr = add_noise(arr, sigma=8)
    arr = rotate_img(arr, angle=-2.4, bg=252)  # skew → triggers deskew
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 2. Electricity utility bill  — uneven illumination + light noise
# ─────────────────────────────────────────────────────────────────────────────
def make_electricity_bill():
    W, H = 620, 820
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    fT = font(SANS_B, 28)
    fB = font(SANS_B, 15)
    fR = font(SANS_R, 14)
    fS = font(SANS_R, 11)
    fM = font(MONO_R, 13)

    # Header band
    d.rectangle([0, 0, W, 70], fill=30)
    text(d, (W//2, 35), "METROPOLITAN ELECTRIC", font(SANS_B, 22), fill=255, anchor="mm")
    text(d, (W//2, 56), "Your trusted energy provider", fS, fill=200, anchor="mm")

    y = 90
    text(d, (30, y), "ELECTRICITY BILL", fT); y += 40
    hline(d, y, 20, W - 20, width=2); y += 16

    # Customer block
    text(d, (30, y), "Account Number:", fB);   text(d, (220, y), "AC-2087-4413", fR); y += 24
    text(d, (30, y), "Customer Name:", fB);    text(d, (220, y), "James R. Holloway", fR); y += 24
    text(d, (30, y), "Service Address:", fB);  text(d, (220, y), "88 Elmwood Drive, Unit 3B", fR); y += 24
    text(d, (30, y), "City / State:", fB);     text(d, (220, y), "Portland, OR  97201", fR); y += 24
    text(d, (30, y), "Billing Period:", fB);   text(d, (220, y), "01 Apr 2025 – 30 Apr 2025", fR); y += 24
    text(d, (30, y), "Due Date:", fB);         text(d, (220, y), "20 May 2025", fR); y += 32
    hline(d, y, 20, W - 20); y += 20

    # Meter readings
    text(d, (30, y), "METER READINGS", fB); y += 22
    headers = ["Reading Type", "Previous", "Current", "Usage"]
    col_x = [30, 200, 340, 460]
    for h_txt, cx in zip(headers, col_x):
        text(d, (cx, y), h_txt, fR)
    y += 20
    hline(d, y, 20, W - 20); y += 10
    for t_item in [("Peak (kWh)", "14,820", "15,307", "487"),
                   ("Off-Peak (kWh)", "9,440", "9,713", "273")]:
        text(d, (col_x[0], y), t_item[0], fR); text(d, (col_x[1], y), t_item[1], fM)
        text(d, (col_x[2], y), t_item[2], fM); text(d, (col_x[3], y), t_item[3], fM); y += 24
    hline(d, y, 20, W - 20); y += 24

    # Charges
    text(d, (30, y), "CHARGES SUMMARY", fB); y += 24
    charges = [
        ("Energy Charge — Peak (487 kWh × $0.142)", "$ 69.15"),
        ("Energy Charge — Off-Peak (273 kWh × $0.089)", "$ 24.30"),
        ("Distribution charge", "$ 14.50"),
        ("Transmission charge", "$  8.75"),
        ("Meter reading & data", "$  4.00"),
        ("Renewable energy levy", "$  3.20"),
    ]
    for label, amount in charges:
        text(d, (30, y), label, fR); text(d, (W - 30, y), amount, fM, anchor="ra"); y += 22
    hline(d, y + 4, 20, W - 20, width=2); y += 18
    text(d, (30, y), "Subtotal", fB);          text(d, (W - 30, y), "$123.90", fM, anchor="ra"); y += 24
    text(d, (30, y), "State Sales Tax (7%)", fR); text(d, (W - 30, y), "$  8.67", fM, anchor="ra"); y += 24
    hline(d, y + 4, 20, W - 20, width=2); y += 16
    text(d, (30, y), "TOTAL AMOUNT DUE", font(SANS_B, 18)); text(d, (W - 30, y), "$132.57", font(MONO_B, 18), anchor="ra"); y += 42
    hline(d, y, 20, W - 20); y += 18
    text(d, (30, y), "Pay online at www.metroelectric.example.com or call 1-800-555-0199", fS)

    arr = to_gray_arr(img)
    arr = illumination_gradient(arr, left=0.70, right=1.12)   # triggers flatten
    arr = add_noise(arr, sigma=6)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hotel invoice  — small / low-DPI (triggers upscale)
# ─────────────────────────────────────────────────────────────────────────────
def make_hotel_invoice():
    W, H = 480, 660
    img = Image.new("L", (W, H), 250)
    d = ImageDraw.Draw(img)

    fT = font(SERIF_B, 22)
    fB = font(SERIF_B, 13)
    fR = font(SERIF_R, 12)
    fS = font(SERIF_R, 10)
    fM = font(MONO_R, 11)

    text(d, (W//2, 28), "GRAND HARBOUR HOTEL", fT, anchor="mm")
    text(d, (W//2, 48), "1 Harbour View Road, Wellington 6011", fS, anchor="mm")
    text(d, (W//2, 62), "Tel +64 4 555 2200   guestservices@ghhotel.example", fS, anchor="mm")
    hline(d, 76, 20, W - 20, width=2)

    y = 88
    text(d, (20, y), "FOLIO / TAX INVOICE", fB); y += 22
    text(d, (20, y), "Folio No:", fB); text(d, (150, y), "FO-2025-08841", fM); y += 18
    text(d, (20, y), "Guest:", fB);          text(d, (150, y), "Dr. Samantha Lowe", fR); y += 18
    text(d, (20, y), "Room:", fB);           text(d, (150, y), "412 — Deluxe King", fR); y += 18
    text(d, (20, y), "Check-in:", fB);       text(d, (150, y), "08 May 2025", fR); y += 18
    text(d, (20, y), "Check-out:", fB);      text(d, (150, y), "12 May 2025  (4 nights)", fR); y += 18
    hline(d, y + 4, 20, W - 20); y += 16

    headers = ["Date", "Description", "Ref", "Amount"]
    col_x = [20, 100, 330, W - 20]
    for h_txt, cx, anc in zip(headers, col_x, ["la", "la", "la", "ra"]):
        text(d, (cx, y), h_txt, fB, anchor=anc)
    y += 16; hline(d, y, 20, W - 20); y += 10

    rows = [
        ("08 May", "Room rate",             "RM412", "195.00"),
        ("08 May", "Mini-bar",              "MB001", " 18.40"),
        ("09 May", "Room rate",             "RM412", "195.00"),
        ("09 May", "Restaurant — dinner",   "RS044", " 67.50"),
        ("09 May", "Spa treatment (60 min)","SP011", " 90.00"),
        ("10 May", "Room rate",             "RM412", "195.00"),
        ("10 May", "Laundry service",       "LN007", " 24.00"),
        ("10 May", "Telephone / internet",  "TE003", "  8.50"),
        ("11 May", "Room rate",             "RM412", "195.00"),
        ("11 May", "Restaurant — breakfast","RS012", " 28.00"),
        ("11 May", "Parking (4 days)",      "PK019", " 60.00"),
    ]
    for dt, desc, ref, amt in rows:
        text(d, (col_x[0], y), dt, fR)
        text(d, (col_x[1], y), desc, fR)
        text(d, (col_x[2], y), ref, fM)
        text(d, (col_x[3], y), f"${amt}", fM, anchor="ra")
        y += 18
    hline(d, y + 4, 20, W - 20, width=2); y += 16
    text(d, (30, y), "Sub-total", fB);     text(d, (W - 20, y), "$1,076.40", fM, anchor="ra"); y += 20
    text(d, (30, y), "GST (15%)", fR);     text(d, (W - 20, y), "$  161.46", fM, anchor="ra"); y += 20
    hline(d, y + 2, 20, W - 20, width=2); y += 12
    text(d, (30, y), "TOTAL DUE (NZD)", fB); text(d, (W - 20, y), "$1,237.86", font(MONO_B, 13), anchor="ra"); y += 28
    hline(d, y, 20, W - 20); y += 12
    text(d, (20, y), "Paid by: VISA  •••• •••• •••• 4421   12 May 2025  APPROVED", fS)

    arr = to_gray_arr(img)
    arr = add_noise(arr, sigma=5)
    arr = shrink(arr, 0.48)   # renders small → upscale triggered
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 4. Supermarket receipt — light-on-dark (inverted polarity)
# ─────────────────────────────────────────────────────────────────────────────
def make_supermarket_receipt():
    W, H = 360, 620
    img = Image.new("L", (W, H), 18)   # dark background
    d = ImageDraw.Draw(img)
    LIGHT = 230

    fT = font(UBUNTU_B, 18)
    fB = font(UBUNTU_B, 14)
    fR = font(UBUNTU_R, 13)
    fS = font(UBUNTU_R, 11)
    fM = font(MONO_R, 12)

    text(d, (W//2, 26), "FRESHMART", fT, fill=LIGHT, anchor="mm")
    text(d, (W//2, 46), "2210 Oak Street  |  Branch #14", fS, fill=180, anchor="mm")
    text(d, (W//2, 61), "Tel: (503) 555-0143", fS, fill=180, anchor="mm")
    hline(d, 74, 14, W - 14, fill=LIGHT)

    text(d, (14, 84), "Date: 02 Jun 2025     10:17 AM", fS, fill=LIGHT)
    text(d, (14, 100), "Cashier: T. Nguyen    Till: 04", fS, fill=LIGHT)
    hline(d, 114, 14, W - 14, fill=LIGHT)

    items = [
        ("Organic whole milk 2L",   "3.49"),
        ("Brown bread 700g",        "2.89"),
        ("Free-range eggs x12",     "5.99"),
        ("Cheddar cheese 400g",     "4.75"),
        ("Greek yoghurt 500g",      "3.20"),
        ("Butter unsalted 250g",    "3.85"),
        ("Orange juice 1L",         "3.99"),
        ("Chicken breast 600g",     "7.80"),
        ("Basmati rice 1kg",        "4.50"),
        ("Pasta spaghetti 500g",    "1.99"),
        ("Tomato sauce (jar)",      "2.40"),
        ("Olive oil 500ml",         "6.95"),
    ]
    y = 124
    for item, price in items:
        text(d, (14, y), item, fR, fill=LIGHT)
        text(d, (W - 14, y), f"${price}", fM, fill=LIGHT, anchor="ra")
        y += 21
    hline(d, y + 4, 14, W - 14, fill=LIGHT); y += 16
    text(d, (14, y), "Sub-total", fR, fill=LIGHT);  text(d, (W - 14, y), "$51.80", fM, fill=LIGHT, anchor="ra"); y += 22
    text(d, (14, y), "Loyalty discount", fR, fill=180); text(d, (W - 14, y), "-$3.10", fM, fill=180, anchor="ra"); y += 22
    hline(d, y + 2, 14, W - 14, fill=LIGHT, width=2); y += 12
    text(d, (14, y), "TOTAL", fB, fill=LIGHT); text(d, (W - 14, y), "$48.70", font(MONO_B, 15), fill=LIGHT, anchor="ra"); y += 30
    hline(d, y, 14, W - 14, fill=LIGHT); y += 14
    text(d, (14, y), "Cash tendered: $50.00", fS, fill=LIGHT); y += 18
    text(d, (14, y), "Change:         $1.30", fS, fill=LIGHT); y += 24
    hline(d, y, 14, W - 14, fill=LIGHT); y += 12
    text(d, (W//2, y), "Thank you for shopping at FreshMart!", fS, fill=160, anchor="mm")

    arr = to_gray_arr(img)
    # Leave as-is — it's already light-on-dark; pipeline will auto-invert
    arr = add_noise(arr, sigma=10)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 5. Parking ticket  — small + skewed + heavy noise
# ─────────────────────────────────────────────────────────────────────────────
def make_parking_ticket():
    W, H = 340, 480
    img = Image.new("L", (W, H), 250)
    d = ImageDraw.Draw(img)

    fT = font(SANS_B, 20)
    fB = font(SANS_B, 13)
    fR = font(SANS_R, 12)
    fS = font(SANS_R, 10)
    fM = font(MONO_R, 12)

    d.rectangle([0, 0, W, 60], fill=40)
    text(d, (W//2, 16), "CITY PARKING AUTHORITY", font(SANS_B, 15), fill=255, anchor="mm")
    text(d, (W//2, 38), "PARKING INFRINGEMENT NOTICE", font(SANS_B, 13), fill=220, anchor="mm")
    text(d, (W//2, 54), "Notice No: CPA-2025-084417", fS, fill=180, anchor="mm")

    y = 74
    fields = [
        ("Vehicle Reg:", "GHK 2284"),
        ("Make / Model:", "Toyota Corolla"),
        ("Color:", "Silver"),
        ("Location:", "Market St, Zone 3B"),
        ("Date:", "07 May 2025"),
        ("Time issued:", "14:22"),
        ("Officer badge:", "PO-1187"),
        ("Infringement:", "Meter expired"),
        ("Fine amount:", "$85.00"),
        ("Due by:", "07 Jun 2025"),
    ]
    for label, val in fields:
        text(d, (16, y), label, fB); text(d, (160, y), val, fM); y += 22
        hline(d, y - 4, 16, W - 16, fill=210)
    y += 10
    hline(d, y, 16, W - 16, width=2); y += 12
    text(d, (16, y), "Payment methods:", fB); y += 18
    text(d, (16, y), "  • Online: www.cityparking.example/pay", fS); y += 16
    text(d, (16, y), "  • Phone: 1-800-PARK-NOW (Option 3)", fS); y += 16
    text(d, (16, y), "  • In-person: 1 City Hall Plaza, Counter 6", fS); y += 20
    hline(d, y, 16, W - 16); y += 12
    text(d, (16, y), "Dispute within 28 days. Late payment attracts a $40 surcharge.", fS)

    arr = to_gray_arr(img)
    arr = add_noise(arr, sigma=14)
    arr = gaussian_blur(arr, ksize=3, sigma=0.9)
    arr = rotate_img(arr, angle=3.1, bg=250)  # skew
    arr = shrink(arr, 0.55)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 6. Medical prescription form — clean but very low DPI (upscale only)
# ─────────────────────────────────────────────────────────────────────────────
def make_prescription():
    W, H = 520, 640
    img = Image.new("L", (W, H), 254)
    d = ImageDraw.Draw(img)

    fT = font(SERIF_B, 22)
    fB = font(SERIF_B, 14)
    fR = font(SERIF_R, 13)
    fS = font(SERIF_R, 11)
    fM = font(MONO_R, 12)

    d.rectangle([0, 0, W, 80], fill=245)
    hline(d, 80, 0, W, fill=180, width=2)
    text(d, (W//2, 22), "RIVERSIDE MEDICAL CLINIC", fT, anchor="mm")
    text(d, (W//2, 46), "Dr. Karen S. Mitchell, MD  (Lic: NM-2019-4412)", fR, anchor="mm")
    text(d, (W//2, 64), "22 Parkland Blvd, Suite 4  |  Tel: (505) 555-0321", fS, anchor="mm")

    y = 96
    text(d, (30, y), "PRESCRIPTION", fB); y += 4
    hline(d, y + 18, 20, W - 20, width=2); y += 30

    text(d, (30, y), "Patient:", fB);      text(d, (160, y), "Nathan P. Grossman", fR); y += 22
    text(d, (30, y), "Date of birth:", fB);text(d, (160, y), "19 Sep 1978", fR); y += 22
    text(d, (30, y), "Date issued:", fB);  text(d, (160, y), "05 May 2025", fR); y += 22
    text(d, (30, y), "Script No:", fB);    text(d, (160, y), "RX-25-084419", fM); y += 32

    hline(d, y, 20, W - 20); y += 20
    text(d, (30, y), "MEDICATION", fB); y += 22

    rx = [
        ("Rx 1", "Amoxicillin 500 mg", "1 capsule every 8 hours for 7 days",  "#30 caps",  "3 refills"),
        ("Rx 2", "Ibuprofen 400 mg",   "1 tablet with food every 6 hrs (max 3/day)", "#20 tabs", "0 refills"),
        ("Rx 3", "Cetirizine 10 mg",   "1 tablet daily at bedtime",           "#30 tabs",  "2 refills"),
    ]
    for num, drug, sig, qty, refill in rx:
        d.rectangle([20, y, W - 20, y + 62], outline=180, width=1)
        text(d, (30, y + 8),  num, fB)
        text(d, (80, y + 8),  drug, fB)
        text(d, (30, y + 28), f"Sig: {sig}", fS)
        text(d, (30, y + 46), f"Qty: {qty}", fS); text(d, (200, y + 46), f"Refills: {refill}", fS)
        y += 76

    y += 14
    hline(d, y, 20, W - 20); y += 18
    text(d, (30, y), "Prescriber signature:  _________________________", fR); y += 30
    text(d, (30, y), "THIS PRESCRIPTION IS VALID FOR 12 MONTHS FROM DATE ISSUED", fS)

    arr = to_gray_arr(img)
    arr = shrink(arr, 0.40)  # tiny → strong upscale needed
    arr = add_noise(arr, sigma=3)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 7. Shipping / packing slip — heavy skew + gradient
# ─────────────────────────────────────────────────────────────────────────────
def make_shipping_slip():
    W, H = 560, 700
    img = Image.new("L", (W, H), 252)
    d = ImageDraw.Draw(img)

    fT = font(UBUNTU_B, 24)
    fB = font(UBUNTU_B, 14)
    fR = font(UBUNTU_R, 13)
    fS = font(UBUNTU_R, 11)
    fM = font(MONO_R, 12)

    # header
    d.rectangle([0, 0, W, 72], fill=22)
    text(d, (22, 36), "RAPID DISPATCH CO.", font(UBUNTU_B, 20), fill=255, anchor="lm")
    text(d, (W - 22, 28), "PACKING SLIP", fB, fill=200, anchor="rm")
    text(d, (W - 22, 50), "Order: ORD-2025-188341", fM, fill=180, anchor="rm")

    y = 88
    # Sender / recipient
    col1_x, col2_x = 22, W//2 + 10
    text(d, (col1_x, y), "SENDER", fB); text(d, (col2_x, y), "RECIPIENT", fB); y += 20
    hline(d, y, 18, W - 18); y += 10
    sender = ["Peak Performance Supplies", "800 Industrial Loop", "Houston, TX 77002", "Tel: (713) 555-0244"]
    recipient = ["Ms. Priya Sharma", "34 Willow Close, Apt 7", "Chicago, IL 60601", "Tel: (312) 555-0890"]
    for s, r in zip(sender, recipient):
        text(d, (col1_x, y), s, fR); text(d, (col2_x, y), r, fR); y += 20
    y += 12
    hline(d, y, 18, W - 18, width=2); y += 16

    text(d, (22, y), "ITEMS SHIPPED", fB); y += 20
    hdrs = ["#", "SKU", "Description", "Qty", "Unit wt."]
    col_x = [22, 48, 130, 390, 450]
    for h_t, cx in zip(hdrs, col_x):
        text(d, (cx, y), h_t, fB)
    y += 16; hline(d, y, 18, W - 18); y += 10
    products = [
        ("1", "PP-BND-001", "Resistance bands set (5 levels)",     "2", "0.4 kg"),
        ("2", "PP-MAT-002", "Yoga mat non-slip 6mm",               "1", "1.2 kg"),
        ("3", "PP-BTL-007", "Stainless water bottle 750ml",        "3", "0.3 kg"),
        ("4", "PP-GLV-003", "Weightlifting gloves M/L",            "1", "0.2 kg"),
        ("5", "PP-BLT-011", "Adjustable lifting belt (L)",         "1", "0.6 kg"),
        ("6", "PP-ROL-005", "Foam roller 60cm high-density",       "2", "0.9 kg"),
    ]
    for item in products:
        for val, cx in zip(item, col_x):
            text(d, (cx, y), val, fR)
        y += 20
    hline(d, y + 4, 18, W - 18, width=2); y += 18
    text(d, (22, y), "Total items: 10    Total weight: 7.0 kg", fR); y += 30
    hline(d, y, 18, W - 18); y += 16
    text(d, (22, y), "Carrier: FedEx Ground   Tracking: 7489 2814 6630 3", fM); y += 20
    text(d, (22, y), "Dispatch date: 09 May 2025   Est. delivery: 13 May 2025", fR); y += 28
    hline(d, y, 18, W - 18); y += 14
    text(d, (22, y), "Questions? support@rapiddispatch.example  |  1-888-RAP-DISP", fS)

    arr = to_gray_arr(img)
    arr = illumination_gradient(arr, left=0.65, right=1.15)
    arr = add_noise(arr, sigma=7)
    arr = rotate_img(arr, angle=4.2, bg=252)  # heavy skew
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 8. Bank statement — faint text / low contrast (CLAHE retry triggered)
# ─────────────────────────────────────────────────────────────────────────────
def make_bank_statement():
    W, H = 640, 860
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    fT = font(SANS_B, 22)
    fB = font(SANS_B, 13)
    fR = font(SANS_R, 12)
    fS = font(SANS_R, 10)
    fM = font(MONO_R, 12)

    text(d, (30, 28), "NORTHBROOK BANK", fT)
    text(d, (30, 56), "ACCOUNT STATEMENT", fB)
    hline(d, 74, 20, W - 20, width=2)

    y = 86
    text(d, (30, y), "Account holder:", fB);   text(d, (220, y), "Elena V. Vasquez", fR); y += 22
    text(d, (30, y), "Account number:", fB);   text(d, (220, y), "NB-CHK-5521-8804", fM); y += 22
    text(d, (30, y), "Statement period:", fB); text(d, (220, y), "01 Apr 2025 – 30 Apr 2025", fR); y += 22
    text(d, (30, y), "Opening balance:", fB);  text(d, (220, y), "$4,210.55", fM); y += 36

    hline(d, y, 20, W - 20, width=2); y += 14
    text(d, (30, y), "TRANSACTIONS", fB); y += 22

    hdrs = ["Date", "Description", "Ref", "Debit", "Credit", "Balance"]
    col_x = [26, 110, 330, 406, 476, W - 26]
    ancs  = ["la", "la", "la", "ra", "ra", "ra"]
    for h_t, cx, anc in zip(hdrs, col_x, ancs):
        text(d, (cx, y), h_t, fB, anchor=anc)
    y += 16; hline(d, y, 20, W - 20); y += 8

    txns = [
        ("01 Apr", "Opening balance",          "OPB",  "",        "",        "4,210.55"),
        ("02 Apr", "Direct debit — EasyRent",  "DD01", "1,200.00","",        "3,010.55"),
        ("03 Apr", "Salary credit",             "SC01", "",        "4,850.00","7,860.55"),
        ("05 Apr", "Supermarket — FreshMart",  "POS",  "  94.30", "",        "7,766.25"),
        ("07 Apr", "Netflix subscription",     "DD02", "  14.99", "",        "7,751.26"),
        ("09 Apr", "ATM withdrawal",            "ATM",  " 200.00", "",        "7,551.26"),
        ("11 Apr", "Online transfer — savings","OT01", "1,500.00","",        "6,051.26"),
        ("14 Apr", "Electricity — MetroElec",  "DD03", " 132.57", "",        "5,918.69"),
        ("17 Apr", "Coffee subscription",      "DD04", "  24.00", "",        "5,894.69"),
        ("20 Apr", "Freelance payment",         "CR01", "",        "  620.00","6,514.69"),
        ("22 Apr", "Pharmacy — MedCare",       "POS",  "  53.80", "",        "6,460.89"),
        ("25 Apr", "Restaurant — The Bistro",  "POS",  "  83.94", "",        "6,376.95"),
        ("28 Apr", "Internet — FiberNet",      "DD05", "  59.00", "",        "6,317.95"),
        ("30 Apr", "Closing balance",          "CLB",  "",        "",        "6,317.95"),
    ]
    for row in txns:
        dt, desc, ref, deb, cre, bal = row
        values = [dt, desc, ref, deb, cre, bal]
        for val, cx, anc in zip(values, col_x, ancs):
            text(d, (cx, y), val, fM if cx in (col_x[3], col_x[4], col_x[5]) else fR, anchor=anc)
        y += 20
        hline(d, y, 20, W - 20, fill=225)
    y += 12
    hline(d, y, 20, W - 20, width=2); y += 16
    text(d, (30, y), "Closing balance:", fB); text(d, (W - 26, y), "$6,317.95", font(MONO_B, 13), anchor="ra"); y += 34
    hline(d, y, 20, W - 20); y += 12
    text(d, (30, y), "This statement was generated electronically and does not require a signature.", fS)

    arr = to_gray_arr(img)
    # Simulate a very faded / low-contrast photocopy → triggers CLAHE retry
    arr = np.clip(arr.astype(np.float32) * 0.45 + 140, 0, 255).astype(np.uint8)
    arr = add_noise(arr, sigma=6)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 9. Tax invoice (B2B) — multi-column, slightly skewed, normal quality
# ─────────────────────────────────────────────────────────────────────────────
def make_tax_invoice():
    W, H = 680, 900
    img = Image.new("L", (W, H), 252)
    d = ImageDraw.Draw(img)

    fT  = font(SANS_B, 30)
    fB  = font(SANS_B, 14)
    fR  = font(SANS_R, 13)
    fS  = font(SANS_R, 11)
    fM  = font(MONO_R, 13)
    fMB = font(MONO_B, 14)

    text(d, (40, 42), "TAX INVOICE", fT)
    text(d, (W - 40, 20), "Precision Tech Solutions Pty Ltd", fB, anchor="ra")
    text(d, (W - 40, 38), "ABN  71 842 930 011", fR, anchor="ra")
    text(d, (W - 40, 56), "Level 12, 580 George St, Sydney NSW 2000", fS, anchor="ra")
    text(d, (W - 40, 72), "accounts@precisiontech.example   |   02 9055 2200", fS, anchor="ra")
    hline(d, 92, 30, W - 30, width=2)

    y = 106
    # Two-column block: bill-to on left, invoice details on right
    text(d, (40, y), "BILL TO:", fB)
    text(d, (W//2 + 20, y), "INVOICE DETAILS:", fB)
    y += 22

    bill_lines = ["Oceania Media Group Ltd", "ABN  43 108 765 092", "Attn: Accounts Payable",
                  "PO Box 441, Brisbane QLD 4000"]
    detail_lines = [("Invoice No:", "INV-2025-0933"), ("Date:", "15 May 2025"),
                    ("Payment due:", "14 Jun 2025"),  ("Purchase order:", "PO-OMG-8814")]
    for bl, (dl, dv) in zip(bill_lines, detail_lines):
        text(d, (40, y), bl, fR)
        text(d, (W//2 + 20, y), dl, fB); text(d, (W//2 + 180, y), dv, fM)
        y += 20
    y += 14
    hline(d, y, 30, W - 30, width=2); y += 14

    text(d, (40, y), "SERVICES RENDERED", fB); y += 22
    col_x = [40, 70, 400, 460, 540, W - 30]
    hdrs  = ["#", "Description", "Unit", "Qty", "Rate", "Amount"]
    ancs  = ["la", "la", "la", "ra", "ra", "ra"]
    for h_t, cx, anc in zip(hdrs, col_x, ancs):
        text(d, (cx, y), h_t, fB, anchor=anc)
    y += 16; hline(d, y, 30, W - 30); y += 10

    services = [
        ("1", "Strategic digital audit & competitor analysis",   "Project",  "1",  "$3,200.00",  "$3,200.00"),
        ("2", "UX/UI redesign — homepage & 3 landing pages",     "Project",  "1",  "$4,800.00",  "$4,800.00"),
        ("3", "SEO optimisation (3-month managed service)",      "Month",    "3",  "$  900.00",  "$2,700.00"),
        ("4", "Social media content calendar (Q2 2025)",         "Package",  "1",  "$1,500.00",  "$1,500.00"),
        ("5", "Email marketing setup & first campaign",          "Project",  "1",  "$  750.00",  "$  750.00"),
        ("6", "Analytics dashboard configuration & training",    "Hours",    "4",  "$  180.00",  "$  720.00"),
    ]
    for svc in services:
        num, desc, unit, qty, rate, amt = svc
        vals = [num, desc, unit, qty, rate, amt]
        for val, cx, anc in zip(vals, col_x, ancs):
            text(d, (cx, y), val, fR, anchor=anc)
        y += 22
    hline(d, y + 4, 30, W - 30, width=2); y += 20

    text(d, (30, y), "Sub-total (excl. GST)", fR);    text(d, (W - 30, y), "$13,670.00", fM, anchor="ra"); y += 22
    text(d, (30, y), "GST (10%)", fR);                text(d, (W - 30, y), "$ 1,367.00", fM, anchor="ra"); y += 22
    hline(d, y + 2, 30, W - 30, width=2); y += 14
    text(d, (30, y), "TOTAL (AUD) incl. GST", fB);    text(d, (W - 30, y), "$15,037.00", fMB, anchor="ra"); y += 38

    hline(d, y, 30, W - 30); y += 14
    text(d, (30, y), "Payment: EFT to BSB 062-001  Account 1044 5820  Ref: INV-2025-0933", fS); y += 18
    text(d, (30, y), "Late payments attract 1.5% per month. Thank you for your business.", fS)

    arr = to_gray_arr(img)
    arr = add_noise(arr, sigma=5)
    arr = rotate_img(arr, angle=-1.6, bg=252)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 10. Gym membership invoice — dark + noisy + slight skew (multiple defects)
# ─────────────────────────────────────────────────────────────────────────────
def make_gym_invoice():
    W, H = 500, 660
    img = Image.new("L", (W, H), 18)   # dark background
    d = ImageDraw.Draw(img)
    INK = 238

    fT = font(UBUNTU_B, 26)
    fB = font(UBUNTU_B, 15)
    fR = font(UBUNTU_R, 13)
    fS = font(UBUNTU_R, 11)
    fM = font(MONO_R, 13)

    d.rectangle([0, 0, W, 76], fill=42)
    text(d, (W//2, 26), "IRONCLAD FITNESS", fT, fill=230, anchor="mm")
    text(d, (W//2, 54), "Your #1 training destination", fS, fill=160, anchor="mm")
    text(d, (W//2, 68), "15 Stadium Road  |  Tel: (202) 555-0188", fS, fill=140, anchor="mm")

    hline(d, 78, 0, W, fill=80, width=2)
    y = 92
    text(d, (W//2, y), "MEMBERSHIP INVOICE", fB, fill=INK, anchor="mm"); y += 32
    hline(d, y, 20, W - 20, fill=70); y += 16

    text(d, (22, y), "Member:", fB, fill=INK);       text(d, (200, y), "Carlos E. Mendez", fR, fill=INK); y += 22
    text(d, (22, y), "Member ID:", fB, fill=INK);    text(d, (200, y), "MEM-10041", fM, fill=INK); y += 22
    text(d, (22, y), "Membership:", fB, fill=INK);   text(d, (200, y), "Elite — All Access", fR, fill=INK); y += 22
    text(d, (22, y), "Invoice No:", fB, fill=INK);   text(d, (200, y), "GYM-INV-2025-0514", fM, fill=INK); y += 22
    text(d, (22, y), "Invoice date:", fB, fill=INK); text(d, (200, y), "01 May 2025", fR, fill=INK); y += 22
    text(d, (22, y), "Period:", fB, fill=INK);       text(d, (200, y), "01 May – 31 May 2025", fR, fill=INK); y += 30
    hline(d, y, 20, W - 20, fill=70); y += 16

    items = [
        ("Monthly membership fee", "Elite All Access", "$89.00"),
        ("Personal training session", "PT — Coach Rivera (x2)", "$60.00"),
        ("Locker rental (monthly)", "Locker #47B", "$12.00"),
        ("Protein shake pack", "Whey Pro 30-pack", "$44.00"),
        ("Guest pass × 2", "Weekend day passes", "$20.00"),
    ]
    for label, detail, amt in items:
        text(d, (22, y), label, fB, fill=INK); y += 18
        text(d, (30, y), detail, fS, fill=140);  text(d, (W - 22, y), amt, fM, fill=INK, anchor="ra"); y += 24
        hline(d, y, 22, W - 22, fill=46); y += 6
    y += 8
    hline(d, y, 20, W - 20, fill=80, width=2); y += 16
    text(d, (22, y), "Sub-total", fR, fill=INK);   text(d, (W - 22, y), "$225.00", fM, fill=INK, anchor="ra"); y += 22
    text(d, (22, y), "Tax (8%)", fR, fill=INK);    text(d, (W - 22, y), "$ 18.00", fM, fill=INK, anchor="ra"); y += 22
    hline(d, y, 20, W - 20, fill=80, width=2); y += 12
    text(d, (22, y), "TOTAL", fB, fill=INK);       text(d, (W - 22, y), "$243.00", font(MONO_B, 16), fill=INK, anchor="ra"); y += 36
    hline(d, y, 20, W - 20, fill=70); y += 14
    text(d, (22, y), "Auto-debit on 01 Jun 2025 to card ending ••••7732", fS, fill=160); y += 20
    text(d, (W//2, y), "Stay consistent. Stay strong.", fS, fill=100, anchor="mm")

    arr = to_gray_arr(img)
    arr = add_noise(arr, sigma=12)
    arr = rotate_img(arr, angle=1.8, bg=18)
    return arr


# ── generate all ──────────────────────────────────────────────────────────
DOCS = [
    ("img_01_restaurant_receipt.png",   make_restaurant_receipt,  "Skewed + noisy"),
    ("img_02_electricity_bill.png",     make_electricity_bill,    "Uneven illumination"),
    ("img_03_hotel_invoice.png",        make_hotel_invoice,       "Low-DPI (upscale)"),
    ("img_04_supermarket_receipt.png",  make_supermarket_receipt, "Light-on-dark (invert)"),
    ("img_05_parking_ticket.png",       make_parking_ticket,      "Small + skewed + noisy"),
    ("img_06_prescription.png",         make_prescription,        "Very small text (upscale)"),
    ("img_07_shipping_slip.png",        make_shipping_slip,       "Heavy skew + gradient"),
    ("img_08_bank_statement.png",       make_bank_statement,      "Faded/low-contrast (CLAHE)"),
    ("img_09_tax_invoice.png",          make_tax_invoice,         "Multi-column + slight skew"),
    ("img_10_gym_invoice.png",          make_gym_invoice,         "Dark bg + noise + skew"),
]

print(f"\nGenerating {len(DOCS)} demo images → {OUT}\n")
for filename, fn, defect in DOCS:
    arr = fn()
    save(arr, filename)
    print(f"    defect: {defect}")

print(f"\nDone. Open ~/Desktop/ocr_demo/ to view all images.")
