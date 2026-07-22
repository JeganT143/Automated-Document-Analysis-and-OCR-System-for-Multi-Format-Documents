#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Install Tesseract OCR into a user-local prefix WITHOUT root / sudo.
#
# On machines where you can run `sudo apt install tesseract-ocr`, do that
# instead – it is simpler. This script is for locked-down environments where
# only `apt-get download` (no install) is permitted.
#
# It downloads the tesseract + leptonica .deb packages and unpacks them under
#   ~/.local/opt/tesseract
# The Python helper src/tesseract_setup.py then auto-detects this location.
# ---------------------------------------------------------------------------
set -euo pipefail

PREFIX="${TESSERACT_PREFIX:-$HOME/.local/opt/tesseract}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo ">> Downloading Tesseract packages into $WORK ..."
cd "$WORK"
apt-get download \
    libleptonica6 \
    libtesseract5 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd

echo ">> Extracting into $PREFIX ..."
rm -rf "$PREFIX"
mkdir -p "$PREFIX"
for d in *.deb; do
    dpkg-deb -x "$d" "$PREFIX"
done

BIN="$PREFIX/usr/bin/tesseract"
LIBDIR="$(dirname "$(find "$PREFIX" -name 'libtesseract.so*' | head -1)")"
TESSDATA="$(dirname "$(find "$PREFIX" -name 'eng.traineddata' | head -1)")"

echo ">> Verifying ..."
LD_LIBRARY_PATH="$LIBDIR" TESSDATA_PREFIX="$TESSDATA" "$BIN" --version | head -3

cat <<EOF

Done. Tesseract installed at:
  $BIN

The Python pipeline auto-detects this via src/tesseract_setup.py.
To use the CLI directly, add to your shell:
  export PATH="$PREFIX/usr/bin:\$PATH"
  export LD_LIBRARY_PATH="$LIBDIR:\$LD_LIBRARY_PATH"
  export TESSDATA_PREFIX="$TESSDATA"
EOF
