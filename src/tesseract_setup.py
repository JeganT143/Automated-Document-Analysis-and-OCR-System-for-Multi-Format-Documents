"""
Locate and configure the Tesseract OCR engine.

Tesseract is a *system* binary, but on machines without root access it can be
unpacked into a local prefix (see scripts/install_tesseract_local.sh). This
module makes the pipeline find Tesseract in either case:

    1. A normal system install on PATH (``tesseract``), or
    2. A user-local extraction under ``~/.local/opt/tesseract``.

It wires up ``pytesseract.tesseract_cmd``, ``TESSDATA_PREFIX`` and
``LD_LIBRARY_PATH`` (the last is needed so the unpacked binary can find its
bundled ``libtesseract``/``libleptonica`` at run time).

Call :func:`ensure_tesseract` once at start-up; it is idempotent.
"""

import os
import shutil
import glob

# Local extraction prefix used by scripts/install_tesseract_local.sh
_LOCAL_PREFIX = os.path.expanduser("~/.local/opt/tesseract")

_configured = False


def _find_local_binary():
    cand = os.path.join(_LOCAL_PREFIX, "usr", "bin", "tesseract")
    return cand if os.path.isfile(cand) else None


def _find_local_libdir():
    pattern = os.path.join(_LOCAL_PREFIX, "usr", "lib", "*", "libtesseract.so*")
    hits = glob.glob(pattern)
    return os.path.dirname(hits[0]) if hits else None


def _find_local_tessdata():
    pattern = os.path.join(_LOCAL_PREFIX, "usr", "share", "tesseract-ocr", "*", "tessdata")
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def ensure_tesseract():
    """Configure pytesseract to use whichever Tesseract is available.

    Returns the resolved tesseract command path, or ``None`` if Tesseract
    could not be found anywhere.
    """
    global _configured
    try:
        import pytesseract
    except Exception:
        return None

    # System install takes precedence – nothing extra to wire up.
    sys_bin = shutil.which("tesseract")
    if sys_bin:
        pytesseract.pytesseract.tesseract_cmd = sys_bin
        _configured = True
        return sys_bin

    # Fall back to the user-local extraction.
    local_bin = _find_local_binary()
    if local_bin:
        pytesseract.pytesseract.tesseract_cmd = local_bin

        libdir = _find_local_libdir()
        if libdir:
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            if libdir not in existing.split(os.pathsep):
                os.environ["LD_LIBRARY_PATH"] = (
                    libdir + (os.pathsep + existing if existing else "")
                )

        tessdata = _find_local_tessdata()
        if tessdata:
            os.environ.setdefault("TESSDATA_PREFIX", tessdata)

        _configured = True
        return local_bin

    return None


def tesseract_version():
    """Return the Tesseract version string, or None if unavailable."""
    if ensure_tesseract() is None:
        return None
    try:
        import pytesseract
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return None
