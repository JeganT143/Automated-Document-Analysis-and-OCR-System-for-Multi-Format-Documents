"""
Document layout analysis (the "analysis" half of the project).

Given a clean grayscale page it finds:
  * character-level connected components (for visualisation), and
  * text regions, each classified as text / table / image / header-footer.

This is classical computer vision, but built for speed: the old pure-Python
Run-Length-Smoothing and recursive X-Y cut were O(H*W) Python loops (seconds
per page). They are replaced here by equivalent OpenCV morphology, which is
both faster and cleaner. Layout is descriptive metadata for the UI and the
structured output; recognition itself is handled by the OCR engine.
"""

import cv2
import numpy as np


class ConnectedComponentAnalyzer:
    """8-connected component extraction over a foreground (white-on-black) mask."""

    def find_components(self, fg_binary):
        num, _, stats, _ = cv2.connectedComponentsWithStats(fg_binary, connectivity=8)
        comps = []
        for i in range(1, num):  # skip background label 0
            x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]),
                          int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]),
                          int(stats[i, cv2.CC_STAT_HEIGHT]))
            area = int(stats[i, cv2.CC_STAT_AREA])
            comps.append({
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": w / float(h) if h else 0.0,
            })
        return comps

    def filter_char_components(self, comps, img_w, img_h):
        """Keep blobs that are plausibly individual characters."""
        max_w = max(40, img_w // 8)
        max_h = max(60, img_h // 12)
        return [c for c in comps
                if 8 <= c["area"] <= max_w * max_h
                and c["bbox"][2] <= max_w and c["bbox"][3] <= max_h
                and 0.05 <= c["aspect_ratio"] <= 15.0]


class RegionClassifier:
    """Classify a region as text / table / image / header-footer."""

    def __init__(self, header_footer_ratio=0.12):
        self.header_footer_ratio = header_footer_ratio

    def classify_region(self, binary_text0, bbox):
        """binary_text0: text=0, background=255 (Otsu output)."""
        x, y, w, h = bbox
        roi = binary_text0[y:y + h, x:x + w]
        if roi.size == 0:
            return "unknown"

        img_h = binary_text0.shape[0]
        rel_top = y / img_h
        rel_bot = (y + h) / img_h
        density = float((roi == 0).mean())
        aspect = w / float(h) if h else 1.0
        h_lines = self._count_horizontal_lines(roi)

        if rel_top < self.header_footer_ratio or rel_bot > 1 - self.header_footer_ratio:
            return "header_footer"
        if h_lines >= 3 and aspect > 1.5:
            return "table"
        if density < 0.02:
            return "image"
        return "text"

    @staticmethod
    def _count_horizontal_lines(roi):
        proj = (roi == 0).sum(axis=1)
        threshold = roi.shape[1] * 0.6
        count, in_line = 0, False
        for v in proj:
            if v > threshold:
                if not in_line:
                    count += 1
                    in_line = True
            else:
                in_line = False
        return count


class LayoutAnalyzer:
    """Facade: char components + classified text regions from a grayscale page."""

    def __init__(self):
        self.cca = ConnectedComponentAnalyzer()
        self.classifier = RegionClassifier()

    def analyze(self, gray):
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Otsu binary in the classic "text=0, background=255" convention.
        binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        fg = cv2.bitwise_not(binary)  # foreground (ink) = 255 for morphology/CC

        # Character-level components (for the recognition visualisation).
        char_comps = self.cca.filter_char_components(
            self.cca.find_components(fg), w, h)

        # Region smear: close horizontally so characters merge into lines, then
        # dilate vertically so adjacent lines merge into paragraph/table blocks.
        kx = max(10, w // 40)
        smear = cv2.morphologyEx(
            fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
        ky = max(3, h // 250)
        smear = cv2.dilate(
            smear, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)))

        # Region boxes from the smeared image.
        min_area = w * h * 0.0004
        regions = []
        for c in self.cca.find_components(smear):
            x, y, bw, bh = c["bbox"]
            if c["area"] < min_area or bw < w * 0.03:
                continue
            regions.append({
                "bbox": (x, y, bw, bh),
                "type": self.classifier.classify_region(binary, (x, y, bw, bh)),
            })
        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

        return {
            "binary": binary,
            "smoothed": smear,
            "components": char_comps,
            "regions": regions,
        }
