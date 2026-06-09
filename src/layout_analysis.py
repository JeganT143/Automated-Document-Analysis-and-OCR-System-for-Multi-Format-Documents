import cv2
import numpy as np


class ConnectedComponentAnalyzer:
    def find_components(self, binary_image):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        components = []
        for i in range(1, num_labels):  # skip background (label 0)
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            density = area / float(w * h) if w * h > 0 else 0.0
            aspect_ratio = w / float(h) if h > 0 else 0.0
            components.append({
                'label': i,
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': (cx, cy),
                'density': density,
                'aspect_ratio': aspect_ratio,
            })
        return components, labels

    def filter_components(self, components, min_area=50, max_area=100000,
                          min_aspect=0.1, max_aspect=20.0):
        return [
            c for c in components
            if min_area <= c['area'] <= max_area
            and min_aspect <= c['aspect_ratio'] <= max_aspect
        ]

    def draw_components(self, image, components, color=(0, 255, 0), thickness=1):
        out = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for c in components:
            x, y, w, h = c['bbox']
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        return out


class ContourDetector:
    def detect_contours(self, binary_image):
        contours, hierarchy = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
            regions.append({
                'contour': cnt,
                'bbox': (x, y, w, h),
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
            })
        return regions

    def filter_text_contours(self, regions, min_area=100, min_aspect=0.1, max_aspect=15.0):
        filtered = []
        for r in regions:
            x, y, w, h = r['bbox']
            aspect = w / float(h) if h > 0 else 0
            if r['area'] >= min_area and min_aspect <= aspect <= max_aspect:
                filtered.append(r)
        return filtered

    def draw_contours(self, image, regions, color=(255, 0, 0), thickness=1):
        out = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for r in regions:
            x, y, w, h = r['bbox']
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        return out


class RLSAProcessor:
    """Run-Length Smoothing Algorithm for grouping text into blocks."""

    def _apply_rlsa(self, binary_row_or_col, threshold):
        result = binary_row_or_col.copy()
        last_black = -1
        for i in range(len(result)):
            if result[i] == 0:
                if last_black >= 0 and (i - last_black) <= threshold:
                    result[last_black:i] = 0
                last_black = i
        return result

    def horizontal_rlsa(self, binary_image, threshold=50):
        # Binary: text=0, background=255.
        # Bridge background gaps <= threshold between adjacent text runs.
        result = binary_image.copy()
        for row_idx in range(binary_image.shape[0]):
            result[row_idx] = self._apply_rlsa(binary_image[row_idx], threshold)
        return result

    def vertical_rlsa(self, binary_image, threshold=20):
        result = binary_image.copy()
        for col_idx in range(binary_image.shape[1]):
            result[:, col_idx] = self._apply_rlsa(binary_image[:, col_idx], threshold)
        return result

    def combined_rlsa(self, binary_image, h_thresh=50, v_thresh=20):
        h_smooth = self.horizontal_rlsa(binary_image, h_thresh)
        v_smooth = self.vertical_rlsa(binary_image, v_thresh)
        combined = cv2.bitwise_and(h_smooth, v_smooth)
        return combined


class XYCutSegmenter:
    """Recursive X-Y Cut document segmentation."""

    def __init__(self, min_region_width=50, min_region_height=30, gap_threshold=5):
        self.min_region_width = min_region_width
        self.min_region_height = min_region_height
        self.gap_threshold = gap_threshold

    def _find_cut(self, profile, min_gap):
        gaps = []
        in_gap = False
        start = 0
        for i, v in enumerate(profile):
            if v == 0:
                if not in_gap:
                    start = i
                    in_gap = True
            else:
                if in_gap:
                    gap_len = i - start
                    if gap_len >= min_gap:
                        gaps.append((start, i, gap_len))
                    in_gap = False
        if gaps:
            best = max(gaps, key=lambda g: g[2])
            return (best[0] + best[1]) // 2
        return None

    def _segment(self, binary, x, y, w, h, regions):
        if w < self.min_region_width or h < self.min_region_height:
            regions.append((x, y, w, h))
            return

        roi = binary[y:y + h, x:x + w]

        # Horizontal projection (rows)
        h_proj = (roi == 0).sum(axis=1)
        h_cut = self._find_cut(h_proj, self.gap_threshold)
        if h_cut is not None:
            self._segment(binary, x, y, w, h_cut, regions)
            self._segment(binary, x, y + h_cut, w, h - h_cut, regions)
            return

        # Vertical projection (columns)
        v_proj = (roi == 0).sum(axis=0)
        v_cut = self._find_cut(v_proj, self.gap_threshold)
        if v_cut is not None:
            self._segment(binary, x, y, v_cut, h, regions)
            self._segment(binary, x + v_cut, y, w - v_cut, h, regions)
            return

        regions.append((x, y, w, h))

    def segment(self, binary_image):
        h, w = binary_image.shape
        regions = []
        self._segment(binary_image, 0, 0, w, h, regions)
        return regions


class RegionClassifier:
    """Classify document regions as text, table, image, or header/footer."""

    def __init__(self):
        self.header_footer_ratio = 0.12

    def classify_region(self, binary_image, bbox):
        x, y, w, h = bbox
        roi = binary_image[y:y + h, x:x + w]
        if roi.size == 0:
            return 'unknown'

        img_h = binary_image.shape[0]
        rel_y_top = y / img_h
        rel_y_bot = (y + h) / img_h

        black_pixels = (roi == 0).sum()
        total_pixels = roi.size
        density = black_pixels / total_pixels if total_pixels > 0 else 0

        # Horizontal line density for table detection
        h_lines = self._count_horizontal_lines(roi)
        aspect = w / float(h) if h > 0 else 1.0

        if rel_y_top < self.header_footer_ratio or rel_y_bot > (1 - self.header_footer_ratio):
            return 'header_footer'
        if h_lines >= 3 and aspect > 1.5:
            return 'table'
        if density < 0.02:
            return 'image'
        return 'text'

    def _count_horizontal_lines(self, roi):
        h_proj = (roi == 0).sum(axis=1)
        threshold = roi.shape[1] * 0.6
        count = 0
        in_line = False
        for v in h_proj:
            if v > threshold:
                if not in_line:
                    count += 1
                    in_line = True
            else:
                in_line = False
        return count

    def classify_all(self, binary_image, regions):
        return [
            {'bbox': r, 'type': self.classify_region(binary_image, r)}
            for r in regions
        ]


class LayoutAnalyzer:
    """Facade combining all layout analysis steps."""

    def __init__(self):
        self.cc_analyzer = ConnectedComponentAnalyzer()
        self.contour_detector = ContourDetector()
        self.rlsa = RLSAProcessor()
        self.xy_cut = XYCutSegmenter()
        self.classifier = RegionClassifier()

    def analyze(self, binary_image):
        # binary_image convention: text=0, background=255.
        # connectedComponentsWithStats needs foreground=255, so invert first.
        inv = cv2.bitwise_not(binary_image)
        components, labels = self.cc_analyzer.find_components(inv)
        # Keep only character-sized blobs: not too small (noise) nor too wide
        # (merged words). Width cap = image_width / 10 ~ one char per 10-col grid.
        img_h, img_w = binary_image.shape[:2]
        max_char_w = max(40, img_w // 10)
        max_char_h = max(60, img_h // 10)
        components = self.cc_analyzer.filter_components(
            components,
            min_area=20, max_area=max_char_w * max_char_h,
            min_aspect=0.05, max_aspect=10.0,
        )
        components = [c for c in components
                      if c['bbox'][2] <= max_char_w and c['bbox'][3] <= max_char_h]

        # RLSA bridges gaps between text runs (operates on text=0 image)
        smoothed = self.rlsa.combined_rlsa(binary_image)

        # X-Y cut on smoothed image
        regions = self.xy_cut.segment(smoothed)

        # Classify each region
        classified = self.classifier.classify_all(binary_image, regions)

        return {
            'components': components,
            'labels': labels,
            'smoothed': smoothed,
            'regions': classified,
        }
