import cv2
import numpy as np
from skimage.filters import threshold_sauvola


class ImageLoader:
    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return image

    def convert_to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def resize_image(self, image, max_width=1500):
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        return image


class NoiseReducer:
    def gaussian_blur(self, image, kernel_size=5, sigma=1.0):
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def bilateral_filter(self, image, diameter=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

    def morphological_noise_removal(self, binary_image):
        # Only a mild opening to remove isolated salt noise (1-pixel blobs).
        # Closing is intentionally omitted: a 3×3 close destroys thin strokes
        # in small-font text (≤10pt equivalents at typical scan resolutions).
        k_open = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, k_open, iterations=1)

    def median_blur(self, image, kernel_size=3):
        return cv2.medianBlur(image, kernel_size)


class Binarizer:
    def otsu_threshold(self, image):
        thresh_val, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary, thresh_val

    def adaptive_threshold(self, image, block_size=11, C=2):
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, C
        )

    def sauvola_threshold(self, image, window_size=15, k=0.5):
        thresh = threshold_sauvola(image, window_size=window_size, k=k)
        return (image > thresh).astype(np.uint8) * 255

    def apply_morphology(self, binary, operation='open', kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        ops = {
            'erode':   lambda i: cv2.erode(i, kernel, iterations=1),
            'dilate':  lambda i: cv2.dilate(i, kernel, iterations=1),
            'open':    lambda i: cv2.morphologyEx(i, cv2.MORPH_OPEN, kernel),
            'close':   lambda i: cv2.morphologyEx(i, cv2.MORPH_CLOSE, kernel),
        }
        if operation not in ops:
            raise ValueError(f"Unknown operation: {operation}")
        return ops[operation](binary)


class GeometricCorrector:
    def detect_skew_angle(self, binary_image):
        edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return 0.0
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if abs(angle) < 45:
                angles.append(angle)
        return float(np.median(angles)) if angles else 0.0

    def deskew_image(self, image, angle):
        if abs(angle) < 0.1:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    def detect_corners(self, image):
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_f = np.float32(gray)
        harris = cv2.cornerHarris(gray_f, blockSize=5, ksize=3, k=0.04)
        harris = cv2.dilate(harris, None)
        threshold = 0.01 * harris.max()
        corner_coords = np.argwhere(harris > threshold)
        if len(corner_coords) == 0:
            h, w = gray.shape
            return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        points = corner_coords[:, ::-1].astype(np.float32)
        return self._select_document_corners(points, gray.shape)

    def _select_document_corners(self, points, shape):
        h, w = shape
        cx, cy = w / 2, h / 2
        sectors = {
            'tl': points[(points[:, 0] < cx) & (points[:, 1] < cy)],
            'tr': points[(points[:, 0] >= cx) & (points[:, 1] < cy)],
            'br': points[(points[:, 0] >= cx) & (points[:, 1] >= cy)],
            'bl': points[(points[:, 0] < cx) & (points[:, 1] >= cy)],
        }
        corners = []
        defaults = [[0, 0], [w, 0], [w, h], [0, h]]
        for sector, default in zip(['tl', 'tr', 'br', 'bl'], defaults):
            pts = sectors[sector]
            if len(pts) > 0:
                dists = np.sqrt((pts[:, 0] - default[0])**2 + (pts[:, 1] - default[1])**2)
                corners.append(pts[np.argmin(dists)])
            else:
                corners.append(np.array(default, dtype=np.float32))
        return np.array(corners, dtype=np.float32)

    def perspective_correction(self, image, corners):
        ordered = self._order_points(corners)
        tl, tr, br, bl = ordered
        w = int(max(
            np.linalg.norm(br - bl),
            np.linalg.norm(tr - tl)
        ))
        h = int(max(
            np.linalg.norm(tr - br),
            np.linalg.norm(tl - bl)
        ))
        dst = np.array([
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(image, M, (w, h))

    def _order_points(self, pts):
        pts = np.array(pts, dtype=np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect


class Preprocessor:
    """Facade that runs the full preprocessing pipeline on a document image."""

    def __init__(self):
        self.loader = ImageLoader()
        self.noise_reducer = NoiseReducer()
        self.binarizer = Binarizer()
        self.corrector = GeometricCorrector()

    def process(self, image_path, binarization='otsu', correct_skew=True):
        image = self.loader.load_image(image_path)
        image = self.loader.resize_image(image)
        gray = self.loader.convert_to_grayscale(image)
        denoised = self.noise_reducer.bilateral_filter(gray)

        if binarization == 'otsu':
            binary, _ = self.binarizer.otsu_threshold(denoised)
        elif binarization == 'adaptive':
            binary = self.binarizer.adaptive_threshold(denoised)
        elif binarization == 'sauvola':
            binary = self.binarizer.sauvola_threshold(denoised)
        else:
            raise ValueError(f"Unknown binarization method: {binarization}")

        binary = self.noise_reducer.morphological_noise_removal(binary)

        if correct_skew:
            angle = self.corrector.detect_skew_angle(binary)
            binary = self.corrector.deskew_image(binary, angle)
            gray = self.corrector.deskew_image(gray, angle)
            image = self.corrector.deskew_image(image, angle)

        return {
            'original': image,
            'gray': gray,
            'binary': binary,
        }
