"""
Automated Document Analysis and OCR Pipeline
Stage 1: Preprocessing -> Stage 2: Layout Analysis ->
Stage 3: Character Recognition -> Stage 4: Post-Processing
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np

# Allow running as a script from any directory
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import Preprocessor
from layout_analysis import LayoutAnalyzer, RLSAProcessor, ConnectedComponentAnalyzer
from recognition import CharacterRecognizer, CharacterNormalizer, WordAssembler
from postprocessing import PostProcessor


def _tesseract_available():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


class OCRPipeline:
    def __init__(
        self,
        binarization='otsu',
        correct_skew=True,
        classifier='svm',
        confidence_threshold=0.55,
        spell_max_distance=2,
        use_tesseract=True,
        model_path=None,
        output_format='json',
    ):
        self.binarization = binarization
        self.correct_skew = correct_skew
        self.output_format = output_format

        # Auto-detect Tesseract; warn and fall back when not installed
        if use_tesseract and not _tesseract_available():
            import warnings
            warnings.warn(
                "Tesseract not found – falling back to custom HOG+classifier pipeline. "
                "Install tesseract-ocr for better accuracy.",
                RuntimeWarning, stacklevel=2,
            )
            use_tesseract = False
        self.use_tesseract = use_tesseract

        self.preprocessor = Preprocessor()
        self.layout_analyzer = LayoutAnalyzer()
        self.char_recognizer = CharacterRecognizer(
            classifier=classifier,
            confidence_threshold=confidence_threshold,
        )
        self.normalizer = CharacterNormalizer()
        self.word_assembler = WordAssembler()
        self.post_processor = PostProcessor(spell_max_distance=spell_max_distance)

        if model_path and os.path.isfile(model_path):
            self.char_recognizer.load_model(model_path)

    # ------------------------------------------------------------------
    # Stage 1 – Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, image_path):
        result = self.preprocessor.process(
            image_path,
            binarization=self.binarization,
            correct_skew=self.correct_skew,
        )
        return result

    # ------------------------------------------------------------------
    # Stage 2 – Layout Analysis
    # ------------------------------------------------------------------
    def analyze_layout(self, binary_image):
        return self.layout_analyzer.analyze(binary_image)

    # ------------------------------------------------------------------
    # Stage 3 – Text Recognition
    # ------------------------------------------------------------------
    def recognize_text(self, binary_image, gray_image, layout_result):
        if self.use_tesseract:
            return self._tesseract_recognition(gray_image)

        components = layout_result['components']
        if not components:
            # No character-level components found – fall back to word blocks
            return self._word_block_extraction(binary_image)

        patches = self.normalizer.extract_char_patches(binary_image, components)
        if self.char_recognizer.clf.is_trained:
            results = self.char_recognizer.recognize_batch(patches)
            labels = [r[0] for r in results]
        else:
            # No trained model – use word block detection for meaningful output
            return self._word_block_extraction(binary_image)

        words = self.word_assembler.assemble(components, labels)
        return words, labels

    def _tesseract_recognition(self, gray_image):
        import pytesseract
        config = '--oem 3 --psm 6'
        raw_text = pytesseract.image_to_string(gray_image, config=config)
        words = raw_text.split()
        return words, []

    def _word_block_extraction(self, binary_image):
        """Fallback: RLSA to find word-level blobs; label each with '[word]'."""
        rlsa = RLSAProcessor()
        # threshold = ~1.5x avg character width; 30px works for typical 600px-wide docs
        h, w = binary_image.shape[:2]
        h_thresh = max(15, w // 20)
        smoothed = rlsa.horizontal_rlsa(binary_image, threshold=h_thresh)
        # After RLSA, text=0 blobs span full words; invert for CC analysis
        inv = cv2.bitwise_not(smoothed)
        cca = ConnectedComponentAnalyzer()
        comps, _ = cca.find_components(inv)
        comps = cca.filter_components(comps, min_area=50, max_area=w * h // 4,
                                       min_aspect=0.2, max_aspect=40.0)
        comps_sorted = sorted(comps, key=lambda c: (c['bbox'][1], c['bbox'][0]))
        return ['[word]' for _ in comps_sorted], comps_sorted

    # ------------------------------------------------------------------
    # Stage 4 – Post-Processing
    # ------------------------------------------------------------------
    def post_process(self, words, regions=None):
        return self.post_processor.format_output(
            words,
            regions=regions,
            fmt=self.output_format,
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        t0 = time.time()

        # Stage 1
        prep = self.preprocess(image_path)
        binary = prep['binary']
        gray = prep['gray']
        t1 = time.time()

        # Stage 2
        layout = self.analyze_layout(binary)
        regions = layout['regions']
        t2 = time.time()

        # Stage 3
        words, char_labels = self.recognize_text(binary, gray, layout)
        t3 = time.time()

        # Stage 4
        output = self.post_process(words, regions=regions)
        t4 = time.time()

        metrics = {
            'preprocessing_s': round(t1 - t0, 3),
            'layout_analysis_s': round(t2 - t1, 3),
            'recognition_s': round(t3 - t2, 3),
            'postprocessing_s': round(t4 - t3, 3),
            'total_s': round(t4 - t0, 3),
            'word_count': len(words),
            'region_count': len(regions),
        }

        return {
            'output': output,
            'metrics': metrics,
            'preprocessed': prep,
            'layout': layout,
            'words': words,
        }

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def compute_cer(ground_truth, hypothesis):
        """Character Error Rate = edit_distance / len(ground_truth)"""
        from postprocessing import levenshtein_distance
        gt = ground_truth.replace(' ', '')
        hyp = hypothesis.replace(' ', '')
        if len(gt) == 0:
            return 0.0
        return levenshtein_distance(gt, hyp) / len(gt)

    @staticmethod
    def compute_wer(ground_truth, hypothesis):
        """Word Error Rate = edit_distance_on_words / len(gt_words)"""
        from postprocessing import levenshtein_distance
        gt_words = ground_truth.split()
        hyp_words = hypothesis.split()
        if len(gt_words) == 0:
            return 0.0
        return levenshtein_distance(' '.join(gt_words), ' '.join(hyp_words)) / len(gt_words)

    @staticmethod
    def compute_iou(boxA, boxB):
        """Intersection over Union for two (x, y, w, h) bounding boxes."""
        ax1, ay1 = boxA[0], boxA[1]
        ax2, ay2 = ax1 + boxA[2], ay1 + boxA[3]
        bx1, by1 = boxB[0], boxB[1]
        bx2, by2 = bx1 + boxB[2], by1 + boxB[3]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Automated Document OCR Pipeline'
    )
    parser.add_argument('image', help='Path to input document image')
    parser.add_argument(
        '--binarization', choices=['otsu', 'adaptive', 'sauvola'],
        default='otsu', help='Binarization method (default: otsu)'
    )
    parser.add_argument(
        '--no-deskew', action='store_true',
        help='Disable automatic skew correction'
    )
    parser.add_argument(
        '--classifier', choices=['svm', 'knn'],
        default='svm', help='Character classifier (default: svm)'
    )
    parser.add_argument(
        '--no-tesseract', action='store_true',
        help='Use custom HOG+SVM/k-NN instead of Tesseract'
    )
    parser.add_argument(
        '--model', default=None,
        help='Path to pre-trained classifier model (.pkl)'
    )
    parser.add_argument(
        '--format', dest='output_format',
        choices=['json', 'txt', 'csv'],
        default='json', help='Output format (default: json)'
    )
    parser.add_argument(
        '--output', default=None,
        help='Save output to file (default: print to stdout)'
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    pipeline = OCRPipeline(
        binarization=args.binarization,
        correct_skew=not args.no_deskew,
        classifier=args.classifier,
        use_tesseract=not args.no_tesseract,
        model_path=args.model,
        output_format=args.output_format,
    )

    result = pipeline.run(args.image)

    print('=== OCR Pipeline Results ===')
    print(f"Words found      : {result['metrics']['word_count']}")
    print(f"Regions found    : {result['metrics']['region_count']}")
    print(f"Total time       : {result['metrics']['total_s']} s")
    print()

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result['output'])
        print(f"Output saved to: {args.output}")
    else:
        print(result['output'])


if __name__ == '__main__':
    main()