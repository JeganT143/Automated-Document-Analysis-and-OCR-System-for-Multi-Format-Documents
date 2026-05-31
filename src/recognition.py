import cv2
import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


CHAR_SIZE = 32  # normalized character patch size


class CharacterNormalizer:
    def normalize(self, char_image, size=CHAR_SIZE):
        if char_image is None or char_image.size == 0:
            return np.zeros((size, size), dtype=np.uint8)
        gray = char_image if len(char_image.shape) == 2 else cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        return resized

    def extract_char_patches(self, binary_image, components, padding=2):
        patches = []
        for comp in components:
            x, y, w, h = comp['bbox']
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(binary_image.shape[1], x + w + padding)
            y2 = min(binary_image.shape[0], y + h + padding)
            patch = binary_image[y1:y2, x1:x2]
            patches.append(self.normalize(patch))
        return patches


class HOGFeatureExtractor:
    def __init__(self, win_size=CHAR_SIZE, block_size=8, block_stride=4,
                 cell_size=4, nbins=9):
        self.win_size = (win_size, win_size)
        self.hog = cv2.HOGDescriptor(
            _winSize=self.win_size,
            _blockSize=(block_size, block_size),
            _blockStride=(block_stride, block_stride),
            _cellSize=(cell_size, cell_size),
            _nbins=nbins
        )

    def extract(self, image):
        if len(image.shape) == 2:
            img = image
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.win_size)
        descriptor = self.hog.compute(img)
        return descriptor.flatten()

    def extract_batch(self, images):
        return np.array([self.extract(img) for img in images])


class SVMClassifier:
    def __init__(self, kernel='rbf', C=10.0, gamma='scale'):
        self.clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def train(self, features, labels):
        encoded = self.label_encoder.fit_transform(labels)
        self.clf.fit(features, encoded)
        self.is_trained = True

    def predict(self, features):
        if not self.is_trained:
            raise RuntimeError("SVM not trained yet")
        encoded = self.clf.predict(features)
        return self.label_encoder.inverse_transform(encoded)

    def predict_proba(self, features):
        if not self.is_trained:
            raise RuntimeError("SVM not trained yet")
        return self.clf.predict_proba(features)

    def evaluate(self, features, labels):
        predictions = self.predict(features)
        return classification_report(labels, predictions)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'clf': self.clf, 'encoder': self.label_encoder}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.clf = data['clf']
        self.label_encoder = data['encoder']
        self.is_trained = True


class KNNClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def train(self, features, labels):
        encoded = self.label_encoder.fit_transform(labels)
        self.clf.fit(features, encoded)
        self.is_trained = True

    def predict(self, features):
        if not self.is_trained:
            raise RuntimeError("k-NN not trained yet")
        encoded = self.clf.predict(features)
        return self.label_encoder.inverse_transform(encoded)

    def predict_proba(self, features):
        if not self.is_trained:
            raise RuntimeError("k-NN not trained yet")
        return self.clf.predict_proba(features)

    def evaluate(self, features, labels):
        predictions = self.predict(features)
        return classification_report(labels, predictions)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'clf': self.clf, 'encoder': self.label_encoder}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.clf = data['clf']
        self.label_encoder = data['encoder']
        self.is_trained = True


class TemplateMatcher:
    """Fallback character recognition using normalized cross-correlation."""

    def __init__(self, template_dir=None):
        self.templates = {}  # label -> list of normalized patches
        if template_dir and os.path.isdir(template_dir):
            self._load_templates(template_dir)

    def _load_templates(self, directory):
        for fname in os.listdir(directory):
            if fname.endswith('.png') or fname.endswith('.jpg'):
                label = os.path.splitext(fname)[0]
                img = cv2.imread(os.path.join(directory, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (CHAR_SIZE, CHAR_SIZE))
                    self.templates.setdefault(label, []).append(img)

    def add_template(self, label, image):
        norm = cv2.resize(image, (CHAR_SIZE, CHAR_SIZE))
        self.templates.setdefault(label, []).append(norm)

    def match(self, char_image):
        if not self.templates:
            return None, 0.0
        query = cv2.resize(char_image, (CHAR_SIZE, CHAR_SIZE)).astype(np.float32)
        best_label, best_score = None, -1.0
        for label, tmps in self.templates.items():
            for tmp in tmps:
                tmp_f = tmp.astype(np.float32)
                result = cv2.matchTemplate(query, tmp_f, cv2.TM_CCOEFF_NORMED)
                score = float(result.max())
                if score > best_score:
                    best_score = score
                    best_label = label
        return best_label, best_score


class CharacterRecognizer:
    """Combines SVM and k-NN with template-matching fallback and confidence gating."""

    def __init__(self, classifier='svm', confidence_threshold=0.6,
                 template_dir=None):
        self.normalizer = CharacterNormalizer()
        self.hog = HOGFeatureExtractor()
        self.classifier_type = classifier
        self.confidence_threshold = confidence_threshold
        self.template_matcher = TemplateMatcher(template_dir)

        if classifier == 'svm':
            self.clf = SVMClassifier()
        elif classifier == 'knn':
            self.clf = KNNClassifier()
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

    def train(self, images, labels):
        normalized = [self.normalizer.normalize(img) for img in images]
        features = self.hog.extract_batch(normalized)
        self.clf.train(features, labels)

    def recognize(self, char_image):
        norm = self.normalizer.normalize(char_image)
        feat = self.hog.extract(norm).reshape(1, -1)

        if not self.clf.is_trained:
            # Pure template matching when no classifier is trained
            label, score = self.template_matcher.match(norm)
            return label or '?', score

        probas = self.clf.predict_proba(feat)[0]
        confidence = probas.max()
        predicted = self.clf.predict(feat)[0]

        if confidence < self.confidence_threshold:
            tm_label, tm_score = self.template_matcher.match(norm)
            if tm_label is not None and tm_score > confidence:
                return tm_label, tm_score

        return predicted, float(confidence)

    def recognize_batch(self, char_images):
        results = []
        for img in char_images:
            label, conf = self.recognize(img)
            results.append((label, conf))
        return results

    def load_model(self, path):
        self.clf.load(path)

    def save_model(self, path):
        self.clf.save(path)


class WordAssembler:
    """Group character bounding boxes into words and lines based on spacing."""

    def __init__(self, word_gap_factor=1.5, line_gap_factor=2.0):
        self.word_gap_factor = word_gap_factor
        self.line_gap_factor = line_gap_factor

    def assemble(self, components, labels):
        if not components or not labels:
            return []

        paired = sorted(
            zip(components, labels),
            key=lambda t: (t[0]['bbox'][1], t[0]['bbox'][0])
        )

        lines = self._group_into_lines(paired)
        words = []
        for line in lines:
            words.extend(self._group_line_into_words(line))
        return words

    def _avg_char_width(self, paired):
        widths = [p[0]['bbox'][2] for p in paired]
        return float(np.median(widths)) if widths else 10.0

    def _avg_char_height(self, paired):
        heights = [p[0]['bbox'][3] for p in paired]
        return float(np.median(heights)) if heights else 10.0

    def _group_into_lines(self, paired):
        if not paired:
            return []
        avg_h = self._avg_char_height(paired)
        gap = avg_h * self.line_gap_factor
        lines = []
        current_line = [paired[0]]
        for item in paired[1:]:
            prev_y = current_line[-1][0]['bbox'][1]
            curr_y = item[0]['bbox'][1]
            if abs(curr_y - prev_y) > gap:
                lines.append(current_line)
                current_line = [item]
            else:
                current_line.append(item)
        lines.append(current_line)
        return lines

    def _group_line_into_words(self, line):
        if not line:
            return []
        line_sorted = sorted(line, key=lambda t: t[0]['bbox'][0])
        avg_w = self._avg_char_width(line_sorted)
        word_gap = avg_w * self.word_gap_factor

        words = []
        current_word_chars = [line_sorted[0][1]]
        prev_bbox = line_sorted[0][0]['bbox']

        for comp, label in line_sorted[1:]:
            x, y, w, h = comp['bbox']
            px, py, pw, ph = prev_bbox
            gap = x - (px + pw)
            if gap > word_gap:
                words.append(''.join(str(c) for c in current_word_chars))
                current_word_chars = [label]
            else:
                current_word_chars.append(label)
            prev_bbox = (x, y, w, h)

        words.append(''.join(str(c) for c in current_word_chars))
        return words
