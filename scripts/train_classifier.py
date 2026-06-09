"""
Train SVM and k-NN character classifiers.

Data sources (tried in order, or pick with --dataset):
  1. emnist   – EMNIST Balanced via OpenML (47 classes, 131,600 samples) – needs internet
  2. mnist    – MNIST digits via OpenML   (10 classes,  70,000 samples) – needs internet
  3. synthetic – Generated offline with OpenCV (47 classes, configurable samples)

Usage:
    python scripts/train_classifier.py                          # auto-pick best source
    python scripts/train_classifier.py --dataset synthetic      # guaranteed offline
    python scripts/train_classifier.py --dataset mnist          # digits only, fast
    python scripts/train_classifier.py --classifier svm --samples 20000
    python scripts/train_classifier.py --classifier knn
"""

import sys, os, time, argparse, pickle, warnings
import numpy as np
import cv2
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from recognition import HOGFeatureExtractor, SVMClassifier, KNNClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

MODELS_DIR = os.path.join(ROOT, 'data', 'models')
RAW_DIR    = os.path.join(ROOT, 'data', 'raw')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# EMNIST Balanced class mapping (47 classes)
EMNIST_CHARS = (list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
                list('abdefghnqrt'))


# ─────────────────────────────────────────────────────────────────
#  Data loaders
# ─────────────────────────────────────────────────────────────────

def _load_openml(name, version=1):
    from sklearn.datasets import fetch_openml
    print(f"  Downloading {name} from OpenML (cached after first run)…", end='', flush=True)
    t0 = time.time()
    ds = fetch_openml(name, version=version, as_frame=False,
                      data_home=RAW_DIR)
    print(f" {time.time()-t0:.0f}s")
    return ds.data.astype(np.uint8), ds.target


def load_emnist(n_samples=None):
    data, target = _load_openml('EMNIST_Balanced')
    X = data.reshape(-1, 28, 28).transpose(0, 2, 1)
    y = np.array([EMNIST_CHARS[int(t)] if int(t) < len(EMNIST_CHARS)
                  else str(t) for t in target])
    if n_samples:
        idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X, y = X[idx], y[idx]
    return X, y


def load_mnist(n_samples=None):
    data, target = _load_openml('mnist_784')
    X = data.reshape(-1, 28, 28)
    y = np.array([str(int(t)) for t in target])
    if n_samples:
        idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X, y = X[idx], y[idx]
    print(f"  MNIST: {len(X)} samples, classes: {sorted(set(y))}")
    return X, y


def load_synthetic(n_per_class=200, img_size=28):
    """
    Generate character images offline using OpenCV text rendering.
    Classes: 0-9, A-Z, a-z  (62 classes).
    Augmentation: 4 fonts × slight rotation/scale jitter.
    """
    chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]

    print(f"  Generating {len(chars)} × {n_per_class} synthetic chars "
          f"({len(chars)*n_per_class} total)…", end='', flush=True)
    t0 = time.time()

    images, labels = [], []
    rng = np.random.default_rng(42)

    for ch in chars:
        per_font = max(1, n_per_class // len(fonts))
        for font in fonts:
            for _ in range(per_font):
                canvas = np.ones((img_size, img_size), np.uint8) * 255
                scale  = rng.uniform(0.5, 0.9)
                thick  = rng.integers(1, 3)
                # centre the character
                (tw, th), _ = cv2.getTextSize(ch, font, scale, thick)
                ox = max(0, (img_size - tw) // 2)
                oy = max(0, (img_size + th) // 2)
                cv2.putText(canvas, ch, (ox, oy), font, scale, 0, thick, cv2.LINE_AA)

                # Augment: small rotation + mild noise
                angle  = rng.uniform(-8, 8)
                M      = cv2.getRotationMatrix2D((img_size//2, img_size//2), angle, 1.0)
                canvas = cv2.warpAffine(canvas, M, (img_size, img_size), borderValue=255)
                noise  = rng.normal(0, 8, canvas.shape).astype(np.int16)
                canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                images.append(canvas)
                labels.append(ch)

    X = np.array(images)
    y = np.array(labels)
    # shuffle
    idx = rng.permutation(len(X))
    print(f" {time.time()-t0:.1f}s  |  {len(X)} samples, {len(chars)} classes")
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────────────────────────

def extract_hog(images):
    hog     = HOGFeatureExtractor()
    imgs_32 = np.array([cv2.resize(img, (32, 32)) for img in images])
    print(f"  Extracting HOG features…", end='', flush=True)
    t0 = time.time()
    feats = hog.extract_batch(imgs_32)
    print(f" done ({time.time()-t0:.1f}s)  dim={feats.shape[1]}")
    return feats


# ─────────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────────

def train_svm(X_tr, y_tr, X_te, y_te):
    print("\n── SVM (RBF kernel, C=10) ─────────────────────────────────────")
    clf = SVMClassifier(kernel='rbf', C=10.0, gamma='scale')
    t0  = time.time()
    clf.train(X_tr, y_tr)
    print(f"  Trained in {time.time()-t0:.1f}s")
    preds = clf.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    print(f"  Test accuracy: {acc*100:.2f}%")
    # Print per-class report (truncated for readability)
    report = classification_report(y_te, preds, zero_division=0)
    lines  = report.split('\n')
    print('\n'.join(lines[:min(20, len(lines))]))
    if len(lines) > 20:
        print(f"  … ({len(lines)-20} more lines)")
    return clf, acc


def train_knn(X_tr, y_tr, X_te, y_te, k=5):
    print(f"\n── k-NN (k={k}) ────────────────────────────────────────────────")
    clf = KNNClassifier(n_neighbors=k)
    t0  = time.time()
    clf.train(X_tr, y_tr)
    print(f"  Trained in {time.time()-t0:.1f}s")
    preds = clf.predict(X_te)
    acc   = accuracy_score(y_te, preds)
    print(f"  Test accuracy: {acc*100:.2f}%")
    return clf, acc


def save_model(clf, scaler, acc, name, clf_type):
    path = os.path.join(MODELS_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'clf': clf.clf, 'encoder': clf.label_encoder,
                     'scaler': scaler, 'accuracy': acc, 'type': clf_type}, f)
    size_kb = os.path.getsize(path) // 1024
    print(f"  Saved → {path}  ({size_kb} KB,  accuracy: {acc*100:.2f}%)")
    return path


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train OCR classifiers')
    parser.add_argument('--dataset', choices=['emnist', 'mnist', 'synthetic', 'auto'],
                        default='auto',
                        help='Dataset to train on (default: auto – tries emnist → mnist → synthetic)')
    parser.add_argument('--classifier', choices=['svm', 'knn', 'both'],
                        default='both')
    parser.add_argument('--samples', type=int, default=None,
                        help='Max training samples (default: all)')
    parser.add_argument('--samples-per-class', type=int, default=200,
                        help='Synthetic dataset: samples per class (default: 200)')
    parser.add_argument('--test-split', type=float, default=0.15)
    args = parser.parse_args()

    np.random.seed(42)

    print("=" * 60)
    print("  OCR Classifier Training")
    print("=" * 60)

    # ── 1. Load data ────────────────────────────────────────────────
    X, y = None, None
    dataset_used = args.dataset

    def try_load(fn, name):
        try:
            return fn(), name
        except Exception as e:
            print(f"  [{name}] failed: {e}")
            return None, None

    if args.dataset in ('emnist', 'auto'):
        (X, y), dataset_used = try_load(
            lambda: load_emnist(args.samples), 'emnist')

    if X is None and args.dataset in ('mnist', 'auto'):
        (X, y), dataset_used = try_load(
            lambda: load_mnist(args.samples), 'mnist')

    if X is None:
        print("  Falling back to offline synthetic dataset…")
        X, y = load_synthetic(n_per_class=args.samples_per_class)
        dataset_used = 'synthetic'
        if args.samples:
            X, y = X[:args.samples], y[:args.samples]

    print(f"\nDataset: {dataset_used}  |  samples: {len(X)}  |  "
          f"classes: {len(set(y))}  |  shape: {X.shape}")

    # ── 2. Extract HOG features ─────────────────────────────────────
    features = extract_hog(X)

    # ── 3. Scale and split ──────────────────────────────────────────
    scaler   = StandardScaler()
    features = scaler.fit_transform(features)

    X_tr, X_te, y_tr, y_te = train_test_split(
        features, y,
        test_size=args.test_split,
        random_state=42,
        stratify=y,
    )
    print(f"Split  →  train: {len(X_tr)}  |  test: {len(X_te)}")

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {scaler_path}")

    results = {}

    # ── 4. Train ────────────────────────────────────────────────────
    if args.classifier in ('svm', 'both'):
        clf, acc  = train_svm(X_tr, y_tr, X_te, y_te)
        model_name = f'svm_{dataset_used}'
        path       = save_model(clf, scaler, acc, model_name, 'svm')
        results['svm'] = {'path': path, 'acc': acc}

    if args.classifier in ('knn', 'both'):
        clf, acc  = train_knn(X_tr, y_tr, X_te, y_te)
        model_name = f'knn_{dataset_used}'
        path       = save_model(clf, scaler, acc, model_name, 'knn')
        results['knn'] = {'path': path, 'acc': acc}

    # ── 5. Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, info in results.items():
        target = "✓ meets 90% target" if info['acc'] >= 0.90 else f"({info['acc']*100:.1f}% – retrain with more data)"
        print(f"  {name.upper()}  accuracy: {info['acc']*100:.2f}%  {target}")
        print(f"       model: {info['path']}")
    print()
    best_model = max(results.values(), key=lambda v: v['acc'])['path']
    print("Next step – test with a real image:")
    print(f"  python scripts/test_with_image.py --image /path/to/doc.jpg --model {best_model}")
    print()
    print("Or test with the generated synthetic invoice:")
    print(f"  python scripts/test_with_image.py --generate --model {best_model}")


if __name__ == '__main__':
    main()
