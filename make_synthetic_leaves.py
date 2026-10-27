"""Synthetic leaf images WITH ground-truth lesion masks (for validating the segmentation code; not a real benchmark).
classes: healthy (0 lesions) | leaf_spot (6-15 small brown spots) | blight (1-3 large irregular necrotic patches)"""
import argparse
from pathlib import Path
import cv2
import numpy as np

SIZE = 128


def make_leaf(rng, cls):
    img = np.full((SIZE, SIZE, 3), 245, np.uint8)
    leaf = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(leaf, (64 + int(rng.integers(-6, 7)), 64), (int(rng.integers(40, 52)), int(rng.integers(26, 36))),
                float(rng.uniform(-25, 25)), 0, 360, 1, -1)
    green = np.array([int(rng.integers(45, 70)), int(rng.integers(130, 170)), int(rng.integers(35, 60))])  # BGR
    tex = rng.normal(0, 7, (SIZE, SIZE, 1))
    img[leaf > 0] = np.clip(green + tex[leaf > 0], 0, 255).astype(np.uint8)
    gt = np.zeros((SIZE, SIZE), np.uint8)
    ys, xs = np.nonzero(leaf)
    if cls == "leaf_spot":
        for _ in range(int(rng.integers(6, 16))):
            i = rng.integers(len(ys)); cv2.circle(gt, (int(xs[i]), int(ys[i])), int(rng.integers(2, 5)), 1, -1)
    elif cls == "blight":
        for _ in range(int(rng.integers(1, 4))):
            i = rng.integers(len(ys)); pts = []
            for ang in np.linspace(0, 2 * np.pi, 9)[:-1]:
                r = rng.uniform(7, 15); pts.append([xs[i] + r * np.cos(ang), ys[i] + r * np.sin(ang)])
            cv2.fillPoly(gt, [np.array(pts, np.int32)], 1)
    gt &= leaf
    brown = np.array([int(rng.integers(20, 45)), int(rng.integers(50, 85)), int(rng.integers(95, 140))])
    img[gt > 0] = np.clip(brown + rng.normal(0, 6, (int((gt > 0).sum()), 3)), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, gt.astype(bool), leaf.astype(bool)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/leaves"); ap.add_argument("--per_class", type=int, default=150)
    a = ap.parse_args(); rng = np.random.default_rng(3)
    for cls in ("healthy", "leaf_spot", "blight"):
        d = Path(a.out) / cls; d.mkdir(parents=True, exist_ok=True)
        (Path(a.out).parent / "masks" / cls).mkdir(parents=True, exist_ok=True)
        for k in range(a.per_class):
            img, gt, _ = make_leaf(rng, cls)
            cv2.imwrite(str(d / f"{k:04d}.png"), img); cv2.imwrite(str(Path(a.out).parent / "masks" / cls / f"{k:04d}.png"), gt.astype(np.uint8) * 255)
    print("wrote", 3 * a.per_class, "images")
