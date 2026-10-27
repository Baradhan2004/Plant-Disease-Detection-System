"""Classical lesion segmentation (OpenCV, HSV). Used for explainable lesion-area features and as a baseline.

leaf mask   = pixels that are not background (background is near-white / low saturation)
healthy     = hue in the green band
lesion mask = leaf pixels that are not healthy green (brown / yellow / dark necrotic tissue)
"""
import cv2
import numpy as np

GREEN_LO, GREEN_HI = (35, 40, 30), (85, 255, 255)


def segment(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    leaf = cv2.inRange(hsv, (0, 35, 15), (180, 255, 255)) > 0       # saturated pixels = leaf tissue
    leaf = cv2.morphologyEx(leaf.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI) > 0
    lesion = leaf & ~green
    lesion = cv2.morphologyEx(lesion.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return leaf, lesion


def lesion_features(bgr):
    leaf, lesion = segment(bgr)
    n, _, stats, _ = cv2.connectedComponentsWithStats(lesion.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], leaf.astype(np.uint8), [12], [0, 180]).ravel()
    hist = hist / max(hist.sum(), 1)
    leaf_px = max(int(leaf.sum()), 1)
    return np.concatenate([[lesion.sum() / leaf_px, n - 1, areas.mean() / leaf_px, areas.max() / leaf_px], hist])


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0
