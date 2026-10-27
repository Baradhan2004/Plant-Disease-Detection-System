"""Runnable (no TensorFlow): segmentation quality (IoU vs ground truth) + classical classifier on lesion features."""
import sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.segmentation import lesion_features, segment, iou

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/leaves"); mroot = root.parent / "masks"
X, y, ious, area_err = [], [], [], []
names = sorted(p.name for p in root.iterdir()); t = time.time()
for ci, c in enumerate(names):
    for p in sorted((root / c).glob("*.png")):
        img = cv2.imread(str(p)); gt = cv2.imread(str(mroot / c / p.name), 0) > 0
        X.append(lesion_features(img)); y.append(ci)
        _, les = segment(img)
        if gt.any():
            ious.append(iou(les, gt)); area_err.append(abs(les.sum() - gt.sum()) / gt.sum())
dt = time.time() - t
X, y = np.array(X), np.array(y)
print(f"{len(y)} images, {len(names)} classes, feature dim {X.shape[1]}, {dt/len(y)*1000:.1f} ms/image (segment+features)")
print(f"Lesion segmentation vs ground truth: mean IoU {np.mean(ious):.3f}, median {np.median(ious):.3f}, mean |area error| {100*np.mean(area_err):.1f}%  (n={len(ious)} diseased leaves)")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
rf = RandomForestClassifier(300, random_state=0).fit(Xtr, ytr); p = rf.predict(Xte)
print(classification_report(yte, p, target_names=names, digits=3)); print("confusion:\n", confusion_matrix(yte, p))
imp = rf.feature_importances_[:4]; print("importance of [lesion_frac, n_lesions, mean_area, max_area]:", np.round(imp, 3))
