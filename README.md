# Plant Disease Detection (Python · CNN · RNN)

Classifies leaf images into healthy/diseased classes with two Keras architectures and a **classical OpenCV lesion-segmentation pipeline** that gives explainable lesion-area features.

| Model | Idea |
|---|---|
| `cnn` | 4 double-conv blocks (32-64-128-128, BN+ReLU) -> GAP -> softmax |
| `cnn_rnn` | same conv stack; final feature map read as a **sequence of rows** (H steps x W*C features) -> Bidirectional GRU(96) -> softmax |

Also: augmentation (flip, rotation, zoom, contrast), Grad-CAM (`src/gradcam.py`), EarlyStopping / ReduceLROnPlateau / ModelCheckpoint, per-class precision/recall + confusion matrix.

> TensorFlow was unavailable where this was authored: `src/model.py`, `src/train.py`, `src/gradcam.py` are written but **not executed**. The OpenCV segmentation, features and baseline are executed and tested.

## Run
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v                                  # 6 tests
python scripts/make_synthetic_leaves.py --out data/leaves --per_class 150
python scripts/baseline_segmentation.py data/leaves                      # no TensorFlow needed
# real data: PlantVillage, folder-per-class under data/leaves/<class>/*.jpg
python -m src.train --data data/leaves --arch cnn
python -m src.train --data data/leaves --arch cnn_rnn
```

## Measured here (synthetic leaves with ground-truth masks; validates the code, not a real benchmark)
- HSV lesion segmentation vs ground-truth masks on 300 diseased leaves: **mean IoU 0.860, median 0.891, mean absolute lesion-area error 13.1%**.
- Segmentation + 16-d feature extraction: **~1.0 ms per 128x128 image**.
- Random forest on lesion features separates the 3 synthetic classes perfectly (135 test images) because the classes are designed to differ in lesion count/size; treat this as a sanity check only.

| Real-data result (fill in after training on PlantVillage) | CNN | CNN-RNN |
|---|---|---|
| #classes / #images / split | | |
| Test accuracy · macro-F1 · top-3 | | |
| Parameters (`model.count_params()`) | | |
