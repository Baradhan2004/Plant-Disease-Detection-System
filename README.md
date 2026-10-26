# Plant Disease Detection | EfficientNet-B0 + FastAPI

A reproducible leaf-image classification project with a training pipeline, held-out evaluation, single-image inference, and an HTTP API. This is a new portfolio implementation based on the project title in the resume; it is not a reconstruction of an unavailable original model or dataset.

## Highlights
- Transfer learning with EfficientNet-B0 and configurable class labels.
- Separate `train`, `val`, and optional `test` directories; test data is not used for model selection.
- Validation macro-F1, per-class metrics, confusion matrix, and saved best checkpoint.
- FastAPI `/health`, `/classes`, and `/predict` endpoints, upload size/type checks, Dockerfile, and smoke tests.
- Class names come from directory names; no disease classes or accuracy figures are assumed.

## Dataset layout
```text
data/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg  # optional, final evaluation only
```
Use a licensed dataset and document its source, version, and licence. Keep near-duplicate images from the same plant/session in a single split to reduce leakage.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py --data-dir data --epochs 10 --batch-size 32
python evaluate.py --data-dir data/test --checkpoint artifacts/best_model.pt
uvicorn api:app --reload
pytest -q
```
Open `http://127.0.0.1:8000/docs` for interactive API docs. `POST /predict` accepts multipart field `file` with a JPEG, PNG, or WebP image. Train first to create `artifacts/best_model.pt` and `artifacts/classes.json`.

## Docker
```bash
docker build -t plant-disease-api .
docker run --rm -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" plant-disease-api
```

## Evaluation and limitations
Do not report performance until a held-out test set has been evaluated. Leaf background, lighting, cultivar, camera, and geography can shift predictions. Confidence is not a calibrated probability, and predictions are not a substitute for expert diagnosis. The API returns `503` until a checkpoint and class list exist.
