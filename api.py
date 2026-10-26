"""FastAPI inference API for a locally trained plant-leaf classifier."""
from pathlib import Path
from functools import lru_cache
from io import BytesIO
import json, os
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS = Path(os.getenv("PLANT_ARTIFACTS", APP_DIR / "artifacts"))
MODEL_PATH = ARTIFACTS / "best_model.pt"
CLASSES_PATH = ARTIFACTS / "classes.json"
app = FastAPI(title="Plant Disease Detection API", version="1.0.0", description="Image classification inference for plant-leaf images. Predictions are advisory, not agronomic diagnoses.")

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists() and CLASSES_PATH.exists()}

@app.get("/classes")
def classes():
    if not CLASSES_PATH.exists():
        raise HTTPException(status_code=503, detail="No trained model classes found. Train a model first.")
    return {"classes": json.loads(CLASSES_PATH.read_text(encoding="utf-8"))}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    payload = await file.read()
    if len(payload) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 8 MB or smaller.")
    try:
        image = Image.open(BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise HTTPException(status_code=503, detail="Model unavailable. Follow README instructions to train one.")
    try:
        import torch
        from torchvision import models, transforms
        class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        image_size = checkpoint.get("image_size", 224)
        transform = transforms.Compose([transforms.Resize((image_size,image_size)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        with torch.inference_mode():
            probabilities = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
        idx = int(probabilities.argmax())
        return {"class_name": class_names[idx], "confidence": round(float(probabilities[idx]), 5), "disclaimer": "Model output is a screening aid; verify with an expert."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}")
