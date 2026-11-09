import base64
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image

# Vercel Python Function: api/index.py
REPO_ID = os.getenv(
    "HF_MODEL_REPO",
    "bharghavairigela/isic2018-segmentation-models",
)
IMAGE_SIZE = 224
THRESHOLD = 0.5
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MODEL_FILES = {
    "unet": "unet_scripted.pt",
    "swin_unet": "swin_unet_scripted.pt",
    "transunet": "transunet_scripted.pt",
}
MODEL_NAMES = {
    "unet": "U-Net",
    "swin_unet": "Swin-UNet",
    "transunet": "TransUNet",
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
torch.set_num_threads(1)

app = FastAPI(
    title="ISIC 2018 Multi-Model Segmentation API",
    version="1.0.0",
    description="Research prototype: U-Net, Swin-UNet and TransUNet segmentation.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_models = {}
_model_paths = {}
_lock = threading.Lock()


def _load_model(key):
    """Download a public TorchScript artifact once per warm function instance."""
    if key in _models:
        return _models[key]

    filename = MODEL_FILES[key]
    local_path = _model_paths.get(key)
    if not local_path or not Path(local_path).is_file():
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"models/{filename}",
            repo_type="model",
            token=os.getenv("HF_TOKEN") or None,
            cache_dir="/tmp/huggingface",
        )
        _model_paths[key] = local_path

    model = torch.jit.load(local_path, map_location="cpu")
    model.eval()
    _models[key] = model
    return model


def _png_b64(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay(rgb, mask):
    rgba = Image.fromarray(rgb).convert("RGBA")
    color = Image.new("RGBA", rgba.size, (255, 45, 60, 0))
    alpha = Image.fromarray(mask.astype(np.uint8) * 95, mode="L")
    color.putalpha(alpha)
    return np.asarray(Image.alpha_composite(rgba, color).convert("RGB"))


def _preprocess(image):
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return (tensor - MEAN) / STD, np.asarray(resized, dtype=np.uint8)


@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": "cpu",
        "image_size": IMAGE_SIZE,
        "threshold": THRESHOLD,
        "models_loaded": list(_models.keys()),
        "available_models": list(MODEL_FILES.keys()),
    }


@app.post("/api/predict-all")
@app.post("/predict-all")
async def predict_all(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 4 MB or smaller.")

    try:
        original = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unreadable image.") from exc

    original_width, original_height = original.size
    tensor, display_rgb = _preprocess(original)
    model_results = {}
    masks = {}
    start_all = time.perf_counter()

    try:
        with _lock, torch.inference_mode():
            for key in MODEL_FILES:
                start = time.perf_counter()
                model = _load_model(key)
                logits = model(tensor)
                probabilities = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
                binary = (probabilities >= THRESHOLD).astype(np.uint8)
                elapsed_ms = (time.perf_counter() - start) * 1000
                masks[key] = binary

                # Return only 224x224 result images to stay below Vercel's
                # 4.5 MB function response limit. The browser scales them.
                mask_u8 = binary * 255
                probability_u8 = np.clip(probabilities * 255, 0, 255).astype(np.uint8)
                overlay = _overlay(display_rgb, binary)

                model_results[key] = {
                    "name": MODEL_NAMES[key],
                    "inference_ms": round(elapsed_ms, 3),
                    "lesion_area_percent": round(float(binary.mean() * 100), 4),
                    "mask_png": _png_b64(mask_u8),
                    "probability_png": _png_b64(probability_u8),
                    "overlay_png": _png_b64(overlay),
                }

                # Drop transient tensors before the next model.
                del model, logits, probabilities, binary

            stack = np.stack(list(masks.values()), axis=0)
            consensus = (stack.sum(axis=0) >= 2).astype(np.uint8)
            consensus_overlay = _overlay(display_rgb, consensus)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed ({type(exc).__name__}): {exc}",
        ) from exc

    return {
        "status": "success",
        "image": {
            "width": original_width,
            "height": original_height,
            "processed_width": IMAGE_SIZE,
            "processed_height": IMAGE_SIZE,
        },
        "configuration": {
            "threshold": THRESHOLD,
            "device": "cpu",
            "precision": "FP32",
            "normalization": "ImageNet",
        },
        "recommended_reference_model": "unet",
        "selection_note": (
            "U-Net is the configured reference model. This is not an "
            "automatic per-image ground-truth ranking."
        ),
        "total_inference_ms": round((time.perf_counter() - start_all) * 1000, 3),
        "models": model_results,
        "consensus": {
            "method": "Majority vote (at least 2 of 3 models)",
            "mask_png": _png_b64(consensus * 255),
            "overlay_png": _png_b64(consensus_overlay),
            "lesion_area_percent": round(float(consensus.mean() * 100), 4),
        },
        "disclaimer": (
            "Research and educational prototype only. Not clinically "
            "validated and not intended for diagnosis."
        ),
    }
