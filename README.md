# ISIC 2018 Multi-Model Skin Lesion Segmentation

A research and educational prototype for binary skin-lesion segmentation on ISIC 2018 Task 1 images. It compares three TorchScript models:

- U-Net
- Swin-UNet
- TransUNet

The web interface is designed for Vercel. The FastAPI inference service is designed to run separately (for example, on a Hugging Face Docker Space).

## Architecture

- **Frontend:** static HTML/CSS/JavaScript, deployed on Vercel.
- **Backend:** FastAPI + PyTorch, deployed separately.
- **Model artifacts:** three TorchScript `.pt` files, stored outside this Git repository and mounted/downloaded by the backend into its `models/` directory.

The backend expects these files:

```text
models/
├── unet_scripted.pt
├── swin_unet_scripted.pt
└── transunet_scripted.pt
```

Do not commit model weights or private access tokens to this repository.

## API

- `GET /health` — backend status and loaded model information.
- `POST /predict-all` — run all three models and return their masks, probability maps, overlays, and a majority-vote consensus.
- `POST /predict/{model_key}` — run one model. Supported keys: `unet`, `swin_unet`, `transunet`.

The frontend sends a multipart image field named `file` to `/predict-all`. Supported image formats are JPEG and PNG; maximum upload size is 12 MB.

## Input preprocessing

- RGB image
- Resize to 224 × 224 using bilinear interpolation
- ImageNet normalization:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- Binary mask threshold: 0.5

## Local backend setup

Use Python 3.11 or a compatible Python version supported by the installed PyTorch release.

1. Install dependencies from the backend requirements file.
2. Place the three model files in `backend/models/` (or set `MODEL_DIR`).
3. Start the API with Uvicorn.

Example:

```bash
cd backend
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 7860
```

## Frontend setup

The static frontend is served from the repository root. In the frontend, enter the deployed backend base URL (without a trailing slash), check API health, and upload a JPEG or PNG image.

## Deployment notes

- Deploy the repository root as a static Vercel project.
- Deploy the FastAPI backend separately using a Docker-based service.
- Configure backend CORS with `ALLOWED_ORIGINS` set to the exact deployed frontend origin(s).
- Configure `MODEL_DIR` to point to the directory containing the TorchScript files.
- GPU availability depends on the backend host and its selected hardware plan.

## Evaluation and reproducibility

The repository includes model-export and inference benchmark metadata. These records describe the reported experiments and should not be interpreted as clinical validation.

## Disclaimer

This project is for research and educational demonstration only. It is not clinically validated and must not be used to diagnose skin cancer or make treatment decisions. Model probability maps are not calibrated clinical confidence estimates; the consensus mask is a majority-vote combination, not an independently validated segmentation.
