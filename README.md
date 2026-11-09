# ISIC 2018 Multi-Model Skin Lesion Segmentation

A research and educational prototype for binary lesion segmentation on ISIC 2018 Task 1 images, comparing three TorchScript models:

- U-Net
- Swin-UNet
- TransUNet

## Deployment architecture

- **Frontend:** static HTML dashboard served by Vercel.
- **Inference API:** Vercel Python Function in `api/index.py` (FastAPI).
- **Model storage:** Hugging Face model repository: [bharghavairigela/isic2018-segmentation-models](https://huggingface.co/bharghavairigela/isic2018-segmentation-models).

The API downloads the three public TorchScript files from the Hugging Face repository on demand and caches them in the warm function instance. No model weights are committed to this Git repository.

## API endpoints

- `GET /api/health` — API health and model loading status.
- `POST /api/predict-all` — run all three models and return masks, probability maps, overlays, and majority-vote consensus.

The prediction endpoint accepts a multipart form field named `file`, with a JPEG or PNG image. The current frontend limits uploads to 4 MB to stay below Vercel Function request-body limits.

## Input preprocessing

- RGB image
- Resize to 224 × 224 using bilinear interpolation
- ImageNet normalization:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- Binary mask threshold: 0.5

The API returns result images at 224 × 224. The browser scales them for display. This keeps responses compact and avoids returning full-resolution base64 images.

## Deploy on Vercel

1. Import this GitHub repository into Vercel.
2. Set Root Directory to `./`.
3. Use the **Other** framework preset.
4. Keep the Python runtime files at the repository root:
   - `api/index.py`
   - `requirements.txt`
   - `.python-version`
   - `vercel.json`
5. Deploy and open the generated Vercel URL.
6. Leave the dashboard's API URL blank (or use the default same-origin URL), then click **Check API status**.

The Hugging Face model repository is public, so no access token is required for downloading these model artifacts. If the model repository is made private later, configure a Hugging Face read token as the Vercel environment variable `HF_TOKEN`; never commit the token.

## Runtime considerations

The API uses CPU inference and one PyTorch thread. The models are cached in a warm function instance after first use. Cold starts can take longer because model files must be downloaded and loaded. Vercel function bundle-size, memory, duration, and account-plan limits apply. Deployment must be tested on the selected Vercel plan; successful build does not guarantee that three-model inference will fit all runtime limits.

## Model artifacts

The three TorchScript files are stored in the Hugging Face repository under:

```text
models/unet_scripted.pt
models/swin_unet_scripted.pt
models/transunet_scripted.pt
```

## Evaluation

The repository's benchmark and export metadata describe the reported experiments. They are not clinical validation.

## Disclaimer

This project is for research and educational demonstration only. It is not clinically validated and must not be used to diagnose skin cancer or make treatment decisions. Probability maps are model outputs, not calibrated clinical confidence estimates. The consensus mask is a majority-vote combination, not an independently validated segmentation.
