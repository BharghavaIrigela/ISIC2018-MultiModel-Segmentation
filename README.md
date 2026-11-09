# ISIC 2018 Multi-Model Skin Lesion Segmentation

A research and educational prototype for binary lesion segmentation on ISIC 2018 Task 1 images, comparing three trained models:

- U-Net
- Swin-UNet
- TransUNet

## Deployment architecture

- **Frontend and inference:** static HTML dashboard on Vercel.
- **Model storage:** public Hugging Face repository: [bharghavairigela/isic2018-segmentation-models](https://huggingface.co/bharghavairigela/isic2018-segmentation-models).
- **Runtime:** ONNX Runtime Web using the WebAssembly execution provider.

The browser downloads ONNX model files from Hugging Face and performs inference locally. Uploaded images are not sent to an inference API. The Vercel deployment does not bundle PyTorch or model weights.

## Model artifacts

The ONNX models are stored in the Hugging Face repository:

```text
onnx_models/unet.onnx
onnx_models/swin_unet.onnx
onnx_models/transunet.onnx
```

Original TorchScript exports are also retained under `models/`.

## Input preprocessing

- RGB image
- Resize to 224 × 224 using bilinear interpolation
- ImageNet normalization:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- Binary mask threshold: 0.5
- ONNX output is converted to probabilities using sigmoid.

## Browser inference

1. Open the Vercel deployment.
2. Select a PNG or JPEG dermoscopic image (maximum 4 MB).
3. Click **Compare all three models**.
4. The browser downloads each ONNX model and runs it sequentially using ONNX Runtime Web.
5. View each model's overlay, binary mask, and probability map, alongside the majority-vote consensus.

The first run downloads approximately 338 MB of model files. Browser memory, available CPU resources, and network speed affect load and inference times. Models are loaded sequentially to reduce peak memory use. A modern desktop browser and a stable connection are recommended.

## Validation

All three ONNX exports were checked against their corresponding TorchScript models. Combined numerical validation reported 100% binary-mask agreement for each model on the same input tensor. This verifies export consistency for the tested input; it is not clinical validation or a substitute for broader dataset evaluation.

## Disclaimer

This project is for research and educational demonstration only. It is not clinically validated and must not be used to diagnose skin cancer or make treatment decisions. Probability maps are model outputs, not calibrated clinical confidence estimates. The consensus mask is a majority-vote combination, not an independently validated segmentation.
