# ISIC 2018 — Multi-Model Skin Lesion Segmentation

<p align="center">
  <strong>U-Net · Swin-UNet · TransUNet · Attention U-Net · DeepLabV3+ · FPN · ONNX Runtime Web · Vercel</strong>
</p>

<p align="center">
  <a href="https://isic-2018-multi-model-segmentation-qzlpm3mka.vercel.app/"><strong>🚀 Live Demo</strong></a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/bharghavairigela/isic2018-segmentation-models">🤗 Hugging Face Models</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/BharghavaIrigela/ISIC2018-MultiModel-Segmentation">💻 Source Code</a>
</p>

A browser-based research application for binary skin-lesion segmentation using three deep-learning architectures trained for the ISIC 2018 Challenge — Task 1. Upload a dermoscopic image, compare predictions from six segmentation models, inspect probability maps, and download the generated outputs.

> **Research and educational use only.** This project is not clinically validated. It must not be used to diagnose skin cancer, assess an individual's cancer risk, or make treatment decisions.

## Table of contents

- [Live demo](#-live-demo)
- [Features](#-features)
- [How to use](#-how-to-use)
- [Models](#-models)
- [Dataset and experimental setup](#-dataset-and-experimental-setup)
- [Test-set results](#-test-set-results)
- [Preprocessing and inference](#-preprocessing-and-inference)
- [Understanding the probability score](#-understanding-the-probability-score)
- [Deployment architecture](#-deployment-architecture)
- [Model files](#-model-files)
- [Run locally](#-run-locally)
- [Repository structure](#-repository-structure)
- [Limitations](#-limitations)
- [Acknowledgements](#-acknowledgements)

## 🚀 Live demo

**Application:** https://isic-2018-multi-model-segmentation-qzlpm3mka.vercel.app/

The application runs inference in the user's browser using ONNX Runtime Web and WebAssembly. Uploaded images are processed locally and are not sent to a project inference API.

## ✨ Features

- **Six-model comparison:** run U-Net, Swin-UNet, TransUNet, Attention U-Net, DeepLabV3+, and FPN on the same image, or select a subset.
- **Interactive visualizations:** switch between overlay, binary mask, and probability map for each model.
- **Per-model metrics:** inference time, predicted lesion area, and mean lesion probability within the predicted lesion region.
- **Majority-vote consensus:** a pixel is included when at least two of the three binary masks classify it as lesion.
- **Downloadable results:** download each model's binary mask, probability map, and overlay, plus the consensus mask and consensus overlay.
- **Browser-side inference:** ONNX models are fetched from Hugging Face and executed locally with ONNX Runtime Web.
- **Static deployment:** Vercel serves the frontend; no PyTorch inference server is required.

## 🧪 How to use

1. Open the [live application](https://isic-2018-multi-model-segmentation-qzlpm3mka.vercel.app/).
2. Select a dermoscopic image in PNG or JPEG format (maximum size: 4 MB).
3. Select the models to run (all six are selected by default), then click **Compare selected models**.
4. Wait for the selected models to load and run. The first visit may download approximately 723 MB when all six models are selected.
5. Inspect the original image and majority-vote consensus. The consensus uses a strict majority of the selected model masks.
6. For each model, select **Overlay**, **Mask**, or **Probability**.
7. Review inference time, lesion area, and mean lesion probability.
8. Use the download links to save individual model outputs or consensus outputs.

A modern desktop browser and a stable internet connection are recommended. Download and inference times depend on network speed, browser resources, and the device.

## 🧠 Models

| Model | Architecture summary | Parameters |
|---|---|---:|
| U-Net | Convolutional encoder-decoder with skip connections | 31,037,633 |
| Swin-UNet | Hierarchical shifted-window transformer-based segmentation model | 27,166,405 |
| TransUNet | Compact CNN encoder with a transformer bottleneck and decoder | 29,770,209 |
| Attention U-Net | U-Net with attention gates | 31,389,165 |
| DeepLabV3+ | Atrous convolutional encoder-decoder | 40,347,041 |
| FPN | Feature Pyramid Network segmentation model | 29,212,737 |

**TransUNet implementation note:** this project uses a custom compact CNN–Transformer implementation. It is not the original TransUNet configuration with a pretrained ResNet-50 encoder.

## 📊 Dataset and experimental setup

The project uses the ISIC 2018 Challenge Task 1 skin-lesion segmentation data.

- Labelled images used: **2,594**
- Training split: **1,815 images (70%)**
- Validation split: **389 images (15%)**
- Test split: **390 images (15%)**
- Input size: **224 × 224**
- Task: binary segmentation (lesion versus background).

The validation split was used during model development. The metrics below are the recorded results on the held-out test split.

## 📈 Test-set results

| Metric | U-Net | Swin-UNet | TransUNet |
|---|---:|---:|---:|
| Dice coefficient | 0.8108 | **0.8332** | 0.8301 |
| Intersection over Union (IoU) | 0.7155 | **0.7444** | 0.7376 |
| Precision | 0.8685 | 0.8768 | 0.8262 |
| Recall | 0.8234 | 0.8521 | **0.9005** |
| Specificity | **0.9842** | 0.9747 | 0.9569 |
| Hausdorff Distance 95 (HD95) | 24.5237 | 19.1677 | **17.8193** |

### Observations

- Swin-UNet achieved the highest Dice coefficient and IoU in these runs.
- TransUNet achieved the highest recall and lowest HD95.
- U-Net achieved the highest specificity.
- Swin-UNet and TransUNet Dice scores are close; the difference should not be overstated.

These results describe this experiment and split. They do not establish clinical performance or guarantee generalization to images from other devices, populations, or acquisition conditions.

## ⚙️ Preprocessing and inference

The browser applies the following preprocessing to the selected image:

1. Read the image as RGB.
2. Resize to **224 × 224** using bilinear interpolation.
3. Convert pixel values to the range [0, 1].
4. Apply ImageNet channel-wise normalization:
   - Mean: [0.485, 0.456, 0.406]
   - Standard deviation: [0.229, 0.224, 0.225]
5. Create a float32 input tensor with shape [1, 3, 224, 224].

The model output is converted to pixel-wise scores using sigmoid. A threshold of **0.5** is used to create the binary mask. The selected models are run sequentially to reduce peak browser memory use. The original U-Net, Swin-UNet, and TransUNet ONNX exports have fixed batch size 1; the added Attention U-Net, DeepLabV3+, and FPN exports support dynamic batch size, although the application uses batch size 1 for all models.

### Output types

| Output | Description |
|---|---|
| Overlay | Resized input image with predicted lesion region highlighted |
| Binary mask | White pixels indicate predicted lesion; black pixels indicate background |
| Probability map | Grayscale visualization of per-pixel sigmoid output |
| Consensus mask | Foreground where at least two model masks agree |
| Consensus overlay | Consensus mask highlighted on the resized input |

## 🎯 Understanding the probability score

For each model, **mean lesion probability** is the average sigmoid output over pixels that the same model classified as lesion (sigmoid output ≥ 0.5):

$$
P_{mean}=\frac{1}{N_{lesion}}\sum_{i\in lesion}\sigma(z_i)
$$

This is a summary of pixel-level model outputs inside the predicted lesion region. It is **not** a calibrated probability that a lesion is malignant, a patient-level cancer probability, or a diagnostic confidence score. The probability map is not guaranteed to be calibrated.

**Lesion area** is the percentage of the 224 × 224 output pixels classified as lesion by that model.

## 🏗️ Deployment architecture

```text
                   User's browser
                        |
                 Static web frontend
                      Vercel
                        |
               ONNX Runtime Web (WASM)
                        |
          +-------------+-------------+
          |             |             |
        U-Net        Swin-UNet      TransUNet
          |             |             |
          +-------------+-------------+
                        |
               Segmentation outputs
                        |
          Masks · Probability maps · Overlays
                        |
              Majority-vote consensus

       ONNX model files are hosted on Hugging Face
```

| Component | Technology / service | Responsibility |
|---|---|---|
| Frontend | HTML, CSS, JavaScript | Upload, visualizations, metrics, downloads |
| Inference | ONNX Runtime Web | Execute exported models in the browser |
| Execution provider | WebAssembly (WASM) | Browser-side CPU inference |
| Model hosting | Hugging Face Hub | Host ONNX and TorchScript artifacts |
| Application hosting | Vercel | Serve the static frontend |
| Source control | GitHub | Application code and documentation |

The browser fetches the hosted model files and ONNX Runtime Web library. Uploaded images are not sent to a backend inference endpoint.

## 🤗 Model files

**Hugging Face repository:**  
https://huggingface.co/bharghavairigela/isic2018-segmentation-models

ONNX deployment files:

- onnx_models/unet.onnx
- onnx_models/swin_unet.onnx
- onnx_models/transunet.onnx
- onnx_models/attention_unet.onnx
- onnx_models/deeplabv3plus.onnx
- onnx_models/fpn.onnx
- onnx_models/six_model_config.json

Original TorchScript exports are also retained in the model repository under the models directory.

## 💻 Run locally

The application is a static frontend and does not require a Python inference server.

1. Clone the repository:

   ```bash
   git clone https://github.com/BharghavaIrigela/ISIC2018-MultiModel-Segmentation.git
   cd ISIC2018-MultiModel-Segmentation
   ```

2. Start a local static HTTP server (Python example):

   ```bash
   python -m http.server 8000
   ```

3. Open http://localhost:8000 in your browser.
4. Upload a supported image and run the comparison.

The browser must be able to access Hugging Face model URLs and the ONNX Runtime Web CDN. Use a local HTTP server rather than opening the HTML directly as a file.

## 📁 Repository structure

```text
ISIC2018-MultiModel-Segmentation/
├── index.html       # Static dashboard and browser-side ONNX inference
├── vercel.json      # Vercel static deployment configuration
└── README.md        # Project documentation
```

Large model artifacts are hosted separately in the Hugging Face repository rather than committed to this source repository.

## ⚠️ Limitations

- Research and educational prototype only; not a medical device.
- The models perform binary lesion segmentation; they do not classify lesions as benign or malignant.
- Mean pixel probability is not a calibrated diagnostic probability.
- Majority voting is a simple ensemble rule, not a separately trained or independently validated model.
- Resizing to 224 × 224 may discard fine image detail.
- Browser inference requires downloading large model files and sufficient device memory.
- External-dataset and real-world clinical generalization have not been established by the reported test metrics.
- Do not use this application to make clinical decisions.

## 🙏 Acknowledgements

- [ISIC Archive](https://www.isic-archive.com/) and [ISIC Challenge](https://challenge.isic-archive.com/) for the skin-lesion imaging research ecosystem and challenge data.
- [Hugging Face](https://huggingface.co/) for model artifact hosting.
- [ONNX Runtime Web](https://onnxruntime.ai/) for browser inference.
- [Vercel](https://vercel.com/) for frontend deployment.

---

<p align="center">
  Built as a deep-learning research project for skin-lesion segmentation and multi-architecture comparison.
</p>
