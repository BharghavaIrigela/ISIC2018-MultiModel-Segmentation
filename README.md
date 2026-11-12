# ISIC 2018 — Six-Model Skin Lesion Segmentation

<p align="center">
  <strong>U-Net · Swin-UNet · TransUNet · Attention U-Net · DeepLabV3+ · FPN</strong>
</p>

<p align="center">
  <a href="https://isic-2018-multi-model-segmentation.vercel.app/"><strong>🚀 Live Application</strong></a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/bharghavairigela/isic2018-segmentation-models">🤗 Hugging Face Models</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/BharghavaIrigela/ISIC2018-MultiModel-Segmentation">💻 Source Code</a>
</p>

A browser-based research application for **binary skin-lesion segmentation** using six deep-learning architectures evaluated on the ISIC 2018 Challenge Task 1 dataset. Upload a dermoscopic image, select one or more models, compare predicted masks and probability maps, inspect a majority-vote consensus, and download the generated results. Model inference runs locally in the browser using ONNX Runtime Web; model files are hosted on Hugging Face and the static application is deployed on Vercel.

> **Research and educational use only.** This prototype has not been clinically validated. It does not diagnose skin cancer, estimate malignancy, or support treatment decisions.

## Contents

- [Live application](#live-application)
- [Features](#features)
- [Models](#models)
- [Dataset and experimental protocol](#dataset-and-experimental-protocol)
- [Held-out test results](#held-out-test-results)
- [Explainability and visual analysis](#explainability-and-visual-analysis)
- [Browser inference pipeline](#browser-inference-pipeline)
- [Outputs and consensus](#outputs-and-consensus)
- [Deployment architecture](#deployment-architecture)
- [Hugging Face model artifacts](#hugging-face-model-artifacts)
- [Run locally](#run-locally)
- [Repository structure](#repository-structure)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Acknowledgements](#acknowledgements)

## Live application

**Production demo:** https://isic-2018-multi-model-segmentation.vercel.app/

The dashboard supports all six models and lets users select a subset or run the complete comparison. Images are resized and processed in the browser. The image itself is not uploaded to a project inference server. The browser does download the selected ONNX model files from Hugging Face.

## Features

- **Six segmentation architectures:** U-Net, Swin-UNet, TransUNet, Attention U-Net, DeepLabV3+, and FPN.
- **Flexible model selection:** run all six models or select a subset.
- **Per-model visualizations:** switch between a color overlay, binary mask, and grayscale sigmoid probability map.
- **Per-model summaries:** inference duration, predicted lesion area, and mean sigmoid output over predicted lesion pixels.
- **Consensus segmentation:** strict majority voting across the selected model masks.
- **Downloadable outputs:** save each model's mask, probability map, and overlay, plus the consensus mask and overlay.
- **Browser-side execution:** ONNX Runtime Web with the WebAssembly execution provider.
- **Static deployment:** GitHub source, Hugging Face model hosting, and Vercel frontend hosting; no dedicated PyTorch inference server.

## Models

| Model | Architecture description | Parameters |
|---|---|---:|
| U-Net | Convolutional encoder–decoder with skip connections | 31,037,633 |
| Swin-UNet | Hierarchical shifted-window transformer segmentation model | 27,166,405 |
| TransUNet | Custom compact CNN encoder with transformer bottleneck and decoder | 29,770,209 |
| Attention U-Net | U-Net variant with attention gates | 31,389,165 |
| DeepLabV3+ | Atrous-convolution encoder–decoder with multi-scale context | 40,347,041 |
| FPN | Feature Pyramid Network-based segmentation model | 29,212,737 |

**Implementation note:** the TransUNet in this project is a custom compact CNN–Transformer implementation. It is not the original TransUNet configuration with a pretrained ResNet-50 encoder. Model names describe the implemented architecture family; they should not be interpreted as claims that every implementation detail matches the corresponding paper.

## Dataset and experimental protocol

The experiments use the labelled training portion of the **ISIC 2018 Challenge Task 1: Lesion Boundary Segmentation** dataset.

| Split | Images | Proportion |
|---|---:|---:|
| Training | 1,815 | 70% |
| Validation | 389 | 15% |
| Held-out test | 390 | 15% |
| **Total** | **2,594** | **100%** |

- Task: binary segmentation (lesion versus background).
- Input resolution: 224 × 224 pixels.
- Input channels: RGB.
- Split: fixed, reproducible seed-42 split; the same held-out test subset was used for all six models.
- Validation data was used during model development; the reported test metrics are from the held-out test split.

The three additional architectures (Attention U-Net, DeepLabV3+, and FPN) were trained for 8 epochs in the recorded experiment. Their best checkpoints were selected using validation performance. The results below describe these specific runs and should not be read as a claim that all architectures received identical optimization histories or training budgets.

## Held-out test results

The following metrics are macro-averaged per-image results on the same 390-image test split. Dice, IoU, precision, recall, and specificity use a 0.5 sigmoid threshold. HD95 is reported in pixel units at the 224 × 224 evaluation resolution.

| Model | Dice ↑ | IoU ↑ | Precision ↑ | Recall ↑ | Specificity ↑ | Mean HD95 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **Attention U-Net** | **0.8557** | **0.7726** | **0.8728** | 0.8847 | **0.9772** | 19.0704 |
| **FPN** | 0.8406 | 0.7533 | 0.8552 | 0.8876 | 0.9643 | **17.6790** |
| **Swin-UNet** | 0.8330 | 0.7462 | 0.8613 | 0.8655 | 0.9665 | 24.1566 |
| **DeepLabV3+** | 0.8268 | 0.7343 | 0.8461 | 0.8779 | 0.9649 | 19.4760 |
| **U-Net** | 0.8099 | 0.7162 | 0.7910 | 0.9109 | 0.9609 | 26.9785 |
| **TransUNet** | 0.7889 | 0.6863 | 0.7488 | **0.9242** | 0.9450 | 24.8865 |

**Metric highlights from this test run**

- Attention U-Net recorded the highest Dice, IoU, precision, and specificity.
- TransUNet recorded the highest recall.
- FPN recorded the lowest mean HD95.
- No single model leads every metric; model selection depends on the segmentation criterion being prioritized.

**HD95 note:** HD95 is the maximum of the two directional 95th-percentile boundary distances, calculated using 8-neighbour boundary extraction. Empty-mask cases are treated as invalid rather than assigned a distance of zero. The valid HD95 counts were 390 for Attention U-Net, FPN, and DeepLabV3+; 388 for Swin-UNet; 389 for U-Net and TransUNet. Mean HD95 is calculated over valid cases.

These are experimental results on one held-out split. They do not establish statistical significance, clinical performance, or generalization to external datasets and acquisition settings.

## Explainability and visual analysis

The project includes perturbation-based **occlusion sensitivity** visualizations for all six models.

- Three held-out test examples were selected for the recorded XAI comparison.
- A 32 × 32 occlusion patch with stride 16 was used.
- The analysis measures changes in the model's mean sigmoid output under local input occlusions.
- The method is model-agnostic and was used instead of gradient-based saliency because the custom Swin-UNet attention path did not support the required backward operation in the notebook environment.

Occlusion maps are qualitative model-behaviour visualizations. They are not causal explanations, clinical evidence, or proof that a model relies on medically valid features.

The experiment also generated six-model segmentation comparison figures for six held-out examples, showing the input image, ground-truth mask, and each model's predicted segmentation.

## Browser inference pipeline

The web application applies the following preprocessing:

1. Read the selected image as RGB.
2. Resize to 224 × 224 using the browser canvas.
3. Scale channel values to [0, 1].
4. Apply ImageNet normalization:
   - Mean: `[0.485, 0.456, 0.406]`
   - Standard deviation: `[0.229, 0.224, 0.225]`
5. Construct a float32 tensor with shape `[1, 3, 224, 224]`.
6. Run the selected ONNX model in ONNX Runtime Web using WebAssembly.
7. Apply sigmoid to the output logits and threshold at 0.5 to produce the binary mask.

Models are executed sequentially to limit peak browser memory usage. The original U-Net, Swin-UNet, and TransUNet ONNX exports have fixed batch size 1. The newly exported Attention U-Net, DeepLabV3+, and FPN models support dynamic batch size. The application sends batch size 1 to all six models.

The UI's per-model inference duration measures the model execution call. The displayed total elapsed duration includes image preparation, model download/session initialization where applicable, model execution, and result rendering; it should not be compared directly with the per-model execution durations.

## Outputs and consensus

| Output | Meaning |
|---|---|
| Overlay | Input image with predicted lesion pixels highlighted |
| Binary mask | White = predicted lesion; black = predicted background |
| Probability map | Grayscale visualization of per-pixel sigmoid output |
| Consensus mask | Pixel classified as lesion by a strict majority of the selected models |
| Consensus overlay | Consensus mask highlighted on the resized input |

For (n) selected models, the consensus threshold is \\(\\lfloor n/2 \\rfloor + 1\\). For example, with all six models selected, at least four models must classify a pixel as lesion. With three models, at least two must agree.

The consensus is a simple voting rule. It is not a separately trained model and has not been independently validated as superior to the individual models.

### Interpreting the displayed scores

- **Lesion area** is the percentage of the 224 × 224 output pixels classified as lesion.
- **Mean lesion probability** is the mean sigmoid output over pixels classified as lesion by that model.

Mean lesion probability is not a calibrated probability of malignancy, a patient-level cancer probability, or a diagnostic confidence score. Pixel-wise sigmoid values may be uncalibrated.

## Deployment architecture

```text
                    User's browser
                          |
                   Static web app
                       Vercel
                          |
                 ONNX Runtime Web
                    WebAssembly
                          |
          +---------------+----------------+
          |       |       |       |        |
        U-Net   Swin    Trans   Attention  DeepLab   FPN
                UNet    UNet      U-Net     V3+
          |       |       |       |        |       |
          +---------------+----------------+
                          |
              Masks / probability maps
                    / overlays
                          |
                Majority-vote consensus

       ONNX model artifacts hosted on Hugging Face
       Source code and documentation hosted on GitHub
```

| Component | Technology | Responsibility |
|---|---|---|
| Frontend | HTML, CSS, JavaScript | Image upload, model selection, visualizations, downloads |
| Inference runtime | ONNX Runtime Web | Execute ONNX graphs in the browser |
| Execution provider | WebAssembly (WASM) | Browser-side CPU inference |
| Model hosting | Hugging Face Hub | Store ONNX and TorchScript artifacts |
| Application hosting | Vercel | Serve the static application |
| Source control | GitHub | Maintain frontend and project documentation |

The first full six-model run downloads approximately **723.36 MB** of ONNX model files, in addition to the ONNX Runtime Web assets. Download size and time vary with caching and network conditions. The selected image is processed locally by the frontend and is not sent to a project inference API.

## Hugging Face model artifacts

**Model repository:** https://huggingface.co/bharghavairigela/isic2018-segmentation-models

ONNX deployment files:

- `onnx_models/unet.onnx`
- `onnx_models/swin_unet.onnx`
- `onnx_models/transunet.onnx`
- `onnx_models/attention_unet.onnx`
- `onnx_models/deeplabv3plus.onnx`
- `onnx_models/fpn.onnx`
- `onnx_models/six_model_config.json`

The original TorchScript artifacts are also retained in the model repository under `models/`.

All six ONNX files were loaded and checked using ONNX Runtime in the notebook. The three newly exported models were additionally compared numerically with their PyTorch outputs; the maximum absolute differences on the validation input were approximately 3.81e-06 (Attention U-Net), 1.86e-05 (DeepLabV3+), and 1.17e-05 (FPN). The original three ONNX models were runtime-validated using their fixed batch size of 1.

## Run locally

The application is a static frontend and does not require a Python inference server.

1. Clone the repository:

   ```bash
   git clone https://github.com/BharghavaIrigela/ISIC2018-MultiModel-Segmentation.git
   cd ISIC2018-MultiModel-Segmentation
   ```

2. Start a local static HTTP server:

   ```bash
   python -m http.server 8000
   ```

3. Open http://localhost:8000 in a modern browser.
4. Select a PNG or JPEG dermoscopic image (maximum 4 MB).
5. Choose one or more models and run the comparison.

The browser needs internet access to retrieve the ONNX Runtime Web library and the selected model files from Hugging Face. Use an HTTP server rather than opening `index.html` directly with a `file://` URL.

## Repository structure

```text
ISIC2018-MultiModel-Segmentation/
├── index.html       # Static six-model dashboard and browser inference
├── vercel.json      # Vercel static deployment configuration
└── README.md        # Project documentation
```

Large model artifacts are hosted separately in the Hugging Face model repository rather than committed to this frontend repository.

## Limitations and responsible use

- This is a research and educational prototype, not a medical device.
- The models perform binary lesion segmentation; they do not classify lesions as benign or malignant.
- No clinical validation or external-dataset generalization is established by these experiments.
- Resizing to 224 × 224 may remove fine image detail.
- Pixel-wise sigmoid outputs are not guaranteed to be calibrated.
- Occlusion sensitivity is a qualitative perturbation analysis, not a causal explanation.
- Majority voting is a simple ensemble heuristic and may reinforce shared model errors.
- Browser inference requires downloading large model files and sufficient device memory; runtime varies by browser, hardware, and network.
- Do not use this application to diagnose, triage, or make treatment decisions.

## Acknowledgements

- [ISIC Archive](https://www.isic-archive.com/) and [ISIC Challenge](https://challenge.isic-archive.com/) for the skin-lesion imaging research ecosystem and challenge data.
- [Hugging Face](https://huggingface.co/) for model artifact hosting.
- [ONNX Runtime](https://onnxruntime.ai/) for browser inference.
- [Vercel](https://vercel.com/) for static application hosting.

---

<p align="center">
  Built as a deep-learning research project for multi-architecture skin-lesion segmentation and browser-based model comparison.
</p>
