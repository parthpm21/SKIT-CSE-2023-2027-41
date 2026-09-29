# Interpretability Support Research: Grad-CAM & Attention Rollout (Part 1)

## 1. Objective
The objective of this research is to analyze the existing TruthLens codebase and prepare the theoretical design for interpretability techniques (**Grad-CAM** and **Attention Rollout**) ahead of future model integration. This document establishes what is currently implemented in the codebase versus what is planned for future parts (Parts 2, 3, and 4).

---

## 2. Codebase Audit & Implementation Verification

### Currently Implemented Architecture

#### Image Pipeline
- **Implementation File**: [`backend/data/image_preprocessor.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/image_preprocessor.py)
- **Classes**:
  - `ImagePreprocessor`: Performs image decoding (RGB, RGBA blending, Grayscale), EXIF forensic metadata extraction (`extract_exif`), high-frequency Laplacian noise residual extraction (`extract_noise_residual`), resizing (`_resize_image` with `LETTERBOX`, `CENTER_CROP`, `DIRECT`), and tensor normalization (`IMAGENET`, `ZERO_TO_ONE`, `MINUS_ONE_TO_ONE`).
  - `ProcessedImage`: Container dataclass for preprocessed tensor `(3, H, W)`, dimensions, padding offsets, scale factor, and EXIF/noise metadata.
- **Gateway Endpoint**: `POST /api/v1/ingest/image` in [`backend/gateway/routes/ingest.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/gateway/routes/ingest.py).

#### Video Pipeline & Frame Extraction
- **Implementation File**: [`backend/data/video_preprocessor.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/video_preprocessor.py)
- **Classes**:
  - `VideoPreprocessor`: Handles video decoding via OpenCV `cv2.VideoCapture`.
  - `VideoPreprocessor._determine_frame_indices()`: Implements frame extraction strategies (`UNIFORM`, `FPS`, `ALL`, `KEYFRAME_FIRST`).
  - `VideoPreprocessor.process()`: Iterates through frames, decodes BGR to RGB, processes each frame with `ImagePreprocessor`, and stacks tensors into temporal sequence `(T, C, H, W)`.
  - `ProcessedVideo`: Dataclass storing temporal tensor `(T, C, H, W)`, frame metadata (`VideoFrameMetadata`), FPS, and helper methods `to_batch()` and `to_channel_first_temporal()` for `(C, T, H, W)` shape conversion.
- **Gateway Endpoint**: `POST /api/v1/ingest/video` in [`backend/gateway/routes/ingest.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/gateway/routes/ingest.py).

#### Validation & Benchmark Infrastructure
- **Implementation Files**:
  - [`backend/data/validator.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/validator.py): `PreprocessingValidator` for checking resolution, byte size, file corruption, and label/verdict consistency.
  - [`backend/data/dataset_manifest.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/dataset_manifest.py): `DatasetManifest` and `BenchmarkSample` data models.

#### Relevant Test Files
- [`tests/test_image_preprocessor.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/tests/test_image_preprocessor.py): Unit tests for image preprocessing.
- [`tests/test_video_preprocessor.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/tests/test_video_preprocessor.py): Unit tests for video frame extraction.
- [`tests/test_gateway_api.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/tests/test_gateway_api.py): Integration tests for FastAPI endpoints.

### Models & Inference Status in Current Codebase
- **CLIP Implementation**: `Not found in current codebase`
- **ViT Implementation**: `Not found in current codebase`
- **DINO / DINOv2 Implementation**: `Not found in current codebase`
- **Model Inference / Prediction Flow**: `Not found in current codebase`

*Note: Preprocessing normalization modes in `ImagePreprocessor` explicitly define `IMAGENET` mean/std parameters in preparation for CLIP/ViT/ResNet models, but model definition files, weights, and inference handlers have not yet been integrated into this repository branch.*

---

## 3. Theoretical Interpretability Research

Because model implementations are not yet present in the codebase, the target layers and hooks described below represent standard PyTorch/HuggingFace conventions for future integration.

### Grad-CAM

#### How It Works
Grad-CAM computes gradients of target score $y^c$ with respect to spatial feature map activations $A^k$ of a layer:
1. Gradients: $\frac{\partial y^c}{\partial A^k_{i, j}}$
2. Importance weights: $\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{i, j}}$
3. Heatmap generation: $L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$

#### Target Layers & Hooks (For Future Models)
- **CNN Backbones (ResNet / EfficientNet)**: Final convolutional layer (e.g., `layer4[-1]` in PyTorch ResNet). Forward hook captures $A^k$; backward hook captures $\frac{\partial y^c}{\partial A^k}$.
- **ViT / CLIP (Vision Transformer)**: Last Transformer block normalization layer or key/value projection layer (e.g., `blocks[-1].norm1`). Spatial patch tokens $[1:]$ must be reshaped from 1D sequence $(N_{\text{patches}}, C)$ to 2D grid $(H_{\text{patches}}, W_{\text{patches}}, C)$.
- **DINO / DINOv2**: Output of last self-attention block layer norm before classification head or projection layer.
- **Frame Branch Integration**: Apply Grad-CAM to each preprocessed frame tensor in `ProcessedVideo` individually, creating a sequence of 2D heatmaps corresponding to `VideoFrameMetadata`.

#### Limitations
- High sensitivity to layer selection (shallow layers yield low-level edges; deep layers yield coarse resolution).
- Requires class target score and backward pass (gradient dependent).
- ViT patch resolution is coarse (e.g., $14 \times 14$ for $224 \times 224$ input with $16 \times 16$ patches), requiring bilinear upsampling.

---

### Attention Rollout

#### How It Works
Attention Rollout propagates multi-head self-attention matrices across Transformer layers to compute total attention flow from input patches to output tokens:
1. Average multi-head attention weights at layer $l$: $\bar{A}^{(l)} = \frac{1}{H} \sum_{h=1}^H A_{h}^{(l)}$
2. Account for residual connection identity matrix: $\hat{A}^{(l)} = 0.5 \bar{A}^{(l)} + 0.5 I$
3. Normalize row sums: $\hat{A}^{(l)} = \frac{\hat{A}^{(l)}}{\sum_j \hat{A}^{(l)}_{ij}}$
4. Compute cumulative matrix across layers: $R^{(l)} = \hat{A}^{(l)} \cdot R^{(l-1)}$
5. Extract row for `[CLS]` token from $R^{(L)}$, drop the `[CLS]` index, and reshape patch values to 2D grid $(H_{\text{patches}}, W_{\text{patches}})$.

#### Target Attention Layers & Tokens (For Future Models)
- **CLIP Vision Transformer**: Hooks into `vision_model.encoder.layers[*].self_attn`. Extract `[CLS]` token attention over spatial patches.
- **Standard ViT**: Hooks into `blocks[*].attn` softmax attention maps.
- **DINO / DINOv2**: Hooks into `blocks[*].attn`. DINO self-attention maps naturally highlight scene objects without supervised labels.
- **Frame Branch Integration**: Run Attention Rollout per frame on extracted video frame sequence to track spatial attention changes across video timestamps.

#### Limitations
- Structural and class-agnostic by default (shows what the model attends to, not necessarily what decision it made).
- Uniform residual mixing ($0.5 \bar{A} + 0.5 I$) can cause attention map diffusion in deep networks.
- Requires internal hook access to attention weight matrices before softmax/linear projections.

---

## 4. Applicability Summary

| Branch / Model Architecture | Currently Implemented? | Grad-CAM Applicability (Future) | Attention Rollout Applicability (Future) | Expected Output Format |
| :--- | :--- | :--- | :--- | :--- |
| **Image Preprocessing** | Yes ([`image_preprocessor.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/image_preprocessor.py)) | Target input tensor provider | Target input tensor provider | Normalized `(3, H, W)` tensor |
| **Video Frame Pipeline** | Yes ([`video_preprocessor.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/video_preprocessor.py)) | Frame-wise batch tensor provider | Frame-wise batch tensor provider | Frame sequence `(T, C, H, W)` |
| **CLIP (Vision Transformer)** | `Not found in current codebase` | Applicable (Target: final block layer norm + patch reshape) | Applicable (Target: `self_attn` maps, `[CLS]` token) | 2D Spatial Heatmap |
| **ViT (Vision Transformer)** | `Not found in current codebase` | Applicable (Target: `blocks[-1].norm1` + patch reshape) | Applicable (Target: `blocks[*].attn` rollout) | 2D Spatial Heatmap |
| **DINO / DINOv2** | `Not found in current codebase` | Applicable (Target: last Transformer block norm) | Applicable (Target: self-attention rollout) | 2D Segmentation / Heatmap |
| **Model Inference Flow** | `Not found in current codebase` | Requires forward & backward pass hooks | Requires forward attention hooks | Class score & explanation map |

---

## 5. Relevant Files, Classes, and Functions

### Current Codebase Components
- **Image Preprocessor**: [`ImagePreprocessor`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/image_preprocessor.py#L55) in [`backend/data/image_preprocessor.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/image_preprocessor.py)
- **Video Preprocessor**: [`VideoPreprocessor`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/video_preprocessor.py#L73) in [`backend/data/video_preprocessor.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/video_preprocessor.py)
- **Frame Extraction**: [`VideoPreprocessor._determine_frame_indices()`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/video_preprocessor.py#L104) and [`VideoPreprocessor.process()`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/data/video_preprocessor.py#L136)
- **Ingestion Routes**: `ingest_image` and `ingest_video` in [`backend/gateway/routes/ingest.py`](file:///c:/Users/ashok/OneDrive/Desktop/TruthLens_New/backend/gateway/routes/ingest.py)

### Planned Future Components (Parts 2, 3, 4)
- `backend/interpretability/gradcam.py`: Will contain `GradCAMEngine` for PyTorch model hook registration.
- `backend/interpretability/attention_rollout.py`: Will contain `AttentionRolloutEngine` for Transformer attention matrix collection.
- `backend/interpretability/utils.py`: Will contain heatmap colormap rendering and letterbox padding compensation.

---

## 6. Plan for Parts 2, 3, and 4

- **Part 2 — Grad-CAM Implementation**: Implement `GradCAMEngine` with PyTorch hooks for CNN and ViT spatial patch reshaping. Add unit tests with mock model backbones.
- **Part 3 — Attention Rollout Implementation**: Implement `AttentionRolloutEngine` to aggregate multi-head attention weights and compute recursive rollout matrices. Add unit tests for rollout output shapes.
- **Part 4 — Video/Frame Integration & Testing**: Integrate interpretability engines with `VideoPreprocessor` frame sequences, create FastAPI gateway endpoints, handle spatial upsampling to original image bounds (respecting `pad_offsets`), and run end-to-end tests.
