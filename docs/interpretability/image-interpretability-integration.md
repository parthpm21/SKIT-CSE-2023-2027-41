# Interpretability Support: Image Interpretability Integration (User Story 3 - Part 2)

## 1. Objective & Purpose

Part 2 of User Story 3 integrates visual interpretability techniques (**Grad-CAM** and **Attention Rollout**) into the existing TruthLens image analysis workflow. 

### Simple Viva Explanation
> "In Part 1 of User Story 3, we implemented the spatial Attention Rollout algorithm to transform Transformer attention matrices into 2D heatmaps. In Part 2, we built the image interpretability integration pipeline (`ImageInterpretabilityPipeline` and `explain_image`). It takes any image input, runs it through the standard preprocessing workflow, and safely generates visual explanation heatmaps using Grad-CAM (for CNN backbones) and Attention Rollout (for Vision Transformers). When models or attention matrices are not yet loaded, it truthfully reports their absence without fabricating predictions or generating fake heatmaps."

---

## 2. Image Interpretability Architecture & Workflow

The image interpretability workflow connects the ingestion and preprocessing pipeline with the interpretability engines:

```
[ Raw Image Input: File Path, Bytes, PIL Image, or NumPy Array ]
                               │
                               ▼
            [ ImagePreprocessor.process() ]
                               │
            ├── Decodes & standardizes color channels (RGB)
            ├── Letterbox resizing (preserves aspect ratio, records pad_offsets)
            ├── Tensor normalization (ImageNet mean & std)
            ├── EXIF metadata extraction & forensic software flags
            └── High-frequency Laplacian noise residual extraction
                               │
                               ▼
                     [ ProcessedImage ]
                   Tensor Shape: (3, H, W)
                               │
                               ▼
             [ ImageInterpretabilityPipeline ]
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             ▼
 [ CNN Branch: Grad-CAM ]             [ Transformer Branch: Attention Rollout ]
 • Checks if CNN model & layer exist  • Checks if Transformer engine or matrices exist
 • If available:                      • If available:
     Forward & backward hooks             compute_attention_rollout() -> (N, N)
     Channel weights via mean grad        rollout_to_heatmap() -> (H, W)
     ReLU & bilinear interpolation    • If unavailable:
 • If unavailable:                        attention_status = "unavailable"
     gradcam_status = "unavailable"
        │                                             │
        └──────────────────────┬──────────────────────┘
                               │
                               ▼
                 [ Spatial Heatmap Alignment ]
       • If align_to_original_resolution = True:
           unpad_and_resize_heatmap() strips letterbox padding
           and resizes heatmaps back to the original image dimensions.
                               │
                               ▼
                     [ ImageExplanation ]
       Container holding:
       - gradcam_heatmap & gradcam_status
       - attention_heatmap & attention_rollout & attention_status
       - prediction metadata (target_class)
       - forensic image metadata (original_size, pad_offsets, EXIF, noise)
       - structured status_message
       - .to_dict() JSON serialization
```

---

## 3. How Grad-CAM Fits Into the Image Workflow

- **Role**: Explains which spatial regions influenced a CNN backbone's prediction on the input image.
- **Execution Condition**: Runs only when `gradcam_engine.is_available() == True` (i.e. a differentiable PyTorch model with a target convolutional layer is attached).
- **Execution**:
  1. Expands tensor to batch `(1, 3, H, W)`.
  2. Forwards batch through the model and computes backward gradients for the target class.
  3. Pools gradients to calculate channel importance weights.
  4. Computes ReLU-activated weighted sum of feature activations.
  5. Resizes heatmap to image resolution and normalizes values to `[0.0, 1.0]`.
- **Truthful Reporting**: When no model is attached in TruthLens, `gradcam_heatmap` remains `None` and records `gradcam_status = "unavailable: no compatible model/target layer loaded in TruthLens"`.

---

## 4. How Attention Rollout Fits Into the Image Workflow

- **Role**: Provides a gradient-free, structural explanation of how multi-head self-attention information flows through Transformer layers to input image patches.
- **Execution Condition**: Runs when either:
  1. An active `AttentionRolloutEngine` is hooked into a Transformer vision backbone (e.g., CLIP-ViT, ViT-B, or DINOv2), or
  2. Multi-layer self-attention weight matrices `[A_0, A_1, ..., A_{L-1}]` are provided from upstream Transformer inference.
- **Execution**:
  1. Averages attention across multi-head dimensions: $\bar{A} = \frac{1}{H}\sum A_h$.
  2. Accounts for residual skip connections: $\bar{A} + I$.
  3. Row-normalizes matrices: $A_{\text{norm}, ij} = \frac{A_{ij}}{\sum_k A_{ik}}$.
  4. Recursively multiplies layers: $R_l = A_l \times R_{l-1}$ to produce cumulative matrix $R$ of shape $(N, N)$.
  5. Isolates `[CLS]` token attention over spatial patches and reshapes to 2D patch grid $(H_{\text{patches}}, W_{\text{patches}})$.
  6. Bilinearly interpolates patch grid to image resolution and normalizes to $[0.0, 1.0]$.
- **Truthful Reporting**: When no Transformer model or attention matrices are present, `attention_heatmap` and `attention_rollout` remain `None` and record `attention_status = "unavailable: no compatible Transformer model or attention matrices available"`.

---

## 5. Spatial Alignment & Letterbox Unpadding

TruthLens employs aspect-ratio preserving letterboxing in `ImagePreprocessor` to prevent distortion. However, letterboxing adds black border padding:
- If a heatmap were directly overlaid onto the original unpadded image, suspicious regions would be misaligned.
- `unpad_and_resize_heatmap` strips `pad_offsets` `(top, bottom, left, right)` and resizes the active crop directly to `original_size` `(width, height)`.
- Setting `align_to_original_resolution=True` guarantees pixel-perfect overlay alignment on the authentic source photo.

---

## 6. Error Handling & Safe Execution

The pipeline enforces rigorous defensive programming:
- **Null Input**: Passing `None` raises a clear `ValueError`.
- **Missing File**: Non-existent file paths raise a standard `FileNotFoundError`.
- **Unsupported Types**: Passing invalid types raises a clear `TypeError`.
- **Corrupted Tensors**: Tensors with `NaN` or `Inf` or wrong color channels (e.g. 1 channel or 4 channels) are safely isolated with `error="Invalid image tensor: ..."` and statuses set to `"skipped"`.
- **Model / Engine Failures**: Runtime errors during model inference or hook execution are captured per-technique (e.g. `gradcam_status = "error: <details>"`), preventing application crashes.

---

## 7. Files Created or Modified

1. **`backend/interpretability/image_integration.py`** *(Created)*:
   - `ImageExplanation`: Dataclass container with `.gradcam_available`, `.attention_available`, and `.to_dict()`.
   - `ImageInterpretabilityPipeline`: Pipeline integrating `ImagePreprocessor`, `GradCAMEngine`, and `AttentionRolloutEngine`.
   - `explain_image`: Top-level convenience function for single-call execution.
   - `unpad_and_resize_heatmap`: Heatmap spatial alignment and letterbox unpadding utility.
2. **`backend/interpretability/__init__.py`** *(Updated)*:
   - Exported `ImageExplanation`, `ImageInterpretabilityPipeline`, `explain_image`, and `unpad_and_resize_heatmap`.
3. **`tests/test_image_interpretability.py`** *(Created)*:
   - 23 comprehensive unit tests validating real-world execution without models, format support, mock engine execution, synthetic attention rollout, error isolation, letterbox unpadding, and serialization.
4. **`docs/interpretability/image-interpretability-integration.md`** *(Created)*:
   - Technical documentation and viva preparation guide.

---

## 8. Test Commands & Verification

### Run Image Interpretability Tests
```bash
pytest tests/test_image_interpretability.py -v
```
**Result**: 23/23 PASSED (100%) in 0.46s.

### Run All Interpretability Tests
```bash
pytest tests/test_gradcam.py tests/test_attention_rollout.py tests/test_video_frame_integration.py tests/test_image_interpretability.py -v
```
**Result**: 65 PASSED, 1 SKIPPED (PyTorch tensor check when running on pure NumPy environment) in 2.04s.

### Run All Ingestion & Preprocessing Tests
```bash
pytest tests/test_image_preprocessor.py tests/test_video_preprocessor.py tests/test_validator.py tests/test_dataset_manifest.py -v
```
**Result**: 34/34 PASSED in 0.77s.

---

## 9. Viva Questions & Answers

**Q1: What did you implement in Part 2 of User Story 3?**
> "I built the Image Interpretability Integration module (`image_integration.py`). It unifies the image preprocessing pipeline (`ImagePreprocessor`) with our visual explainability tools—Grad-CAM for CNNs and Attention Rollout for Vision Transformers. It handles format conversions, spatial padding alignment, and error isolation without producing fake predictions."

**Q2: How does the pipeline behave since no deep learning model is loaded yet?**
> "Following TruthLens engineering principles, it never fakes heatmaps or fabricates synthetic predictions. Instead, it returns an `ImageExplanation` object where `gradcam_available` and `attention_available` are `False`, with clear status messages indicating that no model is loaded. When a model or synthetic matrices are provided, it generates real, normalized heatmaps."

**Q3: How do you solve the spatial mismatch caused by letterbox resizing?**
> "During letterbox resizing, padding is added to preserve the aspect ratio. When heatmaps are generated on the letterboxed tensor, directly overlaying them on the original image causes spatial shifts. Our `unpad_and_resize_heatmap` function crops out the padding using the recorded `pad_offsets` and resizes the active region back to the image's original dimensions."
