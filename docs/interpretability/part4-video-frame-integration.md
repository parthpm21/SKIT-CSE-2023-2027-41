# Interpretability Support: Video & Frame Integration (Part 4)

## 1. What Part 4 Is and Its Purpose

Part 4 connects the existing TruthLens video and frame ingestion pipeline with the interpretability utilities built in Part 2 (**Grad-CAM**) and Part 3 (**Attention Rollout**).

Simple viva explanation:
> "In Parts 2 and 3, we implemented Grad-CAM for CNN backbones and Attention Rollout for Transformer layers. In Part 4, we built the integration layer that takes video streams or frame sequences, preserves their chronological order, frame numbers, and timestamps, and safely pairs each frame with its corresponding visual explanation without fabricating any data."

Instead of treating frames in isolation or duplicating OpenCV extraction logic, Part 4 provides a unified, reusable pipeline (`VideoInterpretabilityPipeline` and `explain_video_frames`) that works seamlessly with `VideoPreprocessor` and `ImagePreprocessor`.

---

## 2. How Video Frames Move Through the Existing Pipeline

The frame processing pipeline follows these sequential stages:

```
[ Raw Video File / Bytes ]
         │
         ▼
[ VideoPreprocessor.process() ]  <── SamplingStrategy (UNIFORM, FPS, ALL)
         │
         ├── Extracted Frame Sequence: (T, C, H, W)
         └── VideoFrameMetadata:
                 - frame_index (e.g., 0, 15, 30...)
                 - timestamp_seconds (e.g., 0.0s, 0.625s, 1.25s...)
                 - original_size (width, height)
                 - pad_offsets (top, bottom, left, right)
                 - scale_factor
         │
         ▼
[ VideoInterpretabilityPipeline ]
         │
         ├── 1. Frame Tensor Validation (checks 3D shape, finiteness, channels)
         │
         ├── 2. CNN Branch (Grad-CAM):
         │       • If a compatible CNN model is loaded:
         │           batch_tensor -> GradCAMEngine -> heatmap (H, W) -> spatial unpadding
         │       • If no model is loaded:
         │           gradcam_status = "unavailable" (truthful, no fake heatmaps)
         │
         ├── 3. Transformer Branch (Attention Rollout):
         │       • If Transformer attention matrices are available:
         │           compute_attention_rollout() -> rollout matrix (N, N)
         │       • If no matrices are available:
         │           attention_status = "unavailable" (truthful, no fake rollout)
         │
         └── 4. Temporal Association:
                 Each frame receives a FrameExplanation container preserving its
                 frame_index, timestamp, status, and metadata.
         │
         ▼
[ VideoExplanationResult ] (Aggregated multi-frame result & summary statistics)
```

---

## 3. How Part 2 (Grad-CAM) Fits Into the Workflow

- **Input**: Individual preprocessed frame tensors of shape `(3, H, W)` extracted from the video stream.
- **Role in Video Forensics**: Grad-CAM highlights which spatial regions (e.g., facial boundaries, hair lines, compression seams) influenced a CNN backbone's prediction on that specific frame.
- **Condition for Execution**: Runs only when `gradcam_engine.is_available()` is `True` (i.e., a PyTorch `nn.Module` with a registered target convolutional layer is present).
- **Spatial Alignment**: When frames have been letterbox-resized during ingestion, `unpad_and_resize_heatmap` removes padding offsets (`pad_offsets`) and rescales the heatmap back to `original_size` so the explanation directly overlays onto the original unpadded video frame.
- **Truthful Behavior**: In the absence of an integrated model, Grad-CAM returns `None` for the heatmap and records `gradcam_status = "unavailable: no compatible model/target layer loaded in TruthLens"`.

---

## 4. How Part 3 (Attention Rollout) Fits Into the Workflow

- **Input**: Multi-layer self-attention weight matrices $[A_0, A_1, \dots, A_{L-1}]$ captured from Transformer vision backbones (e.g., CLIP-ViT, ViT-B, DINOv2).
- **Role in Video Forensics**: Attention Rollout tracks how information flows through multi-head self-attention layers from input patch tokens to final representations across video frames.
- **Condition for Execution**: Runs when either an active `AttentionRolloutEngine` is hooked into a live Transformer backbone, or when pre-extracted attention matrices are supplied for each frame.
- **Per-Frame Processing**: Sequential multi-head averaging, residual identity mixing ($A + I$), row normalization ($\sum_j A_{ij} = 1$), and recursive matrix multiplication are computed for each frame independently.
- **Truthful Behavior**: When no Transformer model or attention matrices are present, Attention Rollout returns `None` and records `attention_status = "unavailable: no compatible Transformer model or attention matrices available"`.

---

## 5. Files Created or Modified

1. **`backend/interpretability/video_integration.py`** *(Created)*:
   - `FrameExplanation`: Dataclass storing per-frame explanation outputs, execution status, timestamps, and metadata.
   - `VideoExplanationResult`: Dataclass summarizing full video interpretability results.
   - `unpad_and_resize_heatmap`: Utility function removing letterbox padding and aligning heatmaps with original frame dimensions.
   - `VideoInterpretabilityPipeline`: Main pipeline coordinating `VideoPreprocessor`, `GradCAMEngine`, and `AttentionRolloutEngine`.
   - `explain_video_frames`: Modular top-level convenience function supporting file paths, raw bytes, `ProcessedVideo` containers, and frame sequences.

2. **`tests/test_video_frame_integration.py`** *(Created)*:
   - 18 comprehensive unit tests covering:
     - Real-world execution without models (truthful reporting).
     - Preservation of frame indices and timestamps.
     - End-to-end integration with `VideoPreprocessor` decoding real MP4 bytes.
     - Mock Grad-CAM engine integration and error isolation.
     - Synthetic Attention Rollout per-frame aggregation and error isolation.
     - Malformed/corrupted frame tensor isolation.
     - Letterbox unpadding and spatial alignment.
     - JSON dictionary serialization via `.to_dict()`.

3. **`docs/interpretability/part4-video-frame-integration.md`** *(Created)*:
   - Comprehensive documentation for viva defense and project handover.

*(Note: Teammates' existing files, Parts 1–3 implementations, and `__pycache__` folders were strictly preserved without modification.)*

---

## 6. How to Run the Tests

To run the Part 4 video/frame integration test suite:

```powershell
pytest tests/test_video_frame_integration.py -v
```

To run all unit tests across the repository:

```powershell
pytest tests/test_gradcam.py tests/test_attention_rollout.py tests/test_image_preprocessor.py tests/test_video_preprocessor.py tests/test_validator.py tests/test_dataset_manifest.py tests/test_video_frame_integration.py
```

To run the entire test suite including the FastAPI gateway endpoints:

```powershell
pytest
```

---

## 7. What Works Now

- **End-to-End Frame Flow**: Video files (MP4/bytes) and preprocessed frames flow through `VideoPreprocessor` into the interpretability pipeline seamlessly.
- **Temporal & Metadata Fidelity**: Frame indices (e.g., sampled frames 0, 15, 30) and precise floating-point timestamps (e.g., 0.0s, 0.625s, 1.25s) are 100% preserved.
- **Robust Error Isolation**: A corrupted frame (e.g., NaN values or wrong dimensions) or a model failure on frame $k$ is safely caught and recorded without interrupting or misassociating explanations for frame $k+1$.
- **Letterbox Spatial Alignment**: Heatmaps can be unpadded and interpolated back to original image dimensions respecting `pad_offsets`.
- **Truthful Status Reporting**: The system honestly reports `unavailable` when models or attention matrices are absent, completely avoiding fake heatmaps or fabricated deepfake scores.
- **Full Serialization**: All results provide `.to_dict()` for integration with API routes or logging.

---

## 8. What Requires a Real Compatible Model

While the integration pipeline and interfaces are complete and tested, live visual explanations require:

1. **PyTorch Vision Model Backbone**: A trained classification backbone (such as ResNet-50, EfficientNet, CLIP-ViT, or DINOv2) loaded into the backend.
2. **Target Layer Specification**: Identifying the final convolutional layer (for Grad-CAM) or Transformer attention modules (for Attention Rollout).
3. **Patch-to-Pixel Reshaping for Transformers**: For Vision Transformers, extracting the `[CLS]` token attention vector and reshaping the 1D patch sequence into a 2D spatial grid ($H_{\text{patches}} \times W_{\text{patches}}$).

---

## 9. Current Limitations

- **No Model Weights Integrated Yet**: TruthLens currently lacks a trained deepfake detection model; therefore, live inference remains unavailable outside of mock tests.
- **Frame-by-Frame Scope**: Explanations are currently computed frame-by-frame on 2D spatial representations rather than temporal 3D attention tubes.
- **Hardware Acceleration**: GPU batch acceleration for Grad-CAM forward/backward passes will need to be configured once CUDA hardware and PyTorch backbones are integrated.
