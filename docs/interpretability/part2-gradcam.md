# Interpretability Support: Grad-CAM Implementation & Readiness (Part 2)

## 1. What Grad-CAM Is
Grad-CAM (Gradient-weighted Class Activation Mapping) is a visual explanation technique for deep neural networks.

Simple viva explanation:
"Grad-CAM helps us understand which part of an image influenced a model's prediction. It uses the activations and gradients from a selected layer to create a heatmap. The heatmap highlights important image regions."

## 2. Why We Need It
In deep learning for media forensics and deepfake detection:
- It transforms "black-box" model classification decisions into interpretable, human-understandable visual heatmaps.
- It enables forensic investigators to verify whether the model is focusing on genuine facial/manipulation artifacts (such as boundary blending, unnatural texture, or frequency anomalies) rather than irrelevant background context.

## 3. Whether It Can Currently Be Used in TruthLens
**No, Grad-CAM cannot currently generate actual heatmaps on live images in TruthLens.**

Audit Result:
The codebase contains the ingestion pipeline (`ImagePreprocessor`, `VideoPreprocessor`) and API routes, but **does NOT yet contain a deep learning model or prediction pipeline** (such as CLIP, ViT, ResNet, or DINOv2).

Per design directives:
- No new model was invented or added to force a fake demonstration.
- No fake predictions or fake heatmaps were generated.
- A clean, reusable `GradCAMEngine` interface (`backend/interpretability/gradcam.py`) was created. It provides hooks and gradient calculation logic ready for execution as soon as a PyTorch model is integrated.

## 4. Model and Target Layer Used
- **Currently Available Model**: None (Model inference pipeline not yet integrated).
- **Target Layer Configuration (For Future Integration)**:
  - CNN (ResNet): Final convolutional layer (e.g., `layer4[-1]`).
  - ViT / CLIP: Last Transformer block layer norm (`blocks[-1].norm1`) with patch sequence reshaping to 2D grid.

## 5. Simple Implementation Explanation
The Grad-CAM processing flow is:
```
Image -> Preprocessing -> PyTorch Model -> Target Layer -> Activations + Gradients -> Grad-CAM Weights -> Heatmap -> Highlighted Image
```

Implementation Steps in `GradCAMEngine`:
1. **Hook Registration**: Attach forward and backward hooks (`register_forward_hook`, `register_full_backward_hook`) to target layer.
2. **Activation Capture**: Forward pass captures feature maps $A^k$.
3. **Gradient Capture**: Backward pass computes gradients $\frac{\partial y^c}{\partial A^k}$ relative to target class score $y^c$.
4. **Weight Calculation**: Global average pooling over spatial dimensions yields channel weights $\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{i,j}}$.
5. **Heatmap Generation**: Weighted sum $\sum_k \alpha_k^c A^k$ is passed through `ReLU` to highlight features with positive influence.
6. **Bilinear Upsampling**: Interpolate heatmap to input image resolution and normalize values to $[0.0, 1.0]$.

## 6. Test Result
- **Test File**: `tests/test_gradcam.py`
- **Result**: **4/4 PASSED** (Overall suite: **47/47 PASSED**).
- **Verified Behavior**:
  - `GradCAMEngine` initialization and default state.
  - `is_available()` correctly returns `False` when no model is attached.
  - Calling `generate_heatmap()` cleanly raises `RuntimeError` explaining missing model infrastructure without producing fake outputs.

## 7. Limitations
- **Model Dependency**: Requires a differentiable PyTorch model with backpropagation support.
- **Coarse Resolution**: Transformer-based models (ViT/CLIP) produce coarse patch-level heatmaps (e.g., $14 \times 14$) requiring spatial interpolation.
- **Single-layer Scope**: Grad-CAM captures explanations from one specific target layer at a time.

## 8. What Is Required for Future Model Integration
To activate live Grad-CAM visual explanations:
1. Integrate a PyTorch classification backbone (e.g. CLIP, ResNet, or ViT) into `backend/models/`.
2. Define inference prediction endpoints that instantiate `GradCAMEngine(model, target_layer)`.
3. Pass input preprocessed tensor `(1, C, H, W)` from `ProcessedImage.to_batch()` into `GradCAMEngine.generate_heatmap()`.
