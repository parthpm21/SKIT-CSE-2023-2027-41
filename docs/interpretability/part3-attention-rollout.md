# Interpretability Support: Attention Rollout Implementation & Readiness (Part 3)

## 1. What Attention Rollout Is

Attention Rollout is an interpretability technique designed specifically for Transformer architectures (such as Vision Transformers [ViT], CLIP, and DINO/DINOv2). 

Simple viva explanation:
> "Attention Rollout helps us understand how attention information flows through transformer layers. It combines attention matrices from multiple layers to estimate which input tokens receive attention. To display attention on an image, we also need to know how the model's image patches correspond to spatial locations."

In standard self-attention, each layer allows tokens to attend to one another. However, deep Transformers have multiple layers and residual skip connections. Looking at only the final layer's attention map can be misleading because information has already mixed across previous layers. Attention Rollout mathematically tracks and propagates attention weights forward layer by layer from the input tokens to the final representations.

---

## 2. Why TruthLens May Need It

In deepfake detection and media forensics:
- State-of-the-art vision models (e.g., CLIP vision encoders, ViT backbones, and self-supervised DINOv2 models) rely heavily on multi-head self-attention mechanisms.
- While Grad-CAM relies on gradients and class scores for a single target layer, Attention Rollout provides a **gradient-free structural explanation** of how the model distributes its attention across image patches.
- It enables forensic examiners to verify whether a Vision Transformer is attending to subtle artifact regions (such as facial boundary seams, hair textures, or eye reflections) across all processing depths, rather than relying on an isolated final layer.

---

## 3. Whether a Compatible Transformer Model Currently Exists

**No, a compatible Transformer model does NOT currently exist in the TruthLens codebase.**

### Audit Finding
- The codebase contains the data ingestion pipeline ([`backend/data/image_preprocessor.py`](../../backend/data/image_preprocessor.py), [`backend/data/video_preprocessor.py`](../../backend/data/video_preprocessor.py)) and ingestion API routes ([`backend/gateway/routes/ingest.py`](../../backend/gateway/routes/ingest.py)).
- **No Transformer model (CLIP, ViT, or DINO/DINOv2), model weights, or inference pipeline** have been integrated into the repository yet.

### Approach Taken (Case B Directive)
- No fake model, mock weights, or fake heatmaps were created to simulate a live system.
- We implemented a **clean, model-independent mathematical utility** (`compute_attention_rollout`) alongside an `AttentionRolloutEngine` interface ready to receive attention matrices once a Transformer backbone is introduced.
- Live model integration is not yet possible, and is clearly documented as such.

---

## 4. How the Implementation Works

The implementation is located in [`backend/interpretability/attention_rollout.py`](../../backend/interpretability/attention_rollout.py).

### The 5 Standard Simplified Steps
Given a sequence of attention matrices $[A_0, A_1, \dots, A_{L-1}]$ from $L$ Transformer layers:

1. **Multi-Head Averaging**:
   When a layer matrix includes multiple attention heads $(H, N, N)$, we average across the heads along dimension 0:
   $$\bar{A} = \frac{1}{H} \sum_{h=1}^H A_h \quad \rightarrow \quad (N, N)$$
2. **Residual Connection Addition**:
   In Transformers, residual connections add the identity mapping to the layer output ($x_l = x_{l-1} + \text{Attn}(x_{l-1})$). To reflect this in attention flow, we add the identity matrix $I$:
   $$A_{\text{residual}} = \bar{A} + I$$
3. **Row Normalization**:
   We re-normalize each row across the last dimension so that row sums equal $1.0$ (ensuring a valid stochastic matrix):
   $$A_{\text{norm}, ij} = \frac{A_{\text{residual}, ij}}{\sum_k A_{\text{residual}, ik}}$$
4. **Layer Combination via Sequential Multiplication**:
   We propagate attention across layers by recursively multiplying normalized matrices in layer order:
   $$R_0 = A_{\text{norm}}^{(0)}$$
   $$R_l = A_{\text{norm}}^{(l)} \times R_{l-1} \quad \text{for } l = 1, \dots, L-1$$
5. **Rollout Matrix Return**:
   The resulting matrix $R$ of shape $(N, N)$ represents the cumulative attention flow.

---

## 5. What the Output Represents

- The output of `compute_attention_rollout` is a 2D matrix of shape $(N, N)$, where $N$ is the total number of tokens (e.g., $1$ `[CLS]` token + $P^2$ patch tokens).
- Each entry $R_{i, j}$ represents the total propagated attention from token $i$ at the final layer back to input token $j$.
- **Important Distinction:** A rollout matrix is **not automatically a pixel-level heatmap**. Converting patch-token attention into a 2D spatial visualization requires knowing the model's exact token layout (e.g., `[CLS]` index, register token count) and the spatial patch grid dimensions ($H_{\text{patches}} \times W_{\text{patches}}$). Because no model exists in the codebase yet, spatial patch reshaping is deferred to future model integration.

---

## 6. How Synthetic Test Data Was Used

Because no real Transformer model or prediction pipeline exists in the codebase:
- All unit tests in [`tests/test_attention_rollout.py`](../../tests/test_attention_rollout.py) use **small, explicitly synthetic attention matrices** generated using fixed seeds.
- These matrices are clearly labeled as unit test data, **not real model outputs**.
- Tests verify mathematical properties: stochastic row sums ($\sum_j R_{ij} = 1.0$), non-negativity ($R_{ij} \ge 0$), finite values, multi-head reduction, multi-layer propagation, exact step-by-step calculations on known matrices, and error handling for empty/invalid shapes.

---

## 7. Test Results

### Test Suite: `tests/test_attention_rollout.py`
Verified unit tests:
1. `test_single_layer_valid_matrix`: Validates output shape $(N, N)$, row sums $= 1.0$, and non-negativity.
2. `test_multiple_attention_layers`: Validates recursive multi-layer rollout across multiple layers.
3. `test_mathematical_step_by_step_exactness`: Confirms exact numerical values matching analytical hand calculations on a $2 \times 2$ synthetic test case.
4. `test_multiple_attention_heads`: Confirms multi-head averaging from $(H, N, N)$ to $(N, N)$.
5. `test_4d_batch_attention_matrix`: Confirms handling of 4D batch inputs $(1, H, N, N)$.
6. `test_pytorch_tensor_inputs`: Confirms PyTorch tensor compatibility and tensor output preservation.
7. `test_empty_input_handling`: Verifies descriptive `ValueError` on empty sequence.
8. `test_none_input_handling`: Verifies descriptive `ValueError` on `None`.
9. `test_non_square_attention_matrix`: Verifies validation error when token dimensions are not square.
10. `test_inconsistent_sequence_length_across_layers`: Verifies validation error when layer sequence lengths mismatch.
11. `test_non_finite_values_handling`: Verifies rejection of matrices with NaNs or Infs.
12. `test_invalid_dimension_count`: Verifies rejection of 1D or 5D inputs.
13. `test_attention_rollout_engine_initialization`: Verifies initial state and `is_available() == False` without model.
14. `test_attention_rollout_engine_raises_without_model`: Verifies descriptive `RuntimeError` when running without a loaded model.

---

## 8. Limitations

- **Class-Agnostic**: Standard Attention Rollout reflects raw self-attention flow and does not inherently reflect class targets or classification gradients (unlike Grad-CAM).
- **Residual Diffusion**: Uniform residual identity mixing ($A + I$) in extremely deep networks can diffuse attention across many tokens.
- **Model Token Layout Dependency**: Visualizing attention over image pixels requires knowing patch size, grid resolution, and special token ordering.
- **Dependency on Model Hook Access**: Attention Rollout requires accessing raw softmax self-attention matrices before output projection.

---

## 9. What Is Needed for Future Model Integration

To enable end-to-end Attention Rollout visualizations in TruthLens:
1. **Transformer Backbone**: Integrate a vision transformer model (e.g., CLIP-ViT, ViT-B/16, or DINOv2) into `backend/models/`.
2. **Attention Hook Registration**: Hook into the Transformer self-attention modules (`attn.softmax` or equivalent) to capture attention matrices during the forward pass.
3. **Patch Grid Reshaping**: Extract the `[CLS]` token's attention row ($R_{0, 1:}$), drop special tokens, and reshape 1D sequence into $(H_{\text{patches}}, W_{\text{patches}})$.
4. **Bilinear Upsampling & Letterbox Alignment**: Interpolate the patch grid to the input image resolution, adjusting for any letterbox padding offsets (`pad_offsets` from `ProcessedImage`).
