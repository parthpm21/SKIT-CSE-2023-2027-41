# TruthLens — Video Deepfake Detection Pipeline Evaluation & Stress Suite
**Author**: Pratham Yadav (Deep Learning & Computer Vision Lead)  
**Sprint 3**: User Story 3 — Parts 5 & 6 (Robustness Testing, CLI Tool & Frame Timeline Visualizer)  
**Date**: October 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary & Story Alignment

User Story 3 focuses on the **"Development of the video deepfake detection pipeline"** (`04/10/2026 – 31/10/2026`).  
In accordance with the project roadmap and Form-1/Form-2 descriptions:
> *"Built frame-extraction/face-alignment pipeline; trained Xception/EfficientNet ensemble on FaceForensics++, added noise-fingerprint extraction."*

Parts 5 and 6 conclude the video deepfake detection engineering lifecycle by providing:
1. **Automated Stress & Adversarial Test Suite** (`backend/tests/test_video_deepfake_stress.py`): Validating pipeline resilience against heavy H.264 compression (C23/C40), dropped frames, duplicate freeze frames, partial face occlusions, and strobing lighting.
2. **Video Deepfake Evaluation CLI & Forensic Visualizer** (`backend/run_video_deepfake_eval.py`): Command-line batch execution engine producing diagnostic sequence timeline images, frame anomaly trajectory graphs, and Form-3 compliant CSV/JSON audit reports.

---

## 2. Multi-Branch Pipeline Architecture

The video deepfake detection engine unifies spatial, spectral noise, and temporal attention signals into a single calibrated verdict:

```
[Input Video / Frames]
         │
         ▼
[1. 5-Point Affine Face Alignment Pipeline] ──────── Canonical 224x224 Face Crops
         │                                                         │
         ├─────────────────────────────────────────────┐           │
         ▼                                             ▼           ▼
[2. Dual-Stream Spatial Backbone]         [3. SRM Noise Residual Extractor]
 (Xception + EfficientNet-MBConv)           (3x3 High-Pass Sensor PRNU Filter)
         │                                             │
   Spatial Logits &                               Residual Energy &
   512-d Frame Embeddings                      Face/Background Discrepancy
         │                                             │
         └──────────────────────┬──────────────────────┘
                                │
                                ▼
         [4. Temporal Attention Aggregator & Sequence Voter]
            • Scaled Dot-Product Self-Attention
            • Cosine Trajectory Feature Drift (1 - cos(e_t, e_t-1))
            • Top-K Keyframe Anomaly Pooling
                                │
                                ▼
         [VideoEnsembleVerdict & Forensic Diagnostic Timeline]
```

### Ensemble Mathematical Formulation:
The final sequence anomaly score $S_{\text{ensemble}} \in [0, 1]$ is computed as:
$$S_{\text{ensemble}} = w_{\text{spatial}} \cdot \left(0.6 \cdot \bar{S}_{\text{top-}k} + 0.4 \cdot \bar{S}_{\text{sequence}}\right) + w_{\text{noise}} \cdot \Delta N_{\text{inconsistency}} + w_{\text{temporal}} \cdot D_{\text{drift}}$$
where:
- $w_{\text{spatial}} = 0.50$, $w_{\text{noise}} = 0.25$, $w_{\text{temporal}} = 0.25$
- $\Delta N_{\text{inconsistency}}$ measures inter-frame sensor pattern noise jumps: $\|N_t - N_{t-1}\|_2$
- $D_{\text{drift}}$ quantifies inter-frame feature cosine drift: $1.0 - \cos(e_t, e_{t-1})$

---

## 3. Part 5: Adversarial, Edge-Case & Stress Test Suite

Implemented in [`backend/tests/test_video_deepfake_stress.py`](file:///c:/Users/shivr/Desktop/skit%204%20project/TruthLens/backend/tests/test_video_deepfake_stress.py).

### Stress Scenarios Tested:
1. **H.264 / Quantization Macroblock Compression**:
   - Simulates 8x8 block discrete cosine quantization and high-frequency loss corresponding to FaceForensics++ C23 and C40 compression levels.
   - Asserts that predictions remain finite, without NaNs or numerical degradation.
2. **Frame Stutter & Duplicate Freeze Sequences**:
   - Repeated identical frames across the sequence (e.g., packet drop freeze).
   - Validates that normalized temporal cosine drift remains near zero ($< 0.20$) without division by zero.
3. **Variable Sequence Lengths & Frame Drops**:
   - Evaluates sub-sequences of length 1, 2, 3, and 5 frames under simulated frame dropouts.
4. **Partial Facial Occlusion**:
   - Top-half eye region blacked out (simulating dark sunglasses).
   - Bottom-half mouth/jaw region masked (simulating surgical/cloth mask).
   - Validates that the 5-point landmark pipeline falls back smoothly to central canonical cropping.
5. **Strobing & Erratic Exposure Fluctuations**:
   - Alternating frames shifted by $\pm 0.7$ luminance.
   - Asserts that noise inconsistency and temporal drift metrics stay normalized in $[0, 1]$.
6. **FaceForensics++ Deepfake vs. Authentic Profiles**:
   - Validates that synthetic deepfake manipulation profiles trigger higher noise inconsistency and higher peak anomaly scores compared to smooth authentic sequences.
7. **Numerical Boundary Conditions**:
   - All-black ($-1.0$), all-white ($+1.0$), and empty ($0$ frame) tensors handled safely.

---

## 4. Part 6: Video Deepfake Evaluation CLI & Timeline Visualizer

Implemented in [`backend/run_video_deepfake_eval.py`](file:///c:/Users/shivr/Desktop/skit%204%20project/TruthLens/backend/run_video_deepfake_eval.py).

### Command Reference:

#### 1. Single Video Forensic Evaluation:
```bash
python backend/run_video_deepfake_eval.py --video data/samples/ff_deepfakes_c23.mp4 --visualize outputs/timeline.png --output outputs/report.json
```

#### 2. Batch Evaluation via Dataset Manifest:
```bash
python backend/run_video_deepfake_eval.py --manifest data/manifests/benchmark_manifest.json --output outputs/batch_report.json --export-csv outputs/metrics.csv --viz-dir outputs/timelines/
```

### Timeline Visualizer Panels:
The generated PNG timeline diagnostic image displays:
- **Header Banner**: Video filename, classification verdict (`REAL` in green or `FAKE` in red), confidence percentage, peak anomaly score, noise inconsistency score, temporal drift score, and anomalous frame tally.
- **Keyframe Facial Alignment Strip**: High-resolution face crops sampled across the video sequence, annotated with frame index and colored anomaly score tags.
- **SRM Residual Heatmap Strip**: Colormapped (Inferno/Jet simulation) high-pass sensor residuals highlighting localized noise discrepancies and boundary blending artifacts.
- **Frame Anomaly Trajectory Plot**: Continuous sequence curve displaying frame anomaly progression across time against the 0.50 decision threshold line.

### Form-3 Report Compliance:
The exported CSV output provides sample-level verification metrics directly ingestible into weekly progress documentation:
`sample_id`, `file_name`, `label`, `confidence`, `peak_score`, `mean_score`, `temporal_drift_score`, `noise_inconsistency_score`, `anomalous_frames_count`, `ground_truth_label`, `correct`, `inference_ms`.

---

## 5. Verification & Test Execution

- **Stress Suite**: `python -m pytest backend/tests/test_video_deepfake_stress.py` (10/10 passed in 6.47s)
- **CLI & Visualizer Suite**: `python -m pytest backend/tests/test_run_video_deepfake_eval.py` (4/4 passed in 1.80s)
- **Ensemble & Noise Suite**: `python -m pytest backend/tests/test_temporal_ensemble.py` (13/13 passed in 0.97s)
- **Face & Backbone Suite**: `python -m pytest backend/tests/test_video_pipeline.py` (15/15 passed in 1.58s)
- **CLI Self-Test**: `python backend/run_video_deepfake_eval.py --help` (Exited 0)
