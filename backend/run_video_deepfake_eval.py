"""
TruthLens — Video Deepfake Detection Pipeline Evaluation CLI & Timeline Visualizer
Author: Pratham Yadav (Deep Learning & Computer Vision Lead)
Sprint 3: User Story 3 - Part 6 (CLI Evaluation & Video Forensic Visualizer)

Provides an automated command-line evaluation suite for video deepfake detection.
Executes the full multi-branch pipeline (Facial Alignment + Dual-Stream Xception/EfficientNet
Backbone + SRM Noise-Fingerprinting + Temporal Attention Pooling), computes frame-level
anomaly trajectories and inter-frame feature drift, renders multi-panel forensic timeline
visualizations, and exports structured JSON/CSV reports compliant with Form-3 reporting.

Usage:
    python backend/run_video_deepfake_eval.py --video sample.mp4 --visualize timeline.png
    python backend/run_video_deepfake_eval.py --manifest data/manifests/benchmark_manifest.json --output report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Support execution both as module and standalone script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.ingestion.preprocessing import preprocess_video_file
from backend.inference.temporal_ensemble import (
    VideoEnsemblePipeline,
    VideoEnsembleVerdict,
    video_ensemble_pipeline,
)
from backend.inference.noise_fingerprint import (
    NoiseFingerprintResult,
    noise_fingerprint_extractor,
)
from backend.data.dataset_manifest import DatasetManifest, BenchmarkSample, MediaType


def _render_inferno_colormap(grayscale: np.ndarray) -> np.ndarray:
    """
    Simulates high-contrast Inferno/Jet colormap for high-frequency noise heatmaps.
    grayscale: (H, W) float in [0, 1] -> RGB uint8 (H, W, 3)
    """
    norm = np.clip(grayscale * 255.0, 0, 255).astype(np.uint8)
    rgb = np.zeros((*grayscale.shape, 3), dtype=np.uint8)
    # Red channel: dominant high energy
    rgb[:, :, 0] = norm
    # Green channel: mid-frequency transition
    rgb[:, :, 1] = np.clip(norm * 0.65, 0, 255).astype(np.uint8)
    # Blue channel: inverted background baseline
    rgb[:, :, 2] = np.clip(255 - norm * 0.85, 0, 255).astype(np.uint8)
    return rgb


def _create_video_timeline_visualization(
    video_path: Path,
    frame_tensor: np.ndarray,
    verdict: VideoEnsembleVerdict,
    output_path: Path,
    num_sampled_keyframes: int = 5,
) -> None:
    """
    Renders a multi-panel forensic diagnostic sequence timeline image:
    1. Header banner: Verdict, Confidence, Peak Score, Noise Inconsistency, Temporal Drift.
    2. Row 1: Keyframe face crops with timestamp/frame tags and anomaly score color bars.
    3. Row 2: SRM high-frequency noise residual heatmaps exposing sensor PRNU suppression.
    4. Row 3: Temporal anomaly trajectory graph showing frame scores and 0.50 decision threshold.
    """
    T = frame_tensor.shape[0]
    if T == 0:
        return

    # Select representative keyframe indices across sequence
    indices = np.linspace(0, T - 1, min(num_sampled_keyframes, T), dtype=int)
    k_count = len(indices)

    thumb_w, thumb_h = 160, 160
    pad = 12
    banner_h = 75
    graph_h = 120
    canvas_w = max(900, k_count * (thumb_w + pad) + pad)
    canvas_h = banner_h + (thumb_h * 2) + graph_h + (pad * 5)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 22, 30))
    draw = ImageDraw.Draw(canvas)

    # 1. Header Banner
    is_fake = verdict.label.lower() == "fake"
    status_color = (245, 75, 75) if is_fake else (50, 205, 120)
    title_text = f"TruthLens Video Deepfake Diagnostic | Verdict: {verdict.label.upper()} ({verdict.confidence*100:.1f}%)"
    meta_text = (
        f"File: {video_path.name} | Frames: {T} | Anomalous: {verdict.anomalous_frames_count}/{T} | "
        f"Peak Anomaly: {verdict.peak_score*100:.1f}% | Noise Inconsistency: {verdict.noise_inconsistency_score*100:.1f}% | "
        f"Temporal Drift: {verdict.temporal_drift_score*100:.1f}%"
    )
    draw.text((pad, 14), title_text, fill=status_color)
    draw.text((pad, 40), meta_text, fill=(170, 185, 210))
    draw.line([(pad, banner_h - 10), (canvas_w - pad, banner_h - 10)], fill=(45, 55, 75), width=1)

    # 2. Keyframe Thumbnails & SRM Residual Heatmaps
    row1_y = banner_h + 5
    row2_y = row1_y + thumb_h + 30

    draw.text((pad, row1_y - 18), "Keyframe Facial Alignment Strip:", fill=(200, 215, 235))
    draw.text((pad, row2_y - 18), "SRM High-Frequency Sensor Noise Residuals:", fill=(200, 215, 235))

    for col, idx in enumerate(indices):
        x = pad + col * (thumb_w + pad)

        # Convert frame from [-1, 1] float32 tensor (3, H, W) to PIL RGB
        f_raw = frame_tensor[idx]
        f_uint8 = np.clip((f_raw.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        f_thumb = Image.fromarray(f_uint8).resize((thumb_w, thumb_h))
        canvas.paste(f_thumb, (x, row1_y))

        # Score bar below frame
        frame_score = verdict.frame_scores[idx] if idx < len(verdict.frame_scores) else 0.0
        score_color = (245, 75, 75) if frame_score >= 0.50 else (50, 205, 120)
        draw.rectangle([x, row1_y + thumb_h - 22, x + thumb_w, row1_y + thumb_h], fill=(10, 12, 16))
        draw.text((x + 6, row1_y + thumb_h - 18), f"F#{idx}  Score: {frame_score:.2f}", fill=score_color)

        # High-pass SRM noise residual thumbnail
        res_map, _, _ = noise_fingerprint_extractor.extract_frame_noise(f_raw)
        res_rgb = _render_inferno_colormap(res_map)
        res_thumb = Image.fromarray(res_rgb).resize((thumb_w, thumb_h))
        canvas.paste(res_thumb, (x, row2_y))
        draw.rectangle([x, row2_y + thumb_h - 22, x + thumb_w, row2_y + thumb_h], fill=(10, 12, 16))
        draw.text((x + 6, row2_y + thumb_h - 18), f"SRM Residual F#{idx}", fill=(200, 210, 225))

    # 3. Anomaly Trajectory Graph across Sequence
    graph_y = row2_y + thumb_h + 35
    draw.text((pad, graph_y - 18), "Frame Anomaly Trajectory & Detection Threshold (0.50):", fill=(200, 215, 235))

    gx1, gy1 = pad, graph_y
    gx2, gy2 = canvas_w - pad, graph_y + graph_h - 10
    draw.rectangle([gx1, gy1, gx2, gy2], fill=(12, 15, 22), outline=(50, 60, 80))

    # 0.50 Threshold dashed line
    thresh_y = int(gy2 - 0.50 * (gy2 - gy1))
    draw.line([(gx1, thresh_y), (gx2, thresh_y)], fill=(180, 140, 50), width=1)
    draw.text((gx1 + 6, thresh_y - 14), "Threshold (0.50)", fill=(210, 170, 70))

    # Plot scores line
    scores = verdict.frame_scores
    if len(scores) > 1:
        points = []
        for i, s in enumerate(scores):
            px = int(gx1 + (i / (len(scores) - 1)) * (gx2 - gx1 - 10) + 5)
            py = int(gy2 - (float(s) * (gy2 - gy1 - 10)) - 5)
            points.append((px, py))

        for i in range(len(points) - 1):
            seg_fake = scores[i] >= 0.50 or scores[i + 1] >= 0.50
            seg_color = (245, 80, 80) if seg_fake else (60, 210, 130)
            draw.line([points[i], points[i + 1]], fill=seg_color, width=2)

        # Plot point markers
        for px, py in points:
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(240, 240, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


class VideoDeepfakeEvaluationRunner:
    """Batch evaluation engine for video deepfake detection across files and manifests."""

    def __init__(
        self,
        pipeline: Optional[VideoEnsemblePipeline] = None,
        max_frames: int = 16,
    ) -> None:
        self.pipeline = pipeline or video_ensemble_pipeline
        self.max_frames = max_frames

    def evaluate_video_file(
        self,
        file_path: Path | str,
        visualize_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Evaluates a single video file, records timings, and optionally creates visualization."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        start_time = time.perf_counter()
        frame_tensor = preprocess_video_file(path, num_frames=self.max_frames)
        preproc_ms = (time.perf_counter() - start_time) * 1000

        infer_start = time.perf_counter()
        verdict = self.pipeline.predict(frame_tensor)
        infer_ms = (time.perf_counter() - infer_start) * 1000

        if visualize_path:
            _create_video_timeline_visualization(path, frame_tensor, verdict, Path(visualize_path))

        res_dict = verdict.to_dict()
        res_dict["file_path"] = str(path)
        res_dict["file_name"] = path.name
        res_dict["preprocessing_ms"] = round(preproc_ms, 2)
        res_dict["inference_ms"] = round(infer_ms, 2)
        res_dict["total_ms"] = round(preproc_ms + infer_ms, 2)
        res_dict["fps"] = round((frame_tensor.shape[0] / max(0.001, (infer_ms / 1000.0))), 1)

        return res_dict

    def evaluate_manifest(
        self,
        manifest_path: Path | str,
        export_report: Optional[Path | str] = None,
        export_csv: Optional[Path | str] = None,
        viz_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Batch evaluates all video samples specified in a DatasetManifest."""
        m_path = Path(manifest_path)
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {m_path}")

        manifest = DatasetManifest.load_from_json(m_path)
        video_samples = [s for s in manifest.samples if s.media_type == MediaType.VIDEO]

        evaluated_records: List[Dict[str, Any]] = []
        correct_count = 0
        total_count = 0
        total_infer_time = 0.0

        viz_path_dir = Path(viz_dir) if viz_dir else None
        if viz_path_dir:
            viz_path_dir.mkdir(parents=True, exist_ok=True)

        for s in video_samples:
            s_file = Path(s.file_path)
            # Resolve relative to manifest directory or project root
            if not s_file.exists():
                candidate = m_path.parent / s.file_path
                if candidate.exists():
                    s_file = candidate
                else:
                    candidate = Path(_PROJECT_ROOT) / s.file_path
                    if candidate.exists():
                        s_file = candidate

            if not s_file.exists():
                # Skip missing video files gracefully
                continue

            v_out = viz_path_dir / f"{s_file.stem}_timeline.png" if viz_path_dir else None
            record = self.evaluate_video_file(s_file, visualize_path=v_out)
            record["sample_id"] = s.id
            record["ground_truth_label"] = s.label
            record["manipulation_category"] = s.manipulation_category.value
            record["generator_source"] = s.generator_source.value

            pred_label_code = 1 if record["label"] == "fake" else 0
            is_correct = pred_label_code == s.label
            record["correct"] = is_correct

            if is_correct:
                correct_count += 1
            total_count += 1
            total_infer_time += record["inference_ms"]

            evaluated_records.append(record)

        accuracy = float(correct_count / total_count) if total_count > 0 else 0.0
        avg_infer_ms = float(total_infer_time / total_count) if total_count > 0 else 0.0

        summary = {
            "manifest_name": manifest.name,
            "total_video_samples": len(video_samples),
            "evaluated_samples": total_count,
            "accuracy": round(accuracy, 4),
            "avg_inference_ms": round(avg_infer_ms, 2),
            "model_version": self.pipeline.model_version,
            "records": evaluated_records,
        }

        # Export JSON report
        if export_report:
            rep_path = Path(export_report)
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        # Export CSV report formatted for Form-3 documentation
        if export_csv:
            csv_path = Path(export_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "sample_id",
                "file_name",
                "label",
                "confidence",
                "peak_score",
                "mean_score",
                "temporal_drift_score",
                "noise_inconsistency_score",
                "anomalous_frames_count",
                "ground_truth_label",
                "correct",
                "inference_ms",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for rec in evaluated_records:
                    writer.writerow(rec)

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="TruthLens — Video Deepfake Detection Pipeline Evaluation CLI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to single video file for evaluation")
    group.add_argument("--manifest", type=str, help="Path to DatasetManifest JSON for batch evaluation")

    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation JSON report")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to save metrics CSV report")
    parser.add_argument("--visualize", type=str, default=None, help="Output image path for single video timeline visualization")
    parser.add_argument("--viz-dir", type=str, default=None, help="Output directory to save batch visualizations")
    parser.add_argument("--max-frames", type=int, default=16, help="Maximum keyframes to extract per video")
    parser.add_argument("--summary", action="store_true", help="Print human-readable summary table")

    args = parser.parse_args()
    runner = VideoDeepfakeEvaluationRunner(max_frames=args.max_frames)

    if args.video:
        print(f"\n[TruthLens] Evaluating Video: {args.video} ...")
        res = runner.evaluate_video_file(args.video, visualize_path=args.visualize)
        print("\n" + "=" * 60)
        print(f"Verdict:              {res['label'].upper()}")
        print(f"Confidence:           {res['confidence']*100:.2f}%")
        print(f"Peak Anomaly Score:   {res['peak_score']*100:.2f}%")
        print(f"Sequence Mean Score:  {res['mean_score']*100:.2f}%")
        print(f"Noise Inconsistency:  {res['noise_inconsistency_score']*100:.2f}%")
        print(f"Temporal Cosine Drift:{res['temporal_drift_score']*100:.2f}%")
        print(f"Anomalous Frames:     {res['anomalous_frames_count']}/{res['total_frames']}")
        print(f"Inference Latency:    {res['inference_ms']:.1f} ms ({res['fps']} FPS)")
        print("=" * 60)

        if args.visualize:
            print(f"[TruthLens] Timeline visualization saved -> {args.visualize}")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"[TruthLens] JSON report saved -> {args.output}")

    elif args.manifest:
        print(f"\n[TruthLens] Running Batch Evaluation on Manifest: {args.manifest} ...")
        summary = runner.evaluate_manifest(
            args.manifest,
            export_report=args.output,
            export_csv=args.export_csv,
            viz_dir=args.viz_dir,
        )
        print("\n" + "=" * 60)
        print(f"Manifest:             {summary['manifest_name']}")
        print(f"Evaluated Samples:    {summary['evaluated_samples']} / {summary['total_video_samples']}")
        print(f"Model Accuracy:       {summary['accuracy']*100:.2f}%")
        print(f"Average Latency:      {summary['avg_inference_ms']:.1f} ms")
        print("=" * 60)

        if args.output:
            print(f"[TruthLens] Batch JSON report saved -> {args.output}")
        if args.export_csv:
            print(f"[TruthLens] Metrics CSV report saved -> {args.export_csv}")


if __name__ == "__main__":
    main()
