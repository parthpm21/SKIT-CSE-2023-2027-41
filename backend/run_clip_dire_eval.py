"""
TruthLens — CLIP:ViT Probing & DDIM/DIRE Dual-Branch Evaluation CLI Tool
Author: Pratham Yadav
Sprint 2: User Story 2 - Part 6 (CLI Evaluation & Forensic Visualizer)

Provides an automated command-line evaluation suite for benchmark datasets and images.
Executes dual-branch CLIP:ViT probing and DDIM/DIRE inversion, calculates quantitative
metrics (MSE, PSNR, SSIM, subtype distribution), produces side-by-side forensic visualizer
heatmaps, and exports JSON/CSV reports formatted for Form-3 progress documentation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Support execution both as module and standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.inference.clip_dire_service import (
    CLIPDIREImageService,
    DualBranchVerdict,
    clip_dire_service,
)
from backend.data.dataset_manifest import DatasetManifest, BenchmarkSample


def _create_side_by_side_visualization(
    original_img: Image.Image,
    reconstructed_tensor: np.ndarray,
    residual_heatmap: np.ndarray,
    fused_heatmap: np.ndarray,
    verdict: DualBranchVerdict,
    output_path: Path,
) -> None:
    """
    Renders a 4-panel diagnostic forensic visualizer image:
    [1. Original Image | 2. DDIM Reconstruction | 3. DIRE Residual Map | 4. Fused Anomaly Heatmap]
    """
    panel_size = (224, 224)
    orig_panel = original_img.convert("RGB").resize(panel_size)

    # 2. Reconstructed image from [-1, 1] tensor -> uint8 RGB
    recon_clamped = np.clip((reconstructed_tensor.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    recon_panel = Image.fromarray(recon_clamped).resize(panel_size)

    # 3. DIRE residual heatmap colorized (Inferno/Jet simulation)
    dire_norm = np.clip(residual_heatmap * 255.0, 0, 255).astype(np.uint8)
    dire_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    dire_rgb[:, :, 0] = dire_norm  # Red channel for error intensity
    dire_rgb[:, :, 1] = np.clip(dire_norm * 0.4, 0, 255).astype(np.uint8)
    dire_rgb[:, :, 2] = np.clip(255 - dire_norm, 0, 255).astype(np.uint8)
    dire_panel = Image.fromarray(dire_rgb).resize(panel_size)

    # 4. Composite anomaly heatmap
    fused_norm = np.clip(fused_heatmap * 255.0, 0, 255).astype(np.uint8)
    fused_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    fused_rgb[:, :, 0] = fused_norm
    fused_rgb[:, :, 1] = np.clip(fused_norm * 0.7, 0, 255).astype(np.uint8)
    fused_rgb[:, :, 2] = np.clip(180 - fused_norm * 0.5, 0, 255).astype(np.uint8)
    fused_panel = Image.fromarray(fused_rgb).resize(panel_size)

    # Compose 4-panel canvas (width: 4 * 224 = 896, height: 224 + 60 banner)
    canvas_w = panel_size[0] * 4 + 40
    canvas_h = panel_size[1] + 80
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 24, 33))
    draw = ImageDraw.Draw(canvas)

    # Paste panels
    x_offsets = [10, 10 + 224 + 10, 10 + 224 * 2 + 20, 10 + 224 * 3 + 30]
    canvas.paste(orig_panel, (x_offsets[0], 65))
    canvas.paste(recon_panel, (x_offsets[1], 65))
    canvas.paste(dire_panel, (x_offsets[2], 65))
    canvas.paste(fused_panel, (x_offsets[3], 65))

    # Header banner text
    title_text = f"TruthLens Forensic Diagnostic | Verdict: {verdict.label.upper()} ({verdict.confidence*100:.1f}%) | Subtype: {verdict.synthetic_subtype.upper()}"
    metrics_text = f"DIRE MSE: {verdict.dire_mse:.5f} | SSIM: {verdict.dire_ssim:.3f} | CLIP Score: {verdict.clip_probe_score:.3f}"
    draw.text((15, 12), title_text, fill=(240, 240, 255))
    draw.text((15, 34), metrics_text, fill=(160, 180, 210))

    # Panel Subtitles
    labels = ["1. Original Input", "2. DDIM Recon (x'_0)", "3. DIRE Residual Map", "4. Fused Anomaly Map"]
    for i, lbl in enumerate(labels):
        draw.text((x_offsets[i] + 5, 48), lbl, fill=(200, 210, 225))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


class CLIPDIREEvaluationRunner:
    """Orchestrates batch dual-branch evaluation across files or benchmark manifests."""

    def __init__(
        self,
        service: Optional[CLIPDIREImageService] = None,
        num_dire_steps: int = 15,
    ) -> None:
        self.service = service or clip_dire_service
        self.num_dire_steps = num_dire_steps

    def evaluate_image_file(
        self,
        image_path: Path,
        ground_truth: Optional[str] = None,
        save_viz_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluates a single image file through dual-branch pipeline."""
        img = Image.open(image_path).convert("RGB")
        start = time.perf_counter()
        verdict = self.service.predict(img, num_dire_steps=self.num_dire_steps)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        viz_saved = None
        if save_viz_dir:
            out_name = f"diag_{image_path.stem}.png"
            viz_path = save_viz_dir / out_name
            reconstructed = verdict.details.get("dire", {}).get("reconstructed_image", None)
            if reconstructed is None:
                # Retrieve from dire analyzer if not stored in details
                dire_res = self.service.dire_analyzer.analyze(img, num_steps=self.num_dire_steps)
                reconstructed = dire_res.reconstructed_image
                dire_map = dire_res.residual_map
            else:
                dire_map = verdict.details.get("dire", {}).get("residual_map", np.zeros((224, 224)))

            _create_side_by_side_visualization(
                original_img=img,
                reconstructed_tensor=reconstructed,
                residual_heatmap=dire_map,
                fused_heatmap=verdict.fused_heatmap,
                verdict=verdict,
                output_path=viz_path,
            )
            viz_saved = str(viz_path)

        res_record = {
            "file_name": image_path.name,
            "file_path": str(image_path),
            "label": verdict.label,
            "confidence": round(float(verdict.confidence), 4),
            "synthetic_subtype": verdict.synthetic_subtype,
            "clip_probe_score": round(float(verdict.clip_probe_score), 4),
            "dire_score": round(float(verdict.dire_score), 4),
            "dire_mse": round(float(verdict.dire_mse), 6),
            "dire_ssim": round(float(verdict.dire_ssim), 4),
            "inference_time_ms": round(elapsed_ms, 2),
            "visualization_path": viz_saved,
        }

        if ground_truth:
            res_record["ground_truth"] = ground_truth.lower()
            res_record["is_correct"] = verdict.label == ground_truth.lower()

        return res_record

    def run_evaluation(
        self,
        input_path: Optional[str | Path] = None,
        manifest_path: Optional[str | Path] = None,
        output_report_path: Optional[str | Path] = "clip_dire_eval_report.json",
        export_csv_path: Optional[str | Path] = None,
        save_viz_dir: Optional[str | Path] = None,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """Runs batch evaluation over a directory, file, or manifest."""
        viz_dir = Path(save_viz_dir) if save_viz_dir else None
        records: List[Dict[str, Any]] = []

        if not quiet:
            print("=" * 70)
            print("TruthLens — Dual-Branch (CLIP:ViT + DIRE) Benchmark Evaluation CLI")
            print("=" * 70)

        # 1. Collect targets
        targets: List[Tuple[Path, Optional[str]]] = []
        if manifest_path:
            manifest = DatasetManifest.load_from_json(manifest_path)
            for sample in manifest.samples:
                if sample.media_type == "image":
                    p = Path(sample.file_path)
                    if p.exists():
                        targets.append((p, sample.label))
        elif input_path:
            inp = Path(input_path)
            if inp.is_file():
                targets.append((inp, None))
            elif inp.is_dir():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]:
                    for p in inp.glob(ext):
                        targets.append((p, None))

        # Fallback to simulated test set if no files found
        if not targets:
            if not quiet:
                print("[*] No existing image files provided. Generating simulated benchmark samples...")
            tmp_dir = Path("eval_temp_samples")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for i, label in enumerate(["real", "fake", "fake", "real", "fake"]):
                p = tmp_dir / f"sample_{i+1}_{label}.png"
                img = Image.new("RGB", (224, 224), color=((i * 40) % 255, (i * 70) % 255, (i * 90) % 255))
                img.save(p)
                targets.append((p, label))

        if not quiet:
            print(f"[*] Processing {len(targets)} sample(s) with {self.num_dire_steps} DDIM inversion steps...")

        # 2. Execute evaluation loop
        start_all = time.perf_counter()
        for idx, (img_p, gt) in enumerate(targets):
            rec = self.evaluate_image_file(img_p, ground_truth=gt, save_viz_dir=viz_dir)
            records.append(rec)
            if not quiet:
                gt_str = f" (GT: {gt})" if gt else ""
                print(f" [{idx+1}/{len(targets)}] {img_p.name} -> {rec['label'].upper()}{gt_str} | Subtype: {rec['synthetic_subtype']} | Conf: {rec['confidence']*100:.1f}% | Time: {rec['inference_time_ms']}ms")

        total_time = time.perf_counter() - start_all
        fps = len(records) / max(total_time, 1e-4)

        # 3. Compute Summary Statistics
        total = len(records)
        real_count = sum(1 for r in records if r["label"] == "real")
        fake_count = sum(1 for r in records if r["label"] == "fake")
        subtypes = {}
        for r in records:
            st = r["synthetic_subtype"]
            subtypes[st] = subtypes.get(st, 0) + 1

        avg_mse = float(np.mean([r["dire_mse"] for r in records]))
        avg_ssim = float(np.mean([r["dire_ssim"] for r in records]))
        avg_clip_score = float(np.mean([r["clip_probe_score"] for r in records]))
        avg_dire_score = float(np.mean([r["dire_score"] for r in records]))

        summary: Dict[str, Any] = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": total,
            "total_time_seconds": round(total_time, 2),
            "throughput_fps": round(fps, 2),
            "distribution": {
                "real_verdicts": real_count,
                "fake_verdicts": fake_count,
                "real_ratio": round(real_count / total, 4) if total else 0.0,
                "fake_ratio": round(fake_count / total, 4) if total else 0.0,
                "subtype_breakdown": subtypes,
            },
            "forensic_metrics": {
                "avg_dire_mse": round(avg_mse, 6),
                "avg_dire_ssim": round(avg_ssim, 4),
                "avg_clip_probe_score": round(avg_clip_score, 4),
                "avg_dire_diffusion_score": round(avg_dire_score, 4),
            },
            "parameters": {
                "num_dire_steps": self.num_dire_steps,
                "clip_backbone": "CLIP-ViT-B/16-Frozen",
                "dire_scheduler": "DDIM-Deterministic-ODE",
            },
        }

        # Accuracy if ground truth present
        gt_records = [r for r in records if "is_correct" in r]
        if gt_records:
            correct = sum(1 for r in gt_records if r["is_correct"])
            acc = correct / len(gt_records)
            summary["ground_truth_validation"] = {
                "evaluated_with_gt": len(gt_records),
                "correct_predictions": correct,
                "accuracy": round(acc, 4),
                "accuracy_percent": f"{acc * 100:.2f}%",
            }

        full_report = {
            "summary": summary,
            "samples": records,
        }

        # 4. Save Output JSON & CSV
        if output_report_path:
            out_p = Path(output_report_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(full_report, f, indent=2)
            if not quiet:
                print(f"\n[+] Full evaluation report saved to: {out_p}")

        if export_csv_path:
            csv_p = Path(export_csv_path)
            csv_p.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_p, "w", newline="", encoding="utf-8") as f:
                if records:
                    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                    writer.writeheader()
                    writer.writerows(records)
            if not quiet:
                print(f"[+] Per-sample CSV report exported to: {csv_p}")

        if not quiet:
            print("-" * 70)
            print("Evaluation Summary:")
            print(f"  Processed: {total} images in {total_time:.2f}s ({fps:.1f} img/s)")
            print(f"  Verdicts : {real_count} Real, {fake_count} Fake | Subtypes: {subtypes}")
            print(f"  DIRE Res : Avg MSE={avg_mse:.5f}, Avg SSIM={avg_ssim:.3f}")
            if "ground_truth_validation" in summary:
                print(f"  Accuracy : {summary['ground_truth_validation']['accuracy_percent']}")
            print("=" * 70)

        return full_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TruthLens CLIP:ViT Probing & DDIM/DIRE Dual-Branch Evaluation CLI"
    )
    parser.add_argument("--input", "-i", type=str, help="Path to input image or directory of images")
    parser.add_argument("--manifest", "-m", type=str, help="Path to dataset manifest JSON")
    parser.add_argument("--output", "-o", type=str, default="clip_dire_eval_report.json", help="Output JSON report path")
    parser.add_argument("--export-csv", type=str, help="Path to export per-sample results CSV")
    parser.add_argument("--visualize", "-v", type=str, help="Directory to save 4-panel diagnostic visualizer images")
    parser.add_argument("--steps", "-s", type=int, default=15, help="Number of DDIM inversion steps (default: 15)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()
    runner = CLIPDIREEvaluationRunner(num_dire_steps=args.steps)
    runner.run_evaluation(
        input_path=args.input,
        manifest_path=args.manifest,
        output_report_path=args.output,
        export_csv_path=args.export_csv,
        save_viz_dir=args.visualize,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
