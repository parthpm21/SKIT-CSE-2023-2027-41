"""
TruthLens — Automated Tests for CLIP:ViT + DIRE Evaluation CLI Tool
Author: Pratham Yadav
Sprint 2: User Story 2 - Part 6 Validation

Validates batch evaluation runner, JSON/CSV report serialization,
4-panel side-by-side diagnostic visualizer image generation, and CLI parsing.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.run_clip_dire_eval import (
    CLIPDIREEvaluationRunner,
    _create_side_by_side_visualization,
)
from backend.inference.clip_dire_service import DualBranchVerdict


class TestCLIPDIREEvalCLI:
    """Test suite for the dual-branch evaluation CLI runner and visualizer."""

    @pytest.fixture
    def runner(self) -> CLIPDIREEvaluationRunner:
        return CLIPDIREEvaluationRunner(num_dire_steps=5)

    def test_create_side_by_side_visualization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = Image.new("RGB", (224, 224), color=(100, 150, 200))
            recon_tensor = np.zeros((3, 224, 224), dtype=np.float32)
            res_map = np.random.uniform(0.0, 1.0, (224, 224)).astype(np.float32)
            fused_map = np.random.uniform(0.0, 1.0, (224, 224)).astype(np.float32)
            verdict = DualBranchVerdict(
                label="fake",
                confidence=0.88,
                synthetic_subtype="diffusion",
                model_version="test-v1",
                clip_probe_score=0.85,
                dire_score=0.91,
                dire_mse=0.012,
                dire_ssim=0.82,
                fused_heatmap=fused_map,
            )

            out_path = Path(tmpdir) / "diag_test.png"
            _create_side_by_side_visualization(
                original_img=orig,
                reconstructed_tensor=recon_tensor,
                residual_heatmap=res_map,
                fused_heatmap=fused_map,
                verdict=verdict,
                output_path=out_path,
            )

            assert out_path.exists()
            with Image.open(out_path) as saved_img:
                assert saved_img.size[0] >= 224 * 4  # 4 panels wide
                assert saved_img.size[1] > 224

    def test_runner_evaluates_single_image(self, runner: CLIPDIREEvaluationRunner):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_sample.png"
            Image.new("RGB", (224, 224), color=(120, 180, 240)).save(img_path)

            viz_dir = Path(tmpdir) / "visualizations"
            record = runner.evaluate_image_file(
                img_path, ground_truth="real", save_viz_dir=viz_dir
            )

            assert record["file_name"] == "test_sample.png"
            assert record["label"] in {"real", "fake"}
            assert 0.0 <= record["confidence"] <= 1.0
            assert "synthetic_subtype" in record
            assert "dire_mse" in record
            assert record["ground_truth"] == "real"
            assert "is_correct" in record
            assert record["visualization_path"] is not None
            assert Path(record["visualization_path"]).exists()

    def test_runner_batch_execution_and_reports(self, runner: CLIPDIREEvaluationRunner):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 test images in a directory
            img_dir = Path(tmpdir) / "batch_images"
            img_dir.mkdir()
            for i in range(3):
                p = img_dir / f"img_{i}.png"
                Image.new("RGB", (128, 128), color=(i * 50, i * 60, i * 70)).save(p)

            out_json = Path(tmpdir) / "eval_report.json"
            out_csv = Path(tmpdir) / "eval_report.csv"
            out_viz = Path(tmpdir) / "viz_output"

            full_report = runner.run_evaluation(
                input_path=img_dir,
                output_report_path=out_json,
                export_csv_path=out_csv,
                save_viz_dir=out_viz,
                quiet=True,
            )

            assert "summary" in full_report
            assert "samples" in full_report
            assert full_report["summary"]["total_samples"] == 3
            assert len(full_report["samples"]) == 3

            # Check JSON file written
            assert out_json.exists()
            with open(out_json, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
                assert loaded_json["summary"]["total_samples"] == 3

            # Check CSV file written
            assert out_csv.exists()
            with open(out_csv, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 4  # 1 header + 3 rows
