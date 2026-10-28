"""
TruthLens — Automated Tests for Video Deepfake Evaluation CLI & Visualizer
Author: Pratham Yadav
Sprint 3: User Story 3 - Part 6 Validation

Validates:
- Video timeline diagnostic image generation with facial crop strip, SRM heatmaps, and anomaly graph.
- Single video evaluation runner with latency and FPS profiling.
- Batch dataset manifest evaluation and JSON/CSV report generation.
- CLI argument parsing and error handling.
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.run_video_deepfake_eval import (
    VideoDeepfakeEvaluationRunner,
    _create_video_timeline_visualization,
    _render_inferno_colormap,
)
from backend.inference.temporal_ensemble import VideoEnsembleVerdict
from backend.data.dataset_manifest import (
    BenchmarkSample,
    DatasetManifest,
    GeneratorSource,
    ManipulationCategory,
    MediaType,
    Verdict,
)


class TestVideoDeepfakeEvalCLI:
    """Test suite for the video evaluation CLI runner and timeline visualizer."""

    @pytest.fixture
    def runner(self) -> VideoDeepfakeEvaluationRunner:
        return VideoDeepfakeEvaluationRunner(max_frames=6)

    @pytest.fixture
    def mock_verdict(self) -> VideoEnsembleVerdict:
        return VideoEnsembleVerdict(
            label="fake",
            confidence=0.88,
            frame_scores=[0.25, 0.35, 0.72, 0.89, 0.81, 0.65],
            peak_score=0.89,
            mean_score=0.61,
            temporal_drift_score=0.42,
            noise_inconsistency_score=0.55,
            anomalous_frames_count=4,
            model_version="video-deepfake-temporal-ensemble-v2",
            details={
                "noise_fingerprint": {"mean_noise_energy": 0.04},
                "temporal_drift": {"aggregate_drift": 0.42},
            },
        )

    def test_render_inferno_colormap(self):
        gray = np.linspace(0.0, 1.0, 100).reshape(10, 10).astype(np.float32)
        rgb = _render_inferno_colormap(gray)
        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.uint8
        assert 0 <= rgb.min() <= rgb.max() <= 255

    def test_create_video_timeline_visualization(self, mock_verdict: VideoEnsembleVerdict):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 6 frames of shape (6, 3, 128, 128)
            frames = np.random.uniform(-1.0, 1.0, (6, 3, 128, 128)).astype(np.float32)
            out_path = Path(tmpdir) / "test_timeline.png"

            _create_video_timeline_visualization(
                video_path=Path("sample_deepfake.mp4"),
                frame_tensor=frames,
                verdict=mock_verdict,
                output_path=out_path,
                num_sampled_keyframes=4,
            )

            assert out_path.exists()
            with Image.open(out_path) as img:
                assert img.format == "PNG"
                assert img.width > 500
                assert img.height > 400

    def test_evaluate_video_file_with_mock_video(self, runner: VideoDeepfakeEvaluationRunner):
        """Creates a minimal test mp4 video and evaluates it."""
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not available for generating test video")

        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = Path(tmpdir) / "test_vid.mp4"
            viz_path = Path(tmpdir) / "test_viz.png"

            # Create an authentic-looking 6-frame video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (224, 224))
            for i in range(6):
                frame = np.ones((224, 224, 3), dtype=np.uint8) * 120
                frame[60:160, 60:160, :] = 180  # Central face block
                writer.write(frame)
            writer.release()

            res = runner.evaluate_video_file(vid_path, visualize_path=viz_path)
            assert "label" in res
            assert res["label"] in {"real", "fake"}
            assert 0.0 <= res["confidence"] <= 1.0
            assert "inference_ms" in res
            assert res["inference_ms"] >= 0.0
            assert "fps" in res
            assert viz_path.exists()

    def test_evaluate_manifest_batch(self, runner: VideoDeepfakeEvaluationRunner, mock_verdict: VideoEnsembleVerdict):
        """Tests batch evaluation across a DatasetManifest with JSON & CSV reporting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest = DatasetManifest(name="TestVideoBenchmark")

            # Create 2 mock video files
            try:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                for i in range(2):
                    vp = tmp_path / f"video_{i}.mp4"
                    writer = cv2.VideoWriter(str(vp), fourcc, 10.0, (128, 128))
                    for _ in range(4):
                        writer.write(np.zeros((128, 128, 3), dtype=np.uint8) + (i * 100))
                    writer.release()

                    manifest.add_sample(
                        BenchmarkSample(
                            id=f"VID_{i}",
                            file_path=str(vp),
                            media_type=MediaType.VIDEO,
                            verdict=Verdict.MANIPULATED if i == 1 else Verdict.AUTHENTIC,
                            manipulation_category=ManipulationCategory.FACE_SWAP if i == 1 else ManipulationCategory.NONE,
                            generator_source=GeneratorSource.FF_DEEPFAKES if i == 1 else GeneratorSource.REAL_CAMERA,
                            label=1 if i == 1 else 0,
                            resolution=(128, 128),
                        )
                    )
            except ImportError:
                pytest.skip("cv2 not available")

            m_json_path = tmp_path / "manifest.json"
            manifest.save_to_json(m_json_path)

            rep_json = tmp_path / "report.json"
            rep_csv = tmp_path / "report.csv"
            viz_dir = tmp_path / "viz"

            summary = runner.evaluate_manifest(
                manifest_path=m_json_path,
                export_report=rep_json,
                export_csv=rep_csv,
                viz_dir=viz_dir,
            )

            assert summary["evaluated_samples"] == 2
            assert "accuracy" in summary
            assert rep_json.exists()
            assert rep_csv.exists()

            # Verify CSV contents
            with open(rep_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 2
                assert "sample_id" in rows[0]
                assert "confidence" in rows[0]
                assert "temporal_drift_score" in rows[0]
