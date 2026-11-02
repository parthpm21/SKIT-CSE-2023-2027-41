"""
TruthLens — Tests for synthetic samples and the manifest-driven dataset builder
Author: Parth Maheshwari
"""
import json

import numpy as np
import pytest

from backend.fusion.contracts import FusionInput, ImageBranchInput, VideoBranchInput
from backend.fusion.training.build_dataset import build_from_manifest
from backend.fusion.training.schema import read_jsonl
from backend.fusion.training.synthetic import make_synthetic_samples


def fake_image_extractor(path, sample_id):
    return FusionInput(sample_id=sample_id, image=ImageBranchInput(cls_embedding=[0.1] * 4, probe_score=0.7))


def fake_video_extractor(path, sample_id):
    return FusionInput(sample_id=sample_id, video=VideoBranchInput(
        frame_embeddings=[[0.1] * 4] * 5, frame_scores=[0.4] * 5))


def failing_extractor(path, sample_id):
    raise RuntimeError("boom")


def write_manifest(tmp_path, rows):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(rows))
    return p


@pytest.mark.unit
class TestSynthetic:
    def test_deterministic_for_seed(self):
        a = make_synthetic_samples(6, seed=3, clip_dim=16, video_dim=8)
        b = make_synthetic_samples(6, seed=3, clip_dim=16, video_dim=8)
        assert [s.to_json() for s in a] == [s.to_json() for s in b]

    def test_contains_image_and_video_samples_with_targets(self):
        samples = make_synthetic_samples(9, seed=0, clip_dim=16, video_dim=8)
        assert any(s.fusion_input.video is not None for s in samples)
        assert any(s.fusion_input.image is not None for s in samples)
        for s in samples:
            if s.fusion_input.video is not None:
                assert s.frame_labels is not None and s.pixel_mask is None
                assert s.verdict == int(s.frame_labels.max() > 0)
            else:
                assert s.pixel_mask is not None

    def test_both_classes_present(self):
        verdicts = {s.verdict for s in make_synthetic_samples(30, seed=1, clip_dim=16, video_dim=8)}
        assert verdicts == {0, 1}


@pytest.mark.unit
class TestBuilder:
    def test_builds_jsonl_with_weak_labels(self, tmp_path):
        img = tmp_path / "a.png"; img.write_bytes(b"x")
        vid = tmp_path / "b.mp4"; vid.write_bytes(b"x")
        manifest = write_manifest(tmp_path, [
            {"sample_id": "a", "media_type": "image", "label": "fake", "file_path": str(img), "dataset_origin": "GenImage"},
            {"sample_id": "b", "media_type": "video", "label": "real", "file_path": str(vid), "dataset_origin": "FF++"},
        ])
        out = tmp_path / "out.jsonl"
        stats = build_from_manifest(manifest, out, image_extractor=fake_image_extractor, video_extractor=fake_video_extractor)
        assert stats == {"written": 2, "skipped_missing": 0, "skipped_error": 0}
        a, b = read_jsonl(out)
        assert a.verdict == 1 and a.pixel_mask.min() == 1.0 and a.source == "GenImage"
        assert b.verdict == 0 and b.frame_labels.tolist() == [0.0] * 5

    def test_missing_and_failing_files_are_skipped(self, tmp_path):
        img = tmp_path / "a.png"; img.write_bytes(b"x")
        manifest = write_manifest(tmp_path, [
            {"sample_id": "gone", "media_type": "image", "label": "real", "file_path": str(tmp_path / "nope.png")},
            {"sample_id": "a", "media_type": "image", "label": "real", "file_path": str(img)},
        ])
        stats = build_from_manifest(manifest, tmp_path / "o.jsonl", image_extractor=failing_extractor)
        assert stats == {"written": 0, "skipped_missing": 1, "skipped_error": 1}

    def test_limit(self, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"{i}.png"; f.write_bytes(b"x"); files.append(f)
        manifest = write_manifest(tmp_path, [
            {"sample_id": str(i), "media_type": "image", "label": "real", "file_path": str(f)} for i, f in enumerate(files)])
        stats = build_from_manifest(manifest, tmp_path / "o.jsonl", limit=2, image_extractor=fake_image_extractor)
        assert stats["written"] == 2


@pytest.mark.integration
def test_real_extractors_end_to_end(tmp_path):
    """Runs the actual CLIP/DIRE and video branches on a tiny generated image + clip."""
    cv2 = pytest.importorskip("cv2")
    from PIL import Image
    from backend.fusion.training.build_dataset import extract_image_input, extract_video_input

    rng = np.random.default_rng(0)
    img = tmp_path / "t.png"
    Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).save(img)
    fi = extract_image_input(str(img), "img", num_dire_steps=2)
    assert len(fi.image.cls_embedding) == 768 and fi.diffusion is not None

    vid = tmp_path / "t.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    for _ in range(12):
        w.write((rng.random((64, 96, 3)) * 255).astype("uint8"))
    w.release()
    fv = extract_video_input(str(vid), "vid", num_frames=6)
    assert len(fv.video.frame_scores) == len(fv.video.frame_embeddings) == len(fv.noise.frame_noise_energies)