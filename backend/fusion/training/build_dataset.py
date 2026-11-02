"""
TruthLens — Build a fusion training set from the benchmark manifest
Author: Parth Maheshwari
Sprint: Development of Cross-Modal Fusion Transformer with dual Decoders.

Reads the manifest written by backend/ingestion/curation.py, runs every
sample through the upstream branches (CLIP:ViT probe + DIRE for images;
face alignment + dual-stream backbone + noise fingerprint for videos),
converts the results with the fusion adapters and writes a JSONL training file.

    python -m backend.fusion.training.build_dataset \
        --manifest backend/data/benchmark_manifest.json --out data/fusion/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from backend.fusion.adapters import build_fusion_input
from backend.fusion.contracts import FusionInput
from backend.fusion.training.schema import (
    FusionSample, weak_frame_labels, weak_pixel_mask, write_jsonl,
)

log = logging.getLogger("fusion.build_dataset")


def extract_image_input(path: str, sample_id: str, num_dire_steps: int = 15) -> FusionInput:
    """Run the image branches (CLIP:ViT probe + DDIM/DIRE) on one image file."""
    from backend.inference.clip_dire_service import clip_dire_service

    clip_res = clip_dire_service.clip_probe.probe(path)
    dire_res = clip_dire_service.dire_analyzer.analyze(path, num_steps=num_dire_steps)
    return build_fusion_input(sample_id, clip_result=clip_res, dire_result=dire_res)


def extract_video_input(path: str, sample_id: str, num_frames: int = 16) -> FusionInput:
    """Run the video branches (backbone embeddings/scores + noise fingerprint) on one clip."""
    from backend.ingestion.preprocessing import preprocess_video_file
    from backend.inference.face_aligner import video_face_pipeline
    from backend.inference.video_backbone import dual_stream_backbone
    from backend.inference.noise_fingerprint import noise_fingerprint_extractor

    frames = preprocess_video_file(path, num_frames=num_frames)
    if frames.shape[0] == 0:
        raise ValueError(f"no frames could be read from {path}")
    aligned = video_face_pipeline.process_frames(frames).aligned_frames
    spatial = dual_stream_backbone.predict_video_sequence(aligned)
    noise = noise_fingerprint_extractor.analyze_sequence(aligned)
    return build_fusion_input(
        sample_id, frame_embeddings=spatial.frame_embeddings, frame_scores=spatial.frame_scores,
        noise_result=noise,
    )


def build_from_manifest(
    manifest_path,
    out_path,
    limit: Optional[int] = None,
    image_extractor: Callable[[str, str], FusionInput] = extract_image_input,
    video_extractor: Callable[[str, str], FusionInput] = extract_video_input,
) -> Dict[str, int]:
    """Returns counts: written / skipped_missing / skipped_error."""
    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if limit is not None:
        entries = entries[:limit]
    samples, stats = [], {"written": 0, "skipped_missing": 0, "skipped_error": 0}

    for entry in entries:
        path = entry["file_path"]
        if not Path(path).exists():
            log.warning("missing file, skipped: %s", path)
            stats["skipped_missing"] += 1
            continue
        verdict = 1 if entry["label"] == "fake" else 0
        try:
            if entry["media_type"] == "image":
                fi = image_extractor(path, entry["sample_id"])
                sample = FusionSample(fi, verdict, pixel_mask=weak_pixel_mask(verdict),
                                      source=entry.get("dataset_origin", ""))
            else:
                fi = video_extractor(path, entry["sample_id"])
                sample = FusionSample(fi, verdict, frame_labels=weak_frame_labels(verdict, len(fi.video.frame_scores)),
                                      source=entry.get("dataset_origin", ""))
        except Exception as exc:  # one bad file must not abort a long extraction run
            log.warning("extraction failed for %s: %s", path, exc)
            stats["skipped_error"] += 1
            continue
        samples.append(sample)
        stats["written"] += 1

    write_jsonl(samples, out_path)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(build_from_manifest(args.manifest, args.out, args.limit))


if __name__ == "__main__":
    main()