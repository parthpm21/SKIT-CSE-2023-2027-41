"""
TruthLens - Benchmark Dataset Validation CLI Tool
Author: Pratham Yadav (Deep Learning & Computer Vision Lead)
Sprint 1: Benchmark Assembly & Ingestion API (User Story 1 - Item 6)

Usage:
    python backend/run_benchmark_validation.py [--manifest PATH] [--output REPORT_JSON] [--export-csv CSV_PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.data.dataset_manifest import (
    BenchmarkSample,
    DatasetManifest,
    GeneratorSource,
    ManipulationCategory,
    MediaType,
    Verdict,
)
from backend.data.image_preprocessor import NormalizationMode
from backend.data.validator import PreprocessingValidator, ValidationSummary


def create_sample_curated_manifest() -> DatasetManifest:
    """Curate a multi-source benchmark manifest covering GAN, Diffusion, and Forensics sources."""
    manifest = DatasetManifest(name="TruthLens-MultiSource-Benchmark-v1.0")

    samples = [
        BenchmarkSample(
            id="BENCH_IMG_001",
            file_path="data/samples/authentic_portrait_01.jpg",
            media_type=MediaType.IMAGE,
            verdict=Verdict.AUTHENTIC,
            manipulation_category=ManipulationCategory.NONE,
            generator_source=GeneratorSource.REAL_CAMERA,
            label=0,
            resolution=(1920, 1080),
        ),
        BenchmarkSample(
            id="BENCH_DIFF_002",
            file_path="data/samples/midjourney_v6_synth.png",
            media_type=MediaType.IMAGE,
            verdict=Verdict.MANIPULATED,
            manipulation_category=ManipulationCategory.GENERATIVE_DIFFUSION,
            generator_source=GeneratorSource.MIDJOURNEY,
            label=1,
            resolution=(1024, 1024),
        ),
        BenchmarkSample(
            id="BENCH_GAN_003",
            file_path="data/samples/stylegan2_ffhq_1024.png",
            media_type=MediaType.IMAGE,
            verdict=Verdict.MANIPULATED,
            manipulation_category=ManipulationCategory.GENERATIVE_GAN,
            generator_source=GeneratorSource.STYLEGAN2,
            label=1,
            resolution=(1024, 1024),
        ),
        BenchmarkSample(
            id="BENCH_VID_004",
            file_path="data/samples/ff_deepfakes_c23.mp4",
            media_type=MediaType.VIDEO,
            verdict=Verdict.MANIPULATED,
            manipulation_category=ManipulationCategory.FACE_SWAP,
            generator_source=GeneratorSource.FF_DEEPFAKES,
            label=1,
            resolution=(1280, 720),
            compression_quality="c23",
        ),
        BenchmarkSample(
            id="BENCH_SPLICE_005",
            file_path="data/samples/invoice_tampered.jpg",
            media_type=MediaType.IMAGE,
            verdict=Verdict.MANIPULATED,
            manipulation_category=ManipulationCategory.SPLICING,
            generator_source=GeneratorSource.REAL_CAMERA,
            label=1,
            resolution=(2480, 3508),
        ),
    ]

    for s in samples:
        manifest.add_sample(s)
    return manifest


def run_benchmark_validation(
    manifest_path: str | None = None,
    output_report: str | None = None,
    export_csv: str | None = None,
    run_preprocessing: bool = False,
) -> int:
    """Execute batch validation on benchmark manifest and log formatted statistical summaries."""
    print("=" * 65)
    print("TruthLens - Benchmark Dataset Validation & Profiling Engine")
    print("=" * 65)

    if manifest_path and Path(manifest_path).exists():
        print(f"[*] Loading manifest from: {manifest_path}")
        manifest = DatasetManifest.load_from_json(manifest_path)
    else:
        print("[*] Curating default multi-source benchmark manifest (5 sample records)...")
        manifest = create_sample_curated_manifest()

    if export_csv:
        manifest.export_to_csv(export_csv)
        print(f"[+] Exported manifest CSV index to: {export_csv}")

    validator = PreprocessingValidator(default_normalization=NormalizationMode.IMAGENET)
    summary: ValidationSummary = validator.validate_manifest(
        manifest, run_preprocessing=run_preprocessing
    )
    stats = summary.statistics

    print("\n--- Dataset Distribution & Integrity Profiling ---")
    print(f"Total Samples Evaluated : {summary.total_evaluated}")
    print(f"Valid Samples           : {summary.passed_count}")
    print(f"Failed / Corrupt        : {summary.failed_count}")
    print(f"Authentic vs Manipulated: {stats.authentic_count} ({stats.authentic_ratio*100:.1f}%) / {stats.manipulated_count} ({stats.manipulated_ratio*100:.1f}%)")
    print(f"Media Breakdown         : {stats.image_count} image(s), {stats.video_count} video(s)")
    print(f"Source Distribution     : {json.dumps(stats.source_distribution)}")
    print(f"Resolution Spread       : Min={stats.resolution_spread.get('min_resolution')}, Max={stats.resolution_spread.get('max_resolution')}, Avg={stats.resolution_spread.get('avg_resolution')}")

    if output_report:
        Path(output_report).parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"\n[+] Validation report saved to: {output_report}")

    print("\n[SUCCESS] Benchmark validation complete.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="TruthLens Benchmark Dataset Validation CLI")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest JSON file")
    parser.add_argument("--output", type=str, default="benchmark_validation_report.json", help="Path to output report JSON")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export manifest as CSV")
    parser.add_argument("--preprocess", action="store_true", help="Execute full preprocessing validation")
    args = parser.parse_args()

    exit_code = run_benchmark_validation(
        manifest_path=args.manifest,
        output_report=args.output,
        export_csv=args.export_csv,
        run_preprocessing=args.preprocess,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
