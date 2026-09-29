"""Measure VideoSeal message replacement on decoded, saved MP4 artifacts.

The pair of carriers is the synthetic benchmark gradient and a committed,
publication-cleared video fixture. Each clip is encoded before it is used by
the next stage, so detector and fidelity measurements refer to the retained
MP4 bytes. Outputs belong outside the repository under ``.local-eval/``.

    uv run --extra dev python scripts/rewatermarking_video_study.py \
      --output-dir /path/to/.local-eval/rewatermarking-video-2026-09-27
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import videoseal_oracle  # noqa: E402
from videoseal_temporal_study import real_carrier_frames  # noqa: E402
from watermark_benchmark import decode_video_artifact, repository_state, sha256_file  # noqa: E402
from watermark_benchmark_video_cohort import encode_clip, synth_carrier  # noqa: E402

log = logging.getLogger(__name__)


def bit_accuracy(decoded: Sequence[int], expected: Sequence[int]) -> float:
    """Return matching-bit fraction for equal-length messages."""
    if not decoded or len(decoded) != len(expected):
        raise ValueError("bit strings must be non-empty and have equal length")
    return sum(int(bit == want) for bit, want in zip(decoded, expected, strict=True)) / len(expected)


def mean_video_psnr(reference: np.ndarray[Any, Any], artifact: np.ndarray[Any, Any]) -> float | None:
    """Match the benchmark kernel's mean of per-frame decoded RGB PSNR."""
    if reference.shape != artifact.shape or reference.ndim != 4 or reference.shape[-1] != 3:
        raise ValueError("video frames must share a (T, H, W, 3) shape")
    per_frame_mse = np.empty(reference.shape[0], dtype=np.float64)
    for index, (reference_frame, artifact_frame) in enumerate(zip(reference, artifact, strict=True)):
        delta = np.subtract(artifact_frame, reference_frame, dtype=np.float32)
        np.square(delta, out=delta)
        per_frame_mse[index] = np.mean(delta, dtype=np.float64)
    finite = per_frame_mse[per_frame_mse > 0.0]
    if finite.size == 0:
        return None
    return float(np.mean(10.0 * np.log10(1.0 / finite)))


def _measure(
    model: object,
    frames: np.ndarray[Any, Any],
    message_a: Sequence[int],
    message_b: Sequence[int],
) -> dict[str, Any]:
    reading = videoseal_oracle.read(model, frames, message=message_a)
    accuracy_b = bit_accuracy(reading.decoded_bits, message_b)
    return {
        "bit_accuracy_vs_a": reading.bit_accuracy,
        "bit_accuracy_vs_b": accuracy_b,
        "detected_as_a": reading.bit_accuracy >= videoseal_oracle.DETECTION_BIT_ACCURACY_THRESHOLD,
        "detected_as_b": accuracy_b >= videoseal_oracle.DETECTION_BIT_ACCURACY_THRESHOLD,
    }


def run_study(output_dir: Path, *, torch_threads: int = 2) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory {output_dir}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the video study")
    import torch

    torch.set_num_threads(torch_threads)
    model = videoseal_oracle.load_model()
    message_a = videoseal_oracle.message_bits(7)
    message_b = videoseal_oracle.message_bits(8)
    output_dir.mkdir(parents=True)
    carriers = {
        "moving_gradient": (
            synth_carrier("moving_gradient")[:64],
            {"source": "deterministic synthetic benchmark carrier"},
        ),
        "real_sora": real_carrier_frames("sora"),
    }
    rows: list[dict[str, Any]] = []
    for name, (source_frames, provenance) in carriers.items():
        log.info("Measuring %s", name)
        case_dir = output_dir / name
        case_dir.mkdir()
        clean_path = encode_clip(ffmpeg, case_dir / "clean.mp4", source_frames)
        clean, clean_sha256 = decode_video_artifact(clean_path)
        marked_path = encode_clip(
            ffmpeg,
            case_dir / "marked-a.mp4",
            np.asarray(videoseal_oracle.embed(model, clean, message_a)),
        )
        marked, marked_sha256 = decode_video_artifact(marked_path)
        rewatermarked_path = encode_clip(
            ffmpeg,
            case_dir / "rewatermarked-b.mp4",
            np.asarray(videoseal_oracle.embed(model, marked, message_b)),
        )
        rewatermarked, rewatermarked_sha256 = decode_video_artifact(rewatermarked_path)
        baseline = _measure(model, clean, message_a, message_b)
        marked_reading = _measure(model, marked, message_a, message_b)
        rewatermarked_reading = _measure(model, rewatermarked, message_a, message_b)
        row = {
            "carrier": name,
            "provenance": provenance,
            "frame_shape": list(clean.shape),
            "clean": {"sha256": clean_sha256, **baseline},
            "marked": {
                "sha256": marked_sha256,
                **marked_reading,
                "mean_psnr_from_clean_db": mean_video_psnr(clean, marked),
            },
            "rewatermarked": {
                "sha256": rewatermarked_sha256,
                **rewatermarked_reading,
                "mean_psnr_from_clean_db": mean_video_psnr(clean, rewatermarked),
                "mean_psnr_from_marked_db": mean_video_psnr(marked, rewatermarked),
            },
            "attack_success": (
                marked_reading["detected_as_a"]
                and not rewatermarked_reading["detected_as_a"]
                and rewatermarked_reading["detected_as_b"]
            ),
        }
        rows.append(row)

    cases_path = output_dir / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    report = {
        "repository_head": repository_state()["commit"],
        "study_script_sha256": sha256_file(Path(__file__)),
        "case_rows_sha256": sha256_file(cases_path),
        "videoseal_jit_sha256": videoseal_oracle.JIT_SHA256,
        "torch_threads": torch_threads,
        "carriers": len(rows),
        "attack_successes": sum(int(row["attack_success"]) for row in rows),
        "mean_psnr_from_clean_db": statistics.fmean(row["rewatermarked"]["mean_psnr_from_clean_db"] for row in rows),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("Wrote %s", report_path)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_study(args.output_dir.resolve(), torch_threads=args.torch_threads)
    except (FileExistsError, ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
