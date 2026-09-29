"""Prepare native-resolution, hash-bound SynthID re-watermarking candidates.

This local-only script first removes every detected visible AI mark and AI
metadata, then makes one cleaned pixel control and one VideoSeal image-mode
overlay for each of two publication-cleared, historically provider-verified
SynthID originals. It does not contact a provider oracle.

    uv run --extra dev python scripts/rewatermarking_synthid_candidates.py \
      --output-dir /path/to/.local-eval/rewatermarking-synthid-2026-09-27
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import videoseal_oracle  # noqa: E402
from watermark_benchmark import sha256_file  # noqa: E402

log = logging.getLogger(__name__)

SELECTED = {
    "openai": "05b836ecfe40fd689177fda74384ae4fdcc446505bbc4281cd3cbb6523eb669e",
    "gemini": "4affd7f27767a445db6abf741355743ba8d95108ad922c9fff045feed8492236",
}
MESSAGE_SEED = 8


def selected_sources() -> dict[str, Path]:
    """Resolve only the two cleared and historically oracle-positive originals."""
    manifest = REPO / "data" / "synthid" / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    sources: dict[str, Path] = {}
    for provider, expected_hash in SELECTED.items():
        matches = [row for row in rows if row["sha256"] == expected_hash]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one manifest row for {provider}")
        row = matches[0]
        required_verifier = "openai-verify" if provider == "openai" else "gemini-app"
        if row["label"] != "pos" or row["verified_via"] != required_verifier:
            raise ValueError(f"{provider} source lacks the required positive-oracle provenance")
        source = manifest.parent / "originals" / row["filename"]
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"{provider} source bytes differ from the manifest")
        sources[provider] = source
    return sources


def bit_accuracy(decoded: tuple[int, ...], expected: tuple[int, ...]) -> float:
    if len(decoded) != len(expected):
        raise ValueError("decoded message length differs from the expected message")
    return sum(int(actual == wanted) for actual, wanted in zip(decoded, expected, strict=True)) / len(expected)


def overlay(model: object, pixels: np.ndarray[Any, Any], message: tuple[int, ...]) -> np.ndarray[Any, Any]:
    """Embed VideoSeal B in image mode and return 8-bit pixels."""
    marked = videoseal_oracle.embed_image(model, pixels.astype(np.float32) / 255.0, message)
    return np.rint(np.clip(marked, 0.0, 1.0) * 255.0).astype(np.uint8)


def decoded_accuracy(model: object, pixels: np.ndarray[Any, Any], message: tuple[int, ...]) -> float:
    decoded = videoseal_oracle.decode_image(model, pixels.astype(np.float32) / 255.0)
    return bit_accuracy(decoded, message)


def cleaned_control(
    source: Path,
    output: Path,
    *,
    required_visible_key: str | None = None,
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
    """Remove visible AI marks and metadata before any SynthID oracle input."""
    from remove_ai_watermarks.api import remove_visible_detailed

    with Image.open(source) as image:
        original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    report = remove_visible_detailed(
        source,
        output,
        sensitivity="auto",
        backend="cv2",
        strip_metadata=True,
        write_noop=True,
    )
    if report.status in {"partial", "unvalidated"}:
        raise RuntimeError(f"visible-mark cleanup was not validated: {report.status}")
    removed_keys = [mark.key for mark in report.marks]
    if required_visible_key is not None and required_visible_key not in removed_keys:
        raise RuntimeError(f"required visible mark was not removed: {required_visible_key}")
    with Image.open(output) as image:
        cleaned = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if cleaned.shape != original.shape:
        raise ValueError("visible-mark cleanup changed image dimensions")
    return cleaned, {
        "status": report.status,
        "removed_keys": removed_keys,
        "removed_labels": report.labels,
        "changed_pixel_fraction": float(np.mean(np.any(original != cleaned, axis=-1))),
        "output_sha256": sha256_file(output),
    }


def run(output_dir: Path, *, torch_threads: int) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory {output_dir}")
    sources = selected_sources()
    import torch

    torch.set_num_threads(torch_threads)
    model = videoseal_oracle.load_model()
    message = videoseal_oracle.message_bits(MESSAGE_SEED)
    output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for provider, source in sources.items():
        log.info("Preparing %s candidate", provider)
        control = output_dir / f"{provider}-pixel-control.png"
        clean, visible_preflight = cleaned_control(
            source,
            control,
            required_visible_key="gemini" if provider == "gemini" else None,
        )
        candidate = output_dir / f"{provider}-videoseal-overlay.png"
        marked = overlay(model, clean, message)
        if marked.shape != clean.shape:
            raise ValueError(f"{provider} overlay changed image dimensions")
        Image.fromarray(marked, "RGB").save(candidate)
        with Image.open(candidate) as image:
            marked = np.asarray(image.convert("RGB"), dtype=np.uint8)
        accuracy = decoded_accuracy(model, marked, message)
        if accuracy < videoseal_oracle.DETECTION_BIT_ACCURACY_THRESHOLD:
            raise RuntimeError(f"{provider} VideoSeal message B did not survive serialization: {accuracy:.4f}")
        delta = np.subtract(marked, clean, dtype=np.float32)
        np.square(delta, out=delta)
        mse = float(np.mean(delta, dtype=np.float64))
        if mse <= 0.0:
            raise RuntimeError(f"{provider} overlay did not change pixels")
        rows.append(
            {
                "provider": provider,
                "source_sha256": sha256_file(source),
                "control": {"path": control.name, "sha256": sha256_file(control)},
                "candidate": {"path": candidate.name, "sha256": sha256_file(candidate)},
                "visible_preflight": visible_preflight,
                "size": [int(clean.shape[1]), int(clean.shape[0])],
                "videoseal_b_bit_accuracy": accuracy,
                "psnr_from_control_db": 20.0 * math.log10(255.0) - 10.0 * math.log10(mse),
                "changed_pixel_fraction": float(np.mean(np.any(clean != marked, axis=-1))),
            }
        )
    report = {
        "source_manifest_sha256": sha256_file(REPO / "data" / "synthid" / "manifest.csv"),
        "study_script_sha256": sha256_file(Path(__file__)),
        "videoseal_jit_sha256": videoseal_oracle.JIT_SHA256,
        "message_seed": MESSAGE_SEED,
        "message_sha256": hashlib.sha256("".join(str(bit) for bit in message).encode()).hexdigest(),
        "rows": rows,
        "note": "Local candidate preparation only; no production SynthID verdict has been requested.",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--torch-threads", default=2, type=int)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report_path = run(args.output_dir.resolve(), torch_threads=args.torch_threads)
    except (FileExistsError, ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    log.info("Wrote %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
