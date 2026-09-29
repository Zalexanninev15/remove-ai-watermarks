"""Measure same-encoder re-watermarking on the project's open image marks.

This development-only study applies message A to publication-cleared images,
serializes the result as PNG, then applies message B with the same encoder to
the decoded A-marked artifact. It measures both messages after each write and
keeps clean-to-output and marginal attack fidelity separate.

The default cohort is twelve images: both providers from six preselected
content strata in the tracked engine-selection matrix. Generated artifacts
and reports belong outside the repository under ``.local-eval/``.

    uv run --extra dev --extra trustmark python \
      scripts/rewatermarking_study.py \
      --output-dir /path/to/.local-eval/rewatermarking-2026-09-27
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import engine_selection_manifest  # noqa: E402
import videoseal_oracle  # noqa: E402
from engine_selection_manifest import ContentFixture  # noqa: E402
from watermark_benchmark import repository_state, sha256_file  # noqa: E402
from watermark_benchmark_cohort import trustmark_runtime  # noqa: E402

log = logging.getLogger(__name__)

STUDY_VERSION = "rewatermarking-study-v1"
DEFAULT_SIZE = 512
DEFAULT_PAIR_IDS = ("00", "02", "07", "08", "11", "17")
DWT_MATCH_THRESHOLD = 44 / 48
TRUSTMARK_BITS = 61


def bit_accuracy(decoded: Sequence[int], expected: Sequence[int]) -> float:
    """Return the exact matching-bit fraction for equal-length messages."""
    if not decoded or len(decoded) != len(expected):
        raise ValueError("bit strings must be non-empty and have equal length")
    return sum(int(actual == wanted) for actual, wanted in zip(decoded, expected, strict=True)) / len(expected)


def image_fidelity(
    reference: np.ndarray[Any, np.dtype[np.uint8]],
    artifact: np.ndarray[Any, np.dtype[np.uint8]],
) -> dict[str, Any]:
    """Return the benchmark kernel's decoded 8-bit image-fidelity metrics."""
    if reference.shape != artifact.shape:
        raise ValueError(f"image shape mismatch: {reference.shape} != {artifact.shape}")
    if reference.dtype != np.uint8 or artifact.dtype != np.uint8:
        raise ValueError("image fidelity requires uint8 decoded pixels")
    delta = artifact.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta)
    mse = float(np.mean(np.square(delta)))
    changed = np.any(reference != artifact, axis=-1)
    identical = mse == 0.0
    return {
        "mae_8bit": float(np.mean(absolute)),
        "changed_fraction": float(np.mean(changed)),
        "psnr_db": None if identical else 20.0 * math.log10(255.0) - 10.0 * math.log10(mse),
        "psnr_status": "unbounded_identical" if identical else "measured",
    }


def seeded_bits(seed: int, count: int) -> tuple[int, ...]:
    """Return a deterministic binary message without touching global RNG state."""
    return tuple(int(bit) for bit in np.random.default_rng(seed).integers(0, 2, count))


def message_sha256(bits: Sequence[int]) -> str:
    return hashlib.sha256("".join(str(bit) for bit in bits).encode()).hexdigest()


def _standardize(carrier: ContentFixture, size: int) -> Image.Image:
    with Image.open(carrier.path) as source:
        return ImageOps.fit(
            source.convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def _save_png(image: Image.Image | np.ndarray[Any, Any], path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    if isinstance(image, Image.Image):
        output = image.convert("RGB")
    else:
        pixels = np.asarray(image)
        if pixels.dtype != np.uint8:
            pixels = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
        output = Image.fromarray(pixels, "RGB")
    output.save(path, format="PNG")
    with Image.open(path) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()


def _artifact_record(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256_file(path)}


def _dwt_messages() -> tuple[tuple[int, ...], tuple[int, ...]]:
    from remove_ai_watermarks.invisible_watermark import _BITS_48

    return tuple(int(bit) for bit in format(_BITS_48["FLUX.2 (Black Forest Labs)"], "048b")), tuple(
        int(bit) for bit in format(_BITS_48["Stable Diffusion XL"], "048b")
    )


def _dwt_embed(
    pixels: np.ndarray[Any, np.dtype[np.uint8]], message: Sequence[int]
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    import cv2
    from imwatermark import WatermarkEncoder

    cv2.setNumThreads(2)
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", list(message))
    marked = encoder.encode(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), "dwtDct")
    return cv2.cvtColor(np.asarray(marked, dtype=np.uint8), cv2.COLOR_BGR2RGB)


def _dwt_decode(pixels: np.ndarray[Any, np.dtype[np.uint8]]) -> tuple[int, ...]:
    import cv2
    from imwatermark import WatermarkDecoder

    decoder = WatermarkDecoder("bits", 48)
    decoded = decoder.decode(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), "dwtDct")
    return tuple(int(bit) for bit in decoded)


def _videoseal_embed(
    model: object,
    pixels: np.ndarray[Any, np.dtype[np.uint8]],
    message: Sequence[int],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.asarray(videoseal_oracle.embed_image(model, pixels.astype(np.float32) / 255.0, message))


def _videoseal_decode(model: object, pixels: np.ndarray[Any, np.dtype[np.uint8]]) -> tuple[int, ...]:
    return videoseal_oracle.decode_image(model, pixels.astype(np.float32) / 255.0)


def _trustmark_decode(runtime: Any, pixels: np.ndarray[Any, np.dtype[np.uint8]]) -> tuple[tuple[int, ...], bool, int]:
    payload, present, schema = runtime.decode(Image.fromarray(pixels, "RGB"), "binary")
    bits = tuple(int(bit) for bit in payload) if payload and set(payload) <= {"0", "1"} else ()
    return bits, bool(present), int(schema)


def _case_row(
    *,
    scheme: str,
    carrier: ContentFixture,
    message_a: Sequence[int],
    message_b: Sequence[int],
    clean: np.ndarray[Any, np.dtype[np.uint8]],
    marked_path: Path,
    rewatermarked_path: Path,
    marked: np.ndarray[Any, np.dtype[np.uint8]],
    rewatermarked: np.ndarray[Any, np.dtype[np.uint8]],
    decoded_marked: Sequence[int],
    decoded_rewatermarked: Sequence[int],
    marked_detected: bool,
    rewatermarked_detected_a: bool,
    rewatermarked_detected_b: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scheme": scheme,
        "carrier": carrier.name,
        "pair_id": carrier.pair_id,
        "provider": carrier.provider,
        "content_stratum": carrier.content_stratum,
        "source_sha256": carrier.sha256,
        "reuse_basis": carrier.reuse_basis,
        "messages": {
            "a_sha256": message_sha256(message_a),
            "b_sha256": message_sha256(message_b),
            "hamming_fraction": 1.0 - bit_accuracy(message_a, message_b),
        },
        "marked": {
            "bit_accuracy_vs_a": bit_accuracy(decoded_marked, message_a),
            "bit_accuracy_vs_b": bit_accuracy(decoded_marked, message_b),
            "detected_as_a": marked_detected,
            "artifact": _artifact_record(marked_path),
            "fidelity_from_clean": image_fidelity(clean, marked),
        },
        "rewatermarked": {
            "bit_accuracy_vs_a": bit_accuracy(decoded_rewatermarked, message_a),
            "bit_accuracy_vs_b": bit_accuracy(decoded_rewatermarked, message_b),
            "detected_as_a": rewatermarked_detected_a,
            "detected_as_b": rewatermarked_detected_b,
            "artifact": _artifact_record(rewatermarked_path),
            "fidelity_from_clean": image_fidelity(clean, rewatermarked),
            "fidelity_from_marked": image_fidelity(marked, rewatermarked),
        },
        "attack_success": marked_detected and not rewatermarked_detected_a and rewatermarked_detected_b,
    }


def _run_dwt(carrier: ContentFixture, clean_path: Path, artifacts: Path) -> list[dict[str, Any]]:
    message_flux, message_sdxl = _dwt_messages()
    with Image.open(clean_path) as image:
        clean = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    for victim_name, message_a, attacker_name, message_b in (
        ("flux", message_flux, "sdxl", message_sdxl),
        ("sdxl", message_sdxl, "flux", message_flux),
    ):
        marked_path = artifacts / f"{carrier.name}--dwt-{victim_name}--marked-a.png"
        marked = _save_png(_dwt_embed(clean, message_a), marked_path)
        rewatermarked_path = artifacts / f"{carrier.name}--dwt-{victim_name}-to-{attacker_name}--rewatermarked-b.png"
        rewatermarked = _save_png(_dwt_embed(marked, message_b), rewatermarked_path)
        decoded_marked = _dwt_decode(marked)
        decoded_rewatermarked = _dwt_decode(rewatermarked)
        rows.append(
            _case_row(
                scheme=f"dwt-dct-{victim_name}-to-{attacker_name}",
                carrier=carrier,
                message_a=message_a,
                message_b=message_b,
                clean=clean,
                marked_path=marked_path,
                rewatermarked_path=rewatermarked_path,
                marked=marked,
                rewatermarked=rewatermarked,
                decoded_marked=decoded_marked,
                decoded_rewatermarked=decoded_rewatermarked,
                marked_detected=bit_accuracy(decoded_marked, message_a) >= DWT_MATCH_THRESHOLD,
                rewatermarked_detected_a=bit_accuracy(decoded_rewatermarked, message_a) >= DWT_MATCH_THRESHOLD,
                rewatermarked_detected_b=bit_accuracy(decoded_rewatermarked, message_b) >= DWT_MATCH_THRESHOLD,
            )
        )
    return rows


def _run_trustmark(runtime: Any, carrier: ContentFixture, clean_path: Path, artifacts: Path) -> dict[str, Any]:
    message_a = seeded_bits(7, TRUSTMARK_BITS)
    message_b = seeded_bits(8, TRUSTMARK_BITS)
    with Image.open(clean_path) as image:
        clean_image = image.convert("RGB")
        clean = np.asarray(clean_image, dtype=np.uint8)
        marked_image = runtime.encode(clean_image, "".join(str(bit) for bit in message_a), MODE="binary")
    marked_path = artifacts / f"{carrier.name}--trustmark--marked-a.png"
    marked = _save_png(marked_image, marked_path)
    rewatermarked_image = runtime.encode(
        Image.fromarray(marked, "RGB"),
        "".join(str(bit) for bit in message_b),
        MODE="binary",
    )
    rewatermarked_path = artifacts / f"{carrier.name}--trustmark--rewatermarked-b.png"
    rewatermarked = _save_png(rewatermarked_image, rewatermarked_path)
    decoded_marked, marked_present, marked_schema = _trustmark_decode(runtime, marked)
    decoded_rewatermarked, rewatermarked_present, rewatermarked_schema = _trustmark_decode(runtime, rewatermarked)
    marked_detected = marked_present and marked_schema == 1 and decoded_marked == message_a
    detected_a = rewatermarked_present and rewatermarked_schema == 1 and decoded_rewatermarked == message_a
    detected_b = rewatermarked_present and rewatermarked_schema == 1 and decoded_rewatermarked == message_b
    return _case_row(
        scheme="trustmark-p-schema1",
        carrier=carrier,
        message_a=message_a,
        message_b=message_b,
        clean=clean,
        marked_path=marked_path,
        rewatermarked_path=rewatermarked_path,
        marked=marked,
        rewatermarked=rewatermarked,
        decoded_marked=decoded_marked,
        decoded_rewatermarked=decoded_rewatermarked,
        marked_detected=marked_detected,
        rewatermarked_detected_a=detected_a,
        rewatermarked_detected_b=detected_b,
    )


def _run_videoseal(model: object, carrier: ContentFixture, clean_path: Path, artifacts: Path) -> dict[str, Any]:
    message_a = videoseal_oracle.message_bits(7)
    message_b = videoseal_oracle.message_bits(8)
    with Image.open(clean_path) as image:
        clean = np.asarray(image.convert("RGB"), dtype=np.uint8)
    marked_path = artifacts / f"{carrier.name}--videoseal-image--marked-a.png"
    marked = _save_png(_videoseal_embed(model, clean, message_a), marked_path)
    rewatermarked_path = artifacts / f"{carrier.name}--videoseal-image--rewatermarked-b.png"
    rewatermarked = _save_png(_videoseal_embed(model, marked, message_b), rewatermarked_path)
    decoded_marked = _videoseal_decode(model, marked)
    decoded_rewatermarked = _videoseal_decode(model, rewatermarked)
    return _case_row(
        scheme="videoseal-image-256b",
        carrier=carrier,
        message_a=message_a,
        message_b=message_b,
        clean=clean,
        marked_path=marked_path,
        rewatermarked_path=rewatermarked_path,
        marked=marked,
        rewatermarked=rewatermarked,
        decoded_marked=decoded_marked,
        decoded_rewatermarked=decoded_rewatermarked,
        marked_detected=bit_accuracy(decoded_marked, message_a) >= videoseal_oracle.DETECTION_BIT_ACCURACY_THRESHOLD,
        rewatermarked_detected_a=(
            bit_accuracy(decoded_rewatermarked, message_a) >= videoseal_oracle.DETECTION_BIT_ACCURACY_THRESHOLD
        ),
        rewatermarked_detected_b=(
            bit_accuracy(decoded_rewatermarked, message_b) >= videoseal_oracle.DETECTION_BIT_ACCURACY_THRESHOLD
        ),
    )


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only validated positive controls, while retaining all counts."""
    summary: dict[str, Any] = {}
    for scheme in sorted({str(row["scheme"]) for row in rows}):
        group = [row for row in rows if row["scheme"] == scheme]
        controls = [row for row in group if row["marked"]["detected_as_a"]]
        successes = [row for row in controls if row["attack_success"]]

        def values(path: tuple[str, ...], source: Sequence[dict[str, Any]] = controls) -> list[float]:
            found: list[float] = []
            for row in source:
                value: Any = row
                for key in path:
                    value = value[key]
                if value is not None:
                    found.append(float(value))
            return found

        def distribution(numbers: Sequence[float]) -> dict[str, float] | None:
            if not numbers:
                return None
            return {
                "min": min(numbers),
                "median": statistics.median(numbers),
                "max": max(numbers),
                "mean": statistics.fmean(numbers),
            }

        summary[scheme] = {
            "cases": len(group),
            "valid_positive_controls": len(controls),
            "attack_successes": len(successes),
            "attack_success_rate_on_valid_controls": len(successes) / len(controls) if controls else None,
            "marked_bit_accuracy_vs_a": distribution(values(("marked", "bit_accuracy_vs_a"))),
            "rewatermarked_bit_accuracy_vs_a": distribution(values(("rewatermarked", "bit_accuracy_vs_a"))),
            "rewatermarked_bit_accuracy_vs_b": distribution(values(("rewatermarked", "bit_accuracy_vs_b"))),
            "rewatermarked_psnr_from_clean": distribution(values(("rewatermarked", "fidelity_from_clean", "psnr_db"))),
            "attack_psnr_from_marked": distribution(values(("rewatermarked", "fidelity_from_marked", "psnr_db"))),
        }
    return summary


def _contact_sheet(rows: Sequence[dict[str, Any]], artifacts: Path, output: Path) -> None:
    selected: list[dict[str, Any]] = []
    for scheme in sorted({str(row["scheme"]) for row in rows}):
        candidates = [row for row in rows if row["scheme"] == scheme and row["marked"]["detected_as_a"]]
        if candidates:
            selected.append(min(candidates, key=lambda row: row["rewatermarked"]["fidelity_from_clean"]["psnr_db"]))
    if not selected:
        return
    thumb = 256
    label_height = 36
    sheet = Image.new("RGB", (thumb * 3, (thumb + label_height) * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(selected):
        clean_path = artifacts / f"{row['carrier']}--clean.png"
        paths = (
            clean_path,
            artifacts / row["marked"]["artifact"]["path"],
            artifacts / row["rewatermarked"]["artifact"]["path"],
        )
        top = row_index * (thumb + label_height)
        for column, path in enumerate(paths):
            with Image.open(path) as image:
                thumbnail = image.convert("RGB").resize((thumb, thumb), Image.Resampling.LANCZOS)
                sheet.paste(thumbnail, (column * thumb, top))
        psnr = row["rewatermarked"]["fidelity_from_clean"]["psnr_db"]
        draw.text((4, top + thumb + 4), f"{row['scheme']} | {row['carrier']} | total PSNR {psnr:.2f} dB", fill="black")
    sheet.save(output, format="PNG")


def run_study(output_dir: Path, *, pair_ids: Sequence[str], size: int, torch_threads: int) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory {output_dir}")
    carriers = engine_selection_manifest.load_content_manifest(
        REPO / "data" / "evaluations" / "engine-selection" / "content-manifest.csv"
    )
    selected = [carrier for carrier in carriers if carrier.pair_id in pair_ids]
    expected = len(pair_ids) * len({carrier.provider for carrier in carriers})
    if len(selected) != expected:
        raise ValueError(f"selected {len(selected)} carriers, expected {expected} for pair ids {pair_ids}")

    import torch

    torch.set_num_threads(torch_threads)
    output_dir.mkdir(parents=True)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    videoseal_model = videoseal_oracle.load_model()
    trustmark = trustmark_runtime()
    rows: list[dict[str, Any]] = []
    for index, carrier in enumerate(selected, start=1):
        log.info("Carrier %s/%s: %s", index, len(selected), carrier.name)
        clean_path = artifacts / f"{carrier.name}--clean.png"
        _save_png(_standardize(carrier, size), clean_path)
        rows.extend(_run_dwt(carrier, clean_path, artifacts))
        rows.append(_run_trustmark(trustmark, carrier, clean_path, artifacts))
        rows.append(_run_videoseal(videoseal_model, carrier, clean_path, artifacts))

    cases_path = output_dir / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = summarize(rows)
    repo_state = repository_state()
    report = {
        "schema_version": 1,
        "study_version": STUDY_VERSION,
        "repository_head": repo_state["commit"],
        "tracked_tree_dirty": repo_state["dirty"],
        "study_script_sha256": sha256_file(Path(__file__)),
        "source_manifest_sha256": sha256_file(
            REPO / "data" / "evaluations" / "engine-selection" / "content-manifest.csv"
        ),
        "videoseal_source_commit": videoseal_oracle.SOURCE_COMMIT,
        "videoseal_jit_sha256": videoseal_oracle.JIT_SHA256,
        "pair_ids": list(pair_ids),
        "carriers": len(selected),
        "size": [size, size],
        "torch_threads": torch_threads,
        "case_rows_sha256": sha256_file(cases_path),
        "summary": summary,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _contact_sheet(rows, artifacts, output_dir / "contact-sheet-worst-valid.png")
    log.info("Wrote %s", report_path)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--pair-id", action="append", dest="pair_ids")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if args.size < 256 or args.size % 16:
        parser.error("--size must be at least 256 and divisible by 16")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_study(
            args.output_dir.resolve(),
            pair_ids=tuple(args.pair_ids or DEFAULT_PAIR_IDS),
            size=args.size,
            torch_threads=args.torch_threads,
        )
    except (FileExistsError, ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
