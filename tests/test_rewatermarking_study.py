"""Pure-logic guards for the development-only re-watermarking study."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from remove_ai_watermarks import api

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rewatermarking_study as study
import rewatermarking_synthid_candidates as synthid_candidates
import rewatermarking_video_study as video_study


def test_bit_accuracy_requires_equal_nonempty_messages() -> None:
    assert study.bit_accuracy((0, 1, 1, 0), (0, 0, 1, 0)) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="non-empty"):
        study.bit_accuracy((), ())
    with pytest.raises(ValueError, match="equal length"):
        study.bit_accuracy((0,), (0, 1))


def test_image_fidelity_uses_decoded_8bit_pixels() -> None:
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    artifact = np.full((2, 2, 3), 10, dtype=np.uint8)
    metrics = study.image_fidelity(reference, artifact)
    assert metrics["mae_8bit"] == pytest.approx(10.0)
    assert metrics["changed_fraction"] == pytest.approx(1.0)
    assert metrics["psnr_db"] == pytest.approx(28.1308036087)


def test_summary_excludes_failed_positive_controls() -> None:
    def row(*, scheme: str, control: bool, success: bool, psnr: float) -> dict[str, object]:
        return {
            "scheme": scheme,
            "marked": {"detected_as_a": control, "bit_accuracy_vs_a": 1.0 if control else 0.5},
            "rewatermarked": {
                "bit_accuracy_vs_a": 0.5,
                "bit_accuracy_vs_b": 1.0,
                "fidelity_from_clean": {"psnr_db": psnr},
                "fidelity_from_marked": {"psnr_db": psnr + 1},
            },
            "attack_success": success,
        }

    summary = study.summarize(
        [
            row(scheme="example", control=True, success=True, psnr=40.0),
            row(scheme="example", control=False, success=True, psnr=10.0),
        ]
    )["example"]
    assert summary["cases"] == 2
    assert summary["valid_positive_controls"] == 1
    assert summary["attack_successes"] == 1
    assert summary["rewatermarked_psnr_from_clean"]["median"] == pytest.approx(40.0)


def test_video_fidelity_uses_mean_per_frame_psnr() -> None:
    reference = np.zeros((2, 2, 2, 3), dtype=np.float32)
    artifact = np.stack(
        (
            np.full((2, 2, 3), 0.1, dtype=np.float32),
            np.full((2, 2, 3), 0.2, dtype=np.float32),
        )
    )
    assert video_study.mean_video_psnr(reference, artifact) == pytest.approx(16.9897, abs=1e-4)
    assert video_study.mean_video_psnr(reference, reference) is None
    with pytest.raises(ValueError, match="share"):
        video_study.mean_video_psnr(reference, artifact[:1])


def test_synthid_candidate_sources_are_manifest_pinned_and_positive() -> None:
    sources = synthid_candidates.selected_sources()
    assert set(sources) == {"openai", "gemini"}
    for provider, source in sources.items():
        assert source.is_file()
        assert synthid_candidates.sha256_file(source) == synthid_candidates.SELECTED[provider]


def test_synthid_candidate_bit_accuracy() -> None:
    assert synthid_candidates.bit_accuracy((1, 0, 1), (1, 1, 1)) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="length"):
        synthid_candidates.bit_accuracy((1,), (1, 0))


def test_synthid_control_removes_visible_mark_and_metadata_before_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "control.png"
    original = np.full((8, 9, 3), 80, dtype=np.uint8)
    cleaned = original.copy()
    cleaned[6:, 7:] = 70
    Image.fromarray(original, "RGB").save(source)

    def fake_remove(
        actual_source: Path,
        actual_output: Path,
        **kwargs: object,
    ) -> SimpleNamespace:
        assert actual_source == source
        assert actual_output == output
        assert kwargs == {
            "sensitivity": "auto",
            "backend": "cv2",
            "strip_metadata": True,
            "write_noop": True,
        }
        Image.fromarray(cleaned, "RGB").save(actual_output)
        return SimpleNamespace(
            status="cleaned",
            marks=(SimpleNamespace(key="gemini"),),
            labels=["Google Gemini visible watermark (sparkle)"],
        )

    monkeypatch.setattr(api, "remove_visible_detailed", fake_remove)

    pixels, preflight = synthid_candidates.cleaned_control(source, output, required_visible_key="gemini")

    assert np.array_equal(pixels, cleaned)
    assert preflight["status"] == "cleaned"
    assert preflight["removed_keys"] == ["gemini"]
    assert preflight["changed_pixel_fraction"] == pytest.approx(4 / 72)


def test_synthid_control_rejects_unvalidated_visible_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "control.png"
    Image.new("RGB", (8, 8), (80, 80, 80)).save(source)
    monkeypatch.setattr(
        api,
        "remove_visible_detailed",
        lambda *_args, **_kwargs: SimpleNamespace(status="partial", marks=(), labels=[]),
    )

    with pytest.raises(RuntimeError, match="not validated"):
        synthid_candidates.cleaned_control(source, output, required_visible_key="gemini")
