"""Tests for the pinned PixelSeal oracle's pure helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pixelseal_oracle  # noqa: E402


class TestMessage:
    def test_message_is_deterministic_and_256_bits(self) -> None:
        first = pixelseal_oracle.message_bits()
        second = pixelseal_oracle.message_bits()

        assert first == second
        assert len(first) == 256
        assert set(first) <= {0, 1}

    def test_seed_changes_the_message(self) -> None:
        assert pixelseal_oracle.message_bits(seed=8) != pixelseal_oracle.message_bits(seed=7)


class TestReading:
    def test_label_is_hex_of_decoded_bits(self) -> None:
        reading = pixelseal_oracle.PixelSealReading(1.0, tuple([1] * 8 + [0] * 8))

        assert reading.label == "ff00"

    def test_detection_rule_is_the_documented_adapter_threshold(self) -> None:
        assert pixelseal_oracle.PixelSealReading(0.899, ()).detected is False
        assert pixelseal_oracle.PixelSealReading(0.9, ()).detected is True


class TestPins:
    @pytest.mark.parametrize(
        "value",
        [pixelseal_oracle.CHECKPOINT_SHA256, pixelseal_oracle.MODEL_CARD_SHA256],
    )
    def test_digests_are_sha256(self, value: str) -> None:
        assert len(value) == 64
        assert all(char in "0123456789abcdef" for char in value)

    def test_source_commit_is_pinned(self) -> None:
        assert len(pixelseal_oracle.SOURCE_COMMIT) == 40


def test_cache_dir_env_override_is_honored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original_cache = pixelseal_oracle.CACHE_DIR
    try:
        with monkeypatch.context() as environment:
            environment.setenv("PIXELSEAL_CACHE_DIR", str(tmp_path / "cache"))
            importlib.reload(pixelseal_oracle)
            assert tmp_path / "cache" == pixelseal_oracle.CACHE_DIR
    finally:
        importlib.reload(pixelseal_oracle)
    assert original_cache == pixelseal_oracle.CACHE_DIR
