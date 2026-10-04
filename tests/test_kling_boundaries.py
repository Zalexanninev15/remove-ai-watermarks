"""Clean UI boundaries must not become automatic Kling removal targets."""

import numpy as np
import pytest

from remove_ai_watermarks._text_mark_engine import TextMarkEngine
from remove_ai_watermarks.kling_engine import KlingEngine, _KlingTextEngine
from remove_ai_watermarks.watermark_registry import Context, _build_candidates, decide


def test_horizontal_boundary_is_not_a_kling_wordmark():
    image = np.full((2244, 1080, 3), (175, 183, 190), dtype=np.uint8)
    image[-40:] = 0
    engine = KlingEngine()
    strict, relaxed = engine.detect_both(image)
    assert not strict.detected
    assert not relaxed.detected
    assert engine.footprint_mask(image) is None
    assert "kling" not in {d.candidate.key for d in decide(_build_candidates(image), Context())}


def test_boundary_fixture_fires_when_glyph_detail_gate_is_removed(monkeypatch: pytest.MonkeyPatch):
    image = np.full((2244, 1080, 3), (175, 183, 190), dtype=np.uint8)
    image[-40:] = 0
    monkeypatch.setattr(_KlingTextEngine, "_ladder_best", TextMarkEngine._ladder_best)
    assert KlingEngine().detect(image).detected
