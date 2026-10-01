"""Shared TC260 producer identity across signatures, metadata, and visible marks."""

from __future__ import annotations

import pytest

from remove_ai_watermarks._internal.tc260_producers import TC260_PRODUCERS, producer_for_code


@pytest.mark.parametrize("producer", TC260_PRODUCERS, ids=lambda row: row.key)
def test_every_identity_resolves_across_consumers(producer):
    from remove_ai_watermarks.video import _tc260_video_platform
    from remove_ai_watermarks.watermark_registry import tc260_producer_mark

    for code in producer.codes:
        for spelling in (code, code.swapcase(), f" {code} "):
            assert producer_for_code(spelling) is producer
            assert _tc260_video_platform(spelling) == producer.video_platform
            mark = tc260_producer_mark(spelling)
            assert (mark.key if mark else None) == producer.image_mark
        if len(code) == 18:
            assert producer_for_code(f"0011{code}00001") is producer
    assert producer_for_code(f"prefix-{producer.codes[0]}-suffix") is None


def test_producer_identities_and_image_marks_are_unambiguous():
    from remove_ai_watermarks.watermark_registry import get_mark, known_marks

    codes = [code.casefold() for row in TC260_PRODUCERS for code in row.codes]
    assert len(codes) == len(set(codes))
    assert len({row.key for row in TC260_PRODUCERS}) == len(TC260_PRODUCERS)
    assert all((row.signer is None) == (row.signing_key_x is None) for row in TC260_PRODUCERS)
    image_marks = [row.image_mark for row in TC260_PRODUCERS if row.image_mark]
    assert len(image_marks) == len(set(image_marks))
    for row in TC260_PRODUCERS:
        if row.image_mark:
            assert get_mark(row.image_mark).tc260_producer_codes == row.codes
    assert all(
        mark.key in image_marks for mark in known_marks() if mark.label_regime == "tc260" and mark.tc260_producer_codes
    )


@pytest.mark.parametrize("mark", ["doubao", "hailuo", "vidu"])
def test_visible_video_confirmation_rejects_every_neighbor(mark):
    from remove_ai_watermarks.video_visible import confirms_video_provenance

    for row in TC260_PRODUCERS:
        for code in row.codes:
            assert confirms_video_provenance(mark, {"aigc_producer": code}) == (row.video_mark == mark)
    for code in ("", "unknown", f"prefix-{mark}-suffix"):
        assert not confirms_video_provenance(mark, {"aigc_producer": code})
