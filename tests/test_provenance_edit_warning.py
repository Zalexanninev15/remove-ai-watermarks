"""Independent vendor fields cannot prove provenance fraud."""

from remove_ai_watermarks.identify import _integrity_clashes


def test_mixed_edit_provenance_preserves_sources_without_alleging_fraud():
    warnings = _integrity_clashes({"c2pa": "Google", "xai": "xAI"}, None, camera_has_ai_marker=True)
    assert len(warnings) == 1
    assert "Google" in warnings[0]
    assert "xAI" in warnings[0]
    assert "edit" in warnings[0].lower()
    assert "spoofed" not in warnings[0]
    assert "laundered" not in warnings[0]


def test_camera_and_ai_fields_are_described_as_a_possible_edit_chain():
    warnings = _integrity_clashes({"iptc": "Adobe Firefly"}, "Leica", camera_has_ai_marker=True)
    assert len(warnings) == 1
    assert "captured image followed by an AI edit" in warnings[0]
    assert "inconsistent" not in warnings[0]
