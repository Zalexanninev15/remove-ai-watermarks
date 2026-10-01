"""Publication-safe reconstructions of metadata observed in public web samples."""

from __future__ import annotations

import json
from pathlib import Path

import piexif
from PIL import Image

from remove_ai_watermarks._internal.tc260_signature import check_tc260_signature
from remove_ai_watermarks.identify import identify, identify_metadata_record
from remove_ai_watermarks.metadata import aigc_label, remove_ai_metadata
from remove_ai_watermarks.metadata_record import collect_metadata_record


def test_honor_yoyo_nested_security_data_round_trip(tmp_path: Path) -> None:
    """Nested RSA-shaped fields remain readable and removable without a trust claim."""
    fixture = Path(__file__).resolve().parents[1] / "data/fixtures/synthetic/honor-yoyo-tc260.json"
    payload = json.loads(fixture.read_text())
    source = tmp_path / "synthetic-yoyo.jpg"
    exif = piexif.dump({"Exif": {piexif.ExifIFD.UserComment: json.dumps(payload).encode()}})
    Image.new("RGB", (32, 32), (50, 100, 150)).save(source, exif=exif)

    label = aigc_label(source)
    assert label is not None
    assert label["ContentProducer"] == payload["AIGC"]["ContentProducer"]
    assert "PubSd" in label["ReservedCode1"]
    report = identify(source, check_visible=False, check_invisible=False)
    assert report.is_ai_generated is True
    signature = check_tc260_signature(label)
    assert signature is not None
    assert signature.label == "unsupported"
    assert signature.signer is None  # Synthetic RSA fields do not establish trust.

    with Image.open(source) as image:
        before = image.tobytes()
    cleaned = remove_ai_metadata(source, tmp_path / "cleaned.jpg")
    assert aigc_label(cleaned) is None
    assert identify(cleaned, check_visible=False, check_invisible=False).is_ai_generated is None
    with Image.open(cleaned) as image:
        assert image.tobytes() == before


def test_nested_security_data_is_preserved_as_json() -> None:
    from remove_ai_watermarks.metadata import parse_tc260_aigc_json

    fixture = Path(__file__).resolve().parents[1] / "data/fixtures/synthetic/honor-yoyo-tc260.json"
    payload = json.loads(fixture.read_text())
    label = parse_tc260_aigc_json(json.dumps(payload["AIGC"]).encode())
    assert label is not None
    assert json.loads(label["ReservedCode1"]) == payload["AIGC"]["ReservedCode1"]


def test_honor_attribution_and_unsupported_content_signature(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "data/fixtures/synthetic/honor-yoyo-tc260.json"
    payload = json.loads(fixture.read_text())
    source = tmp_path / "yoyo.jpg"
    exif = piexif.dump({"Exif": {piexif.ExifIFD.UserComment: json.dumps(payload).encode()}})
    Image.new("RGB", (32, 32), (50, 100, 150)).save(source, exif=exif)
    report = identify(source, check_visible=False, check_invisible=False)
    assert report.platform == "Honor"
    payload["AIGC"]["ContentProducer"] = "001191440300MA5G49LC9K1TK01"
    exif = piexif.dump({"Exif": {piexif.ExifIFD.UserComment: json.dumps(payload).encode()}})
    other_product = tmp_path / "other-honor-product.jpg"
    Image.new("RGB", (32, 32), (50, 100, 150)).save(other_product, exif=exif)
    assert identify(other_product, check_visible=False, check_invisible=False).platform == "Honor"
    assert any("not supported" in caveat for caveat in report.caveats)
    record = json.loads(json.dumps(collect_metadata_record(source)))
    assert identify_metadata_record(record, path=source) == report
    label = aigc_label(source)
    assert label is not None
    signature = check_tc260_signature(label)
    assert signature is not None
    assert signature.label == "unsupported"
    assert signature.signer is None


def test_namespaced_nested_security_data_is_preserved_as_json() -> None:
    from remove_ai_watermarks.metadata import aigc_label_from_metadata

    fixture = Path(__file__).resolve().parents[1] / "data/fixtures/synthetic/honor-yoyo-tc260.json"
    payload = json.loads(fixture.read_text())["AIGC"]
    packet = b"<TC260:AIGC>" + json.dumps(payload).encode() + b"</TC260:AIGC>"
    label = aigc_label_from_metadata(packet)
    assert label is not None
    assert json.loads(label["ReservedCode1"]) == payload["ReservedCode1"]


def test_unsupported_signature_is_distinct_from_absent_signature() -> None:
    fixture = Path(__file__).resolve().parents[1] / "data/fixtures/synthetic/honor-yoyo-tc260.json"
    payload = json.loads(fixture.read_text())["AIGC"]
    payload["ReservedCode1"]["SecurityData"]["PubSd"] = []
    result = check_tc260_signature({"ReservedCode1": json.dumps(payload["ReservedCode1"])})
    assert result is not None
    assert result.label == "absent"
