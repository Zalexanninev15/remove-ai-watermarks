#!/usr/bin/env python3
"""Revision-pinned local Perth oracle for development-only benchmarks.

Perth v1 is an implicit audio watermark: its public model returns one presence
confidence, not a user-selected payload.  This adapter therefore exposes a
thresholded presence reading and deliberately has no message label or forgery
operation.

The official ``resemble-perth`` wheel predates current PyTorch releases and
pins an obsolete stack.  Development runs install the exact source revision
below without dependency resolution, then verify the bundled checkpoint before
loading it.  Nothing here ships in the installed CLI or public Python API.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

SOURCE_COMMIT = "ff1c8ac55a976971245cdd53c18d6131ca00d993"
PACKAGE_NAME = "resemble-perth"
PACKAGE_VERSION = "1.1.0"
CHECKPOINT_SHA256 = "a15bce457ebc53ce5e6c9c3f11df78cf7ee2bf9cdab0a798902135b4c4027670"
CHECKPOINT_RELATIVE_PATH = Path("perth_net/pretrained/implicit/perth_net_250000.pth.tar")
SAMPLE_RATE = 16_000
DETECTION_THRESHOLD = 0.5


@dataclass(frozen=True)
class PerthReading:
    """One implicit-watermark presence score."""

    confidence: float

    @property
    def detected(self) -> bool:
        """Apply the upstream round-at-one-half decision rule."""
        return self.confidence > DETECTION_THRESHOLD


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_failure() -> Exception | None:
    """Return the missing or incompatible dependency error, if any."""
    try:
        import librosa  # noqa: F401
        import perth
        import torch  # noqa: F401
        import torchaudio  # noqa: F401

        if importlib.metadata.version(PACKAGE_NAME) != PACKAGE_VERSION:
            raise RuntimeError(f"{PACKAGE_NAME} must be installed at version {PACKAGE_VERSION}")
        if perth.PerthImplicitWatermarker is None:
            raise RuntimeError("resemble-perth imported without PerthImplicitWatermarker")
    except (ImportError, importlib.metadata.PackageNotFoundError, RuntimeError) as exc:
        return exc
    return None


def available() -> bool:
    """Whether the pinned Perth source stack is importable."""
    return import_failure() is None


def checkpoint_path() -> Path:
    """Return the package checkpoint after verifying its pinned digest."""
    failure = import_failure()
    if failure is not None:
        raise RuntimeError(
            "Perth oracle requires the pinned development source and its local dependencies; "
            f"install resemble-ai/Perth@{SOURCE_COMMIT} without adding it to runtime dependencies"
        ) from failure
    import perth

    path = Path(perth.__file__).resolve().parent / CHECKPOINT_RELATIVE_PATH
    if not path.is_file():
        raise RuntimeError(f"pinned Perth checkpoint is missing: {path}")
    digest = _sha256_file(path)
    if digest != CHECKPOINT_SHA256:
        raise RuntimeError(f"pinned Perth checkpoint digest mismatch: {digest} != {CHECKPOINT_SHA256}")
    return path


def load_model() -> object:
    """Load the pinned Perth implicit watermarker on CPU."""
    checkpoint_path()
    import perth

    return perth.PerthImplicitWatermarker(device="cpu")


def embed(model: object, samples: object) -> object:
    """Embed Perth's fixed implicit watermark into mono float samples."""
    import numpy as np

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    return np.asarray(model.apply_watermark(array, sample_rate=SAMPLE_RATE), dtype=np.float32).reshape(-1)


def read(model: object, samples: object) -> PerthReading:
    """Return the raw implicit-watermark confidence for mono float samples."""
    import numpy as np

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    raw = np.asarray(model.get_watermark(array, sample_rate=SAMPLE_RATE, round=False), dtype=np.float32).reshape(-1)
    if raw.size != 1 or not np.isfinite(raw).all():
        raise RuntimeError(f"Perth returned an invalid confidence shape/value: {raw!r}")
    return PerthReading(confidence=float(raw[0]))
