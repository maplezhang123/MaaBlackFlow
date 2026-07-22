"""Unicode-safe image loading helpers for OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VisionError(ValueError):
    """Raised when an image cannot be validated or processed."""


def load_png(path: str | Path) -> tuple[np.ndarray, Path]:
    source = Path(path)
    if source.suffix.lower() != ".png":
        raise VisionError(f"node detection requires a PNG image: {source}")
    if not source.exists():
        raise VisionError(f"image file does not exist: {source}")
    if not source.is_file():
        raise VisionError(f"image path is not a file: {source}")
    try:
        encoded = np.fromfile(source, dtype=np.uint8)
    except OSError as exc:
        raise VisionError(f"cannot read image '{source}': {exc}") from exc
    if encoded.size == 0:
        raise VisionError(f"image file is empty: {source}")
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise VisionError(f"invalid or unsupported PNG image: {source}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise VisionError(f"unsupported image channel layout: {source}")
    height, width = image.shape[:2]
    if width < 64 or height < 64:
        raise VisionError(f"image is too small for node detection: {width}x{height}")
    return image, source


def encode_png(image: np.ndarray) -> np.ndarray:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise VisionError("failed to encode annotated PNG")
    return encoded