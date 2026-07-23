"""Framework-independent adapter for Maa Custom Recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from maablackflow.vision import NodeDetector, VisionError

from .serialization import build_detail

MAA_CUSTOM_RECOGNITION_NAME = "MaaBlackFlow.MapRecognize"


class MaaAdapterError(ValueError):
    """Raised for invalid Maa image input or adapter parameters."""


@dataclass(frozen=True, slots=True)
class MaaRecognitionPayload:
    success: bool
    box: tuple[int, int, int, int] | None
    detail: dict[str, object]


def image_from_maa(image: Any) -> np.ndarray:
    """Validate the SDK's BGR numpy image without importing MaaFramework."""
    if not isinstance(image, np.ndarray):
        raise MaaAdapterError(
            "Maa Custom Recognition image must be a numpy.ndarray in BGR format"
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise MaaAdapterError("Maa Custom Recognition image must have three BGR channels")
    if image.dtype != np.uint8:
        raise MaaAdapterError("Maa Custom Recognition image must use uint8 pixels")
    if image.shape[0] < 64 or image.shape[1] < 64:
        raise MaaAdapterError("Maa Custom Recognition image is too small")
    return np.ascontiguousarray(image)


class MapRecognitionAdapter:
    """Run the existing detector and create a Maa-compatible pure payload."""

    def __init__(self, detector: NodeDetector | None = None) -> None:
        self._detector = detector or NodeDetector()

    def analyze(
        self,
        image: Any,
        parameters: Mapping[str, object] | None = None,
    ) -> MaaRecognitionPayload:
        params = dict(parameters or {})
        mode = params.get("recognition_mode", "grid_baseline")
        if mode != "grid_baseline":
            raise MaaAdapterError(f"unsupported recognition_mode: {mode!r}")
        if params.get("require_solver_ready") is True:
            raise MaaAdapterError(
                "solver-ready recognition is unavailable in the v0.3a baseline"
            )
        bgr = image_from_maa(image)
        try:
            result = self._detector.detect(bgr)
        except VisionError as exc:
            raise MaaAdapterError(str(exc)) from exc
        detail = build_detail(result)
        box = _primary_box(detail)
        return MaaRecognitionPayload(
            success=(
                detail["grid_fit_status"] == "ok"
                and bool(detail["nodes"])
                and box is not None
            ),
            box=box,
            detail=detail,
        )


def _primary_box(detail: dict[str, object]) -> tuple[int, int, int, int] | None:
    """Prefer the current grid-cell box, otherwise use the fitted map ROI."""
    current = detail.get("current_position")
    if isinstance(current, dict):
        box = current.get("bbox")
        if isinstance(box, dict):
            return _box_tuple(box)
    roi = detail.get("map_roi")
    if isinstance(roi, dict):
        return _box_tuple(roi)
    return None


def _box_tuple(value: Mapping[str, object]) -> tuple[int, int, int, int]:
    items = tuple(int(value[key]) for key in ("x", "y", "width", "height"))
    return items  # type: ignore[return-value]
