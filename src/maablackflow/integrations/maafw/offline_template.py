"""Safe offline MaaFramework TemplateMatch execution helpers.

The optional runtime is imported lazily. The static controller only returns a
caller-supplied image; every input or application operation is rejected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .agent import MaaFrameworkRuntimeError, OPTIONAL_RUNTIME_ERROR


@dataclass(frozen=True, slots=True)
class OfflineTemplateSpec:
    template_id: str
    resource_path: str
    visual_class: str
    scale_group: str


@dataclass(frozen=True, slots=True)
class OfflineTemplateHit:
    template_id: str
    box: tuple[int, int, int, int]
    score: float
    visual_class: str
    scale_group: str


@dataclass(frozen=True, slots=True)
class OfflineTemplateRun:
    hits: tuple[OfflineTemplateHit, ...]
    raw_results: tuple[dict[str, object], ...]
    controller_calls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SnappedTemplateHit:
    template_id: str
    box: tuple[int, int, int, int]
    score: float
    visual_class: str
    grid_row: int
    grid_col: int
    grid_center: tuple[int, int]
    center_distance: float


def template_pipeline(
    templates: Iterable[OfflineTemplateSpec], *, threshold: float = 0.70
) -> dict[str, object]:
    """Build one DoNothing Pipeline v2 node per approved template."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("TemplateMatch threshold must be between zero and one")
    document: dict[str, object] = {}
    for index, template in enumerate(templates, 1):
        if not template.template_id or not template.resource_path:
            raise ValueError("template id and resource path must not be empty")
        entry = f"MaaBlackFlowOfflineTemplate{index:03d}"
        document[entry] = {
            "recognition": {
                "type": "TemplateMatch",
                "param": {
                    "template": [template.resource_path],
                    "threshold": [threshold],
                    "order_by": "Score",
                    "index": 0,
                    "method": 5,
                    "green_mask": False,
                },
            },
            "action": {"type": "DoNothing", "param": {}},
            "next": [],
        }
    return document


def filter_and_snap_hits(
    hits: Iterable[OfflineTemplateHit],
    *,
    grid_rows: Iterable[int],
    grid_columns: Iterable[int],
    grid_spacing: tuple[float, float],
    map_roi: tuple[int, int, int, int],
    ui_rectangles: Iterable[tuple[int, int, int, int]],
    maximum_distance_ratio: float = 0.30,
) -> tuple[SnappedTemplateHit, ...]:
    """Apply ROI/UI checks, grid snapping, and one-hit-per-cell fusion."""
    rows, columns = tuple(grid_rows), tuple(grid_columns)
    if not rows or not columns:
        raise ValueError("grid rows and columns must not be empty")
    sx, sy = grid_spacing
    if sx <= 0 or sy <= 0 or maximum_distance_ratio <= 0:
        raise ValueError("grid spacing and distance ratio must be positive")
    left, top, right, bottom = map_roi
    if right <= left or bottom <= top:
        raise ValueError("map ROI must use positive bounds")
    ui = tuple(ui_rectangles)
    distance_limit = maximum_distance_ratio * min(sx, sy)
    by_cell: dict[tuple[int, int], SnappedTemplateHit] = {}
    for hit in hits:
        x, y, width, height = hit.box
        if width <= 0 or height <= 0:
            continue
        center_x, center_y = x + width / 2.0, y + height / 2.0
        if not (left <= center_x < right and top <= center_y < bottom):
            continue
        box_area = width * height
        if any(
            (ux1 <= center_x < ux2 and uy1 <= center_y < uy2)
            or _intersection_area(hit.box, (ux1, uy1, ux2, uy2))
            > box_area * 0.10
            for ux1, uy1, ux2, uy2 in ui
        ):
            continue
        row = min(range(len(rows)), key=lambda index: abs(rows[index] - center_y))
        column = min(
            range(len(columns)), key=lambda index: abs(columns[index] - center_x)
        )
        grid_x, grid_y = columns[column], rows[row]
        distance = float(
            ((center_x - grid_x) ** 2 + (center_y - grid_y) ** 2) ** 0.5
        )
        if distance > distance_limit:
            continue
        snapped = SnappedTemplateHit(
            hit.template_id,
            hit.box,
            hit.score,
            hit.visual_class,
            row,
            column,
            (grid_x, grid_y),
            distance,
        )
        cell = (row, column)
        current = by_cell.get(cell)
        if current is None or (
            snapped.score,
            snapped.template_id,
            tuple(-value for value in snapped.box),
        ) > (
            current.score,
            current.template_id,
            tuple(-value for value in current.box),
        ):
            by_cell[cell] = snapped
    return tuple(
        by_cell[cell]
        for cell in sorted(by_cell, key=lambda value: (value[0], value[1]))
    )


def execute_template_pipeline(
    resource_dir: str | Path,
    image: np.ndarray,
    templates: Iterable[OfflineTemplateSpec],
    *,
    timeout_seconds: float = 20.0,
) -> OfflineTemplateRun:
    """Run real MaaFramework Resource/Tasker against one immutable image."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("offline controller image must be a uint8 numpy.ndarray")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("offline controller image must have three or four channels")
    specs = tuple(templates)
    if not specs:
        return OfflineTemplateRun((), (), ())
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    try:
        from maa.controller import CustomController
        from maa.define import MaaControllerFeatureEnum
        from maa.pipeline import JRecognitionType, JTemplateMatch
        from maa.resource import Resource
        from maa.tasker import Tasker
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise MaaFrameworkRuntimeError(OPTIONAL_RUNTIME_ERROR) from exc

    class StaticImageController(CustomController):
        def __init__(self, static_image: np.ndarray) -> None:
            self._image = np.ascontiguousarray(static_image)
            self.calls: list[str] = []
            self.forbidden_calls: list[str] = []
            super().__init__()

        def connect(self) -> bool:
            self.calls.append("connect")
            return True

        def connected(self) -> bool:
            return True

        def request_uuid(self) -> str:
            return "maablackflow-offline-static-image"

        def get_features(self) -> int:
            return int(MaaControllerFeatureEnum.Null)

        def screencap(self) -> np.ndarray:
            self.calls.append("screencap")
            return self._image.copy()

        def _reject(self, operation: str) -> bool:
            self.forbidden_calls.append(operation)
            return False

        def start_app(self, intent: str) -> bool:
            return self._reject("start_app")

        def stop_app(self, intent: str) -> bool:
            return self._reject("stop_app")

        def click(self, x: int, y: int) -> bool:
            return self._reject("click")

        def swipe(
            self, x1: int, y1: int, x2: int, y2: int, duration: int
        ) -> bool:
            return self._reject("swipe")

        def touch_down(
            self, contact: int, x: int, y: int, pressure: int
        ) -> bool:
            return self._reject("touch_down")

        def touch_move(
            self, contact: int, x: int, y: int, pressure: int
        ) -> bool:
            return self._reject("touch_move")

        def touch_up(self, contact: int) -> bool:
            return self._reject("touch_up")

        def click_key(self, keycode: int) -> bool:
            return self._reject("click_key")

        def input_text(self, text: str) -> bool:
            return self._reject("input_text")

        def key_down(self, keycode: int) -> bool:
            return self._reject("key_down")

        def key_up(self, keycode: int) -> bool:
            return self._reject("key_up")

        def shell(self, cmd: str, timeout: int) -> None:
            self._reject("shell")
            return None

    resource = Resource()
    _wait_job(resource.post_bundle(Path(resource_dir).resolve()), timeout_seconds)
    if not resource.loaded:
        raise MaaFrameworkRuntimeError("MaaFramework failed to load offline resource")

    controller = StaticImageController(image)
    _wait_job(controller.post_connection(), timeout_seconds)
    tasker = Tasker()
    if not tasker.bind(resource, controller) or not tasker.inited:
        raise MaaFrameworkRuntimeError(
            "MaaFramework failed to bind offline Resource and CustomController"
        )

    hits: list[OfflineTemplateHit] = []
    raw_results: list[dict[str, object]] = []
    for spec in specs:
        task = tasker.post_recognition(
            JRecognitionType.TemplateMatch,
            JTemplateMatch(
                template=[spec.resource_path],
                threshold=[0.70],
                order_by="Score",
                index=0,
                method=5,
                green_mask=False,
            ),
            image,
        )
        _wait_job(task, timeout_seconds)
        detail = task.get()
        if detail is None:
            raise MaaFrameworkRuntimeError(
                f"MaaFramework returned no task detail for {spec.template_id}"
            )
        for node_id in detail.node_id_list:
            node = tasker.get_node_detail(node_id)
            recognition = None if node is None else node.recognition
            if recognition is None:
                continue
            raw = recognition.raw_detail
            raw_results.append(
                {
                    "template_id": spec.template_id,
                    "recognition_hit": bool(recognition.hit),
                    "recognition_box": _rect_tuple(recognition.box),
                    "raw_detail": raw,
                }
            )
            for result in _accepted_results(raw):
                box = _box_tuple(result.get("box"))
                score = result.get("score")
                if box is None or not isinstance(score, (int, float)):
                    continue
                hits.append(
                    OfflineTemplateHit(
                        spec.template_id,
                        box,
                        float(score),
                        spec.visual_class,
                        spec.scale_group,
                    )
                )

    if controller.forbidden_calls:
        operations = ", ".join(controller.forbidden_calls)
        raise MaaFrameworkRuntimeError(
            f"offline TemplateMatch attempted forbidden controller calls: {operations}"
        )
    return OfflineTemplateRun(
        tuple(
            sorted(
                hits,
                key=lambda hit: (
                    hit.template_id,
                    -hit.score,
                    hit.box[1],
                    hit.box[0],
                ),
            )
        ),
        tuple(raw_results),
        tuple(controller.calls),
    )


def _wait_job(job: object, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not job.done:  # type: ignore[attr-defined]
        if time.monotonic() >= deadline:
            raise MaaFrameworkRuntimeError("MaaFramework offline job timed out")
        time.sleep(0.01)
    if not job.succeeded:  # type: ignore[attr-defined]
        raise MaaFrameworkRuntimeError("MaaFramework offline job failed")


def _accepted_results(raw: object) -> list[Mapping[str, object]]:
    if not isinstance(raw, dict):
        return []
    filtered = raw.get("filtered")
    if isinstance(filtered, list):
        values = [item for item in filtered if isinstance(item, dict)]
        if values:
            return values
    best = raw.get("best")
    return [best] if isinstance(best, dict) else []


def _box_tuple(value: object) -> tuple[int, int, int, int] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return tuple(int(item) for item in value)  # type: ignore[return-value]
    if isinstance(value, dict):
        try:
            return (
                int(value["x"]),
                int(value["y"]),
                int(value["width"]),
                int(value["height"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _rect_tuple(value: object) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if all(hasattr(value, name) for name in ("x", "y", "width", "height")):
        return (
            int(value.x),
            int(value.y),
            int(value.width),
            int(value.height),
        )
    return _box_tuple(value)


def _intersection_area(
    box: tuple[int, int, int, int], bounds: tuple[int, int, int, int]
) -> int:
    x, y, width, height = box
    left, top, right, bottom = bounds
    overlap_width = max(0, min(x + width, right) - max(x, left))
    overlap_height = max(0, min(y + height, bottom) - max(y, top))
    return overlap_width * overlap_height
