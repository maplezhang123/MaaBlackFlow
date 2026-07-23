"""Write categorized annotations, JSON, and road-grid debug artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .image_io import VisionError, encode_png
from .models import DetectionResult


CATEGORY_COLORS = {
    "event_node": (40, 220, 40),
    "empty_waypoint": (255, 180, 40),
    "current_position": (255, 255, 255),
    "occluded_node": (80, 80, 255),
    "uncertain": (0, 220, 255),
}


def annotate_image(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    result.validate()
    annotated = image.copy()
    line_width = max(2, round(min(result.image_width, result.image_height) / 360))
    font_scale = max(0.45, min(result.image_width, result.image_height) / 1050)
    occupied_labels: list[tuple[int, int, int, int]] = []
    for node in result.nodes:
        color = CATEGORY_COLORS[node.category]
        box = node.bbox
        cv2.rectangle(
            annotated,
            (box.x, box.y),
            (box.x + box.width - 1, box.y + box.height - 1),
            color,
            line_width,
        )
        cv2.drawMarker(
            annotated,
            (node.center_x, node.center_y),
            (0, 220, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=max(8, line_width * 4),
            thickness=line_width,
        )
        if node.marker_center is not None:
            marker = (node.marker_center.x, node.marker_center.y)
            grid = (node.grid_center.x, node.grid_center.y)
            cv2.circle(annotated, marker, max(4, line_width * 2), color, line_width)
            cv2.arrowedLine(
                annotated,
                marker,
                grid,
                color,
                line_width,
                cv2.LINE_AA,
                tipLength=0.18,
            )
        label = node.id
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        label_box = _place_label(
            node.center_x,
            node.center_y,
            box,
            text_width + 6,
            text_height + baseline + 6,
            result.image_width,
            result.image_height,
            occupied_labels,
        )
        occupied_labels.append(label_box)
        x1, y1, x2, y2 = label_box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            line_width,
            cv2.LINE_AA,
        )
    return annotated


def _place_label(
    center_x: int,
    center_y: int,
    box,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    proposals = (
        (box.x, box.y - height - 5),
        (box.x, box.y + box.height + 5),
        (box.x + box.width + 5, center_y - height // 2),
        (box.x - width - 5, center_y - height // 2),
    )
    best = None
    best_overlap = None
    for proposed_x, proposed_y in proposals:
        x1 = min(max(0, proposed_x), max(0, image_width - width))
        y1 = min(max(0, proposed_y), max(0, image_height - height))
        candidate = (x1, y1, x1 + width, y1 + height)
        overlap = sum(_intersection_area(candidate, prior) for prior in occupied)
        if best_overlap is None or overlap < best_overlap:
            best, best_overlap = candidate, overlap
        if overlap == 0:
            break
    assert best is not None
    return best


def _intersection_area(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> int:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def write_detection_outputs(
    source: Path,
    image: np.ndarray,
    result: DetectionResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    if not result.nodes:
        raise VisionError("refusing to write empty detection outputs")
    annotated = annotate_image(image, result)
    encoded = encode_png(annotated)
    document = json.dumps(
        result.to_dict(input_name=source.name), ensure_ascii=False, indent=2
    ) + "\n"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{source.stem}.nodes.json"
    annotated_path = output / f"{source.stem}.annotated.png"
    try:
        json_path.write_text(document, encoding="utf-8")
        encoded.tofile(annotated_path)
    except OSError as exc:
        raise VisionError(f"cannot write detection outputs to '{output}': {exc}") from exc
    return json_path, annotated_path


def write_grid_debug_output(
    source: Path, debug_image: np.ndarray, output_dir: str | Path
) -> Path:
    encoded = encode_png(debug_image)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    debug_path = output / f"{source.stem}.grid-debug.png"
    try:
        encoded.tofile(debug_path)
    except OSError as exc:
        raise VisionError(f"cannot write grid debug output to '{output}': {exc}") from exc
    return debug_path