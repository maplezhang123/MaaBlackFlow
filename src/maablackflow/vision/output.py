"""Write annotated PNG and JSON artifacts for detector results."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .image_io import VisionError, encode_png
from .models import DetectionResult


def annotate_image(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    result.validate()
    annotated = image.copy()
    line_width = max(2, round(min(result.image_width, result.image_height) / 360))
    font_scale = max(0.5, min(result.image_width, result.image_height) / 900)
    for node in result.nodes:
        box = node.bbox
        cv2.rectangle(
            annotated,
            (box.x, box.y),
            (box.x + box.width - 1, box.y + box.height - 1),
            (40, 220, 40),
            line_width,
        )
        cv2.drawMarker(
            annotated,
            (node.center_x, node.center_y),
            (0, 220, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=max(10, line_width * 5),
            thickness=line_width,
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            node.id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        text_x = min(max(0, box.x), max(0, result.image_width - text_width - 4))
        above_y = box.y - 7
        text_y = above_y if above_y - text_height >= 0 else min(
            result.image_height - baseline - 2, box.y + box.height + text_height + 5
        )
        cv2.rectangle(
            annotated,
            (text_x, text_y - text_height - 3),
            (text_x + text_width + 4, text_y + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            node.id,
            (text_x + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (40, 220, 40),
            line_width,
            cv2.LINE_AA,
        )
    return annotated


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