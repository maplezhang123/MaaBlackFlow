from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from maablackflow.vision import (
    BoundingBox,
    DetectedNode,
    DetectionResult,
    NodeDetector,
    VisionError,
    write_detection_outputs,
)


def synthetic_map() -> np.ndarray:
    image = np.full((360, 640, 3), (12, 24, 18), dtype=np.uint8)
    road = (130, 145, 135)
    cv2.line(image, (120, 120), (500, 120), road, 5, cv2.LINE_AA)
    cv2.line(image, (220, 120), (220, 240), road, 5, cv2.LINE_AA)
    cv2.line(image, (220, 240), (440, 240), road, 5, cv2.LINE_AA)
    for center, radius, color in (
        ((120, 120), 26, (40, 220, 230)),
        ((300, 120), 10, (180, 180, 180)),
        ((500, 120), 25, (220, 90, 60)),
        ((220, 240), 9, (180, 180, 180)),
        ((440, 240), 27, (80, 210, 90)),
    ):
        cv2.circle(image, center, radius, color, 4, cv2.LINE_AA)
        cv2.circle(image, center, max(2, radius // 3), (230, 230, 230), 2, cv2.LINE_AA)
    return image


def write_png(path, image) -> None:
    success, encoded = cv2.imencode(".png", image)
    assert success
    encoded.tofile(path)


def test_detector_supports_chinese_png_path_and_original_coordinates(tmp_path) -> None:
    source = tmp_path / "合成地图节点.png"
    image = synthetic_map()
    write_png(source, image)
    result, loaded, returned_source = NodeDetector().detect_file(source)
    assert returned_source == source
    assert loaded.shape == image.shape
    assert result.image_width == 640
    assert result.image_height == 360
    assert len(result.nodes) >= 3
    for node in result.nodes:
        node.validate(640, 360)
        assert node.bbox.x <= node.center_x < node.bbox.x + node.bbox.width
        assert node.bbox.y <= node.center_y < node.bbox.y + node.bbox.height


def test_image_input_validation_is_clear(tmp_path) -> None:
    with pytest.raises(VisionError, match="does not exist"):
        NodeDetector().detect_file(tmp_path / "missing.png")
    wrong_type = tmp_path / "image.jpg"
    wrong_type.write_bytes(b"not an image")
    with pytest.raises(VisionError, match="requires a PNG"):
        NodeDetector().detect_file(wrong_type)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    with pytest.raises(VisionError, match="invalid or unsupported"):
        NodeDetector().detect_file(broken)


def test_detection_result_json_serialization_and_bounds() -> None:
    result = DetectionResult(
        100,
        80,
        (
            DetectedNode(
                "node_01",
                25,
                30,
                BoundingBox(15, 20, 21, 21),
                0.875,
            ),
        ),
    )
    document = result.to_dict("虚构图片.png")
    encoded = json.dumps(document, ensure_ascii=False)
    assert "虚构图片.png" in encoded
    assert document["nodes"][0]["bbox"] == {
        "x": 15,
        "y": 20,
        "width": 21,
        "height": 21,
    }
    with pytest.raises(ValueError, match="outside"):
        BoundingBox(90, 70, 20, 20).validate(100, 80)


def test_writes_full_size_annotated_png_and_json(tmp_path) -> None:
    source = tmp_path / "合成图.png"
    image = synthetic_map()
    write_png(source, image)
    result, loaded, _ = NodeDetector().detect_file(source)
    json_path, annotated_path = write_detection_outputs(
        source, loaded, result, tmp_path / "output"
    )
    annotated = cv2.imdecode(np.fromfile(annotated_path, dtype=np.uint8), 1)
    assert annotated.shape == image.shape
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["node_count"] == len(result.nodes)
    assert document["image"] == {"width": 640, "height": 360}