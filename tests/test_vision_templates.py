from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from maablackflow.vision.models import (
    BoundingBox,
    DetectedNode,
    DetectionResult,
    Point,
)
from maablackflow.vision.templates import (
    TemplateCandidateError,
    build_template_candidates,
    classify_scale_group,
    perceptual_hash,
)


def _node(
    node_id: str,
    center: tuple[int, int],
    category: str,
    *,
    marker: tuple[int, int] | None = None,
) -> DetectedNode:
    x, y = center
    marker_point = Point(*marker) if marker else None
    marker_box = (
        BoundingBox(marker[0] - 8, marker[1] - 15, 17, 31) if marker else None
    )
    return DetectedNode(
        id=node_id,
        center=Point(x, y),
        grid_center=Point(x, y),
        bbox=BoundingBox(x - 20, y - 20, 41, 41),
        confidence=0.91,
        category=category,
        reliable=True,
        sources=("synthetic_high_confidence",),
        grid_row=0 if y == 200 else 1,
        grid_col=x // 200 - 1,
        marker_center=marker_point,
        marker_bbox=marker_box,
    )


class StubDetector:
    def detect(self, image: np.ndarray) -> DetectionResult:
        assert image.shape == (720, 1280, 3)
        return DetectionResult(
            1280,
            720,
            (
                _node("event", (200, 200), "event_node"),
                _node("empty", (400, 200), "empty_waypoint"),
                _node("current", (600, 300), "current_position", marker=(575, 280)),
            ),
            analysis={
                "grid_fit_status": "ok",
                "grid_spacing": {"sx": 168.0, "sy": 168.0},
                "right_panel_detected": False,
            },
        )


def _write_map(path: Path) -> None:
    image = np.full((720, 1280, 3), (25, 35, 30), np.uint8)
    cv2.line(image, (180, 200), (620, 200), (140, 145, 142), 7)
    cv2.circle(image, (200, 200), 24, (40, 180, 220), 5)
    cv2.circle(image, (400, 200), 10, (150, 155, 152), 3)
    cv2.circle(image, (575, 270), 7, (245, 245, 245), -1)
    cv2.line(image, (575, 277), (575, 295), (245, 245, 245), 5)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(path)


def test_private_candidate_build_excludes_holdout_and_preserves_all_duplicates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Screenshots"
    source.mkdir()
    _write_map(source / "地图一.png")
    _write_map(source / "地图二.png")
    (source / "固定验收.png").write_bytes(b"not an image and must stay unread")
    _write_map(source / "节点内部.png")
    small = np.zeros((100, 100, 3), np.uint8)
    ok, encoded = cv2.imencode(".png", small)
    assert ok
    encoded.tofile(source / "地图小图.png")

    output = tmp_path / "templates_private" / "run03"
    result = build_template_candidates(
        source,
        output,
        excluded_filenames=["固定验收.png"],
        detector=StubDetector(),
    )

    assert result.candidate_count == 6
    assert result.recommended_count == 3
    assert len(list((output / "all_candidates").rglob("*.png"))) == 6
    assert len(list((output / "selected_candidates").rglob("*.png"))) == 3
    assert result.contact_sheet_path.exists()
    assert (result.review_bundle_path / "contact_sheet.png").exists()
    assert (
        output
        / "private_resource_preview"
        / "resource"
        / "pipeline"
        / "run03_template_draft.json"
    ).exists()
    preview_images = output / "private_resource_preview" / "resource" / "image"
    assert len(list(preview_images.rglob("*.png"))) == 3

    sources = json.loads(result.source_manifest_path.read_text("utf-8"))
    holdout = next(item for item in sources["sources"] if item["role"].startswith("holdout"))
    assert holdout == {
        "private_filename": "固定验收.png",
        "processed": False,
        "role": "holdout_excluded_unread",
    }
    internal = next(item for item in sources["sources"] if item["private_filename"] == "节点内部.png")
    assert internal["skip_reason"] == "not_map_overview"
    too_small = next(
        item for item in sources["sources"] if item["private_filename"] == "地图小图.png"
    )
    assert too_small["skip_reason"] == "source_is_not_1280x720"
    template_sources = [
        item for item in sources["sources"] if item["role"] == "template_source"
    ]
    assert sum(item.get("duplicate_of_source_id") is not None for item in template_sources) == 2

    manifest = json.loads(result.candidate_manifest_path.read_text("utf-8"))
    assert manifest["solver_ready"] is False
    assert manifest["template_match_executed"] is False
    assert {item["visual_class"] for item in manifest["candidates"]} == {
        "event_node",
        "empty_waypoint",
        "current_position",
    }
    assert all(item["scale_group"] == "normal" for item in manifest["candidates"])
    assert sum(item["near_duplicate"] for item in manifest["candidates"]) == 6
    current = next(
        item for item in manifest["candidates"] if item["visual_class"] == "current_position"
    )
    assert current["contamination_risks"]["person"] is True

    summary_text = result.summary_path.read_text("utf-8")
    assert "地图一.png" not in summary_text
    assert str(source.resolve()) not in summary_text
    assert "private_source_id" in summary_text


def test_pipeline_draft_is_relative_do_nothing_and_not_executed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_map(source / "地图.png")
    (source / "holdout.png").write_bytes(b"unread")
    result = build_template_candidates(
        source,
        tmp_path / "output",
        excluded_filenames=["holdout.png"],
        detector=StubDetector(),
    )
    pipeline = json.loads(result.pipeline_draft_path.read_text("utf-8"))
    assert pipeline
    for node in pipeline.values():
        assert node["recognition"]["type"] == "TemplateMatch"
        params = node["recognition"]["param"]
        assert params["roi"] == [19, 40, 1095, 579]
        assert all(path.startswith("run03/selected_candidates/") for path in params["template"])
        assert all(not Path(path).is_absolute() for path in params["template"])
        assert node["action"] == {"type": "DoNothing", "param": {}}
        assert node["next"] == []
    serialized = json.dumps(pipeline)
    assert not any(word in serialized for word in ("Click", "Swipe", "Shell", "Command"))


def test_scale_groups_hash_and_output_reuse_guard(tmp_path: Path) -> None:
    assert classify_scale_group(168, 720) == "normal"
    assert classify_scale_group(98, 720) == "zoomed_out"
    assert classify_scale_group(220, 720) == "zoomed_in_unsupported"
    image = np.full((50, 50, 3), 90, np.uint8)
    assert perceptual_hash(image) == perceptual_hash(image.copy())

    source = tmp_path / "source"
    source.mkdir()
    _write_map(source / "地图.png")
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(TemplateCandidateError, match="not empty"):
        build_template_candidates(
            source,
            output,
            excluded_filenames=["holdout.png"],
            detector=StubDetector(),
        )
