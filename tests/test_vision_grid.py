from __future__ import annotations

import cv2
import numpy as np

from maablackflow.vision.grid import CandidateEvidence, find_map_viewport
from maablackflow.vision.providers import (
    OpenCVCircleEvidenceProvider,
    OpenCVCurrentMarkerEvidenceProvider,
)
from maablackflow.vision import (
    NodeDetector,
    analyze_road_grid,
    write_grid_debug_output,
)


def hybrid_synthetic_map() -> np.ndarray:
    image = np.full((450, 800, 3), (12, 22, 16), dtype=np.uint8)
    road = (135, 145, 140)
    cv2.line(image, (140, 140), (500, 140), road, 7, cv2.LINE_AA)
    cv2.line(image, (140, 280), (500, 280), road, 7, cv2.LINE_AA)
    for x in (140, 320, 500):
        cv2.line(image, (x, 140), (x, 280), road, 7, cv2.LINE_AA)

    cv2.circle(image, (140, 140), 27, (40, 190, 225), 5, cv2.LINE_AA)
    cv2.circle(image, (500, 280), 26, (200, 80, 70), 5, cv2.LINE_AA)
    for center in ((320, 140), (500, 140), (140, 280)):
        cv2.circle(image, center, 10, (165, 170, 168), 3, cv2.LINE_AA)
        cv2.circle(image, center, 5, (4, 5, 5), -1, cv2.LINE_AA)

    # One white person-like marker centered on a real grid point.
    cv2.circle(image, (296, 250), 6, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.line(image, (296, 256), (296, 276), (245, 245, 245), 5, cv2.LINE_AA)
    cv2.line(image, (296, 262), (285, 271), (245, 245, 245), 4, cv2.LINE_AA)
    cv2.line(image, (296, 262), (307, 271), (245, 245, 245), 4, cv2.LINE_AA)

    # Busy UI-like panel. It must not become an inferred continuation of the map.
    cv2.rectangle(image, (565, 60), (799, 390), (28, 31, 34), -1)
    for y in range(80, 370, 32):
        cv2.rectangle(image, (585, y), (775, y + 18), (150, 150, 150), 2)
        cv2.circle(image, (750, y + 9), 8, (220, 220, 220), 2)
    return image


def test_road_skeleton_and_grid_axes_are_real_structure() -> None:
    image = hybrid_synthetic_map()
    analysis = analyze_road_grid(image)
    assert np.count_nonzero(analysis.road_mask) > 500
    assert np.count_nonzero(analysis.skeleton) > 100
    assert len(analysis.rows) >= 2
    assert len(analysis.columns) >= 2
    assert analysis.viewport.right_panel_detected
    assert all(
        analysis.viewport.left <= candidate.x < analysis.viewport.right
        for candidate in analysis.candidates
    )


def test_hybrid_detector_fuses_evidence_and_keeps_one_current_position(tmp_path) -> None:
    image = hybrid_synthetic_map()
    result, debug = NodeDetector().detect_with_debug(image)
    categories = [node.category for node in result.nodes]
    assert categories.count("current_position") == 1
    assert "event_node" in categories
    assert "empty_waypoint" in categories
    assert len(result.nodes) >= 5
    assert debug.shape == image.shape
    assert result.analysis["right_panel_detected"] is True
    for node in result.nodes:
        assert node.sources
        assert node.scores
        assert "icon" in node.scores
        assert node.center_x < 565

    debug_path = write_grid_debug_output(
        tmp_path / "合成地图.png", debug, tmp_path / "outputs"
    )
    assert debug_path.exists()
    loaded = cv2.imdecode(np.fromfile(debug_path, dtype=np.uint8), 1)
    assert loaded.shape == image.shape

def test_global_grid_has_unique_axes_and_unique_final_cells() -> None:
    result = NodeDetector().detect(hybrid_synthetic_map())
    rows = result.analysis["grid_rows"]
    columns = result.analysis["grid_columns"]
    sx = result.analysis["grid_spacing"]["sx"]
    sy = result.analysis["grid_spacing"]["sy"]
    assert all(right - left > 0.8 * sx for left, right in zip(columns, columns[1:]))
    assert all(lower - upper > 0.8 * sy for upper, lower in zip(rows, rows[1:]))
    cells = [(node.grid_row, node.grid_col) for node in result.nodes]
    assert len(cells) == len(set(cells))
    assert result.analysis["merged_duplicate_count"] > 0


def test_person_component_is_unique_and_keeps_subject_coordinates() -> None:
    result = NodeDetector().detect(hybrid_synthetic_map())
    people = [node for node in result.nodes if node.category == "current_position"]
    assert len(people) == 1
    assert people[0].center == people[0].grid_center
    assert np.hypot(people[0].grid_center.x - 320, people[0].grid_center.y - 280) <= 8
    assert people[0].marker_center is not None
    assert np.hypot(people[0].marker_center.x - 296, people[0].marker_center.y - 262) <= 8
    assert people[0].marker_bbox is not None
    assert people[0].grid_row is not None
    assert people[0].grid_col is not None


def test_final_boxes_scale_with_grid_and_do_not_span_cells() -> None:
    result = NodeDetector().detect(hybrid_synthetic_map())
    minimum_spacing = min(result.analysis["grid_spacing"].values())
    assert all(node.bbox.width < minimum_spacing * 0.5 for node in result.nodes)
    assert len({node.bbox.width for node in result.nodes}) == 1


def test_blank_image_reports_grid_fit_failed() -> None:
    image = np.zeros((450, 800, 3), dtype=np.uint8)
    result = NodeDetector().detect(image)
    assert result.nodes == ()
    assert result.analysis["grid_fit_status"] == "grid_fit_failed"

def test_upper_right_action_ui_is_an_occluding_mask() -> None:
    image = hybrid_synthetic_map()
    # Add a bright circular UI ornament in the generic upper-right action band.
    cv2.circle(image, (700, 65), 24, (245, 245, 245), 5, cv2.LINE_AA)
    viewport = find_map_viewport(image)
    assert not viewport.contains(700, 65)
    result = NodeDetector().detect(image)
    assert not any(node.center.x >= 640 and node.center.y < 100 for node in result.nodes)


def test_custom_evidence_provider_uses_backend_neutral_interface() -> None:
    class SyntheticProvider:
        name = "synthetic_test_evidence"
        requires_grid = False

        def collect(self, context):
            return (
                CandidateEvidence(320, 140, "synthetic_anchor", 0.9),
            )

    detector = NodeDetector(
        (
            OpenCVCircleEvidenceProvider(),
            OpenCVCurrentMarkerEvidenceProvider(),
            SyntheticProvider(),
        )
    )
    result = detector.detect(hybrid_synthetic_map())
    assert "synthetic_test_evidence" in result.analysis["evidence_providers"]