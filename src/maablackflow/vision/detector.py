"""Two-stage global-grid and visual-evidence map-node detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .grid import CandidateEvidence, RoadGridAnalysis, analyze_road_grid, find_map_viewport
from .image_io import VisionError, load_png
from .models import BoundingBox, DetectedNode, DetectionResult, Point
from .providers import (
    EvidenceContext,
    EvidenceProvider,
    OpenCVCircleEvidenceProvider,
    OpenCVCurrentMarkerEvidenceProvider,
)


@dataclass(frozen=True, slots=True)
class _SnappedEvidence:
    evidence: CandidateEvidence
    row: int
    column: int
    grid_x: int
    grid_y: int
    normalized_distance: float


class NodeDetector:
    """Locate unique visible map cells without identifying event subtypes."""

    def __init__(
        self, evidence_providers: Iterable[EvidenceProvider] | None = None
    ) -> None:
        self.evidence_providers = tuple(
            evidence_providers
            if evidence_providers is not None
            else (
                OpenCVCircleEvidenceProvider(),
                OpenCVCurrentMarkerEvidenceProvider(),
            )
        )

    def detect_file(self, path: str | Path) -> tuple[DetectionResult, np.ndarray, Path]:
        result, image, source, _ = self.detect_file_with_debug(path)
        return result, image, source

    def detect_file_with_debug(
        self, path: str | Path
    ) -> tuple[DetectionResult, np.ndarray, Path, np.ndarray]:
        image, source = load_png(path)
        result, debug = self.detect_with_debug(image)
        if result.analysis.get("grid_fit_status") != "ok":
            raise VisionError(
                "grid_fit_failed: no stable global periodic map grid could be fitted"
            )
        if not result.nodes:
            raise VisionError(
                "no plausible map grid points were detected; the image may not be a map overview"
            )
        return result, image, source, debug

    def detect(self, image: np.ndarray) -> DetectionResult:
        return self.detect_with_debug(image)[0]

    def detect_with_debug(self, image: np.ndarray) -> tuple[DetectionResult, np.ndarray]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise VisionError("detector expects a three-channel BGR image")
        original_height, original_width = image.shape[:2]
        if original_width <= 0 or original_height <= 0:
            raise VisionError("image dimensions must be positive")

        scale = min(1.0, 1600.0 / max(original_width, original_height))
        working = (
            cv2.resize(
                image,
                (round(original_width * scale), round(original_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else image.copy()
        )
        viewport = find_map_viewport(working)
        pre_grid = self._collect_evidence(working, viewport, None, requires_grid=False)
        analysis = analyze_road_grid(
            working, viewport=viewport, visual_evidence=pre_grid
        )
        if not analysis.grid.succeeded:
            result = DetectionResult(
                original_width,
                original_height,
                (),
                analysis={
                    "grid_fit_status": "grid_fit_failed",
                    "raw_evidence_count": len(analysis.evidence),
                    "warning": self._warning(),
                },
            )
            return result, self._render_debug(working, analysis, (), ())

        post_grid = self._collect_evidence(
            working, viewport, analysis, requires_grid=True
        )
        all_evidence = tuple(analysis.evidence) + post_grid
        snapped = self._snap_evidence(all_evidence, analysis)
        groups = self._group_evidence(snapped)
        final = self._classify_groups(working, analysis, groups)

        inverse_scale = 1.0 / scale
        box_half = max(
            6,
            round(
                min(analysis.grid.spacing_x, analysis.grid.spacing_y)
                * 0.18
                * inverse_scale
            ),
        )
        nodes: list[DetectedNode] = []
        for index, item in enumerate(final, 1):
            row, column, category, confidence, reliable, items, scores = item
            grid_x = round(analysis.columns[column] * inverse_scale)
            grid_y = round(analysis.rows[row] * inverse_scale)
            x1, y1 = max(0, grid_x - box_half), max(0, grid_y - box_half)
            x2 = min(original_width, grid_x + box_half + 1)
            y2 = min(original_height, grid_y + box_half + 1)
            person = next(
                (
                    value
                    for value in items
                    if value.evidence.source == "white_person_component"
                ),
                None,
            )
            marker_center = None
            marker_bbox = None
            if person is not None:
                marker_center = Point(
                    round(person.evidence.x * inverse_scale),
                    round(person.evidence.y * inverse_scale),
                )
                if person.evidence.bbox is not None:
                    bx, by, bw, bh = person.evidence.bbox
                    marker_bbox = BoundingBox(
                        round(bx * inverse_scale),
                        round(by * inverse_scale),
                        max(1, round(bw * inverse_scale)),
                        max(1, round(bh * inverse_scale)),
                    )
            grid_center = Point(grid_x, grid_y)
            nodes.append(
                DetectedNode(
                    id=f"node_{index:02d}",
                    center=grid_center,
                    grid_center=grid_center,
                    bbox=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                    confidence=round(confidence, 3),
                    category=category,
                    reliable=reliable,
                    sources=tuple(sorted({value.evidence.source for value in items})),
                    scores={
                        name: round(float(np.clip(value, 0, 1)), 3)
                        for name, value in sorted(scores.items())
                    },
                    evidence=tuple(
                        {
                            "source": value.evidence.source,
                            "x": round(value.evidence.x * inverse_scale),
                            "y": round(value.evidence.y * inverse_scale),
                            "confidence": round(value.evidence.confidence, 3),
                            "snap_distance": round(value.normalized_distance, 3),
                        }
                        for value in sorted(
                            items,
                            key=lambda entry: (
                                entry.evidence.source,
                                entry.evidence.y,
                                entry.evidence.x,
                            ),
                        )
                    ),
                    grid_row=row,
                    grid_col=column,
                    marker_center=marker_center,
                    marker_bbox=marker_bbox,
                )
            )

        snapped_count = len(snapped)
        accepted_evidence_count = sum(len(item[5]) for item in final)
        merged_duplicate_count = sum(max(0, len(item[5]) - 1) for item in final)
        result = DetectionResult(
            original_width,
            original_height,
            tuple(nodes),
            method="opencv_global_periodic_grid_fusion_v3",
            analysis={
                "grid_fit_status": "ok",
                "grid_spacing": {
                    "sx": round(analysis.grid.spacing_x * inverse_scale, 2),
                    "sy": round(analysis.grid.spacing_y * inverse_scale, 2),
                },
                "grid_origin": {
                    "x": round(analysis.grid.origin_x * inverse_scale, 2),
                    "y": round(analysis.grid.origin_y * inverse_scale, 2),
                },
                "grid_rows": [round(value * inverse_scale) for value in analysis.rows],
                "grid_columns": [
                    round(value * inverse_scale) for value in analysis.columns
                ],
                "grid_row_count": len(analysis.rows),
                "grid_column_count": len(analysis.columns),
                "raw_evidence_count": len(all_evidence),
                "snapped_evidence_count": snapped_count,
                "unique_grid_point_count": len(nodes),
                "merged_duplicate_count": merged_duplicate_count,
                "rejected_snapped_evidence_count": (
                    snapped_count - accepted_evidence_count
                ),
                "right_panel_detected": analysis.viewport.right_panel_detected,
                "evidence_providers": [
                    provider.name for provider in self.evidence_providers
                ],
                "warning": self._warning(),
            },
        )
        result.validate()
        debug = self._render_debug(working, analysis, snapped, final)
        if scale < 1.0:
            debug = cv2.resize(
                debug,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return result, debug

    def _collect_evidence(
        self,
        image: np.ndarray,
        viewport,
        analysis: RoadGridAnalysis | None,
        *,
        requires_grid: bool,
    ) -> tuple[CandidateEvidence, ...]:
        context = EvidenceContext(image, viewport, analysis)
        collected: list[CandidateEvidence] = []
        for provider in self.evidence_providers:
            if provider.requires_grid == requires_grid:
                collected.extend(provider.collect(context))
        return tuple(collected)

    @staticmethod
    def _warning() -> str:
        return (
            "detections are an offline baseline, not validated map topology, "
            "and must not be sent to the solver"
        )

    @staticmethod
    def _snap_evidence(
        evidence: tuple[CandidateEvidence, ...], analysis: RoadGridAnalysis
    ) -> tuple[_SnappedEvidence, ...]:
        result: list[_SnappedEvidence] = []
        spacing_x, spacing_y = analysis.grid.spacing_x, analysis.grid.spacing_y
        for item in evidence:
            column = min(
                range(len(analysis.columns)),
                key=lambda index: abs(analysis.columns[index] - item.x),
            )
            row = min(
                range(len(analysis.rows)),
                key=lambda index: abs(analysis.rows[index] - item.y),
            )
            grid_x, grid_y = analysis.columns[column], analysis.rows[row]
            normalized_distance = float(
                np.hypot(
                    (item.x - grid_x) / spacing_x,
                    (item.y - grid_y) / spacing_y,
                )
            )
            limit = 0.36 if item.source == "white_person_component" else 0.30
            if normalized_distance <= limit:
                result.append(
                    _SnappedEvidence(
                        item,
                        row,
                        column,
                        grid_x,
                        grid_y,
                        normalized_distance,
                    )
                )
        return tuple(result)

    @staticmethod
    def _group_evidence(
        snapped: tuple[_SnappedEvidence, ...],
    ) -> dict[tuple[int, int], list[_SnappedEvidence]]:
        groups: dict[tuple[int, int], list[_SnappedEvidence]] = {}
        for item in snapped:
            groups.setdefault((item.row, item.column), []).append(item)
        return groups

    def _classify_groups(self, image, analysis, groups):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 60, 140)
        accepted = []
        for (row, column), items in groups.items():
            grid_x, grid_y = analysis.columns[column], analysis.rows[row]
            if not analysis.viewport.contains(grid_x, grid_y):
                continue
            scores = {"grid": 1.0}
            for item in items:
                for name, value in item.evidence.scores.items():
                    scores[name] = max(scores.get(name, 0.0), value)
            for name, value in self._visual_scores(
                gray, hsv, edges, grid_x, grid_y, image.shape[0]
            ).items():
                scores[name] = max(scores.get(name, 0.0), value)

            circle = scores.get("circle", 0.0)
            road = scores.get("road", 0.0)
            dark = scores.get("dark_center", 0.0)
            icon = scores.get("icon", 0.0)
            max_radius = max(
                (
                    item.evidence.radius
                    for item in items
                    if item.evidence.source == "hough_circle"
                ),
                default=0,
            )
            person = [
                item
                for item in items
                if item.evidence.source == "white_person_component"
            ]
            has_grid_road = any(
                item.evidence.source.startswith("grid_road_intersection")
                for item in items
            )
            if person:
                category = "current_position"
                confidence = max(0.72, person[0].evidence.confidence)
                reliable = True
            else:
                structural = 0.42 * road + 0.28 * dark + 0.18 * circle + 0.12 * icon
                visual = 0.65 * circle + 0.25 * icon + 0.10 * road
                confidence = float(np.clip(max(structural, visual), 0, 1))
                if has_grid_road and dark >= 0.45:
                    confidence = max(confidence, 0.48)
                if not (circle >= 0.50 or has_grid_road) or confidence < 0.45:
                    continue
                if (
                    circle >= 0.56
                    and icon >= 0.38
                    and max_radius >= image.shape[0] * 0.024
                ):
                    category = "event_node"
                elif (
                    has_grid_road
                    and (
                        (road >= 0.28 and (dark >= 0.20 or circle >= 0.50))
                        or dark >= 0.45
                    )
                ) or (
                    max_radius < image.shape[0] * 0.024
                    and circle >= 0.68
                    and icon >= 0.45
                ):
                    category = "empty_waypoint"
                else:
                    category = "uncertain"
                if category == "uncertain" and not has_grid_road:
                    continue
                reliable = confidence >= 0.62 and category != "uncertain"
            scores["snap_consistency"] = max(
                0.0,
                1 - min(item.normalized_distance for item in items) / 0.36,
            )
            accepted.append(
                (row, column, category, confidence, reliable, tuple(items), scores)
            )
        return tuple(sorted(accepted, key=lambda item: (item[0], item[1], item[2])))

    @staticmethod
    def _visual_scores(gray, hsv, edges, x, y, image_height):
        height, width = gray.shape
        radius = max(13, round(image_height * 0.032))
        x1, x2 = max(0, x - radius), min(width, x + radius + 1)
        y1, y2 = max(0, y - radius), min(height, y + radius + 1)
        patch_gray = gray[y1:y2, x1:x2]
        patch_hsv = hsv[y1:y2, x1:x2]
        patch_edges = edges[y1:y2, x1:x2]
        if not patch_gray.size:
            return {"icon": 0.0, "brightness": 0.0}
        yy, xx = np.ogrid[y1:y2, x1:x2]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        value = patch_hsv[:, :, 2][disk]
        saturation = patch_hsv[:, :, 1][disk]
        icon = (
            0.45 * min(float(np.std(patch_gray[disk])) / 65, 1)
            + 0.35 * min(float(np.mean(saturation)) / 110, 1)
            + 0.20 * min(float(np.mean(patch_edges[disk]) / 255) / 0.22, 1)
        )
        return {"icon": icon, "brightness": float(np.mean(value) / 255)}

    def _render_debug(self, image, analysis, snapped, final):
        debug = (image.astype(np.float32) * 0.40).astype(np.uint8)
        debug[analysis.road_mask > 0] = (35, 125, 35)
        debug[analysis.skeleton > 0] = (180, 180, 40)
        viewport = analysis.viewport
        for x1, y1, x2, y2 in viewport.excluded_rectangles:
            cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (35, 35, 180), 2)
        for row in analysis.local_rows:
            cv2.line(
                debug,
                (viewport.left, row),
                (viewport.right - 1, row),
                (80, 65, 55),
                1,
            )
        for column in analysis.local_columns:
            cv2.line(
                debug,
                (column, viewport.top),
                (column, viewport.bottom - 1),
                (65, 55, 80),
                1,
            )
        if analysis.grid.succeeded:
            for row in analysis.rows:
                cv2.line(
                    debug,
                    (viewport.left, row),
                    (viewport.right - 1, row),
                    (255, 90, 30),
                    2,
                )
            for column in analysis.columns:
                cv2.line(
                    debug,
                    (column, viewport.top),
                    (column, viewport.bottom - 1),
                    (190, 40, 230),
                    2,
                )
        source_colors = {
            "hough_circle": (0, 180, 255),
            "white_person_component": (255, 255, 255),
        }
        for item in snapped:
            color = source_colors.get(item.evidence.source, (130, 130, 130))
            marker = (item.evidence.x, item.evidence.y)
            grid = (item.grid_x, item.grid_y)
            cv2.circle(debug, marker, 3, color, -1)
            cv2.arrowedLine(debug, marker, grid, (80, 150, 220), 1, tipLength=0.15)
        final_colors = {
            "event_node": (40, 220, 40),
            "empty_waypoint": (255, 180, 40),
            "current_position": (255, 255, 255),
            "uncertain": (0, 220, 255),
        }
        for row, column, category, _, _, _, _ in final:
            cv2.circle(
                debug,
                (analysis.columns[column], analysis.rows[row]),
                8,
                final_colors[category],
                2,
            )
        cv2.rectangle(
            debug,
            (viewport.left, viewport.top),
            (viewport.right - 1, viewport.bottom - 1),
            (180, 180, 180),
            1,
        )
        return debug