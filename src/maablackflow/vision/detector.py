"""Traditional computer-vision baseline for circular map-node proposals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_io import VisionError, load_png
from .models import BoundingBox, DetectedNode, DetectionResult


@dataclass(frozen=True, slots=True)
class _CircleCandidate:
    x: int
    y: int
    radius: int
    score: float


class NodeDetector:
    """Detect visible circular nodes without game-specific image assets."""

    def detect_file(self, path: str | Path) -> tuple[DetectionResult, np.ndarray, Path]:
        image, source = load_png(path)
        result = self.detect(image)
        if not result.nodes:
            raise VisionError(
                "no plausible map nodes were detected; the image may not be a map overview"
            )
        return result, image, source

    def detect(self, image: np.ndarray) -> DetectionResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise VisionError("detector expects a three-channel BGR image")
        original_height, original_width = image.shape[:2]
        if original_width <= 0 or original_height <= 0:
            raise VisionError("image dimensions must be positive")

        scale = min(1.0, 1600.0 / max(original_width, original_height))
        if scale < 1.0:
            working = cv2.resize(
                image,
                (round(original_width * scale), round(original_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = image.copy()
        candidates = self._find_candidates(working)
        selected = self._select_candidates(candidates, working.shape[0])

        inverse_scale = 1.0 / scale
        ordered = sorted(selected, key=lambda item: (item.y, item.x, item.radius))
        nodes: list[DetectedNode] = []
        for index, candidate in enumerate(ordered, 1):
            center_x = round(candidate.x * inverse_scale)
            center_y = round(candidate.y * inverse_scale)
            radius = max(4, round(candidate.radius * inverse_scale * 1.18))
            x1 = max(0, center_x - radius)
            y1 = max(0, center_y - radius)
            x2 = min(original_width, center_x + radius + 1)
            y2 = min(original_height, center_y + radius + 1)
            nodes.append(
                DetectedNode(
                    id=f"node_{index:02d}",
                    center_x=center_x,
                    center_y=center_y,
                    bbox=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                    confidence=round(float(np.clip(candidate.score, 0.0, 1.0)), 3),
                )
            )
        result = DetectionResult(original_width, original_height, tuple(nodes))
        result.validate()
        return result

    def _find_candidates(self, image: np.ndarray) -> list[_CircleCandidate]:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        edges = cv2.Canny(blurred, 60, 140)
        distance_to_edge = cv2.distanceTransform(
            (edges == 0).astype(np.uint8), cv2.DIST_L2, 3
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # The central map viewport is stable across the observed zoom levels. These
        # proportional margins suppress resource bars and most bottom/side controls.
        left, right = round(width * 0.07), round(width * 0.82)
        top, bottom = round(height * 0.11), round(height * 0.82)
        right_panel = edges[
            round(height * 0.15) : round(height * 0.80),
            round(width * 0.65) : round(width * 0.95),
        ]
        middle_map = edges[
            round(height * 0.15) : round(height * 0.80),
            round(width * 0.35) : round(width * 0.60),
        ]
        right_density = float(np.mean(right_panel > 0))
        middle_density = float(np.mean(middle_map > 0))
        if right_density > 0.045 and right_density > 1.35 * max(middle_density, 0.001):
            right = round(width * 0.66)
        roi_mask = np.zeros_like(gray)
        roi_mask[top:bottom, left:right] = 255
        hough_input = cv2.bitwise_and(blurred, blurred, mask=roi_mask)

        circles = cv2.HoughCircles(
            hough_input,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(18, round(height * 0.035)),
            param1=100,
            param2=max(16, round(height * 0.03)),
            minRadius=max(5, round(height * 0.008)),
            maxRadius=max(12, round(height * 0.082)),
        )
        if circles is None:
            return []

        candidates: list[_CircleCandidate] = []
        angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        yy, xx = np.ogrid[:height, :width]
        for x, y, radius in np.rint(circles[0]).astype(int):
            if not left <= x < right or not top <= y < bottom:
                continue
            support_values = []
            for ring_radius in (max(2, radius - 2), radius, radius + 2):
                sample_x = np.clip(
                    np.rint(x + ring_radius * np.cos(angles)).astype(int),
                    0,
                    width - 1,
                )
                sample_y = np.clip(
                    np.rint(y + ring_radius * np.sin(angles)).astype(int),
                    0,
                    height - 1,
                )
                support_values.append(
                    float(np.mean(distance_to_edge[sample_y, sample_x] <= 2.3))
                )
            circular_support = max(support_values)
            disk = (xx - x) ** 2 + (yy - y) ** 2 <= max(3, 0.65 * radius) ** 2
            texture = float(np.std(gray[disk]))
            saturation = float(np.mean(hsv[:, :, 1][disk]))
            edge_density = float(np.mean(edges[disk]) / 255.0)
            score = (
                0.62 * circular_support
                + 0.14 * min(texture / 55.0, 1.0)
                + 0.12 * min(saturation / 100.0, 1.0)
                + 0.12 * min(edge_density / 0.25, 1.0)
            )
            if radius > height * 0.065:
                score -= 0.08
            candidates.append(_CircleCandidate(x, y, radius, score))
        return candidates

    def _select_candidates(
        self, candidates: list[_CircleCandidate], image_height: int
    ) -> list[_CircleCandidate]:
        if not candidates:
            return []

        # Strong larger circles establish horizontal map rows. Dim/empty nodes can
        # then survive at a lower confidence when aligned with those rows.
        row_seed = [
            item
            for item in candidates
            if item.radius >= image_height * 0.022 and item.score >= 0.63
        ]
        row_tolerance = max(12, round(image_height * 0.035))
        rows: list[list[_CircleCandidate]] = []
        for item in sorted(row_seed, key=lambda candidate: candidate.y):
            if rows and abs(item.y - round(np.mean([part.y for part in rows[-1]]))) <= row_tolerance:
                rows[-1].append(item)
            else:
                rows.append([item])
        row_centers = [round(np.median([item.y for item in row])) for row in rows]

        accepted = [
            item
            for item in candidates
            if item.score >= 0.78
            or (
                item.score >= 0.55
                and any(abs(item.y - row_y) <= row_tolerance for row_y in row_centers)
            )
        ]
        accepted.sort(key=lambda item: (-item.score, item.y, item.x, item.radius))

        selected: list[_CircleCandidate] = []
        for item in accepted:
            if any(
                (item.x - prior.x) ** 2 + (item.y - prior.y) ** 2
                <= max(28.0, 1.4 * max(item.radius, prior.radius)) ** 2
                for prior in selected
            ):
                continue
            selected.append(item)
            if len(selected) >= 64:
                break
        return selected