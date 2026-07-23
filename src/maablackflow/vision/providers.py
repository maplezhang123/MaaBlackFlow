"""Extensible evidence-provider boundary for offline vision backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .grid import CandidateEvidence, MapViewport, RoadGridAnalysis


@dataclass(frozen=True, slots=True)
class EvidenceContext:
    image: np.ndarray
    viewport: MapViewport
    grid_analysis: RoadGridAnalysis | None = None


class EvidenceProvider(Protocol):
    """Backend-neutral source of local observations.

    Future OpenCV template, MaaFramework TemplateMatch, or OCR adapters can
    implement this protocol without changing grid fitting or cell fusion.
    """

    name: str
    requires_grid: bool

    def collect(self, context: EvidenceContext) -> tuple[CandidateEvidence, ...]: ...


class OpenCVCircleEvidenceProvider:
    name = "opencv_circle"
    requires_grid = False

    def collect(self, context: EvidenceContext) -> tuple[CandidateEvidence, ...]:
        image, viewport = context.image, context.viewport
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        edges = cv2.Canny(blurred, 60, 140)
        distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        roi = np.zeros_like(gray)
        roi[viewport.top : viewport.bottom, viewport.left : viewport.right] = 255
        for x1, y1, x2, y2 in viewport.excluded_rectangles:
            roi[y1:y2, x1:x2] = 0
        circles = cv2.HoughCircles(
            cv2.bitwise_and(blurred, blurred, mask=roi),
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(16, round(height * 0.03)),
            param1=100,
            param2=max(15, round(height * 0.029)),
            minRadius=max(4, round(height * 0.007)),
            maxRadius=max(12, round(height * 0.082)),
        )
        if circles is None:
            return ()

        angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        yy, xx = np.ogrid[:height, :width]
        result: list[CandidateEvidence] = []
        for x, y, radius in np.rint(circles[0]).astype(int):
            if not viewport.contains(int(x), int(y)):
                continue
            support = max(
                float(
                    np.mean(
                        distance[
                            np.clip(
                                np.rint(y + ring * np.sin(angles)).astype(int),
                                0,
                                height - 1,
                            ),
                            np.clip(
                                np.rint(x + ring * np.cos(angles)).astype(int),
                                0,
                                width - 1,
                            ),
                        ]
                        <= 2.3
                    )
                )
                for ring in (max(2, radius - 2), radius, radius + 2)
            )
            disk = (xx - x) ** 2 + (yy - y) ** 2 <= max(3, 0.65 * radius) ** 2
            texture = float(np.std(gray[disk]))
            saturation = float(np.mean(hsv[:, :, 1][disk]))
            edge_density = float(np.mean(edges[disk]) / 255.0)
            score = (
                0.62 * support
                + 0.14 * min(texture / 55, 1)
                + 0.12 * min(saturation / 100, 1)
                + 0.12 * min(edge_density / 0.25, 1)
            )
            if radius > height * 0.065:
                score -= 0.08
            if score >= 0.42:
                result.append(
                    CandidateEvidence(
                        int(x),
                        int(y),
                        "hough_circle",
                        float(score),
                        int(radius),
                        {"circle": float(score)},
                    )
                )
        return tuple(result)


class OpenCVCurrentMarkerEvidenceProvider:
    name = "opencv_current_marker"
    requires_grid = True

    def collect(self, context: EvidenceContext) -> tuple[CandidateEvidence, ...]:
        analysis = context.grid_analysis
        if analysis is None or not analysis.grid.succeeded:
            return ()
        image = context.image
        height, _ = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white = ((hsv[:, :, 2] >= 210) & (hsv[:, :, 1] <= 65)).astype(np.uint8) * 255
        roi = np.zeros_like(white)
        viewport = analysis.viewport
        roi[viewport.top : viewport.bottom, viewport.left : viewport.right] = 255
        for x1, y1, x2, y2 in viewport.excluded_rectangles:
            roi[y1:y2, x1:x2] = 0
        white = cv2.morphologyEx(
            cv2.bitwise_and(white, roi), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        count, _, stats, centroids = cv2.connectedComponentsWithStats(white)
        ranked: list[tuple[float, CandidateEvidence]] = []
        for index in range(1, count):
            x, y, box_width, box_height, area = map(int, stats[index])
            if not (
                250 <= area <= 1200
                and 8 <= box_width <= 36
                and 25 <= box_height <= 75
            ):
                continue
            center_x, center_y = np.rint(centroids[index]).astype(int)
            if not viewport.contains(int(center_x), int(center_y)):
                continue
            grid_x = min(analysis.columns, key=lambda value: abs(value - center_x))
            grid_y = min(analysis.rows, key=lambda value: abs(value - center_y))
            normalized_distance = float(
                np.hypot(
                    (center_x - grid_x) / analysis.grid.spacing_x,
                    (center_y - grid_y) / analysis.grid.spacing_y,
                )
            )
            if normalized_distance > 0.36:
                continue
            vertical = min(1.0, box_height / max(box_width, 1) / 1.6)
            fill = area / max(1, box_width * box_height)
            shape = float(
                np.clip(
                    0.30 * vertical
                    + 0.25 * min(area / 500, 1)
                    + 0.20 * min(box_width / 24, 1)
                    + 0.15 * min(fill / 0.45, 1)
                    + 0.10 * (1 - normalized_distance / 0.36),
                    0,
                    1,
                )
            )
            ranked.append(
                (
                    shape,
                    CandidateEvidence(
                        int(center_x),
                        int(center_y),
                        "white_person_component",
                        shape,
                        max(box_width, box_height) // 2,
                        {
                            "person_shape": shape,
                            "grid_proximity": max(0.0, 1 - normalized_distance / 0.36),
                        },
                        (x, y, box_width, box_height),
                    ),
                )
            )
        if not ranked:
            return ()
        return (
            max(
                ranked,
                key=lambda item: (item[0], item[1].y, item[1].x),
            )[1],
        )