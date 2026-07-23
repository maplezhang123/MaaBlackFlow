"""Stable, framework-independent serialization for Maa recognition detail."""

from __future__ import annotations

import json
from typing import Any

from maablackflow.vision.models import DetectionResult

DETAIL_SCHEMA_VERSION = "maablackflow.maafw.detail.v1"


def build_detail(result: DetectionResult) -> dict[str, object]:
    """Convert a vision result to the public Custom Recognition detail schema."""
    result.validate()
    analysis = result.analysis
    fit_status = str(analysis.get("grid_fit_status", "unknown"))
    ordered_nodes = sorted(
        result.nodes,
        key=lambda node: (
            node.grid_row if node.grid_row is not None else 10**9,
            node.grid_col if node.grid_col is not None else 10**9,
            node.id,
        ),
    )
    nodes = [_node_document(node) for node in ordered_nodes]
    current = next(
        (node for node in nodes if node["visual_category"] == "current_position"),
        None,
    )
    warning = analysis.get("warning")
    warnings = [str(warning)] if warning else []
    warnings.append("solver_ready is false: roads, exit, and topology are not validated")
    return {
        "schema_version": DETAIL_SCHEMA_VERSION,
        "reached_detection_stage": "nodes_fused" if fit_status == "ok" else "grid_fit_failed",
        "image_width": result.image_width,
        "image_height": result.image_height,
        "grid_fit_status": fit_status,
        "grid_spacing": _grid_spacing(analysis),
        "map_roi": _map_roi(result),
        "nodes": nodes,
        "current_position": current,
        "warnings": sorted(set(warnings)),
        "solver_ready": False,
    }


def stable_json(document: dict[str, object], *, pretty: bool = True) -> str:
    """Serialize deterministically without accepting unsafe solver readiness."""
    if document.get("solver_ready") is not False:
        raise ValueError("Maa recognition detail must keep solver_ready=false")
    return json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )


def _node_document(node: Any) -> dict[str, object]:
    document: dict[str, object] = {
        "id": node.id,
        "grid_row": node.grid_row,
        "grid_col": node.grid_col,
        "center": {"x": node.grid_center.x, "y": node.grid_center.y},
        "grid_center": {"x": node.grid_center.x, "y": node.grid_center.y},
        "bbox": {
            "x": node.bbox.x,
            "y": node.bbox.y,
            "width": node.bbox.width,
            "height": node.bbox.height,
        },
        "visual_category": node.category,
        "confidence": node.confidence,
        "evidence_sources": list(sorted(node.sources)),
    }
    if node.marker_center is not None:
        document["marker_center"] = {"x": node.marker_center.x, "y": node.marker_center.y}
    if node.marker_bbox is not None:
        document["marker_bbox"] = {
            "x": node.marker_bbox.x,
            "y": node.marker_bbox.y,
            "width": node.marker_bbox.width,
            "height": node.marker_bbox.height,
        }
    return document


def _grid_spacing(analysis: dict[str, object]) -> dict[str, float] | None:
    value = analysis.get("grid_spacing")
    if not isinstance(value, dict):
        return None
    sx, sy = value.get("sx"), value.get("sy")
    if not isinstance(sx, (int, float)) or not isinstance(sy, (int, float)):
        return None
    return {"sx": float(sx), "sy": float(sy)}


def _map_roi(result: DetectionResult) -> dict[str, int] | None:
    analysis = result.analysis
    rows, columns = analysis.get("grid_rows"), analysis.get("grid_columns")
    spacing = _grid_spacing(analysis)
    if (
        not isinstance(rows, list)
        or not isinstance(columns, list)
        or not rows
        or not columns
        or spacing is None
    ):
        return None
    half_x, half_y = spacing["sx"] / 2, spacing["sy"] / 2
    left = max(0, round(min(columns) - half_x))
    top = max(0, round(min(rows) - half_y))
    right = min(result.image_width, round(max(columns) + half_x))
    bottom = min(result.image_height, round(max(rows) + half_y))
    if right <= left or bottom <= top:
        return None
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}
