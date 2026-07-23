"""Private, holdout-safe template candidate preparation.

This module only crops candidates and computes perceptual hashes. It never runs
TemplateMatch and never promotes a candidate into the public Maa resource tree.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .dataset import infer_scene_label, is_detection_candidate, sha256_file
from .detector import NodeDetector
from .image_io import VisionError, encode_png, load_png

_ALLOWED_CLASSES = {"event_node", "empty_waypoint", "current_position"}
_CONFIDENCE_FLOORS = {
    "event_node": 0.62,
    "empty_waypoint": 0.60,
    "current_position": 0.72,
}


class TemplateCandidateError(ValueError):
    """Raised when private candidate preparation cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class TemplateBuildResult:
    output_dir: Path
    source_manifest_path: Path
    candidate_manifest_path: Path
    summary_path: Path
    contact_sheet_path: Path
    pipeline_draft_path: Path
    review_bundle_path: Path
    candidate_count: int
    recommended_count: int


def build_template_candidates(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    excluded_filenames: Iterable[str],
    detector: NodeDetector | None = None,
) -> TemplateBuildResult:
    """Build a new private run without reading excluded holdout images."""
    source = Path(source_dir)
    output = Path(output_dir)
    if not source.is_dir():
        raise TemplateCandidateError(f"template source directory does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise TemplateCandidateError(f"template output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    excluded = {Path(name).name.casefold() for name in excluded_filenames}
    if not excluded:
        raise TemplateCandidateError("at least one holdout filename must be excluded")

    all_dir = output / "all_candidates"
    selected_dir = output / "selected_candidates"
    all_dir.mkdir()
    selected_dir.mkdir()
    engine = detector or NodeDetector()

    paths = sorted(
        (item for item in source.iterdir() if item.is_file()),
        key=lambda item: item.name.casefold(),
    )
    sources: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    first_source_by_hash: dict[str, str] = {}
    candidate_images: dict[str, np.ndarray] = {}

    for path in paths:
        if path.name.casefold() in excluded:
            sources.append(
                {
                    "private_filename": path.name,
                    "role": "holdout_excluded_unread",
                    "processed": False,
                }
            )
            continue
        scene = infer_scene_label(path.name)
        source_record: dict[str, object] = {
            "private_filename": path.name,
            "scene_label": scene,
            "role": "template_source",
            "processed": False,
        }
        sources.append(source_record)
        if path.suffix.lower() != ".png":
            source_record["skip_reason"] = "not_lossless_png"
            continue
        try:
            image, _ = load_png(path)
        except VisionError as exc:
            source_record["skip_reason"] = f"invalid_image: {exc}"
            continue
        height, width = image.shape[:2]
        digest = sha256_file(path)
        source_id = f"source_{digest[:12]}"
        duplicate_of = first_source_by_hash.get(digest)
        first_source_by_hash.setdefault(digest, source_id)
        source_record.update(
            image_format="PNG",
            width=width,
            height=height,
            file_size=path.stat().st_size,
            source_id=source_id,
            sha256=digest,
            duplicate_of_source_id=duplicate_of,
        )
        if (width, height) != (1280, 720):
            source_record["skip_reason"] = "source_is_not_1280x720"
            continue
        if not is_detection_candidate(path.name, scene):
            source_record["skip_reason"] = "not_map_overview"
            continue
        source_record["processed"] = True
        try:
            result = engine.detect(image)
        except (VisionError, cv2.error) as exc:
            source_record.update(processed=False, skip_reason=f"detection_failed: {exc}")
            continue
        if result.analysis.get("grid_fit_status") != "ok":
            source_record.update(processed=False, skip_reason="grid_fit_failed")
            continue
        spacing = result.analysis.get("grid_spacing")
        if not isinstance(spacing, dict):
            source_record.update(processed=False, skip_reason="missing_grid_spacing")
            continue
        minimum_spacing = min(float(spacing["sx"]), float(spacing["sy"]))
        scale_group = classify_scale_group(minimum_spacing, height)
        source_record.update(
            scale_group=scale_group,
            grid_spacing={"sx": float(spacing["sx"]), "sy": float(spacing["sy"])},
            detected_node_count=len(result.nodes),
        )
        if scale_group == "zoomed_in_unsupported":
            source_record.update(processed=False, skip_reason="zoomed_in_not_in_run03_scope")
            continue

        for node in result.nodes:
            if (
                node.category not in _ALLOWED_CLASSES
                or not node.reliable
                or node.confidence < _CONFIDENCE_FLOORS[node.category]
            ):
                continue
            crop_box = candidate_crop_box(node, minimum_spacing, width, height)
            x, y, crop_width, crop_height = crop_box
            crop = image[y : y + crop_height, x : x + crop_width].copy()
            if crop.size == 0:
                continue
            candidate_id = f"candidate_{len(candidates) + 1:04d}"
            risks = contamination_risks(
                image,
                crop,
                crop_box,
                node,
                right_panel=bool(result.analysis.get("right_panel_detected")),
            )
            relative = Path(scale_group) / node.category / f"{candidate_id}.png"
            target = all_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            encode_png(crop).tofile(target)
            phash = perceptual_hash(crop)
            record = {
                "candidate_id": candidate_id,
                "private_source_id": source_id,
                "private_source_filename": path.name,
                "source_node_id": node.id,
                "bbox": {
                    "x": x,
                    "y": y,
                    "width": crop_width,
                    "height": crop_height,
                },
                "grid_center": {"x": node.grid_center.x, "y": node.grid_center.y},
                "scale_group": scale_group,
                "visual_class": node.category,
                "detector_confidence": float(node.confidence),
                "contamination_risks": risks,
                "perceptual_hash": f"{phash:016x}",
                "candidate_path": relative.as_posix(),
                "duplicate_group": None,
                "recommended": False,
            }
            candidates.append(record)
            candidate_images[candidate_id] = crop

    _assign_duplicate_groups(candidates)
    recommended = _choose_recommended(candidates)
    for record in candidates:
        if record["candidate_id"] not in recommended:
            continue
        record["recommended"] = True
        relative = Path(str(record["candidate_path"]))
        target = selected_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(all_dir / relative, target)

    source_manifest = {
        "schema_version": "maablackflow.private_template_sources.v1",
        "source_directory_name": source.name,
        "holdout_count": sum(item.get("role") == "holdout_excluded_unread" for item in sources),
        "source_count": sum(item.get("role") == "template_source" for item in sources),
        "sources": sources,
    }
    candidate_manifest = {
        "schema_version": "maablackflow.private_template_candidates.v1",
        "solver_ready": False,
        "template_match_executed": False,
        "selection_status": "awaiting_manual_review",
        "candidate_count": len(candidates),
        "recommended_count": len(recommended),
        "candidates": candidates,
    }
    source_manifest_path = output / "source_manifest.json"
    candidate_manifest_path = output / "candidate_manifest.json"
    _write_json(source_manifest_path, source_manifest)
    _write_json(candidate_manifest_path, candidate_manifest)

    summary = _manifest_summary(source_manifest, candidate_manifest)
    summary_path = output / "manifest_summary.json"
    _write_json(summary_path, summary)
    contact_sheet_path = output / "contact_sheet.png"
    _write_contact_sheet(contact_sheet_path, candidates, candidate_images)
    pipeline_draft_path = output / "template_match_pipeline_draft.json"
    _write_pipeline_draft(pipeline_draft_path, candidates)
    _write_json(
        output / "pipeline_draft_notes.json",
        {
            "status": "not_executed",
            "solver_ready": False,
            "template_root_when_manually_staged": "resource/image/run03",
            "roi_basis": "1280x720 offline map viewport; review panel variants before use",
            "warning": (
                "private candidates are not approved templates and must not be "
                "copied to public resources"
            ),
        },
    )

    private_resource = output / "private_resource_preview" / "resource"
    (private_resource / "pipeline").mkdir(parents=True)
    shutil.copy2(
        pipeline_draft_path,
        private_resource / "pipeline" / "run03_template_draft.json",
    )
    if any(selected_dir.rglob("*.png")):
        shutil.copytree(
            selected_dir,
            private_resource / "image" / "run03" / "selected_candidates",
        )

    review = output / "review_bundle"
    review.mkdir()
    shutil.copy2(contact_sheet_path, review / contact_sheet_path.name)
    shutil.copy2(summary_path, review / summary_path.name)
    if any(selected_dir.rglob("*.png")):
        shutil.copytree(selected_dir, review / "selected_candidates")

    return TemplateBuildResult(
        output,
        source_manifest_path,
        candidate_manifest_path,
        summary_path,
        contact_sheet_path,
        pipeline_draft_path,
        review,
        len(candidates),
        len(recommended),
    )


def classify_scale_group(grid_spacing: float, image_height: int) -> str:
    ratio = grid_spacing / image_height
    if ratio <= 0.18:
        return "zoomed_out"
    if ratio <= 0.28:
        return "normal"
    return "zoomed_in_unsupported"


def candidate_crop_box(node, spacing: float, width: int, height: int) -> tuple[int, int, int, int]:
    half = max(12, round(spacing * 0.24))
    left = node.grid_center.x - half
    top = node.grid_center.y - half
    right = node.grid_center.x + half + 1
    bottom = node.grid_center.y + half + 1
    if node.category == "current_position" and node.marker_bbox is not None:
        margin = max(4, round(spacing * 0.06))
        left = min(left, node.marker_bbox.x - margin)
        top = min(top, node.marker_bbox.y - margin)
        right = max(right, node.marker_bbox.x + node.marker_bbox.width + margin)
        bottom = max(bottom, node.marker_bbox.y + node.marker_bbox.height + margin)
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    return left, top, right - left, bottom - top


def contamination_risks(image, crop, box, node, *, right_panel: bool) -> dict[str, bool]:
    x, y, width, height = box
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    white = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 190)).astype(np.uint8) * 255
    count, _, stats, _ = cv2.connectedComponentsWithStats(white)
    text_like = 0
    for index in range(1, count):
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if 3 <= area <= max(80, crop.size // 120) and (
            component_width >= 2.2 * max(component_height, 1)
            or component_height >= 2.8 * max(component_width, 1)
        ):
            text_like += 1
    bright_fraction = float(np.mean(hsv[:, :, 2] >= 240))
    marker = node.marker_center
    person_overlap = node.category == "current_position" or (
        marker is not None and x <= marker.x < x + width and y <= marker.y < y + height
    )
    return {
        "person": bool(person_overlap),
        "text": text_like >= 2,
        "panel": bool(right_panel and x + width >= image.shape[1] * 0.58),
        "highlight": bright_fraction >= 0.18 or float(np.mean(gray)) >= 190,
        "clipped": x == 0 or y == 0 or x + width == image.shape[1] or y + height == image.shape[0],
    }


def perceptual_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8]
    median = float(np.median(coefficients.flatten()[1:]))
    bits = (coefficients >= median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _assign_duplicate_groups(candidates: list[dict[str, object]]) -> None:
    groups: list[list[dict[str, object]]] = []
    for candidate in candidates:
        match = None
        value = int(str(candidate["perceptual_hash"]), 16)
        for group in groups:
            exemplar = group[0]
            if (
                exemplar["scale_group"] == candidate["scale_group"]
                and exemplar["visual_class"] == candidate["visual_class"]
                and (value ^ int(str(exemplar["perceptual_hash"]), 16)).bit_count() <= 8
            ):
                match = group
                break
        if match is None:
            groups.append([candidate])
        else:
            match.append(candidate)
    for index, group in enumerate(groups, 1):
        group_id = f"duplicate_group_{index:04d}"
        for candidate in group:
            candidate["duplicate_group"] = group_id
            candidate["near_duplicate"] = len(group) > 1


def _choose_recommended(candidates: list[dict[str, object]]) -> set[str]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["duplicate_group"])].append(candidate)
    winners: list[dict[str, object]] = []
    for group in grouped.values():
        winners.append(
            min(
                group,
                key=lambda item: (
                    _risk_rank(item),
                    -float(item["detector_confidence"]),
                    str(item["candidate_id"]),
                ),
            )
        )
    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for winner in winners:
        buckets[(str(winner["scale_group"]), str(winner["visual_class"]))].append(winner)
    selected: set[str] = set()
    for values in buckets.values():
        ordered = sorted(
            values,
            key=lambda item: (
                _risk_rank(item),
                -float(item["detector_confidence"]),
                str(item["candidate_id"]),
            ),
        )
        selected.update(str(item["candidate_id"]) for item in ordered[:6])
    return selected


def _risk_rank(candidate: dict[str, object]) -> tuple[int, int]:
    risks = candidate["contamination_risks"]
    severe = sum(bool(risks[name]) for name in ("panel", "highlight", "clipped"))
    secondary = sum(bool(risks[name]) for name in ("text",))
    return severe, secondary


def _manifest_summary(source_manifest, candidate_manifest) -> dict[str, object]:
    candidates = candidate_manifest["candidates"]
    class_counts = Counter(item["visual_class"] for item in candidates)
    scale_counts = Counter(item["scale_group"] for item in candidates)
    risk_counts = Counter(
        name
        for item in candidates
        for name, value in item["contamination_risks"].items()
        if value
    )
    duplicate_counts = Counter(item["duplicate_group"] for item in candidates)
    return {
        "schema_version": "maablackflow.private_template_summary.v1",
        "solver_ready": False,
        "template_match_executed": False,
        "holdout_count": source_manifest["holdout_count"],
        "template_source_count": source_manifest["source_count"],
        "duplicate_source_count": sum(
            item.get("duplicate_of_source_id") is not None
            for item in source_manifest["sources"]
        ),
        "candidate_count": candidate_manifest["candidate_count"],
        "recommended_count": candidate_manifest["recommended_count"],
        "class_counts": dict(sorted(class_counts.items())),
        "scale_counts": dict(sorted(scale_counts.items())),
        "risk_counts": dict(sorted(risk_counts.items())),
        "near_duplicate_candidate_count": sum(
            count for count in duplicate_counts.values() if count > 1
        ),
        "near_duplicate_group_count": sum(count > 1 for count in duplicate_counts.values()),
        "recommended": [
            {
                "candidate_id": item["candidate_id"],
                "private_source_id": item["private_source_id"],
                "scale_group": item["scale_group"],
                "visual_class": item["visual_class"],
                "contamination_risks": item["contamination_risks"],
            }
            for item in candidates
            if item["recommended"]
        ],
    }


def _write_contact_sheet(path, candidates, images) -> None:
    columns, cell_width, cell_height = 5, 210, 175
    rows = max(1, (len(candidates) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#202328")
    draw = ImageDraw.Draw(sheet)
    for index, candidate in enumerate(candidates):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        rgb = cv2.cvtColor(images[str(candidate["candidate_id"])], cv2.COLOR_BGR2RGB)
        thumb = Image.fromarray(rgb)
        thumb.thumbnail((190, 115), Image.Resampling.LANCZOS)
        thumb = ImageOps.pad(thumb, (190, 115), color="#101214")
        sheet.paste(thumb, (x + 10, y + 6))
        risks = [name for name, value in candidate["contamination_risks"].items() if value]
        draw.text((x + 10, y + 124), str(candidate["candidate_id"]), fill="white")
        draw.text(
            (x + 10, y + 139),
            f"{candidate['visual_class']} | {candidate['scale_group']}",
            fill="#9dd7ff",
        )
        draw.text(
            (x + 10, y + 154),
            "risk: " + (",".join(risks) if risks else "none"),
            fill="#ffbd70" if risks else "#8ee59b",
        )
    sheet.save(path, format="PNG")


def _write_pipeline_draft(path: Path, candidates: list[dict[str, object]]) -> None:
    nodes: dict[str, object] = {}
    for scale_group in ("normal", "zoomed_out"):
        for current_only, suffix in ((False, "Nodes"), (True, "CurrentPosition")):
            selected = [
                item
                for item in candidates
                if item["recommended"]
                and item["scale_group"] == scale_group
                and (item["visual_class"] == "current_position") == current_only
            ]
            if not selected:
                continue
            templates = [
                "run03/selected_candidates/" + str(item["candidate_path"])
                for item in selected
            ]
            node_name = f"MaaBlackFlowPrivate{scale_group.title().replace('_', '')}{suffix}Draft"
            nodes[node_name] = {
                "recognition": {
                    "type": "TemplateMatch",
                    "param": {
                        "roi": [19, 40, 1095, 579],
                        "template": templates,
                        "threshold": [0.82] * len(templates),
                        "order_by": "Score",
                        "index": 0,
                        "method": 5,
                        "green_mask": False,
                    },
                },
                "action": {"type": "DoNothing", "param": {}},
                "next": [],
            }
    _write_json(path, nodes)


def _write_json(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
