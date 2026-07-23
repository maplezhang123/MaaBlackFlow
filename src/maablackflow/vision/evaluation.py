"""Private ground-truth evaluation for offline node detections."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


class EvaluationError(ValueError):
    """Raised for invalid ground truth or prediction documents."""


@dataclass(frozen=True, slots=True)
class ImageEvaluation:
    name: str
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    mean_center_error: float | None
    max_center_error: float | None
    current_marker_error: float | None
    current_grid_cell_correct: bool
    duplicate_grid_points: int
    tolerance: float
    matched_errors: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    images: tuple[ImageEvaluation, ...]
    total: ImageEvaluation


def evaluate_directories(
    ground_truth_dir: str | Path, predictions_dir: str | Path
) -> EvaluationReport:
    ground_truth_path = Path(ground_truth_dir)
    predictions_path = Path(predictions_dir)
    if not ground_truth_path.is_dir():
        raise EvaluationError(f"ground-truth directory does not exist: {ground_truth_path}")
    if not predictions_path.is_dir():
        raise EvaluationError(f"predictions directory does not exist: {predictions_path}")
    ground_truth_files = sorted(ground_truth_path.glob("*.json"))
    if not ground_truth_files:
        raise EvaluationError("ground-truth directory contains no JSON files")

    results: list[ImageEvaluation] = []
    for ground_truth_file in ground_truth_files:
        truth = _load_json(ground_truth_file, "ground truth")
        stem = _required_string(truth, "prediction_stem", ground_truth_file)
        prediction_file = predictions_path / f"{stem}.nodes.json"
        prediction = _load_json(prediction_file, "prediction")
        results.append(evaluate_documents(stem, truth, prediction))
    return EvaluationReport(tuple(results), _aggregate(results))


def evaluate_documents(
    name: str, ground_truth: dict[str, object], prediction: dict[str, object]
) -> ImageEvaluation:
    truth_points = _point_list(ground_truth.get("grid_points"), "grid_points")
    nodes = prediction.get("nodes")
    if not isinstance(nodes, list):
        raise EvaluationError("prediction nodes must be a list")
    prediction_points: list[tuple[float, float]] = []
    eligible_indices: list[int] = []
    cells: list[tuple[int, int]] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise EvaluationError("each prediction node must be an object")
        point = _point(node.get("grid_center") or node.get("center"), "grid_center")
        prediction_points.append(point)
        row, column = node.get("grid_row"), node.get("grid_col")
        if isinstance(row, int) and isinstance(column, int):
            cells.append((row, column))
        # uncertain remains a prediction (and therefore may be an FP), but can
        # never consume a ground-truth match.
        if node.get("category") != "uncertain":
            eligible_indices.append(index)

    spacing = prediction.get("analysis", {})
    if not isinstance(spacing, dict):
        raise EvaluationError("prediction analysis must be an object")
    grid_spacing = spacing.get("grid_spacing")
    if not isinstance(grid_spacing, dict):
        raise EvaluationError("prediction grid_spacing is missing")
    try:
        minimum_spacing = min(float(grid_spacing["sx"]), float(grid_spacing["sy"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("prediction grid_spacing must contain numeric sx and sy") from exc
    if minimum_spacing <= 0:
        raise EvaluationError("prediction grid spacing must be positive")
    tolerance_ratio = float(ground_truth.get("tolerance_ratio", 0.30))
    tolerance = minimum_spacing * tolerance_ratio

    eligible_points = [prediction_points[index] for index in eligible_indices]
    local_matches = _minimum_cost_maximum_matching(
        truth_points, eligible_points, tolerance
    )
    matches = [
        (truth_index, eligible_indices[prediction_index], distance)
        for truth_index, prediction_index, distance in local_matches
    ]
    errors = tuple(distance for _, _, distance in matches)
    true_positive = len(matches)
    false_positive = len(prediction_points) - true_positive
    false_negative = len(truth_points) - true_positive
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)

    current_truth = ground_truth.get("current_position")
    marker_error = None
    grid_correct = False
    if isinstance(current_truth, dict):
        truth_marker = _point(current_truth.get("marker_center"), "current marker_center")
        truth_grid = _point(current_truth.get("grid_center"), "current grid_center")
        predicted_current = [
            node
            for node in nodes
            if isinstance(node, dict) and node.get("category") == "current_position"
        ]
        if len(predicted_current) == 1:
            marker_value = predicted_current[0].get("marker_center")
            if marker_value is not None:
                marker_error = _distance(
                    _point(marker_value, "predicted marker_center"), truth_marker
                )
            predicted_grid = _point(
                predicted_current[0].get("grid_center")
                or predicted_current[0].get("center"),
                "predicted current grid_center",
            )
            grid_correct = _distance(predicted_grid, truth_grid) <= tolerance

    duplicate_count = len(cells) - len(set(cells))
    return ImageEvaluation(
        name=name,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_center_error=(sum(errors) / len(errors) if errors else None),
        max_center_error=(max(errors) if errors else None),
        current_marker_error=marker_error,
        current_grid_cell_correct=grid_correct,
        duplicate_grid_points=duplicate_count,
        tolerance=tolerance,
        matched_errors=errors,
    )


def _minimum_cost_maximum_matching(
    truth: list[tuple[float, float]],
    predictions: list[tuple[float, float]],
    tolerance: float,
) -> list[tuple[int, int, float]]:
    """One-to-one minimum-cost maximum matching using residual shortest paths."""
    truth_count, prediction_count = len(truth), len(predictions)
    source = 0
    truth_start = 1
    prediction_start = truth_start + truth_count
    sink = prediction_start + prediction_count
    graph: list[list[list[float | int]]] = [[] for _ in range(sink + 1)]
    references: list[tuple[int, int, list[float | int], float]] = []

    def add_edge(start: int, end: int, capacity: int, cost: float):
        forward: list[float | int] = [end, len(graph[end]), capacity, cost]
        reverse: list[float | int] = [start, len(graph[start]), 0, -cost]
        graph[start].append(forward)
        graph[end].append(reverse)
        return forward

    for truth_index in range(truth_count):
        add_edge(source, truth_start + truth_index, 1, 0.0)
    for prediction_index in range(prediction_count):
        add_edge(prediction_start + prediction_index, sink, 1, 0.0)
    for truth_index, truth_point in enumerate(truth):
        for prediction_index, prediction_point in enumerate(predictions):
            distance = _distance(truth_point, prediction_point)
            if distance <= tolerance:
                edge = add_edge(
                    truth_start + truth_index,
                    prediction_start + prediction_index,
                    1,
                    distance,
                )
                references.append((truth_index, prediction_index, edge, distance))

    while True:
        distances = [math.inf] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = 0.0
        for _ in range(len(graph) - 1):
            changed = False
            for start, edges in enumerate(graph):
                if math.isinf(distances[start]):
                    continue
                for edge_index, edge in enumerate(edges):
                    end, _, capacity, cost = edge
                    if int(capacity) <= 0:
                        continue
                    candidate = distances[start] + float(cost)
                    if candidate + 1e-9 < distances[int(end)]:
                        distances[int(end)] = candidate
                        previous[int(end)] = (start, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            start, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[start][edge_index]
            edge[2] = int(edge[2]) - 1
            reverse_index = int(edge[1])
            graph[node][reverse_index][2] = int(graph[node][reverse_index][2]) + 1
            node = start

    matches = [
        (truth_index, prediction_index, distance)
        for truth_index, prediction_index, edge, distance in references
        if int(edge[2]) == 0
    ]
    return sorted(matches)


def _aggregate(results: list[ImageEvaluation]) -> ImageEvaluation:
    true_positive = sum(item.true_positive for item in results)
    false_positive = sum(item.false_positive for item in results)
    false_negative = sum(item.false_negative for item in results)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    errors = tuple(error for item in results for error in item.matched_errors)
    marker_errors = [
        item.current_marker_error
        for item in results
        if item.current_marker_error is not None
    ]
    return ImageEvaluation(
        name="TOTAL",
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_center_error=(sum(errors) / len(errors) if errors else None),
        max_center_error=(max(errors) if errors else None),
        current_marker_error=(
            sum(marker_errors) / len(marker_errors) if marker_errors else None
        ),
        current_grid_cell_correct=all(
            item.current_grid_cell_correct for item in results
        ),
        duplicate_grid_points=sum(item.duplicate_grid_points for item in results),
        tolerance=0.0,
        matched_errors=errors,
    )


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"{label} file does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read {label} JSON '{path}': {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} JSON must contain an object: {path}")
    return value


def _required_string(document: dict[str, object], key: str, path: Path) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{path} must contain non-empty '{key}'")
    return value


def _point_list(value: object, label: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or not value:
        raise EvaluationError(f"{label} must be a non-empty list")
    return [_point(item, label) for item in value]


def _point(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be an object with x and y")
    try:
        x, y = float(value["x"]), float(value["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError(f"{label} must contain numeric x and y") from exc
    return x, y


def _distance(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0