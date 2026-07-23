from __future__ import annotations

import json

import pytest

from maablackflow.vision import EvaluationError, evaluate_directories
from maablackflow.vision.evaluation import evaluate_documents


def prediction_node(
    x: int,
    y: int,
    category: str,
    row: int,
    column: int,
    marker: tuple[int, int] | None = None,
) -> dict[str, object]:
    return {
        "id": f"node_{row}_{column}",
        "center": {"x": x, "y": y},
        "grid_center": {"x": x, "y": y},
        "category": category,
        "grid_row": row,
        "grid_col": column,
        "marker_center": (
            {"x": marker[0], "y": marker[1]} if marker is not None else None
        ),
    }


def test_evaluation_uses_one_to_one_matching_and_counts_uncertain_as_fp() -> None:
    truth = {
        "prediction_stem": "synthetic",
        "tolerance_ratio": 0.30,
        "grid_points": [
            {"x": 0, "y": 0},
            {"x": 100, "y": 0},
            {"x": 200, "y": 0},
        ],
        "current_position": {
            "grid_center": {"x": 100, "y": 0},
            "marker_center": {"x": 88, "y": 0},
        },
    }
    prediction = {
        "analysis": {"grid_spacing": {"sx": 100, "sy": 100}},
        "nodes": [
            prediction_node(2, 0, "event_node", 0, 0),
            prediction_node(4, 0, "event_node", 0, 0),
            prediction_node(101, 0, "current_position", 0, 1, (90, 0)),
            prediction_node(200, 0, "uncertain", 0, 2),
        ],
    }
    result = evaluate_documents("synthetic", truth, prediction)
    assert (result.true_positive, result.false_positive, result.false_negative) == (
        2,
        2,
        1,
    )
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(4 / 7)
    assert result.duplicate_grid_points == 1
    assert result.current_marker_error == pytest.approx(2.0)
    assert result.current_grid_cell_correct
    assert result.mean_center_error == pytest.approx(1.5)
    assert result.max_center_error == pytest.approx(2.0)


def test_evaluate_directories_aggregates_images_and_validates_inputs(tmp_path) -> None:
    ground_truth = tmp_path / "ground_truth"
    predictions = tmp_path / "predictions"
    ground_truth.mkdir()
    predictions.mkdir()
    truth = {
        "prediction_stem": "合成地图",
        "grid_points": [{"x": 50, "y": 50}],
        "current_position": {
            "grid_center": {"x": 50, "y": 50},
            "marker_center": {"x": 40, "y": 45},
        },
    }
    prediction = {
        "analysis": {"grid_spacing": {"sx": 80, "sy": 80}},
        "nodes": [
            prediction_node(51, 50, "current_position", 0, 0, (41, 45))
        ],
    }
    (ground_truth / "合成地图.json").write_text(
        json.dumps(truth, ensure_ascii=False), encoding="utf-8"
    )
    (predictions / "合成地图.nodes.json").write_text(
        json.dumps(prediction, ensure_ascii=False), encoding="utf-8"
    )
    report = evaluate_directories(ground_truth, predictions)
    assert report.total.true_positive == 1
    assert report.total.false_positive == 0
    assert report.total.false_negative == 0
    assert report.total.f1 == pytest.approx(1.0)
    assert report.total.current_marker_error == pytest.approx(1.0)

    with pytest.raises(EvaluationError, match="does not exist"):
        evaluate_directories(tmp_path / "missing", predictions)