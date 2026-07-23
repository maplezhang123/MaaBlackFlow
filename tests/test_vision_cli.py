from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image

from maablackflow.cli import main


def write_synthetic_map(path) -> None:
    image = np.full((380, 520, 3), (12, 22, 16), dtype=np.uint8)
    for y in (80, 180, 280):
        cv2.line(image, (100, y), (420, y), (150, 150, 150), 5, cv2.LINE_AA)
    for x in (100, 260, 420):
        cv2.line(image, (x, 80), (x, 280), (150, 150, 150), 5, cv2.LINE_AA)
        for y in (80, 180, 280):
            radius = 25 if (x, y) in {(100, 80), (420, 280)} else 10
            cv2.circle(image, (x, y), radius, (80, 220, 220), 4, cv2.LINE_AA)
            cv2.circle(image, (x, y), max(3, radius // 3), (230, 230, 230), 2, cv2.LINE_AA)
    success, encoded = cv2.imencode(".png", image)
    assert success
    encoded.tofile(path)


def test_inspect_dataset_cli_success(tmp_path, capsys) -> None:
    source = tmp_path / "截图"
    source.mkdir()
    Image.new("RGB", (100, 60), (10, 20, 30)).save(source / "普通地图.png")
    output = tmp_path / "private"
    code = main(["inspect-dataset", str(source), "--output", str(output)])
    captured = capsys.readouterr()
    assert code == 0
    assert "图片数量: 1" in captured.out
    assert (output / "dataset_manifest.json").exists()
    assert (output / "contact_sheet.png").exists()


def test_detect_nodes_cli_success_and_failure(tmp_path, capsys) -> None:
    source = tmp_path / "中文地图.png"
    write_synthetic_map(source)
    output = tmp_path / "detections"
    code = main(["detect-nodes", str(source), "--output", str(output)])
    captured = capsys.readouterr()
    assert code == 0
    assert "原始分辨率: 520x380" in captured.out
    assert "检测到的节点数量:" in captured.out
    assert (output / "中文地图.nodes.json").exists()
    assert (output / "中文地图.annotated.png").exists()
    assert (output / "中文地图.grid-debug.png").exists()
    assert "道路/网格调试图路径:" in captured.out

    failed_output = tmp_path / "failed"
    code = main(
        [
            "detect-nodes",
            str(tmp_path / "不存在.png"),
            "--output",
            str(failed_output),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "节点检测失败:" in captured.err
    assert not failed_output.exists()

def test_evaluate_detection_cli_success_and_failure(tmp_path, capsys) -> None:
    ground_truth = tmp_path / "ground_truth"
    predictions = tmp_path / "predictions"
    ground_truth.mkdir()
    predictions.mkdir()
    truth = {
        "prediction_stem": "合成评估",
        "grid_points": [{"x": 100, "y": 100}],
        "current_position": {
            "grid_center": {"x": 100, "y": 100},
            "marker_center": {"x": 80, "y": 90},
        },
    }
    prediction = {
        "analysis": {"grid_spacing": {"sx": 100, "sy": 100}},
        "nodes": [
            {
                "grid_center": {"x": 100, "y": 100},
                "category": "current_position",
                "grid_row": 0,
                "grid_col": 0,
                "marker_center": {"x": 80, "y": 90},
            }
        ],
    }
    (ground_truth / "合成评估.json").write_text(
        json.dumps(truth, ensure_ascii=False), encoding="utf-8"
    )
    (predictions / "合成评估.nodes.json").write_text(
        json.dumps(prediction, ensure_ascii=False), encoding="utf-8"
    )
    code = main(
        [
            "evaluate-detection",
            "--ground-truth",
            str(ground_truth),
            "--predictions",
            str(predictions),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "TP=1 FP=0 FN=0" in captured.out
    assert "current_grid_correct=True" in captured.out

    code = main(
        [
            "evaluate-detection",
            "--ground-truth",
            str(tmp_path / "missing"),
            "--predictions",
            str(predictions),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "检测评估失败:" in captured.err