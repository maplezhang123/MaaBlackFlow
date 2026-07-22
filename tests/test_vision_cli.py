from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from maablackflow.cli import main


def write_synthetic_map(path) -> None:
    image = np.full((300, 520, 3), (12, 22, 16), dtype=np.uint8)
    cv2.line(image, (100, 110), (420, 110), (150, 150, 150), 5, cv2.LINE_AA)
    for center, radius in (((100, 110), 25), ((260, 110), 10), ((420, 110), 25)):
        cv2.circle(image, center, radius, (80, 220, 220), 4, cv2.LINE_AA)
        cv2.circle(image, center, max(3, radius // 3), (230, 230, 230), 2, cv2.LINE_AA)
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
    assert "原始分辨率: 520x300" in captured.out
    assert "检测到的节点数量:" in captured.out
    assert (output / "中文地图.nodes.json").exists()
    assert (output / "中文地图.annotated.png").exists()

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