from __future__ import annotations

import hashlib
import json
import shutil

import pytest
from PIL import Image

from maablackflow.vision import (
    DatasetInspectionError,
    infer_scene_label,
    inspect_dataset,
    sha256_file,
)


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    source = tmp_path / "sample.bin"
    source.write_bytes(b"maablackflow-vision")
    expected = hashlib.sha256(b"maablackflow-vision").hexdigest()
    assert sha256_file(source) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("普通地图.png", "地图正常状态"),
        ("第二层缩小.png", "地图缩小状态"),
        ("第二层放大.png", "地图放大状态"),
        ("打开移动工具.png", "移动工具展开状态"),
        ("移动过后.png", "移动后状态"),
        ("事件内部.png", "节点内部界面"),
        ("第三层起点.png", "起点或终点相关界面"),
        ("无法识别.png", "暂时无法判断"),
    ],
)
def test_scene_label_inference(filename, expected) -> None:
    assert infer_scene_label(filename) == expected


def test_inventory_handles_chinese_names_and_duplicates(tmp_path) -> None:
    source = tmp_path / "截图"
    output = tmp_path / "private-output"
    source.mkdir()
    first = source / "地图正常状态.png"
    Image.new("RGB", (96, 64), (20, 40, 60)).save(first)
    duplicate = source / "地图正常状态副本.png"
    shutil.copyfile(first, duplicate)
    Image.new("RGBA", (80, 48), (10, 20, 30, 255)).save(source / "事件内部.png")

    result = inspect_dataset(source, output)

    assert len(result.entries) == 3
    assert result.manifest_path.exists()
    assert result.contact_sheet_path.exists()
    duplicate_entries = [entry for entry in result.entries if entry.duplicate_of]
    assert len(duplicate_entries) == 1
    assert duplicate_entries[0].duplicate_of == "地图正常状态.png"
    rgba = next(entry for entry in result.entries if entry.filename == "事件内部.png")
    assert rgba.channels == 4
    assert not rgba.suitable_for_node_detection
    document = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert document["source_name"] == "截图"
    assert document["duplicate_count"] == 1
    assert all("absolute" not in key for key in document)


def test_inventory_rejects_missing_or_empty_directory(tmp_path) -> None:
    with pytest.raises(DatasetInspectionError, match="does not exist"):
        inspect_dataset(tmp_path / "missing", tmp_path / "out")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetInspectionError, match="no supported images"):
        inspect_dataset(empty, tmp_path / "out")