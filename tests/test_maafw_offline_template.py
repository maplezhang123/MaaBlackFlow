from __future__ import annotations

import json
from importlib import metadata
import sys

import cv2
import numpy as np
import pytest

from maablackflow.integrations.maafw.offline_template import (
    OfflineTemplateSpec,
    execute_template_pipeline,
    filter_and_snap_hits,
    template_pipeline,
)
from maablackflow.integrations.maafw.offline_template import OfflineTemplateHit
from maablackflow.integrations.maafw.agent import (
    MaaFrameworkRuntimeError,
    OPTIONAL_RUNTIME_ERROR,
)


def _spec() -> OfflineTemplateSpec:
    return OfflineTemplateSpec(
        "synthetic_node",
        "approved/normal/event_node/synthetic_node.png",
        "event_node",
        "normal",
    )


def test_offline_template_pipeline_is_fixed_threshold_and_do_nothing() -> None:
    document = template_pipeline([_spec()])
    node = document["MaaBlackFlowOfflineTemplate001"]
    assert node["recognition"] == {
        "type": "TemplateMatch",
        "param": {
            "template": ["approved/normal/event_node/synthetic_node.png"],
            "threshold": [0.70],
            "order_by": "Score",
            "index": 0,
            "method": 5,
            "green_mask": False,
        },
    }
    assert node["action"] == {"type": "DoNothing", "param": {}}
    assert node["next"] == []
    serialized = json.dumps(document)
    for dangerous in ("Click", "Swipe", "Shell", "Command"):
        assert dangerous not in serialized


def test_real_maafw_static_controller_template_match(tmp_path) -> None:
    try:
        version = metadata.version("MaaFw")
    except metadata.PackageNotFoundError:
        pytest.skip("MaaFramework optional runtime is not installed")
    assert tuple(int(part) for part in version.split(".")) >= (5, 12, 2)

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(image, (150, 120), 22, (230, 230, 230), -1)
    cv2.line(image, (130, 100), (170, 140), (30, 30, 30), 3)
    template = image[90:151, 120:181].copy()

    resource = tmp_path / "resource"
    template_path = resource / "image" / _spec().resource_path
    template_path.parent.mkdir(parents=True)
    ok, encoded = cv2.imencode(".png", template)
    assert ok
    encoded.tofile(template_path)
    pipeline_dir = resource / "pipeline"
    pipeline_dir.mkdir()
    (pipeline_dir / "offline.json").write_text(
        json.dumps(template_pipeline([_spec()])), encoding="utf-8"
    )

    result = execute_template_pipeline(resource, image, [_spec()])
    assert result.controller_calls == ("connect",)
    assert len(result.hits) >= 1
    best = max(result.hits, key=lambda hit: hit.score)
    assert best.template_id == "synthetic_node"
    assert best.score >= 0.99
    assert best.box == (120, 90, 61, 61)


def test_template_hits_are_roi_masked_snapped_and_deduplicated() -> None:
    hits = [
        OfflineTemplateHit("low", (82, 82, 36, 36), 0.75, "event_node", "normal"),
        OfflineTemplateHit("high", (84, 84, 36, 36), 0.91, "event_node", "normal"),
        OfflineTemplateHit("ui", (280, 30, 40, 40), 0.99, "event_node", "normal"),
        OfflineTemplateHit("road", (140, 140, 20, 20), 0.98, "empty_waypoint", "normal"),
    ]
    snapped = filter_and_snap_hits(
        hits,
        grid_rows=(100, 200),
        grid_columns=(100, 200, 300),
        grid_spacing=(100.0, 100.0),
        map_roi=(20, 20, 340, 240),
        ui_rectangles=((260, 20, 340, 80),),
    )
    assert len(snapped) == 1
    assert snapped[0].template_id == "high"
    assert snapped[0].grid_center == (100, 100)
    assert (snapped[0].grid_row, snapped[0].grid_col) == (0, 0)


def test_offline_template_runtime_remains_optional(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "maa", None)
    for name in ("maa.controller", "maa.define", "maa.pipeline", "maa.resource", "maa.tasker"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(MaaFrameworkRuntimeError, match=OPTIONAL_RUNTIME_ERROR):
        execute_template_pipeline(
            tmp_path,
            np.zeros((100, 100, 3), dtype=np.uint8),
            [_spec()],
        )
