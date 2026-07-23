from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

from maablackflow import cli
from maablackflow.integrations.maafw.adapter import (
    MaaAdapterError,
    MaaRecognitionPayload,
    MapRecognitionAdapter,
    image_from_maa,
)
from maablackflow.integrations.maafw.agent import (
    OPTIONAL_RUNTIME_ERROR,
    analyze_argv,
    main as agent_main,
    register_custom_recognition,
)
from maablackflow.integrations.maafw.serialization import build_detail, stable_json
from maablackflow.integrations.maafw.template_evidence import (
    MaaTemplateEvidenceProvider,
)
from maablackflow.vision.models import (
    BoundingBox,
    DetectedNode,
    DetectionResult,
    Point,
)


def detection_result() -> DetectionResult:
    current = DetectedNode(
        id="node_02",
        center=Point(300, 200),
        grid_center=Point(300, 200),
        bbox=BoundingBox(280, 180, 41, 41),
        confidence=0.91,
        category="current_position",
        reliable=True,
        sources=("white_person_component", "grid_road_intersection"),
        grid_row=1,
        grid_col=2,
        marker_center=Point(276, 188),
        marker_bbox=BoundingBox(267, 170, 19, 37),
    )
    event = DetectedNode(
        id="node_01",
        center=Point(100, 100),
        grid_center=Point(100, 100),
        bbox=BoundingBox(80, 80, 41, 41),
        confidence=0.82,
        category="event_node",
        reliable=True,
        sources=("hough_circle",),
        grid_row=0,
        grid_col=0,
    )
    return DetectionResult(
        image_width=640,
        image_height=360,
        nodes=(event, current),
        analysis={
            "grid_fit_status": "ok",
            "grid_spacing": {"sx": 100.0, "sy": 100.0},
            "grid_rows": [100, 200],
            "grid_columns": [100, 200, 300],
            "warning": "offline baseline only",
        },
    )


class StubDetector:
    def detect(self, image):
        assert image.shape == (360, 640, 3)
        return detection_result()


def test_detail_serialization_is_stable_safe_and_solver_not_ready() -> None:
    detail = build_detail(detection_result())
    first = stable_json(detail)
    second = stable_json(build_detail(detection_result()))
    assert first == second
    assert detail["solver_ready"] is False
    assert detail["map_roi"] == {"x": 50, "y": 50, "width": 300, "height": 200}
    assert "D:\\rouge" not in first
    assert "Screenshots" not in first
    reversed_result = replace(
        detection_result(), nodes=tuple(reversed(detection_result().nodes))
    )
    assert stable_json(build_detail(reversed_result)) == first


def test_current_marker_and_grid_centers_remain_separate() -> None:
    current = build_detail(detection_result())["current_position"]
    assert current["center"] == current["grid_center"] == {"x": 300, "y": 200}
    assert current["marker_center"] == {"x": 276, "y": 188}


def test_maa_numpy_image_conversion_and_validation() -> None:
    backing = np.zeros((360, 1280, 3), dtype=np.uint8)
    non_contiguous = backing[:, ::2, :]
    converted = image_from_maa(non_contiguous)
    assert converted.shape == (360, 640, 3)
    assert converted.flags.c_contiguous
    with pytest.raises(MaaAdapterError, match="numpy.ndarray"):
        image_from_maa(SimpleNamespace(array=converted))
    with pytest.raises(MaaAdapterError, match="uint8"):
        image_from_maa(converted.astype(np.float32))


def test_fake_custom_recognition_arguments_use_pure_adapter() -> None:
    argv = SimpleNamespace(
        image=np.zeros((360, 640, 3), dtype=np.uint8),
        custom_recognition_param=json.dumps(
            {"recognition_mode": "grid_baseline", "require_solver_ready": False}
        ),
    )
    payload = analyze_argv(argv, MapRecognitionAdapter(StubDetector()))
    assert payload.success
    assert payload.box == (280, 180, 41, 41)
    assert payload.detail["solver_ready"] is False


def test_core_import_survives_without_maafw_and_agent_error_is_clear(
    monkeypatch, capsys
) -> None:
    import maablackflow

    assert maablackflow is not None
    monkeypatch.setitem(sys.modules, "maa", None)
    monkeypatch.delitem(sys.modules, "maa.agent", raising=False)
    monkeypatch.delitem(sys.modules, "maa.agent.agent_server", raising=False)
    code = agent_main(["fake-socket"])
    captured = capsys.readouterr()
    assert code == 2
    assert OPTIONAL_RUNTIME_ERROR in captured.err


def test_fake_sdk_registration_returns_official_analyze_result(monkeypatch) -> None:
    captured = {}

    @dataclass
    class AnalyzeResult:
        box: object
        detail: object

    class CustomRecognition:
        pass

    CustomRecognition.AnalyzeResult = AnalyzeResult

    class AgentServer:
        @staticmethod
        def custom_recognition(name):
            def decorator(cls):
                captured["name"] = name
                captured["class"] = cls
                return cls
            return decorator

    maa = ModuleType("maa")
    agent_package = ModuleType("maa.agent")
    agent_server = ModuleType("maa.agent.agent_server")
    custom_recognition = ModuleType("maa.custom_recognition")
    agent_server.AgentServer = AgentServer
    custom_recognition.CustomRecognition = CustomRecognition
    monkeypatch.setitem(sys.modules, "maa", maa)
    monkeypatch.setitem(sys.modules, "maa.agent", agent_package)
    monkeypatch.setitem(sys.modules, "maa.agent.agent_server", agent_server)
    monkeypatch.setitem(sys.modules, "maa.custom_recognition", custom_recognition)

    server, recognizer = register_custom_recognition()
    assert server is AgentServer
    assert captured["name"] == "MaaBlackFlow.MapRecognize"
    assert captured["class"] is recognizer
    monkeypatch.setattr(
        "maablackflow.integrations.maafw.agent.MapRecognitionAdapter",
        lambda: MapRecognitionAdapter(StubDetector()),
    )
    argv = SimpleNamespace(
        image=np.zeros((360, 640, 3), dtype=np.uint8),
        custom_recognition_param="{}",
    )
    sdk_result = recognizer().analyze(object(), argv)
    assert isinstance(sdk_result, AnalyzeResult)
    assert sdk_result.box == (280, 180, 41, 41)
    assert sdk_result.detail["solver_ready"] is False


def test_fake_template_match_hit_becomes_candidate_evidence() -> None:
    evidence = MaaTemplateEvidenceProvider().from_hits(
        [{"template": "private_future", "box": [10, 20, 40, 30], "score": 0.88}]
    )
    assert len(evidence) == 1
    assert (evidence[0].x, evidence[0].y) == (30, 35)
    assert evidence[0].bbox == (10, 20, 40, 30)
    assert evidence[0].source == "maafw_template:private_future"
    assert evidence[0].scores == {"template": 0.88}


def test_pipeline_v2_is_custom_recognition_only_and_safe() -> None:
    pipeline = json.loads(Path("maafw_project/resource/pipeline/blackflow.json").read_text("utf-8"))
    node = pipeline["MaaBlackFlowMapRecognize"]
    assert node["recognition"]["type"] == "Custom"
    params = node["recognition"]["param"]
    assert params["custom_recognition"] == "MaaBlackFlow.MapRecognize"
    assert params["custom_recognition_param"]["require_solver_ready"] is False
    assert node["action"] == {"type": "DoNothing", "param": {}}
    assert node["next"] == []
    serialized = json.dumps(pipeline)
    for dangerous in ("Click", "Swipe", "Shell", "Command"):
        assert dangerous not in serialized


def test_project_interface_has_agent_but_no_controller() -> None:
    interface = json.loads(Path("maafw_project/interface.json").read_text("utf-8"))
    assert interface["interface_version"] == 2
    assert interface["agent"]["child_exec"] == "python"
    assert interface["controller"] == []


def test_adapter_smoke_cli_success_and_failure(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "中文 Maa 输入.png"
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(source)

    payload = MaaRecognitionPayload(
        success=True,
        box=(280, 180, 41, 41),
        detail=build_detail(detection_result()),
    )

    class StubAdapter:
        def analyze(self, image, parameters):
            assert parameters["require_solver_ready"] is False
            return payload

    monkeypatch.setattr(cli, "MapRecognitionAdapter", StubAdapter)
    output = tmp_path / "private-output"
    code = cli.main(["maa-adapter-smoke", str(source), "--output", str(output)])
    captured = capsys.readouterr()
    assert code == 0
    detail_path = output / "中文 Maa 输入.maa-detail.json"
    assert detail_path.exists()
    detail_text = detail_path.read_text("utf-8")
    assert str(source.resolve()) not in detail_text
    assert json.loads(detail_text)["solver_ready"] is False
    assert "solver_ready: false" in captured.out

    failed = tmp_path / "failed"
    code = cli.main(
        ["maa-adapter-smoke", str(tmp_path / "missing.png"), "--output", str(failed)]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "Maa adapter smoke 失败" in captured.err
    assert not failed.exists()
