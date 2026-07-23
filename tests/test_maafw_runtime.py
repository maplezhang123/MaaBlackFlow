"""Optional real-runtime compatibility tests; skipped when MaaFw is absent."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def _require_maafw() -> str:
    try:
        return metadata.version("MaaFw")
    except metadata.PackageNotFoundError:
        pytest.skip("MaaFramework optional runtime is not installed")


def _run_isolated(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_real_maafw_registration_and_analyze_result_contract() -> None:
    version = tuple(int(part) for part in _require_maafw().split("."))
    assert (5, 12, 2) <= version < (6, 0, 0)
    assert metadata.version("MaaAgentBinary")
    process = _run_isolated(
        """
        import json
        import numpy as np
        from maa.agent.agent_server import AgentServer
        from maa.custom_recognition import CustomRecognition
        from maa.define import Rect
        from maablackflow.integrations.maafw import agent
        from maablackflow.integrations.maafw.adapter import (
            MAA_CUSTOM_RECOGNITION_NAME,
            MapRecognitionAdapter,
        )
        from maablackflow.vision.models import (
            BoundingBox, DetectedNode, DetectionResult, Point,
        )

        class StubDetector:
            def detect(self, image):
                node = DetectedNode(
                    id="node_01",
                    center=Point(300, 200),
                    grid_center=Point(300, 200),
                    bbox=BoundingBox(280, 180, 41, 41),
                    confidence=0.9,
                    category="current_position",
                    reliable=True,
                    sources=("white_person_component",),
                    grid_row=1,
                    grid_col=2,
                    marker_center=Point(276, 188),
                    marker_bbox=BoundingBox(267, 170, 19, 37),
                )
                return DetectionResult(
                    640,
                    360,
                    (node,),
                    analysis={
                        "grid_fit_status": "ok",
                        "grid_spacing": {"sx": 100.0, "sy": 100.0},
                        "grid_rows": [100, 200],
                        "grid_columns": [100, 200, 300],
                    },
                )

        agent.MapRecognitionAdapter = lambda: MapRecognitionAdapter(StubDetector())
        server, recognition_class = agent.register_custom_recognition()
        assert server is AgentServer
        instance = AgentServer._custom_recognition_holder[
            MAA_CUSTOM_RECOGNITION_NAME
        ]
        assert isinstance(instance, recognition_class)
        argv = CustomRecognition.AnalyzeArg(
            task_detail=None,
            node_name="MaaBlackFlowMapRecognize",
            custom_recognition_name=MAA_CUSTOM_RECOGNITION_NAME,
            custom_recognition_param=json.dumps({
                "recognition_mode": "grid_baseline",
                "require_solver_ready": False,
            }),
            image=np.zeros((360, 640, 3), dtype=np.uint8),
            roi=Rect(0, 0, 640, 360),
        )
        result = instance.analyze(object(), argv)
        assert isinstance(result, CustomRecognition.AnalyzeResult)
        assert result.box == (280, 180, 41, 41)
        assert result.detail["solver_ready"] is False
        assert result.detail["current_position"]["marker_center"] != (
            result.detail["current_position"]["grid_center"]
        )
        """
    )
    assert process.returncode == 0, process.stderr or process.stdout


def test_real_maafw_resource_loads_pipeline_v2_without_controller() -> None:
    _require_maafw()
    process = _run_isolated(
        """
        from pathlib import Path
        from maa.resource import Resource

        resource = Resource()
        job = resource.post_bundle(Path("maafw_project/resource").resolve())
        job.wait()
        assert job.succeeded
        assert resource.loaded
        assert resource.node_list == ["MaaBlackFlowMapRecognize"]
        node = resource.get_node_data("MaaBlackFlowMapRecognize")
        assert node["recognition"]["type"] == "Custom"
        assert node["recognition"]["param"]["custom_recognition"] == (
            "MaaBlackFlow.MapRecognize"
        )
        assert node["action"] == {"type": "DoNothing", "param": {}}
        assert node["next"] == []
        resource.clear()
        """
    )
    assert process.returncode == 0, process.stderr or process.stdout
