"""Optional MaaFramework AgentServer entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from .adapter import (
    MAA_CUSTOM_RECOGNITION_NAME,
    MaaAdapterError,
    MapRecognitionAdapter,
)
from .serialization import DETAIL_SCHEMA_VERSION

OPTIONAL_RUNTIME_ERROR = "MaaFramework optional runtime is not installed"
_LOGGER = logging.getLogger("maablackflow.maafw.agent")


class MaaFrameworkRuntimeError(RuntimeError):
    """Raised only when Maa integration is invoked without a usable runtime."""


def parse_custom_parameters(value: str | dict[str, object] | None) -> dict[str, object]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise MaaAdapterError("custom_recognition_param must be a JSON object")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise MaaAdapterError("custom_recognition_param is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise MaaAdapterError("custom_recognition_param must decode to an object")
    return parsed


def analyze_argv(argv: Any, adapter: MapRecognitionAdapter | None = None):
    """Pure handler used by both the real Agent wrapper and fake tests."""
    if not hasattr(argv, "image"):
        raise MaaAdapterError("Custom Recognition arguments do not contain an image")
    parameters = parse_custom_parameters(
        getattr(argv, "custom_recognition_param", None)
    )
    return (adapter or MapRecognitionAdapter()).analyze(argv.image, parameters)


def _failure_detail(exc: Exception) -> dict[str, object]:
    return {
        "schema_version": DETAIL_SCHEMA_VERSION,
        "reached_detection_stage": "failed",
        "image_width": 0,
        "image_height": 0,
        "grid_fit_status": "failed",
        "grid_spacing": None,
        "map_roi": None,
        "nodes": [],
        "current_position": None,
        "warnings": [f"{type(exc).__name__}: {exc}"],
        "solver_ready": False,
    }


def register_custom_recognition() -> tuple[Any, Any]:
    """Import the optional SDK lazily and register exactly one recognizer."""
    try:
        from maa.agent.agent_server import AgentServer
        from maa.custom_recognition import CustomRecognition
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise MaaFrameworkRuntimeError(OPTIONAL_RUNTIME_ERROR) from exc

    class MaaBlackFlowMapRecognition(CustomRecognition):
        def analyze(self, context: Any, argv: Any):
            del context  # The recognition boundary deliberately never uses controller/context.
            try:
                payload = analyze_argv(argv)
            except (MaaAdapterError, ValueError) as exc:
                _LOGGER.error(
                    json.dumps(
                        {
                            "event": "maafw_custom_recognition_failed",
                            "recognizer": MAA_CUSTOM_RECOGNITION_NAME,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                return CustomRecognition.AnalyzeResult(
                    box=None, detail=_failure_detail(exc)
                )
            return CustomRecognition.AnalyzeResult(
                box=payload.box if payload.success else None,
                detail=payload.detail,
            )

    try:
        registered = AgentServer.register_custom_recognition(
            MAA_CUSTOM_RECOGNITION_NAME,
            MaaBlackFlowMapRecognition(),
        )
    except OSError as exc:
        raise MaaFrameworkRuntimeError(OPTIONAL_RUNTIME_ERROR) from exc
    if not registered:
        raise MaaFrameworkRuntimeError(
            f"failed to register custom recognition: {MAA_CUSTOM_RECOGNITION_NAME}"
        )
    return AgentServer, MaaBlackFlowMapRecognition


def run(socket_id: str) -> int:
    if not socket_id.strip():
        raise MaaFrameworkRuntimeError("socket_id must not be empty")
    agent_server, _ = register_custom_recognition()
    if not agent_server.start_up(socket_id):
        raise MaaFrameworkRuntimeError("MaaFramework AgentServer failed to start")
    try:
        agent_server.join()
    finally:
        agent_server.shut_down()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m maablackflow.integrations.maafw.agent",
        description=(
            "Start the optional MaaFramework AgentServer for offline map recognition. "
            "The entry point registers recognition only and performs no actions."
        ),
    )
    parser.add_argument(
        "socket_id",
        help="socket identifier supplied by MaaFramework AgentClient/ProjectInterface",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run(args.socket_id)
    except MaaFrameworkRuntimeError as exc:
        print(f"Maa Agent 启动失败: {exc}", file=sys.stderr)
        print(
            "请准备官方 MaaFramework Python SDK 与对应原生运行库；本项目不会自动下载或修改 PATH。",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
