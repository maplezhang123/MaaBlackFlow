"""PowerShell-friendly command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from maablackflow.integrations.maafw import MaaAdapterError, MapRecognitionAdapter
from maablackflow.integrations.maafw.serialization import stable_json
from maablackflow.io import MapLoadError, load_problem
from maablackflow.solver import RoutePlanner
from maablackflow.vision.image_io import load_png
from maablackflow.vision.templates import (
    TemplateCandidateError,
    build_template_candidates,
)
from maablackflow.vision import (
    DatasetInspectionError,
    EvaluationError,
    NodeDetector,
    VisionError,
    evaluate_directories,
    inspect_dataset,
    write_detection_outputs,
    write_grid_debug_output,
)


DEFAULT_PRIVATE_OUTPUT = Path("data/outputs_private/run01")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maablackflow")
    subcommands = parser.add_subparsers(dest="command", required=True)

    solve = subcommands.add_parser("solve", help="solve a JSON map")
    solve.add_argument("map", type=Path, help="path to the JSON map")

    inspect = subcommands.add_parser(
        "inspect-dataset", help="inventory a private local screenshot directory"
    )
    inspect.add_argument("directory", type=Path, help="directory containing images")
    inspect.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PRIVATE_OUTPUT,
        help="private output directory (default: data/outputs_private/run01)",
    )

    detect = subcommands.add_parser(
        "detect-nodes", help="detect visible map nodes in one offline PNG"
    )
    detect.add_argument("image", type=Path, help="input PNG image")
    detect.add_argument(
        "--output",
        type=Path,
        required=True,
        help="private directory for JSON and annotated PNG",
    )
    evaluate = subcommands.add_parser(
        "evaluate-detection",
        help="evaluate private node predictions against private ground truth",
    )
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)

    maa_smoke = subcommands.add_parser(
        "maa-adapter-smoke",
        help="run the Maa adapter offline without MaaFramework",
    )
    maa_smoke.add_argument("image", type=Path, help="input PNG image")
    maa_smoke.add_argument(
        "--output",
        type=Path,
        required=True,
        help="private directory for the framework-neutral detail JSON",
    )
    templates = subcommands.add_parser(
        "build-template-candidates",
        help="prepare private, holdout-safe template candidates",
    )
    templates.add_argument("directory", type=Path, help="private screenshot directory")
    templates.add_argument("--output", type=Path, required=True)
    templates.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="holdout filename to exclude without reading; repeat for each holdout",
    )

    return parser


def _solve(map_path: Path) -> int:
    try:
        map_state, game_state = load_problem(map_path)
        result = RoutePlanner().solve(map_state, game_state)
    except MapLoadError as exc:
        print(f"地图错误: {exc}", file=sys.stderr)
        return 2

    print(f"是否到达出口: {'是' if result.reached_exit else '否'}")
    print("最优路线:")
    if result.route:
        for index, step in enumerate(result.route, 1):
            print(
                f"  {index}. {step.from_node} -> {step.to_node} "
                f"[{step.movement}] 行动力-{step.action_cost}, 收益+{step.reward_gained}"
            )
    else:
        print("  （无）")
    print(f"总收益: {result.total_reward}")
    print(f"剩余行动力: {result.remaining_action_points}")
    print("加工品使用情况:")
    if result.movement_consumed:
        for rule_id, count in sorted(result.movement_consumed.items()):
            print(f"  {rule_id}: {count}")
    else:
        print("  （无加工品）")
    print(f"无解原因: {result.failure_reason or '无'}")
    return 0 if result.reached_exit else 1


def _inspect_dataset(directory: Path, output: Path) -> int:
    try:
        result = inspect_dataset(directory, output)
    except DatasetInspectionError as exc:
        print(f"数据清点失败: {exc}", file=sys.stderr)
        return 2
    duplicates = sum(entry.duplicate_of is not None for entry in result.entries)
    resolutions = Counter(f"{entry.width}x{entry.height}" for entry in result.entries)
    print(f"图片数量: {len(result.entries)}")
    print(f"分辨率: {', '.join(f'{key} ({count})' for key, count in sorted(resolutions.items()))}")
    print(f"重复图片: {duplicates}")
    print(f"数据清单: {result.manifest_path.resolve()}")
    print(f"Contact sheet: {result.contact_sheet_path.resolve()}")
    return 0


def _detect_nodes(image_path: Path, output: Path) -> int:
    try:
        result, image, source, debug_image = NodeDetector().detect_file_with_debug(image_path)
        json_path, annotated_path = write_detection_outputs(
            source, image, result, output
        )
        debug_path = write_grid_debug_output(source, debug_image, output)
    except VisionError as exc:
        print(f"节点检测失败: {exc}", file=sys.stderr)
        return 2
    print(f"输入图片: {image_path}")
    print(f"原始分辨率: {result.image_width}x{result.image_height}")
    print(f"检测到的节点数量: {len(result.nodes)}")
    print(f"JSON 输出路径: {json_path.resolve()}")
    categories = Counter(node.category for node in result.nodes)
    print(
        "分类统计: "
        + ", ".join(f"{name}={count}" for name, count in sorted(categories.items()))
    )
    print(f"标注图输出路径: {annotated_path.resolve()}")
    print(f"道路/网格调试图路径: {debug_path.resolve()}")
    return 0


def _maa_adapter_smoke(image_path: Path, output: Path) -> int:
    try:
        image, source = load_png(image_path)
        payload = MapRecognitionAdapter().analyze(
            image,
            {
                "output_detail": True,
                "require_solver_ready": False,
                "recognition_mode": "grid_baseline",
            },
        )
        if not payload.success:
            raise MaaAdapterError("adapter did not produce a successful recognition")
        current = payload.detail.get("current_position")
        if not isinstance(current, dict):
            raise MaaAdapterError("current_position was not detected")
        if "marker_center" not in current or "grid_center" not in current:
            raise MaaAdapterError(
                "current_position must contain marker_center and grid_center"
            )
        nodes = payload.detail.get("nodes")
        if not isinstance(nodes, list):
            raise MaaAdapterError("adapter detail nodes are invalid")
        cells = [(node.get("grid_row"), node.get("grid_col")) for node in nodes]
        if len(cells) != len(set(cells)):
            raise MaaAdapterError("adapter detail contains duplicate grid cells")
        document = stable_json(payload.detail)
        if str(source.resolve()) in document:
            raise MaaAdapterError("adapter detail contains a private absolute path")
    except (VisionError, MaaAdapterError, OSError) as exc:
        print(f"Maa adapter smoke 失败: {exc}", file=sys.stderr)
        return 2

    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / f"{source.stem}.maa-detail.json"
    detail_path.write_text(document + "\n", encoding="utf-8")
    print(f"输入分辨率: {payload.detail['image_width']}x{payload.detail['image_height']}")
    print(f"识别节点数量: {len(nodes)}")
    print(f"solver_ready: {str(payload.detail['solver_ready']).lower()}")
    print(f"detail JSON: {detail_path.resolve()}")
    return 0

def _build_template_candidates(
    directory: Path, output: Path, excluded: list[str]
) -> int:
    try:
        result = build_template_candidates(
            directory,
            output,
            excluded_filenames=excluded,
        )
    except TemplateCandidateError as exc:
        print(f"模板候选构建失败: {exc}", file=sys.stderr)
        return 2
    print(f"候选模板数量: {result.candidate_count}")
    print(f"推荐模板数量: {result.recommended_count}")
    print(f"来源清单: {result.source_manifest_path.resolve()}")
    print(f"候选清单: {result.candidate_manifest_path.resolve()}")
    print(f"Contact sheet: {result.contact_sheet_path.resolve()}")
    print(f"Pipeline 草案: {result.pipeline_draft_path.resolve()}")
    print(f"检查包: {result.review_bundle_path.resolve()}")
    return 0

def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _evaluate_detection(ground_truth: Path, predictions: Path) -> int:
    try:
        report = evaluate_directories(ground_truth, predictions)
    except EvaluationError as exc:
        print(f"检测评估失败: {exc}", file=sys.stderr)
        return 2
    for item in (*report.images, report.total):
        print(f"[{item.name}]")
        print(f"  TP={item.true_positive} FP={item.false_positive} FN={item.false_negative}")
        print(f"  precision={item.precision:.3f} recall={item.recall:.3f} F1={item.f1:.3f}")
        print(f"  mean_error={_format_metric(item.mean_center_error)} max_error={_format_metric(item.max_center_error)}")
        print(
            f"  current_marker_error={_format_metric(item.current_marker_error)} "
            f"current_grid_correct={item.current_grid_cell_correct} "
            f"duplicate_grid_points={item.duplicate_grid_points}"
        )
    return 0

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "solve":
        return _solve(args.map)
    if args.command == "inspect-dataset":
        return _inspect_dataset(args.directory, args.output)
    if args.command == "detect-nodes":
        return _detect_nodes(args.image, args.output)
    if args.command == "evaluate-detection":
        return _evaluate_detection(args.ground_truth, args.predictions)
    if args.command == "maa-adapter-smoke":
        return _maa_adapter_smoke(args.image, args.output)
    if args.command == "build-template-candidates":
        return _build_template_candidates(args.directory, args.output, args.exclude)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())