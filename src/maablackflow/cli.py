"""PowerShell-friendly command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from maablackflow.io import MapLoadError, load_problem
from maablackflow.solver import RoutePlanner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maablackflow")
    subcommands = parser.add_subparsers(dest="command", required=True)
    solve = subcommands.add_parser("solve", help="solve a JSON map")
    solve.add_argument("map", type=Path, help="path to the JSON map")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        map_state, game_state = load_problem(args.map)
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


if __name__ == "__main__":
    raise SystemExit(main())
