"""Structured planner output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RouteStep:
    from_node: str
    to_node: str
    movement: str
    action_cost: int
    reward_gained: int


@dataclass(frozen=True, slots=True)
class RouteResult:
    reached_exit: bool
    route: tuple[RouteStep, ...]
    total_reward: int
    remaining_action_points: int
    movement_consumed: dict[str, int]
    failure_reason: str | None = None
