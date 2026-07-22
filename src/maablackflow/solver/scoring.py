"""Deterministic lexicographic comparison for feasible routes."""

from __future__ import annotations

from collections.abc import Sequence

from .result import RouteStep


def route_signature(route: Sequence[RouteStep]) -> tuple[tuple[str, str, str, int], ...]:
    return tuple((step.from_node, step.to_node, step.movement, step.action_cost) for step in route)


def is_better(
    candidate_reward: int,
    candidate_remaining: int,
    candidate_consumed: int,
    candidate_route: Sequence[RouteStep],
    incumbent_reward: int,
    incumbent_remaining: int,
    incumbent_consumed: int,
    incumbent_route: Sequence[RouteStep],
) -> bool:
    candidate_score = (candidate_reward, candidate_remaining, -candidate_consumed)
    incumbent_score = (incumbent_reward, incumbent_remaining, -incumbent_consumed)
    if candidate_score != incumbent_score:
        return candidate_score > incumbent_score
    return route_signature(candidate_route) < route_signature(incumbent_route)
