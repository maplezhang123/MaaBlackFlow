"""Exact memoized search over route-planning states."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from maablackflow.domain.models import GameState, MapState, Node
from maablackflow.domain.validation import validate_problem

from .result import RouteResult, RouteStep
from .scoring import is_better


@dataclass(frozen=True, slots=True)
class _Candidate:
    reward: int
    remaining_ap: int
    consumed: tuple[int, ...]
    route: tuple[RouteStep, ...]


class RoutePlanner:
    """Find the globally optimal exit-reaching route without mutating inputs."""

    def solve(self, map_state: MapState, game_state: GameState) -> RouteResult:
        validate_problem(map_state, game_state)

        nodes = tuple(sorted(map_state.nodes, key=lambda node: node.id))
        rules = tuple(sorted(map_state.movement_rules, key=lambda rule: rule.id))
        node_by_id = {node.id: node for node in nodes}
        index_by_id = {node.id: index for index, node in enumerate(nodes)}
        node_at = {(node.x, node.y): node for node in nodes}

        edges_from: dict[str, list[tuple[str, int]]] = {}
        for edge in map_state.edges:
            edges_from.setdefault(edge.from_node, []).append(
                (edge.to_node, edge.walking_cost)
            )
        for edges in edges_from.values():
            edges.sort()

        completed_ids = {
            node.id for node in nodes if node.completed
        } | set(game_state.completed_nodes) | {game_state.current_node}
        initial_mask = sum(1 << index_by_id[node_id] for node_id in completed_ids)
        initial_uses = tuple(
            (game_state.movement_uses or {}).get(rule.id, rule.remaining_uses)
            for rule in rules
        )

        def has_uncompleted_between(
            start: Node, end: Node, completed_mask: int
        ) -> bool:
            dx = end.x - start.x
            dy = end.y - start.y
            squared_length = dx * dx + dy * dy
            for node in nodes:
                if node.id in (start.id, end.id):
                    continue
                offset_x = node.x - start.x
                offset_y = node.y - start.y
                if offset_x * dy != offset_y * dx:
                    continue
                dot = offset_x * dx + offset_y * dy
                if 0 < dot < squared_length:
                    if not completed_mask & (1 << index_by_id[node.id]):
                        return True
            return False

        @lru_cache(maxsize=None)
        def search(
            current_id: str,
            remaining_ap: int,
            completed_mask: int,
            movement_uses: tuple[int, ...],
        ) -> _Candidate | None:
            current = node_by_id[current_id]
            if current.is_exit:
                return _Candidate(0, remaining_ap, (0,) * len(rules), ())

            best: _Candidate | None = None
            transitions: list[tuple[str, str, int, int | None]] = []

            for target_id, cost in edges_from.get(current_id, ()):
                target = node_by_id[target_id]
                if cost <= remaining_ap and not has_uncompleted_between(
                    current, target, completed_mask
                ):
                    transitions.append((target_id, "walk", cost, None))

            for rule_index, rule in enumerate(rules):
                if movement_uses[rule_index] <= 0 or rule.action_cost > remaining_ap:
                    continue
                for dx, dy in sorted(rule.vectors):
                    target = node_at.get((current.x + dx, current.y + dy))
                    if target is None:
                        continue
                    if (
                        not rule.allow_skip_uncompleted
                        and has_uncompleted_between(current, target, completed_mask)
                    ):
                        continue
                    transitions.append(
                        (target.id, rule.id, rule.action_cost, rule_index)
                    )

            transitions.sort(key=lambda item: (item[0], item[1], item[2]))
            for target_id, movement, cost, rule_index in transitions:
                target_bit = 1 << index_by_id[target_id]
                gained = (
                    0
                    if completed_mask & target_bit
                    else node_by_id[target_id].reward
                )
                next_uses = list(movement_uses)
                if rule_index is not None:
                    next_uses[rule_index] -= 1
                suffix = search(
                    target_id,
                    remaining_ap - cost,
                    completed_mask | target_bit,
                    tuple(next_uses),
                )
                if suffix is None:
                    continue

                consumed = list(suffix.consumed)
                if rule_index is not None:
                    consumed[rule_index] += 1
                step = RouteStep(
                    from_node=current_id,
                    to_node=target_id,
                    movement=movement,
                    action_cost=cost,
                    reward_gained=gained,
                )
                candidate = _Candidate(
                    reward=gained + suffix.reward,
                    remaining_ap=suffix.remaining_ap,
                    consumed=tuple(consumed),
                    route=(step,) + suffix.route,
                )
                if best is None or is_better(
                    candidate.reward,
                    candidate.remaining_ap,
                    sum(candidate.consumed),
                    candidate.route,
                    best.reward,
                    best.remaining_ap,
                    sum(best.consumed),
                    best.route,
                ):
                    best = candidate
            return best

        candidate = search(
            game_state.current_node,
            game_state.action_points,
            initial_mask,
            initial_uses,
        )
        if candidate is None:
            return RouteResult(
                reached_exit=False,
                route=(),
                total_reward=0,
                remaining_action_points=game_state.action_points,
                movement_consumed={rule.id: 0 for rule in rules},
                failure_reason="no route can reach an exit with the available resources",
            )

        return RouteResult(
            reached_exit=True,
            route=candidate.route,
            total_reward=candidate.reward,
            remaining_action_points=candidate.remaining_ap,
            movement_consumed={
                rule.id: candidate.consumed[index]
                for index, rule in enumerate(rules)
            },
            failure_reason=None,
        )
