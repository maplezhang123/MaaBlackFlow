"""Validation shared by input adapters and direct Python callers."""

from __future__ import annotations

from .models import GameState, MapState


def validate_problem(map_state: MapState, game_state: GameState) -> None:
    if not map_state.nodes:
        raise ValueError("map must contain at least one node")

    node_ids = [node.id for node in map_state.nodes]
    if any(not node_id for node_id in node_ids):
        raise ValueError("node id must not be empty")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("node ids must be unique")
    coordinates = [(node.x, node.y) for node in map_state.nodes]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("node coordinates must be unique")
    node_set = set(node_ids)

    if game_state.current_node not in node_set:
        raise ValueError(f"current node does not exist: {game_state.current_node}")
    if game_state.action_points < 0:
        raise ValueError("action_points must be non-negative")
    unknown_completed = set(game_state.completed_nodes) - node_set
    if unknown_completed:
        raise ValueError(f"completed_nodes contains unknown node: {min(unknown_completed)}")

    for edge in map_state.edges:
        if edge.from_node not in node_set or edge.to_node not in node_set:
            raise ValueError(
                f"edge references unknown node: {edge.from_node} -> {edge.to_node}"
            )
        if edge.from_node == edge.to_node:
            raise ValueError(f"self-loop edge is not allowed: {edge.from_node}")
        if edge.walking_cost <= 0:
            raise ValueError("walking_cost must be positive")

    rule_ids = [rule.id for rule in map_state.movement_rules]
    if any(not rule_id for rule_id in rule_ids):
        raise ValueError("movement rule id must not be empty")
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("movement rule ids must be unique")
    for rule in map_state.movement_rules:
        if not rule.vectors:
            raise ValueError(f"movement rule must contain vectors: {rule.id}")
        if any(vector == (0, 0) for vector in rule.vectors):
            raise ValueError(f"movement rule contains zero vector: {rule.id}")
        if rule.action_cost <= 0:
            raise ValueError(f"movement action_cost must be positive: {rule.id}")
        if rule.remaining_uses < 0:
            raise ValueError(f"movement remaining_uses must be non-negative: {rule.id}")

    unknown_rules = set(game_state.movement_uses or {}) - set(rule_ids)
    if unknown_rules:
        raise ValueError(f"movement_uses contains unknown rule: {min(unknown_rules)}")
    for rule_id, uses in (game_state.movement_uses or {}).items():
        if uses < 0:
            raise ValueError(f"movement use count must be non-negative: {rule_id}")
