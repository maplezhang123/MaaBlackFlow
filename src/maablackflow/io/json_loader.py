"""Load and validate planner problems from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from maablackflow.domain.models import Edge, GameState, MapState, Node
from maablackflow.domain.movement import MovementRule
from maablackflow.domain.validation import validate_problem


class MapLoadError(ValueError):
    """Raised when a JSON document cannot describe a valid planning problem."""


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MapLoadError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise MapLoadError(f"{path} must be an array")
    return value


def _field(obj: dict[str, Any], name: str, expected: type, path: str) -> Any:
    if name not in obj:
        raise MapLoadError(f"{path}.{name} is required")
    value = obj[name]
    if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise MapLoadError(f"{path}.{name} must be an integer")
    if expected is not int and not isinstance(value, expected):
        raise MapLoadError(f"{path}.{name} must be a {expected.__name__}")
    return value


def load_problem(path: str | Path) -> tuple[MapState, GameState]:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MapLoadError(f"cannot read map file '{source}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MapLoadError(
            f"invalid JSON in '{source}' at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    try:
        root = _object(raw, "$")
        nodes = []
        for index, raw_node in enumerate(_list(root.get("nodes"), "$.nodes")):
            item = _object(raw_node, f"$.nodes[{index}]")
            nodes.append(
                Node(
                    id=_field(item, "id", str, f"$.nodes[{index}]"),
                    x=_field(item, "x", int, f"$.nodes[{index}]"),
                    y=_field(item, "y", int, f"$.nodes[{index}]"),
                    type=_field(item, "type", str, f"$.nodes[{index}]"),
                    reward=item.get("reward", 0),
                    completed=item.get("completed", False),
                    repeatable=item.get("repeatable", False),
                    is_exit=item.get("is_exit", False),
                )
            )

        edges = []
        for index, raw_edge in enumerate(_list(root.get("edges"), "$.edges")):
            item = _object(raw_edge, f"$.edges[{index}]")
            edges.append(
                Edge(
                    from_node=_field(item, "from_node", str, f"$.edges[{index}]"),
                    to_node=_field(item, "to_node", str, f"$.edges[{index}]"),
                    walking_cost=_field(
                        item, "walking_cost", int, f"$.edges[{index}]"
                    ),
                )
            )

        rules = []
        raw_rules = _list(root.get("movement_rules", []), "$.movement_rules")
        for index, raw_rule in enumerate(raw_rules):
            item = _object(raw_rule, f"$.movement_rules[{index}]")
            vectors = []
            for vector_index, raw_vector in enumerate(
                _list(item.get("vectors"), f"$.movement_rules[{index}].vectors")
            ):
                vector = _list(
                    raw_vector,
                    f"$.movement_rules[{index}].vectors[{vector_index}]",
                )
                if (
                    len(vector) != 2
                    or not all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in vector
                    )
                ):
                    raise MapLoadError(
                        f"$.movement_rules[{index}].vectors[{vector_index}] "
                        "must contain exactly two integers"
                    )
                vectors.append((vector[0], vector[1]))
            rules.append(
                MovementRule(
                    id=_field(item, "id", str, f"$.movement_rules[{index}]"),
                    vectors=tuple(vectors),
                    action_cost=_field(
                        item, "action_cost", int, f"$.movement_rules[{index}]"
                    ),
                    remaining_uses=_field(
                        item, "remaining_uses", int, f"$.movement_rules[{index}]"
                    ),
                    allow_skip_uncompleted=item.get(
                        "allow_skip_uncompleted", False
                    ),
                )
            )

        state_item = _object(root.get("game_state"), "$.game_state")
        raw_completed = _list(
            state_item.get("completed_nodes", []),
            "$.game_state.completed_nodes",
        )
        if not all(isinstance(value, str) for value in raw_completed):
            raise MapLoadError("$.game_state.completed_nodes must contain strings")
        raw_uses = _object(
            state_item.get("movement_uses", {}),
            "$.game_state.movement_uses",
        )
        if not all(
            isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            for key, value in raw_uses.items()
        ):
            raise MapLoadError(
                "$.game_state.movement_uses must map strings to integers"
            )
        game_state = GameState(
            current_node=_field(
                state_item, "current_node", str, "$.game_state"
            ),
            action_points=_field(
                state_item, "action_points", int, "$.game_state"
            ),
            completed_nodes=frozenset(raw_completed),
            movement_uses=raw_uses,
        )
        map_state = MapState(tuple(nodes), tuple(edges), tuple(rules))
        validate_problem(map_state, game_state)
        _validate_optional_types(map_state)
        return map_state, game_state
    except MapLoadError:
        raise
    except (TypeError, ValueError) as exc:
        raise MapLoadError(f"invalid map '{source}': {exc}") from exc


def _validate_optional_types(map_state: MapState) -> None:
    for node in map_state.nodes:
        if not isinstance(node.reward, int) or isinstance(node.reward, bool):
            raise MapLoadError(f"node reward must be an integer: {node.id}")
        for name in ("completed", "repeatable", "is_exit"):
            if not isinstance(getattr(node, name), bool):
                raise MapLoadError(f"node {name} must be boolean: {node.id}")
    for rule in map_state.movement_rules:
        if not isinstance(rule.allow_skip_uncompleted, bool):
            raise MapLoadError(
                f"movement allow_skip_uncompleted must be boolean: {rule.id}"
            )
