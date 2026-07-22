"""Immutable domain models for maps and game state."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .movement import MovementRule


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    x: int
    y: int
    type: str
    reward: int = 0
    completed: bool = False
    repeatable: bool = False
    is_exit: bool = False


@dataclass(frozen=True, slots=True)
class Edge:
    from_node: str
    to_node: str
    walking_cost: int


@dataclass(frozen=True, slots=True)
class MapState:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    movement_rules: tuple[MovementRule, ...] = ()


@dataclass(frozen=True, slots=True)
class GameState:
    current_node: str
    action_points: int
    completed_nodes: frozenset[str] = frozenset()
    movement_uses: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        uses = {} if self.movement_uses is None else dict(self.movement_uses)
        object.__setattr__(self, "movement_uses", MappingProxyType(uses))
