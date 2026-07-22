"""Configuration-driven special movement rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MovementRule:
    id: str
    vectors: tuple[tuple[int, int], ...]
    action_cost: int
    remaining_uses: int
    allow_skip_uncompleted: bool = False
