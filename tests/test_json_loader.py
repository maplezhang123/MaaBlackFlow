from __future__ import annotations

import json

import pytest

from maablackflow.io import MapLoadError, load_problem


def test_invalid_json_has_location_in_error(tmp_path) -> None:
    source = tmp_path / "broken.json"
    source.write_text('{"nodes": [}', encoding="utf-8")
    with pytest.raises(MapLoadError, match=r"invalid JSON.*line 1, column"):
        load_problem(source)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"edges": [{"from_node": "start", "to_node": "missing", "walking_cost": 1}]}, "unknown node"),
        ({"nodes": []}, "at least one node"),
        ({"game_state": {"current_node": "missing", "action_points": 1}}, "current node"),
    ],
)
def test_invalid_map_has_clear_error(tmp_path, change, message) -> None:
    document = {
        "nodes": [
            {"id": "start", "x": 0, "y": 0, "type": "start"},
            {"id": "exit", "x": 1, "y": 0, "type": "exit", "is_exit": True},
        ],
        "edges": [{"from_node": "start", "to_node": "exit", "walking_cost": 1}],
        "game_state": {"current_node": "start", "action_points": 1},
    }
    document.update(change)
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(MapLoadError, match=message):
        load_problem(source)


def test_valid_map_loads_defaults(tmp_path) -> None:
    source = tmp_path / "valid.json"
    source.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "start", "x": 0, "y": 0, "type": "start"},
                    {
                        "id": "exit",
                        "x": 1,
                        "y": 0,
                        "type": "exit",
                        "is_exit": True,
                    },
                ],
                "edges": [
                    {
                        "from_node": "start",
                        "to_node": "exit",
                        "walking_cost": 1,
                    }
                ],
                "game_state": {"current_node": "start", "action_points": 1},
            }
        ),
        encoding="utf-8",
    )
    map_state, game_state = load_problem(source)
    assert len(map_state.nodes) == 2
    assert game_state.completed_nodes == frozenset()
