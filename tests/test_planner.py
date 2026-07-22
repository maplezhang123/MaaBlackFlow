from __future__ import annotations

from maablackflow.domain import Edge, GameState, MapState, MovementRule, Node
from maablackflow.solver import RoutePlanner


def node(
    node_id: str,
    x: int,
    y: int,
    *,
    reward: int = 0,
    completed: bool = False,
    repeatable: bool = False,
    exit: bool = False,
) -> Node:
    return Node(
        node_id,
        x,
        y,
        "exit" if exit else "encounter",
        reward,
        completed,
        repeatable,
        exit,
    )


def solve(
    nodes: list[Node],
    edges: list[Edge],
    ap: int,
    *,
    current: str = "start",
    rules: list[MovementRule] | None = None,
    completed: frozenset[str] = frozenset(),
    uses: dict[str, int] | None = None,
):
    return RoutePlanner().solve(
        MapState(tuple(nodes), tuple(edges), tuple(rules or [])),
        GameState(current, ap, completed, uses),
    )


def test_unique_direct_route_to_exit() -> None:
    result = solve(
        [node("start", 0, 0), node("exit", 1, 0, exit=True)],
        [Edge("start", "exit", 1)],
        1,
    )
    assert result.reached_exit
    assert [step.to_node for step in result.route] == ["exit"]
    assert result.remaining_action_points == 0


def test_detour_reward_is_taken_when_exit_remains_reachable() -> None:
    result = solve(
        [
            node("start", 0, 0),
            node("reward", 0, 1, reward=8),
            node("exit", 1, 0, exit=True),
        ],
        [
            Edge("start", "exit", 1),
            Edge("start", "reward", 1),
            Edge("reward", "exit", 1),
        ],
        2,
    )
    assert result.total_reward == 8
    assert [step.to_node for step in result.route] == ["reward", "exit"]


def test_unreachable_reward_is_rejected_in_favor_of_exit() -> None:
    result = solve(
        [
            node("start", 0, 0),
            node("trap", 0, 1, reward=100),
            node("exit", 1, 0, exit=True),
        ],
        [Edge("start", "trap", 1), Edge("start", "exit", 1)],
        1,
    )
    assert result.reached_exit
    assert [step.to_node for step in result.route] == ["exit"]
    assert result.total_reward == 0


def test_two_tile_item_can_be_required_to_reach_exit() -> None:
    jump = MovementRule(
        "straight_two",
        ((2, 0), (-2, 0), (0, 2), (0, -2)),
        action_cost=1,
        remaining_uses=1,
        allow_skip_uncompleted=True,
    )
    result = solve(
        [
            node("start", 0, 0),
            node("blocker", 1, 0),
            node("exit", 2, 0, exit=True),
        ],
        [],
        1,
        rules=[jump],
    )
    assert result.reached_exit
    assert result.route[0].movement == "straight_two"
    assert result.movement_consumed == {"straight_two": 1}


def test_walking_is_preferred_when_item_has_no_advantage() -> None:
    item = MovementRule("item", ((1, 0),), 1, 1, True)
    result = solve(
        [node("start", 0, 0), node("exit", 1, 0, exit=True)],
        [Edge("start", "exit", 1)],
        1,
        rules=[item],
    )
    assert result.route[0].movement == "walk"
    assert result.movement_consumed["item"] == 0


def test_completed_node_reward_is_not_counted_again() -> None:
    result = solve(
        [
            node("start", 0, 0),
            node("reward", 0, 1, reward=9),
            node("exit", 1, 1, exit=True),
        ],
        [Edge("start", "reward", 1), Edge("reward", "exit", 1)],
        2,
        completed=frozenset({"reward"}),
    )
    assert result.reached_exit
    assert result.total_reward == 0
    assert result.route[0].reward_gained == 0


def test_insufficient_action_points_returns_structured_failure() -> None:
    result = solve(
        [node("start", 0, 0), node("exit", 1, 0, exit=True)],
        [Edge("start", "exit", 2)],
        1,
    )
    assert not result.reached_exit
    assert result.route == ()
    assert result.failure_reason
    assert result.remaining_action_points == 1


def test_same_reward_prefers_more_remaining_action_points() -> None:
    result = solve(
        [
            node("start", 0, 0),
            node("slow", 0, 1),
            node("exit", 1, 0, exit=True),
        ],
        [
            Edge("start", "slow", 1),
            Edge("slow", "exit", 1),
            Edge("start", "exit", 1),
        ],
        2,
    )
    assert [step.to_node for step in result.route] == ["exit"]
    assert result.remaining_action_points == 1


def test_same_reward_and_remaining_ap_prefers_fewer_items() -> None:
    item = MovementRule("item", ((1, 0),), 1, 1, True)
    result = solve(
        [node("start", 0, 0), node("exit", 1, 0, exit=True)],
        [Edge("start", "exit", 1)],
        2,
        rules=[item],
    )
    assert result.route[0].movement == "walk"
    assert result.remaining_action_points == 1


def test_walking_cannot_skip_an_uncompleted_collinear_node() -> None:
    result = solve(
        [
            node("start", 0, 0),
            node("blocker", 1, 0),
            node("exit", 2, 0, exit=True),
        ],
        [Edge("start", "exit", 1)],
        1,
    )
    assert not result.reached_exit


def test_solver_does_not_modify_input_objects() -> None:
    map_state = MapState(
        (node("start", 0, 0), node("exit", 1, 0, exit=True)),
        (Edge("start", "exit", 1),),
        (),
    )
    game_state = GameState("start", 1, frozenset(), {})
    before_map = repr(map_state)
    before_game = repr(game_state)
    RoutePlanner().solve(map_state, game_state)
    assert repr(map_state) == before_map
    assert repr(game_state) == before_game


def test_same_input_is_deterministic() -> None:
    problem = MapState(
        (
            node("start", 0, 0),
            node("a", 0, 1),
            node("b", 1, 0),
            node("exit", 1, 1, exit=True),
        ),
        (
            Edge("start", "b", 1),
            Edge("b", "exit", 1),
            Edge("start", "a", 1),
            Edge("a", "exit", 1),
        ),
    )
    state = GameState("start", 2)
    results = [RoutePlanner().solve(problem, state) for _ in range(5)]
    assert all(result == results[0] for result in results)
    assert [step.to_node for step in results[0].route] == ["a", "exit"]


def test_game_state_can_override_configured_item_uses() -> None:
    item = MovementRule("jump", ((2, 0),), 1, 1, True)
    result = solve(
        [node("start", 0, 0), node("exit", 2, 0, exit=True)],
        [],
        1,
        rules=[item],
        uses={"jump": 0},
    )
    assert not result.reached_exit
