"""Comprehensive verification suite for GridMind Environment API v1.0.

Covers the Developer 1 Phase 13 checklist:
grid/agent creation, movement, boundary/obstacle/same-cell/swap collisions,
goal detection, rewards, reset, seed reproducibility, state, multiple agents,
termination, truncation, and BFS reachability (bounded).
"""

import pytest

from python_backend.environment.constants import Action, Position
from python_backend.environment.grid_world import GridWorld
from python_backend.environment.reward import RewardConfig


@pytest.fixture
def base_env() -> GridWorld:
    """Fixture providing a deterministic 2-agent, 5x5 grid environment."""
    agents = [
        (Position(0, 0), Position(0, 4)),  # Agent 0
        (Position(4, 4), Position(4, 0)),  # Agent 1
    ]
    obstacles = {Position(2, 2)}
    return GridWorld(
        height=5,
        width=5,
        agents_config=agents,
        obstacles=obstacles,
        reward_config=RewardConfig(),
        max_steps=10,
        seed=42,
    )


# ================= Grid & Agent Creation =================


def test_grid_initialization(base_env: GridWorld) -> None:
    assert base_env.height == 5
    assert base_env.width == 5
    assert len(base_env.agents) == 2
    assert base_env.is_obstacle(Position(2, 2))
    assert not base_env.is_obstacle(Position(0, 0))


def test_agent_creation_state(base_env: GridWorld) -> None:
    state = base_env.get_agent_state(0)
    assert state["agent_id"] == 0
    assert state["current_position"] == Position(0, 0)
    assert state["goal_position"] == Position(0, 4)
    assert state["steps"] == 0
    assert state["total_reward"] == 0.0
    assert state["done"] is False

    all_states = base_env.get_all_states()
    assert set(all_states.keys()) == {0, 1}


def test_invalid_placement_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        GridWorld(5, 5, agents_config=[(Position(5, 0), Position(0, 0))])


def test_invalid_placement_on_obstacle() -> None:
    with pytest.raises(ValueError, match="conflicts with an obstacle"):
        GridWorld(
            5, 5,
            agents_config=[(Position(2, 2), Position(0, 0))],
            obstacles={Position(2, 2)},
        )


# ================= Movement & Rewards =================


def test_movement_and_step(base_env: GridWorld) -> None:
    base_env.reset()
    # Agent 0 moves RIGHT, Agent 1 moves LEFT
    obs, rewards, dones, terminated, info = base_env.step({0: Action.RIGHT, 1: Action.LEFT})
    assert base_env.get_agent_position(0) == Position(0, 1)
    assert base_env.get_agent_position(1) == Position(4, 3)
    assert rewards[0] == -1.0
    assert rewards[1] == -1.0
    assert not terminated


def test_stay_reward(base_env: GridWorld) -> None:
    base_env.reset()
    _, rewards, _, _, _ = base_env.step({0: Action.STAY, 1: Action.STAY})
    assert rewards[0] == -2.0
    assert rewards[1] == -2.0


def test_custom_reward_config() -> None:
    config = RewardConfig(goal=50.0, step=-0.5, stay=-0.1)
    env = GridWorld(3, 3, agents_config=[(Position(0, 0), Position(0, 1))],
                    reward_config=config)
    env.reset()
    _, rewards, _, _, _ = env.step({0: Action.RIGHT})
    assert rewards[0] == 50.0
    env.reset()
    _, rewards, _, _, _ = env.step({0: Action.STAY})
    assert rewards[0] == -0.1


# ================= Collisions =================


def test_boundary_collision(base_env: GridWorld) -> None:
    base_env.reset()
    # Agent 0 attempts to move UP outside row 0
    obs, rewards, dones, terminated, info = base_env.step({0: Action.UP, 1: Action.STAY})
    assert base_env.get_agent_position(0) == Position(0, 0)  # Bounced back
    assert base_env.is_collision(0)
    assert rewards[0] == -10.0  # Obstacle/boundary penalty
    assert info["collisions"][0] == "BOUNDARY"


def test_obstacle_collision() -> None:
    agents = [(Position(2, 1), Position(0, 0))]
    obstacles = {Position(2, 2)}
    env = GridWorld(5, 5, agents_config=agents, obstacles=obstacles)
    env.reset()

    # Move RIGHT directly into obstacle at (2, 2)
    obs, rewards, dones, terminated, info = env.step({0: Action.RIGHT})
    assert env.get_agent_position(0) == Position(2, 1)  # Bounced back
    assert env.is_collision(0)
    assert rewards[0] == -10.0


def test_same_cell_collision() -> None:
    # Agent 0 at (1, 2), Agent 1 at (3, 2); both step toward (2, 2)
    agents = [(Position(1, 2), Position(0, 0)), (Position(3, 2), Position(4, 4))]
    env = GridWorld(5, 5, agents_config=agents)
    env.reset()

    obs, rewards, dones, terminated, info = env.step({0: Action.DOWN, 1: Action.UP})
    assert env.get_agent_position(0) == Position(1, 2)
    assert env.get_agent_position(1) == Position(3, 2)
    assert env.is_collision(0)
    assert env.is_collision(1)
    assert rewards[0] == -20.0
    assert rewards[1] == -20.0


def test_swap_collision() -> None:
    # Agent 0 at (1, 1), Agent 1 at (1, 2); try to cross each other
    agents = [(Position(1, 1), Position(0, 0)), (Position(1, 2), Position(4, 4))]
    env = GridWorld(5, 5, agents_config=agents)
    env.reset()

    obs, rewards, dones, terminated, info = env.step({0: Action.RIGHT, 1: Action.LEFT})
    assert env.get_agent_position(0) == Position(1, 1)
    assert env.get_agent_position(1) == Position(1, 2)
    assert env.is_collision(0)
    assert env.is_collision(1)
    assert rewards[0] == -20.0
    assert rewards[1] == -20.0


def test_cascade_collision_bounces_mover_into_bounced_agent() -> None:
    # Agent 1 is blocked by an obstacle and bounces back to (0, 3).
    # Agent 0 simultaneously moves into (0, 3) and must cascade-bounce too.
    agents = [
        (Position(0, 2), Position(4, 4)),
        (Position(0, 3), Position(4, 0)),
    ]
    obstacles = {Position(1, 3)}  # Blocks Agent 1's DOWN move
    env = GridWorld(5, 5, agents_config=agents, obstacles=obstacles)
    env.reset()

    obs, rewards, _, _, info = env.step({0: Action.RIGHT, 1: Action.DOWN})
    assert env.get_agent_position(0) == Position(0, 2)  # Cascaded bounce
    assert env.get_agent_position(1) == Position(0, 3)  # Obstacle bounce
    assert env.is_collision(0)
    assert env.is_collision(1)


# ================= Goal, Termination, Truncation =================


def test_goal_detection_and_termination() -> None:
    agents = [(Position(0, 3), Position(0, 4))]
    env = GridWorld(5, 5, agents_config=agents, max_steps=10)
    env.reset()

    obs, rewards, dones, terminated, info = env.step({0: Action.RIGHT})
    assert env.get_agent_position(0) == Position(0, 4)
    assert dones[0] is True
    assert terminated is True
    assert rewards[0] == 100.0
    assert env.is_goal_reached(0)


def test_truncation(base_env: GridWorld) -> None:
    base_env.reset()
    terminated = False
    info: dict = {}
    for _ in range(base_env.max_steps):
        obs, rewards, dones, terminated, info = base_env.step({0: Action.STAY, 1: Action.STAY})
    assert info["truncated"] is True
    assert not terminated


def test_truncation_flag_not_premature(base_env: GridWorld) -> None:
    base_env.reset()
    obs, rewards, dones, terminated, info = base_env.step({0: Action.STAY, 1: Action.STAY})
    assert info["truncated"] is False
    assert base_env.current_step < base_env.max_steps


# ================= Reset & Seed Reproducibility =================


def test_reset_restores_initial_state(base_env: GridWorld) -> None:
    base_env.reset()
    base_env.step({0: Action.RIGHT, 1: Action.LEFT})
    assert base_env.get_agent_position(0) == Position(0, 1)

    base_env.reset()
    assert base_env.get_agent_position(0) == Position(0, 0)
    assert base_env.get_agent_position(1) == Position(4, 4)
    assert base_env.current_step == 0
    state = base_env.get_agent_state(0)
    assert state["total_reward"] == 0.0
    assert state["steps"] == 0
    assert state["done"] is False
    assert not base_env.is_collision(0)


def test_seed_reproducible_random_layout() -> None:
    env_a = GridWorld(7, 7)
    env_b = GridWorld(7, 7)
    env_a.reset(seed=123, randomize=True, num_agents=2, num_obstacles=5)
    env_b.reset(seed=123, randomize=True, num_agents=2, num_obstacles=5)

    assert env_a.get_all_states() == env_b.get_all_states()
    assert env_a.obstacles == env_b.obstacles


def test_seed_different_seeds_differ() -> None:
    env_a = GridWorld(7, 7)
    env_b = GridWorld(7, 7)
    env_a.reset(seed=1, randomize=True, num_agents=2, num_obstacles=5)
    env_b.reset(seed=2, randomize=True, num_agents=2, num_obstacles=5)

    # With 90 free cells the probability of identical layouts is negligible.
    assert env_a.get_all_states() != env_b.get_all_states()


def test_random_layout_rejects_impossible_configuration() -> None:
    env = GridWorld(2, 2)
    with pytest.raises(ValueError, match="exceed total grid cell capacity"):
        env.reset(randomize=True, num_agents=2, num_obstacles=10)


def test_random_layout_guarantees_reachability() -> None:
    env = GridWorld(6, 6)
    env.reset(seed=7, randomize=True, num_agents=2, num_obstacles=8)
    for aid in env.agents:
        start = env.get_agent_position(aid)
        goal = env.get_goal_position(aid)
        assert GridWorld.is_reachable(start, goal, env.obstacles, env.height, env.width)


# ================= Public API Helpers =================


def test_get_valid_actions_respects_boundaries_and_obstacles() -> None:
    agents = [(Position(0, 0), Position(4, 0))]
    obstacles = {Position(0, 1)}  # Blocks RIGHT
    env = GridWorld(5, 5, agents_config=agents, obstacles=obstacles)
    env.reset()

    valid = env.get_valid_actions(0)
    assert Action.UP not in valid      # Boundary above row 0
    assert Action.LEFT not in valid    # Boundary left of column 0
    assert Action.RIGHT not in valid   # Obstacle at (0, 1)
    assert set(valid) == {Action.DOWN, Action.STAY}

    # Terminal agents may only STAY
    for _ in range(4):
        env.step({0: Action.DOWN})
    assert env.is_goal_reached(0)
    assert env.get_valid_actions(0) == [Action.STAY]


def test_missing_action_defaults_to_stay() -> None:
    env = GridWorld(3, 3, agents_config=[(Position(0, 0), Position(2, 2))])
    env.reset()
    # Omitting agent 0's action entirely must behave as STAY (-2 reward)
    _, rewards, _, _, _ = env.step({})
    assert rewards[0] == -2.0
    assert env.get_agent_position(0) == Position(0, 0)


def test_render_contains_agents_obstacles_and_goals(base_env: GridWorld) -> None:
    base_env.reset()
    rendered = base_env.render()
    assert "A0" in rendered
    assert "A1" in rendered
    assert "G0" in rendered
    assert "G1" in rendered
    assert "#" in rendered


def test_close_clears_environment(base_env: GridWorld) -> None:
    base_env.reset()
    base_env.close()
    assert len(base_env.agents) == 0
    assert len(base_env.obstacles) == 0


# ================= Observations =================


def test_observation_vector_shape_and_values() -> None:
    agents = [(Position(0, 0), Position(1, 1))]
    env = GridWorld(3, 3, agents_config=agents)
    obs = env.reset()
    vec = obs[0]
    assert len(vec) == 12
    # [agent_row, agent_col, goal_row, goal_col, obs_up, obs_down, obs_left, obs_right, ag_up, ag_down, ag_left, ag_right]
    assert vec[0:4] == [0, 0, 1, 1]
    assert vec[4] == 1  # Boundary above is blocked
    assert vec[6] == 1  # Boundary to the left is blocked


def test_observation_detects_obstacle_and_other_agent() -> None:
    # Agent 1 sits directly below Agent 0; obstacle directly to the right.
    agents = [
        (Position(1, 1), Position(0, 0)),
        (Position(2, 1), Position(4, 4)),
    ]
    env = GridWorld(5, 5, agents_config=agents, obstacles={Position(1, 2)})
    obs = env.reset()

    vec0 = obs[0]
    assert vec0[7] == 1  # obstacle_right at (1, 2)
    assert vec0[9] == 1  # agent_down: agent 1 at (2, 1)

    vec1 = obs[1]
    assert vec1[8] == 1  # agent_up: agent 0 seen above at (1, 1)


# ================= BFS Reachability (bounded) =================


def test_bfs_reachability() -> None:
    obstacles = {Position(1, 0), Position(1, 1), Position(1, 2)}
    # Completely blocked horizontal wall across 3-wide grid
    assert not GridWorld.is_reachable(Position(0, 0), Position(2, 0), obstacles, 3, 3)
    # Clear path around obstacle
    cleared_obstacles = {Position(1, 0), Position(1, 1)}
    assert GridWorld.is_reachable(Position(0, 0), Position(2, 0), cleared_obstacles, 3, 3)


def test_bfs_reachability_goal_out_of_bounds() -> None:
    assert not GridWorld.is_reachable(Position(0, 0), Position(5, 0), set(), 3, 3)
    assert not GridWorld.is_reachable(Position(0, 0), Position(-1, 0), set(), 3, 3)


def test_bfs_reachability_goal_on_obstacle() -> None:
    obstacles = {Position(2, 0)}
    assert not GridWorld.is_reachable(Position(0, 0), Position(2, 0), obstacles, 3, 3)


def test_bfs_reachability_trivial_start_equals_goal() -> None:
    assert GridWorld.is_reachable(Position(1, 1), Position(1, 1), set(), 3, 3)
