"""Observation construction for multi-agent systems (Developer 3, Phase 5).

Two observation modes support the LOCAL_ONLY vs LOCAL_PLUS_OTHER_AGENTS
comparison:

LOCAL_ONLY
    The frozen 12-element Environment API v1.0 vector, returned unchanged.
    Fields: agent (row, col), goal (row, col), 4 obstacle flags, 4 agent flags
    (see docs/environment_api.md §3).

LOCAL_PLUS_OTHER_AGENTS
    The frozen 12-element vector followed by one fixed 4-element block per
    OTHER agent (ordered by agent id):
        [goal_delta_row, goal_delta_col, wait_flag, done_flag]
    where goal_delta is the other agent's (goal - position) and wait_flag is 1
    if that agent requested STAY on the previous step.

    Size: 12 + 4 * (num_agents - 1), constant per system configuration, so
    Q-networks keep a fixed input shape.

No environment code is modified: this builder consumes only the frozen
get_all_states() / get_agent_observation() API.
"""

from typing import Dict, List, Optional, Sequence

LOCAL_ONLY = "local_only"
LOCAL_PLUS_OTHER_AGENTS = "local_plus_other_agents"

OBSERVATION_MODES = (LOCAL_ONLY, LOCAL_PLUS_OTHER_AGENTS)

# Size of the per-other-agent block appended in LOCAL_PLUS_OTHER_AGENTS mode.
OTHER_AGENT_BLOCK_SIZE = 4
# Size of the frozen Environment API v1.0 observation vector.
BASE_OBSERVATION_SIZE = 12


def observation_size(num_agents: int, mode: str = LOCAL_ONLY) -> int:
    """Returns the observation vector length for a system configuration.

    Args:
        num_agents: Number of agents in the shared environment.
        mode: LOCAL_ONLY or LOCAL_PLUS_OTHER_AGENTS.

    Raises:
        ValueError: If the mode is unknown or num_agents < 1.
    """
    if mode not in OBSERVATION_MODES:
        raise ValueError(f"Unknown observation mode '{mode}'. Available: {list(OBSERVATION_MODES)}")
    if num_agents < 1:
        raise ValueError("num_agents must be >= 1.")
    if mode == LOCAL_ONLY:
        return BASE_OBSERVATION_SIZE
    return BASE_OBSERVATION_SIZE + OTHER_AGENT_BLOCK_SIZE * (num_agents - 1)


def build_observation(
    env,
    agent_id: int,
    num_agents: int,
    mode: str = LOCAL_ONLY,
    waiting_flags: Optional[Dict[int, bool]] = None,
) -> List[int]:
    """Builds the observation vector for `agent_id` in the given mode.

    Args:
        env: GridWorld instance (frozen API v1.0 only).
        agent_id: Agent to build the observation for.
        num_agents: Total number of agents in the system (must match env).
        mode: LOCAL_ONLY or LOCAL_PLUS_OTHER_AGENTS.
        waiting_flags: Optional {agent_id: requested_stay_last_step} map used
            to fill the wait_flag field of other-agent blocks.

    Returns:
        The observation vector (length given by observation_size()).

    Raises:
        ValueError: On unknown mode or env/num_agents mismatch.
    """
    base = env.get_agent_observation(agent_id)

    if mode == LOCAL_ONLY:
        return base
    if mode != LOCAL_PLUS_OTHER_AGENTS:
        raise ValueError(f"Unknown observation mode '{mode}'. Available: {list(OBSERVATION_MODES)}")

    all_states = env.get_all_states()
    if len(all_states) != num_agents:
        raise ValueError(
            f"Environment has {len(all_states)} agents but system was configured for {num_agents}."
        )

    waiting_flags = waiting_flags or {}
    vector = list(base)
    for other_id in sorted(all_states):
        if other_id == agent_id:
            continue
        state = all_states[other_id]
        pos = state["current_position"]
        goal = state["goal_position"]
        vector.extend(
            [
                goal[0] - pos[0],  # goal_delta_row (other agent)
                goal[1] - pos[1],  # goal_delta_col (other agent)
                1 if waiting_flags.get(other_id, False) else 0,
                1 if state["done"] else 0,
            ]
        )
    return vector
