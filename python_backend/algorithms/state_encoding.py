"""State encoding: converts environment observations into hashable keys.

The environment returns a 12-element integer observation vector (Environment
API v1.0 §3). Tabular Q-learning requires hashable state keys, so observations
are converted with one of the encoders below.

Encoders
--------
encode_full(obs):
    Uses the raw 12-tuple (agent_row, agent_col, goal_row, goal_col,
    8 sensor flags). State space grows with grid size x goal positions.
    Exact but does not generalize across different (start, goal) layouts.

encode_relative(obs):
    Replaces the four absolute position fields with the goal delta
    (goal_row - agent_row, goal_col - agent_col) and keeps the 8 sensor flags.
    Two states that "look the same" relative to the goal share a key, so a
    table learned on one layout transfers to others. State space is bounded
    by (2*height-1) x (2*width-1) x 2^8.

Both encoders return plain tuples of ints: hashable, picklable, deterministic.
"""

from typing import List, Tuple


def encode_full(obs: List[int]) -> Tuple[int, ...]:
    """Encodes the observation as the raw 12-element tuple."""
    return tuple(int(x) for x in obs)


def encode_relative(obs: List[int]) -> Tuple[int, ...]:
    """Encodes the observation as (goal_delta_row, goal_delta_col, *8 flags).

    Indices 0-3 (absolute agent/goal positions) are replaced by the relative
    goal vector; indices 4-11 (obstacle/agent flags) pass through unchanged.
    """
    delta_row = int(obs[2]) - int(obs[0])
    delta_col = int(obs[3]) - int(obs[1])
    return (delta_row, delta_col) + tuple(int(x) for x in obs[4:])


ENCODERS = {
    "full": encode_full,
    "relative": encode_relative,
}


def get_encoder(name: str):
    """Returns the encoder function registered under `name`."""
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(ENCODERS)}")
    return ENCODERS[name]
