"""Reward configuration dataclass for customizable environment feedback."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Configurable reward schedule. Hardcoded constants are forbidden."""

    goal: float = 100.0
    step: float = -1.0
    obstacle_collision: float = -10.0
    agent_collision: float = -20.0
    invalid_move: float = -10.0
    stay: float = -2.0