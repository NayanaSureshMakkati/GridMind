"""Logical agent state entity.

Maintains spatial, step, and cumulative reward bookkeeping without any policy logic.
"""

from python_backend.environment.constants import Position


class Agent:
    """Represents an agent inside the discrete grid environment."""

    def __init__(self, agent_id: int, start_position: Position, goal_position: Position) -> None:
        self.agent_id: int = agent_id
        self.start_position: Position = start_position
        self.current_position: Position = start_position
        self.goal_position: Position = goal_position
        self.total_reward: float = 0.0
        self.steps: int = 0
        self.done: bool = False

    def reset(self, start_position: Position | None = None, goal_position: Position | None = None) -> None:
        """Reset agent metrics and position for a new episode."""
        if start_position is not None:
            self.start_position = start_position
        if goal_position is not None:
            self.goal_position = goal_position

        self.current_position = self.start_position
        self.total_reward = 0.0
        self.steps = 0
        self.done = False

    def record_step(self, new_position: Position, reward: float, reached_goal: bool) -> None:
        """Update step counts, position, and cumulative rewards."""
        self.current_position = new_position
        self.total_reward += reward
        self.steps += 1
        if reached_goal:
            self.done = True