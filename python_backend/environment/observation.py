"""Observation vector generator for GridMind agents."""

from typing import Dict, List, Set
from python_backend.environment.constants import Action, ACTION_DELTAS, Position


class ObservationBuilder:
    """Constructs 12-element compact state representations."""

    @staticmethod
    def build_vector(
        agent_id: int,
        agent_positions: Dict[int, Position],
        goal_positions: Dict[int, Position],
        obstacles: Set[Position],
        grid_height: int,
        grid_width: int,
    ) -> List[int]:
        """Constructs an interpretable vector for a single agent.

        Indices:
        0: agent_row
        1: agent_col
        2: goal_row
        3: goal_col
        4: obstacle_up       (1 if obstacle or boundary, else 0)
        5: obstacle_down     (1 if obstacle or boundary, else 0)
        6: obstacle_left     (1 if obstacle or boundary, else 0)
        7: obstacle_right    (1 if obstacle or boundary, else 0)
        8: agent_up          (1 if other agent present, else 0)
        9: agent_down        (1 if other agent present, else 0)
        10: agent_left       (1 if other agent present, else 0)
        11: agent_right      (1 if other agent present, else 0)
        """
        curr = agent_positions[agent_id]
        goal = goal_positions[agent_id]

        def is_blocked(pos: Position) -> int:
            if not (0 <= pos.row < grid_height and 0 <= pos.col < grid_width):
                return 1
            return 1 if pos in obstacles else 0

        other_agent_positions = {pos for aid, pos in agent_positions.items() if aid != agent_id}

        def has_agent(pos: Position) -> int:
            return 1 if pos in other_agent_positions else 0

        pos_up = curr + ACTION_DELTAS[Action.UP]
        pos_down = curr + ACTION_DELTAS[Action.DOWN]
        pos_left = curr + ACTION_DELTAS[Action.LEFT]
        pos_right = curr + ACTION_DELTAS[Action.RIGHT]

        return [
            curr.row,
            curr.col,
            goal.row,
            goal.col,
            is_blocked(pos_up),
            is_blocked(pos_down),
            is_blocked(pos_left),
            is_blocked(pos_right),
            has_agent(pos_up),
            has_agent(pos_down),
            has_agent(pos_left),
            has_agent(pos_right),
        ]