"""Constants and enumerations for the GridMind environment.

Defines actions, directions, and coordinate primitives.
"""

from enum import IntEnum
from typing import NamedTuple


class Action(IntEnum):
    """Discrete navigation actions available to each agent."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


class Position(NamedTuple):
    """Grid position expressed in (row, column) coordinates."""

    row: int
    col: int

    def __add__(self, other: "Position") -> "Position":  # type: ignore[override]
        return Position(self.row + other.row, self.col + other.col)


# Coordinate delta mapping: row increases downwards, column increases rightwards
ACTION_DELTAS = {
    Action.UP: Position(-1, 0),
    Action.DOWN: Position(1, 0),
    Action.LEFT: Position(0, -1),
    Action.RIGHT: Position(0, 1),
    Action.STAY: Position(0, 0),
}