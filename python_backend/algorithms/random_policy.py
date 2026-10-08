"""Random baseline policy (Phase 1).

Chooses uniformly among the agent's valid actions. Establishes the floor
performance that learned policies must beat.
"""

import random
from typing import List, Optional


class RandomPolicy:
    """Uniformly random policy over valid actions."""

    def __init__(self, n_actions: int = 5, seed: Optional[int] = None) -> None:
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def choose_action(
        self,
        observation: List[int],
        valid_actions: Optional[List[int]] = None,
        greedy: bool = False,
    ) -> int:
        """Returns a uniformly random valid action.

        `greedy` is accepted for interface compatibility with the learning
        agents but has no effect: a random policy has no greedy mode.
        """
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
        if not valid_actions:
            raise ValueError("valid_actions is empty; agent has no legal move.")
        return self.rng.choice(valid_actions)
