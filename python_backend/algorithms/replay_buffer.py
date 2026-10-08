"""Uniform replay buffer (Phase 7).

Stores (state, action, reward, next_state, done) tuples and samples random
minibatches for DQN training.
"""

import random
from collections import deque
from typing import List, Optional, Tuple

import torch

Transition = Tuple[List[int], int, float, List[int], bool]


class ReplayBuffer:
    """Fixed-capacity FIFO buffer with uniform random sampling."""

    def __init__(self, capacity: int = 50000, seed: Optional[int] = None) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        """Stores one transition; oldest is evicted when full."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Returns a random minibatch (no replacement)."""
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}."
            )
        return self.rng.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
