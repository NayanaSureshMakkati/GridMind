"""Q-network architecture (Phase 6).

Small MLP: input(12) -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(5).
Output dimension is fixed at 5 (one Q-value per action id, Environment API §2).
The 12-element observation vector is used directly as network input (no
encoding needed - the network learns its own representation).
"""

import torch
import torch.nn as nn

OBSERVATION_SIZE = 12
NUM_ACTIONS = 5


class QNetwork(nn.Module):
    """Two-hidden-layer MLP mapping observations to per-action Q-values."""

    def __init__(
        self,
        obs_size: int = OBSERVATION_SIZE,
        hidden_size: int = 128,
        n_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q-values with shape (batch, n_actions)."""
        return self.net(x)
