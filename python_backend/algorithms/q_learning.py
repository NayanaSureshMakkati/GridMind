"""Tabular Q-learning agent (Phase 3).

Implements the standard update:

    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') * (1 - done) - Q(s,a)]

Terminal goals mask the bootstrap; time-limit truncation does NOT (the episode
was cut, not finished - see Environment API v1.0 §7). This agent only consumes
the frozen public API: it never touches GridWorld internals.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from python_backend.algorithms.state_encoding import get_encoder


class QLearningConfig:
    """Hyperparameter container for tabular Q-learning."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05,
        encoder: str = "relative",
        seed: Optional[int] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.encoder = encoder
        self.seed = seed


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning agent."""

    def __init__(self, config: Optional[QLearningConfig] = None) -> None:
        self.config = config or QLearningConfig()
        cfg = self.config
        self.alpha = cfg.learning_rate
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon
        self.epsilon_decay = cfg.epsilon_decay
        self.epsilon_min = cfg.epsilon_min
        self.encoder = get_encoder(cfg.encoder)
        self.rng = random.Random(cfg.seed)

        # Q-table: Dict[state_key, List[float]] with one entry per action id 0..4
        self.q_table: Dict[Tuple[int, ...], List[float]] = defaultdict(
            lambda: [0.0] * 5
        )

    # ---------- Action selection ----------

    def choose_action(
        self,
        observation: List[int],
        valid_actions: Optional[List[int]] = None,
        greedy: bool = False,
    ) -> int:
        """Epsilon-greedy action selection over valid actions.

        Exploration samples uniformly from valid_actions; exploitation picks
        the argmax of Q over valid_actions (ties broken by lowest action id).
        With greedy=True, exploration is disabled.
        """
        if valid_actions is None:
            valid_actions = list(range(5))
        if not valid_actions:
            raise ValueError("valid_actions is empty; agent has no legal move.")

        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)

        state_key = self.encoder(observation)
        q_values = self.q_table[state_key]
        best_value = max(q_values[a] for a in valid_actions)
        # Random tie-breaking among (near-)maximal actions. Deterministic
        # lowest-id tie-breaking creates degenerate loops (e.g. UP/DOWN 2-cycles)
        # when many unseen states share Q=0 - which starves exploration on
        # sparse-reward grids.
        tied = [a for a in valid_actions if q_values[a] >= best_value - 1e-9]
        return self.rng.choice(tied)

    # ---------- Learning ----------

    def update(
        self,
        observation: List[int],
        action: int,
        reward: float,
        next_observation: List[int],
        done: bool,
    ) -> None:
        """Applies one Q-learning update from a single transition.

        `done` must be the *goal* termination flag (per-agent dones[aid]),
        not the time-limit truncation flag.
        """
        state_key = self.encoder(observation)
        next_key = self.encoder(next_observation)

        target = reward
        if not done:
            target += self.gamma * max(self.q_table[next_key])

        current = self.q_table[state_key][action]
        self.q_table[state_key][action] = current + self.alpha * (target - current)

    def decay_epsilon(self) -> None:
        """Decays epsilon toward epsilon_min; call once per training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------- Introspection / persistence ----------

    def get_q_value(self, observation: List[int], action: int) -> float:
        """Returns Q(s, a) for the encoded state."""
        return self.q_table[self.encoder(observation)][action]

    def num_states_seen(self) -> int:
        """Number of distinct encoded states encountered so far."""
        return len(self.q_table)

    def save(self, path: str, metadata: Optional[Dict] = None) -> None:
        """Saves the Q-table and hyperparameters (pickle format)."""
        import pickle

        payload = {
            "format": "gridmind.q_learning.v1",
            "algorithm": "q_learning",
            "encoder": self.config.encoder,
            "hyperparameters": {
                "learning_rate": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
            },
            "epsilon": self.epsilon,
            "q_table": dict(self.q_table),
            "metadata": metadata or {},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        """Restores an agent (including epsilon) from a saved file."""
        import pickle

        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("format") != "gridmind.q_learning.v1":
            raise ValueError(f"Not a GridMind Q-learning model file: {path}")

        config = QLearningConfig(
            learning_rate=payload["hyperparameters"]["learning_rate"],
            gamma=payload["hyperparameters"]["gamma"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["hyperparameters"]["epsilon_decay"],
            epsilon_min=payload["hyperparameters"]["epsilon_min"],
            encoder=payload["encoder"],
        )
        agent = cls(config)
        agent.q_table = defaultdict(lambda: [0.0] * 5, payload["q_table"])
        return agent
