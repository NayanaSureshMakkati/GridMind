"""Deep Q-Network agent (Phases 6-10, 12).

Standard DQN with:
  - online network + periodically synchronized target network
  - uniform replay buffer
  - epsilon-greedy exploration over VALID actions
  - Adam optimizer, gradient clipping, MSE loss on 1-step TD targets

Action masking contract (documented limitation):
  - Action SELECTION is restricted to env.get_valid_actions(agent_id).
  - The TD bootstrap max_a' Q(s', a') is computed over ALL 5 actions
    (no masking inside the loss). Invalid actions rarely become greedy
    because selection is always masked, but the target values themselves
    may include invalid-action Q-values. Removing this discrepancy would
    require a mask channel in the replay buffer (kept spec-sized here).

Terminal handling:
  - done=True (goal reached) masks the bootstrap.
  - Time-limit truncation does NOT mask (bootstrapping continues) per
    Environment API v1.0 §7.
"""

import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from python_backend.algorithms.network import QNetwork
from python_backend.algorithms.replay_buffer import ReplayBuffer


# Reward scale for the LEARNER ONLY (see remember()). The environment reward
# schedule (+100 goal, -1 step, ...) is frozen API v1.0 and is never modified;
# metrics/evaluation always use the TRUE environment rewards.
#
# The scale is a monotonic rescaling of the TD target: it leaves the argmax of
# the TRUE Q-function unchanged, but the ABSOLUTE magnitude of the bootstrap
# target decides whether the ReLU network stays numerically stable. Measured on
# the 3-agent 10x10 demo (60-step episodes, gamma 0.99):
#   * scale=10.0 put TD targets ~O(10) and started a positive-feedback
#     Q-value blow-up - max|Q| grew 0.6 -> 1e9 over a 3000-episode run, after
#     which the greedy policy degenerated to all-STAY (train reward -339,
#     success 0.00) regardless of the start position.
#   * scale=100 shrank converged Q-differences below float32 argmax precision.
#   * scale=1.0 leaves targets ~O(1) and stays bounded: max|Q| plateaus at
#     ~1e2 and the greedy policy holds all 3 agents on their goals for the
#     whole run (reward +263).
# See python_backend/evaluation/experiments.py build_agent_dqn_config for the
# matching learning-rate change.
REWARD_SCALE = 1.0


class DQNConfig:
    """Configurable hyperparameters for DQN training."""

    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05,
        learning_rate: float = 5e-4,
        batch_size: int = 64,
        buffer_capacity: int = 50000,
        min_buffer_size: int = 500,
        target_update_frequency: int = 250,
        hidden_size: int = 128,
        observation_size: int = 12,
        seed: Optional[int] = None,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.min_buffer_size = min_buffer_size
        self.target_update_frequency = target_update_frequency
        self.hidden_size = hidden_size
        self.observation_size = observation_size
        self.seed = seed


class DQNAgent:
    """Reusable DQN agent; instantiate one per logical agent (Phase 12).

    Independent DQN = several DQNAgent instances sharing an environment but
    each with its own networks, buffer, and epsilon schedule. Coordination is
    explicitly out of scope (Developer 3).
    """

    def __init__(self, agent_id: int = 0, config: Optional[DQNConfig] = None) -> None:
        self.agent_id = agent_id
        self.config = config or DQNConfig()
        cfg = self.config

        seed = cfg.seed if cfg.seed is not None else random.randrange(2**31)
        self.torch_rng = random.Random(seed)  # python-level RNG for exploration
        torch.manual_seed(seed + 1 + agent_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_net = QNetwork(
            obs_size=cfg.observation_size, hidden_size=cfg.hidden_size
        ).to(self.device)
        self.target_net = QNetwork(
            obs_size=cfg.observation_size, hidden_size=cfg.hidden_size
        ).to(self.device)
        self.sync_target_network()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.learning_rate)
        # Huber (smooth L1) instead of MSE: with goal reward +100, TD errors can
        # be large; squared error produces exploding gradients and catastrophic
        # policy collapse (observed experimentally). Huber bounds the gradient.
        self.loss_fn = nn.HuberLoss(delta=1.0)

        # Buffer RNG is seeded from the agent seed for run-to-run reproducibility
        # (unseeded sampling made identical-config runs diverge experimentally).
        self.replay_buffer = ReplayBuffer(capacity=cfg.buffer_capacity, seed=seed + 2)
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon
        self.epsilon_decay = cfg.epsilon_decay
        self.epsilon_min = cfg.epsilon_min
        self.batch_size = cfg.batch_size
        self.train_step_count = 0

    # ---------- Action selection ----------

    def choose_action(
        self,
        observation: List[int],
        valid_actions: Optional[List[int]] = None,
        greedy: bool = False,
    ) -> int:
        """Epsilon-greedy selection restricted to valid actions.

        Exploration samples uniformly among valid actions; exploitation is
        the argmax of online Q over valid actions (ties -> lowest action id).
        """
        if valid_actions is None:
            valid_actions = list(range(5))
        if not valid_actions:
            raise ValueError("valid_actions is empty; agent has no legal move.")

        if not greedy and self.torch_rng.random() < self.epsilon:
            return self.torch_rng.choice(valid_actions)

        q_values = self._q_values(observation)
        best_action = valid_actions[0]
        best_value = q_values[best_action]
        for action in valid_actions[1:]:
            if q_values[action] > best_value:
                best_value = q_values[action]
                best_action = action
        return best_action

    def _q_values(self, observation: List[int]) -> List[float]:
        """Evaluates the online network on a single observation."""
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.online_net(obs).squeeze(0).tolist()

    # ---------- Learning ----------

    def remember(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        """Stores one transition in the replay buffer (reward scaled - see REWARD_SCALE)."""
        self.replay_buffer.push(state, action, reward / REWARD_SCALE, next_state, done)

    def train_step(self) -> Optional[float]:
        """Performs one gradient update on a sampled minibatch.

        Returns the TD loss value, or None if the buffer is still too small.
        """
        cfg = self.config
        if len(self.replay_buffer) < max(cfg.min_buffer_size, self.batch_size):
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q for the taken actions: Q(s).gather(a)
        q_current = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        # 1-step TD target with DOUBLE-DQN semantics: action selection (argmax)
        # from the online network, action evaluation from the target network.
        # Vanilla max-over-target-NET overestimates Q because the max operator
        # propagates positive bias through bootstrapping; on short episodes with
        # +100 goal reward this diverges run away (Q values doubling every ~100
        # episodes, experimentally observed). Double DQN is the standard fix and
        # uses the same online/target networks required by the phase spec.
        with torch.no_grad():
            next_online_q = self.online_net(next_states_t)
            best_next_actions = next_online_q.argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states_t).gather(1, best_next_actions).squeeze(1)
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % cfg.target_update_frequency == 0:
            self.sync_target_network()

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decays epsilon toward epsilon_min; call once per training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sync_target_network(self) -> None:
        """Copies online network weights into the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ---------- Persistence ----------

    def save_model(self, path: str, metadata: Optional[Dict] = None) -> None:
        """Saves online+target weights, epsilon, hyperparameters, and step count."""
        torch.save(
            {
                "format": "gridmind.dqn.v1",
                "algorithm": "dqn",
                "agent_id": self.agent_id,
                "epsilon": self.epsilon,
                "train_step_count": self.train_step_count,
                "online_net_state": self.online_net.state_dict(),
                "target_net_state": self.target_net.state_dict(),
                "hyperparameters": {
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_decay": self.epsilon_decay,
                    "epsilon_min": self.epsilon_min,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.batch_size,
                    "buffer_capacity": self.config.buffer_capacity,
                    "min_buffer_size": self.config.min_buffer_size,                "target_update_frequency": self.config.target_update_frequency,
                "hidden_size": self.config.hidden_size,
                "observation_size": self.config.observation_size,
                },
                "metadata": metadata or {},
            },
            path,
        )

    def load_model(self, path: str, load_replay_buffer: bool = False) -> None:
        """Restores networks, epsilon, and hyperparameters from a checkpoint.

        The replay buffer is not persisted by default (transient training
        state); pass load_replay_buffer=True to error if it is present.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format") != "gridmind.dqn.v1":
            raise ValueError(f"Not a GridMind DQN model file: {path}")

        self.online_net.load_state_dict(checkpoint["online_net_state"])
        self.target_net.load_state_dict(checkpoint["target_net_state"])
        self.epsilon = checkpoint["epsilon"]
        self.train_step_count = checkpoint["train_step_count"]

        hp = checkpoint["hyperparameters"]
        self.gamma = hp["gamma"]
        self.epsilon_decay = hp["epsilon_decay"]
        self.epsilon_min = hp["epsilon_min"]
        self.batch_size = hp["batch_size"]

        if load_replay_buffer:
            raise NotImplementedError("Replay buffer persistence is not supported.")
