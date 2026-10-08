"""Independent DQN system (Phases 12-13).

Several DQNAgent instances (one per logical agent) share ONE GridWorld but each
has its own observation, networks, replay buffer, and epsilon schedule. No
coordination whatsoever - Developer 3 owns that.

Per-agent semantics (documented for Developer 3):
  - Action selection uses each agent's OWN valid actions.
  - Transitions are stored per-agent: (own obs, own action, own reward,
    own next obs, own goal-done flag). Truncation does NOT mask bootstraps.
  - After an agent's goal is reached (dones[aid]=True) it is excluded from
    action selection (environment defaults its action to STAY) and stops
    learning (terminal reward is 0 forever per API v1.0 §4).
  - Environment `terminated` (all agents done) or `truncated` ends the episode
    for everyone.
"""

from typing import Dict, List, Optional

from python_backend.algorithms.dqn import DQNAgent, DQNConfig


class IndependentDQNSystem:
    """Runs N independent DQN agents in one shared GridWorld environment."""

    def __init__(
        self,
        num_agents: int,
        config: Optional[DQNConfig] = None,
        configs: Optional[Dict[int, DQNConfig]] = None,
    ) -> None:
        """One DQNAgent per logical agent; per-agent configs override the shared one."""
        self.num_agents = num_agents
        self.agents: Dict[int, DQNAgent] = {}
        for aid in range(num_agents):
            cfg = configs.get(aid, config) if configs else config
            self.agents[aid] = DQNAgent(agent_id=aid, config=cfg)

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    def run_episode(self, env, train: bool = True) -> Dict[str, object]:
        """Executes one simultaneous multi-agent episode.

        Returns per-agent and episode-level metrics. Learning updates happen
        inside this loop when `train=True`.
        """
        obs = env.reset()
        active = set(self.agents.keys())

        ep_reward = {aid: 0.0 for aid in self.agents}
        ep_steps = {aid: 0 for aid in self.agents}
        ep_collisions = {aid: 0 for aid in self.agents}
        successes = {aid: False for aid in self.agents}
        losses: List[float] = []

        while True:
            # 1) Collect all actions simultaneously (skip terminal agents;
            #    the environment defaults them to STAY).
            actions: Dict[int, int] = {}
            for aid in active:
                if env.get_agent_state(aid)["done"]:
                    continue
                valid = env.get_valid_actions(aid)
                actions[aid] = self.agents[aid].choose_action(obs[aid], valid)

            # 2) Simultaneous environment step
            next_obs, rewards, dones, terminated, info = env.step(actions)

            # 3) Per-agent transition storage + learning
            for aid in active:
                done = dones[aid]
                reward = rewards[aid]
                if train:
                    self.agents[aid].remember(
                        obs[aid], actions[aid], reward, next_obs[aid], done
                    )
                    loss = self.agents[aid].train_step()
                    if loss is not None:
                        losses.append(loss)

                ep_reward[aid] += reward
                ep_steps[aid] += 1
                if aid in info.get("collisions", {}):
                    ep_collisions[aid] += 1
                if done:
                    successes[aid] = True

            # 4) Episode end conditions
            if terminated or info.get("truncated", False):
                break

            # 5) Retire finished agents (they STAY; no further learning)
            obs = next_obs
            for aid in list(active):
                if dones[aid]:
                    active.discard(aid)
            if not active:
                break

        return {
            "terminated": terminated,
            "truncated": info.get("truncated", False),
            "steps": env.current_step,
            "reward": ep_reward,
            "steps_per_agent": ep_steps,
            "collisions": ep_collisions,
            "successes": successes,
            "all_success": all(successes.values()),
            "mean_loss": (sum(losses) / len(losses)) if losses else 0.0,
        }

    def train(self, env, episodes: int, seed: int = 0) -> List[Dict[str, object]]:
        """Trains all agents independently for `episodes` episodes."""
        history: List[Dict[str, object]] = []
        for episode in range(episodes):
            result = self.run_episode(env, train=True)
            result["episode"] = episode
            for agent in self.agents.values():
                agent.decay_epsilon()
            history.append(result)
        return history

    def evaluate(self, env, episodes: int = 50, seed: int = 0) -> Dict[str, object]:
        """Greedy evaluation over seeded random layouts."""
        totals = {
            "episodes": episodes,
            "all_success": 0,
            "any_success": 0,
            "reward": {aid: 0.0 for aid in self.agents},
            "steps": {aid: 0 for aid in self.agents},
            "collisions": {aid: 0 for aid in self.agents},
        }

        for i in range(episodes):
            env.reset(seed=1000 + i, randomize=True, num_agents=self.num_agents)
            # deterministic episode
            result = self.run_episode(env, train=False)
            totals["all_success"] += int(result["all_success"])
            totals["any_success"] += int(any(result["successes"].values()))
            for aid in self.agents:
                totals["reward"][aid] += result["reward"][aid]
                totals["steps"][aid] += result["steps_per_agent"][aid]
                totals["collisions"][aid] += result["collisions"][aid]

        n = episodes
        return {
            "num_agents": self.num_agents,
            "episodes": n,
            "all_success_rate": totals["all_success"] / n,
            "any_success_rate": totals["any_success"] / n,
            "avg_reward_per_agent": {aid: r / n for aid, r in totals["reward"].items()},
            "avg_collisions_per_agent": {aid: c / n for aid, c in totals["collisions"].items()},
        }

    # ------------------------------------------------------------------
    # Model management (Phase 10, applied per agent)
    # ------------------------------------------------------------------

    def save_models(self, directory: str, tag: str = "independent") -> None:
        """Saves each agent's checkpoint as models/<dir>/<tag>_agent<aid>.pt."""
        import os

        os.makedirs(directory, exist_ok=True)
        for aid, agent in self.agents.items():
            path = os.path.join(directory, f"{tag}_agent{aid}.pt")
            agent.save_model(path, metadata={"system": "independent_dqn", "num_agents": self.num_agents})

    def load_models(self, directory: str, tag: str = "independent") -> None:
        """Loads each agent's checkpoint from models/<dir>/<tag>_agent<aid>.pt."""
        import os

        for aid, agent in self.agents.items():
            path = os.path.join(directory, f"{tag}_agent{aid}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing checkpoint for agent {aid}: {path}")
            agent.load_model(path)
