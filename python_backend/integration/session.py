"""Live simulation session for the Python <-> Unity link (Phases 2/3/5).

A :class:`LiveSession` owns ONE frozen ``GridWorld`` plus the policy that decides
actions, and exposes a step-at-a-time API so Unity can display a running
simulation:

    session.reset_episode()
    session.step_once()                  # Python decides the actions (policy mode)
    session.step_once({0: 3, 1: 2})      # client supplies the actions (external mode)
    session.state_message()              # authoritative state for the 3D view

Python stays the single source of truth: positions, goals, obstacles, rewards,
collisions, deadlocks and episode termination are all produced here and only
visualized by Unity (Master Context 47).

Single-agent methods (``random`` / ``q_learning`` / ``dqn``) drive agent 0 with
the existing policy classes. Multi-agent methods step through the SAME pipeline
order as ``MultiAgentSystem.run_episode`` (Developer 3): observations -> action
selection -> rule-based arbitration -> deadlock recovery -> communication ->
``env.step`` -> collision/deadlock bookkeeping -> waiting-time update. The
coordination, deadlock, communication and observation logic itself is REUSED
from ``python_backend/multi_agent/`` — only the per-step orchestration lives
here, and ``tests/test_integration.py`` asserts that a live-stepped episode ends
identically to ``MultiAgentSystem.run_episode`` for the same configuration, so
the two can never drift apart silently.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from python_backend.environment.constants import ACTION_DELTAS, Action
from python_backend.environment.grid_world import GridWorld
from python_backend.evaluation import model_store
from python_backend.evaluation.experiments import (
    CTDE_INSPIRED,
    DQN,
    INDEPENDENT_DQN,
    METHOD_LABELS,
    Q_LEARNING,
    RANDOM,
    SINGLE_AGENT_METHODS,
    build_agent_dqn_config,
    build_q_learning_config,
    method_family,
    multi_agent_config_kwargs,
)
from python_backend.evaluation.scenarios import DEMO_SCENARIO, Scenario, load_scenario
from python_backend.integration import protocol

ACTION_SOURCES = ("policy", "external")


@dataclass
class SessionConfig:
    """Everything a live session needs, all explicit (no hidden defaults)."""

    algorithm: str = INDEPENDENT_DQN
    scenario_name: str = DEMO_SCENARIO
    num_agents: int = 3
    seed: int = 7
    action_source: str = "policy"
    speed_ms: int = 250
    episode_limit: int = 3
    train_episodes_for_decay: int = 400
    model_path: Optional[str] = None      # single-agent checkpoint (.pt / .pkl)
    model_tag: Optional[str] = None       # multi-agent checkpoint tag
    communication_enabled: bool = False
    coordination_mode: Optional[str] = None  # override for live demos

    def __post_init__(self) -> None:
        if self.algorithm not in METHOD_LABELS:
            raise ValueError(
                f"Unknown algorithm '{self.algorithm}'. Available: {list(METHOD_LABELS)}"
            )
        if self.action_source not in ACTION_SOURCES:
            raise ValueError(f"action_source must be one of {list(ACTION_SOURCES)}")
        if self.speed_ms < 0:
            raise ValueError("speed_ms must be >= 0")
        if self.episode_limit < 1:
            raise ValueError("episode_limit must be >= 1")
        if method_family(self.algorithm) == "single_agent" and self.num_agents != 1:
            # Single-agent algorithms cannot control several agents; the demo
            # visualizes agent 0 only (documented behaviour, not a silent change).
            self.num_agents = 1


@dataclass
class LiveStep:
    """Result of one live step (feeds ``step_result`` and session bookkeeping)."""

    step: int
    rewards: Dict[int, float]
    collisions: List[Dict[str, Any]]
    goals_reached: List[int]
    deadlocks_fired: bool
    terminated: bool
    truncated: bool
    actions: Dict[int, int]
    messages: int = 0
    movements: List[Dict[str, Any]] = field(default_factory=list)


class LiveSession:
    """Step-at-a-time simulation bound to one environment and one policy."""

    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        scenario: Optional[Scenario] = None,
    ) -> None:
        self.config = config or SessionConfig()
        self.scenario = scenario or load_scenario(self.config.scenario_name)
        self.num_agents = self.config.num_agents
        self.family = method_family(self.config.algorithm)
        self.env: GridWorld = self.scenario.with_num_agents(self.num_agents).build_env()

        self.system = None          # MultiAgentSystem for multi-agent methods
        self.agent = None           # policy object for single-agent methods
        self.trained = False
        self._build_policy()

        # Coordination/deadlock/communication helpers (multi-agent only)
        self._waiting_times: Dict[int, int] = {}
        self._waiting_flags: Dict[int, bool] = {}
        self._prev_positions: Dict[int, Any] = {}
        self._deadlock_fired_last_step = False

        # Session bookkeeping for the UI
        self.episode_index = 0
        self.step_index = 0
        self.episodes_completed = 0
        self.successes = 0
        self.total_collisions = 0
        self.total_deadlocks = 0
        self.total_messages = 0
        self.episode_rewards: Dict[int, float] = {}
        self.episode_collisions = 0
        self.episode_deadlocks = 0
        self.episode_messages = 0
        self.episode_distance = 0
        self.history: List[Dict[str, Any]] = []
        self.last_step: Optional[LiveStep] = None
        self.finished = False
        # Start in a clean, fully specified episode 1 state so the very first
        # ``state`` message already reflects a reset environment.
        self.reset_episode(increment=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_policy(self) -> None:
        """Instantiates the policy objects (and loads checkpoints when given)."""
        algorithm = self.config.algorithm
        seed = self.config.seed
        budget = self.config.train_episodes_for_decay

        if self.family == "multi_agent":
            from python_backend.multi_agent.multi_agent_system import (
                MultiAgentConfig,
                MultiAgentSystem,
            )

            dqn_configs = {
                aid: build_agent_dqn_config(seed=seed + aid, episodes=budget)
                for aid in range(self.num_agents)
            }
            kwargs = multi_agent_config_kwargs(algorithm, self.num_agents, dqn_configs, seed)
            if self.config.communication_enabled:
                kwargs["communication_enabled"] = True
            if self.config.coordination_mode:
                kwargs["coordination_mode"] = self.config.coordination_mode
            self.system = MultiAgentSystem(self.env, MultiAgentConfig(**kwargs))
            self._waiting_times = {aid: 0 for aid in self.system.agents}
            self._waiting_flags = {aid: False for aid in self.system.agents}
            if self.config.model_tag:
                model_store.load_multi_agent(self.system, self.config.model_tag)
                self.trained = True
            return

        # ---- single-agent methods ----
        if algorithm == RANDOM:
            from python_backend.algorithms.random_policy import RandomPolicy

            self.agent = RandomPolicy(seed=seed)
            self.trained = True  # nothing to train: the baseline IS the policy
            return
        if algorithm == Q_LEARNING:
            if self.config.model_path:
                self.agent = model_store.load_q_learning(self.config.model_path)
                self.trained = True
            else:
                from python_backend.algorithms.q_learning import QLearningAgent

                self.agent = QLearningAgent(
                    build_q_learning_config(seed=seed, episodes=budget)
                )
            return
        if algorithm == DQN:
            if self.config.model_path:
                self.agent = model_store.load_dqn(self.config.model_path, agent_id=0, seed=seed)
                self.trained = True
            else:
                from python_backend.algorithms.dqn import DQNAgent

                self.agent = DQNAgent(
                    agent_id=0, config=build_agent_dqn_config(seed=seed, episodes=budget)
                )
            return
        raise ValueError(f"Unsupported single-agent algorithm '{algorithm}'.")

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def reset_episode(self, increment: bool = True) -> Dict[str, Any]:
        """Starts a new episode and returns the first state message.

        Args:
            increment: True when this reset starts a NEW episode (advances the
                episode counter); False for the very first reset of a session.
        """
        if increment:
            self.episode_index += 1
        else:
            self.episode_index = max(1, self.episode_index)
        self.step_index = 0
        self.episode_rewards = {}
        self.episode_collisions = 0
        self.episode_deadlocks = 0
        self.episode_messages = 0
        self.episode_distance = 0
        self._deadlock_fired_last_step = False
        self.last_step = None
        self.env.reset(seed=self.config.seed)
        self._prev_positions = {
            aid: self.env.get_agent_position(aid) for aid in self.env.agents
        }
        if self.system is not None:
            self._waiting_times = {aid: 0 for aid in self.system.agents}
            self._waiting_flags = {aid: False for aid in self.system.agents}
            self.system.communication.reset(full=False)
            self.system.deadlock_detector.reset()
            self.system.recovery_manager.reset()
        return self.state_message()

    def step_once(self, external_actions: Optional[Dict[int, int]] = None) -> LiveStep:
        """Performs one simultaneous environment step.

        Args:
            external_actions: Required in ``action_source="external"`` mode
                (the client supplies every active agent's action). Ignored in
                policy mode, where Python decides.

        Raises:
            RuntimeError: if the episode already ended (call ``reset_episode``).
            ValueError: if external actions are malformed or invalid.
        """
        if self.last_step and (self.last_step.terminated or self.last_step.truncated):
            raise RuntimeError("Episode finished: call reset_episode() before stepping again.")

        if self.config.action_source == "external":
            actions = self._resolve_external_actions(external_actions)
        else:
            actions = self._policy_actions()

        if self.system is not None:
            step = self._step_multi_agent(actions)
        else:
            step = self._step_single_agent(actions)

        self.step_index = step.step
        self.last_step = step
        self.episode_rewards = {
            aid: self.episode_rewards.get(aid, 0.0) + float(step.rewards.get(aid, 0.0))
            for aid in self.env.agents
        }
        self.episode_collisions += len(step.collisions)
        self.total_collisions += len(step.collisions)
        self.episode_messages += step.messages
        self.total_messages += step.messages
        if step.deadlocks_fired:
            self.episode_deadlocks += 1
            self.total_deadlocks += 1

        if step.terminated or step.truncated:
            self._finish_episode(step)
        return step

    def _policy_actions(self) -> Dict[int, int]:
        """Actions chosen by the configured policy for every active agent."""
        actions: Dict[int, int] = {}
        if self.system is not None:
            active = self._active_ids()
            for aid in active:
                observation = self._observation(aid)
                valid = self.env.get_valid_actions(aid)
                actions[aid] = int(self.system.agents[aid].choose_action(observation, valid, greedy=True))
            return actions

        aid = 0
        observation = self.env.get_agent_observation(aid)
        valid = self.env.get_valid_actions(aid)
        if self.config.algorithm == RANDOM:
            actions[aid] = int(self.agent.choose_action(observation, valid))
        else:
            actions[aid] = int(self.agent.choose_action(observation, valid, greedy=True))
        return actions

    def _resolve_external_actions(self, external: Optional[Dict[int, int]]) -> Dict[int, int]:
        """Validates client-supplied actions (external/manual mode)."""
        if not external:
            raise ValueError(
                "action_source='external' requires an 'actions' message with at least one agent."
            )
        active = self._active_ids()
        resolved: Dict[int, int] = {}
        for aid, action in external.items():
            if aid not in self.env.agents:
                raise ValueError(f"Action for unknown agent {aid}.")
            if aid not in active:
                continue  # terminal agents are held in place by the environment
            if action not in self.env.get_valid_actions(aid):
                raise ValueError(
                    f"Action {protocol.action_name(action)} is not valid for agent {aid} "
                    f"(position {self.env.get_agent_position(aid)})."
                )
            resolved[aid] = int(action)
        if not resolved:
            raise ValueError("No valid action supplied for any active agent.")
        return resolved

    # ------------------------------------------------------------------
    # Step internals (pipeline order mirrors MultiAgentSystem.run_episode)
    # ------------------------------------------------------------------

    def _step_single_agent(self, actions: Dict[int, int]) -> LiveStep:
        """Steps agent 0; other agents (there are none) are untouched."""
        aid = 0
        before = self.env.get_agent_position(aid)
        _, rewards, dones, terminated, info = self.env.step({aid: actions[aid]})
        after = self.env.get_agent_position(aid)
        self.episode_distance += abs(before.row - after.row) + abs(before.col - after.col)

        movements, collisions = self._build_step_events(
            actions, rewards, {aid: before}, {aid: after}, dict(info.get("collisions", {})),
        )
        goals_reached = [a for a, done in dones.items() if done]
        return LiveStep(
            step=info["step"], rewards={a: float(v) for a, v in rewards.items()},
            collisions=collisions, goals_reached=goals_reached, deadlocks_fired=False,
            terminated=bool(terminated), truncated=bool(info.get("truncated", False)),
            actions=dict(actions), messages=0, movements=movements,
        )

    def _step_multi_agent(self, actions: Dict[int, int]) -> LiveStep:
        """Steps all agents through Developer 3's pipeline order (no learning)."""
        cfg = self.system.config

        # 1) RULE-BASED COORDINATION pre-step yields (explicitly not learned)
        if cfg.coordination_mode == "rule_based" and len(actions) >= 2:
            from python_backend.multi_agent.coordination import RuleBasedArbiter

            arbiter = RuleBasedArbiter(self.env, self._waiting_times)
            actions, _events = arbiter.resolve(self.step_index + 1, actions)

        # 2) Deadlock recovery override (takes precedence over coordination)
        if self._deadlock_fired_last_step and self.system.recovery_manager.should_recover():
            actions = self.system.recovery_manager.recovery_actions(self.env, self._active_ids())

        # 3) Optional communication, AFTER arbitration so messages carry the
        #    final intended action (same ordering as the training loop)
        positions_now = {
            aid: (self.env.get_agent_position(aid).row, self.env.get_agent_position(aid).col)
            for aid in self._active_ids()
        }
        self.system.communication.broadcast_intents(
            self.step_index + 1,
            self._active_ids(),
            positions_now,
            actions,
            goal_reached={
                aid: bool(self.env.get_agent_state(aid)["done"]) for aid in self._active_ids()
            },
        )
        messages = int(self.system.communication.messages_per_step)

        # 4) Simultaneous environment step (frozen API v1.0 resolves conflicts)
        positions_before = {
            aid: self.env.get_agent_position(aid) for aid in self.env.agents
        }
        _, rewards, dones, terminated, info = self.env.step(actions)

        positions_after = {
            aid: self.env.get_agent_position(aid) for aid in self.env.agents
        }

        # 5) Collision analysis + deadlock detection (recorded, never hidden)
        collision_types = dict(info.get("collisions", {}))
        movements, collision_entries = self._build_step_events(
            actions, rewards, positions_before, positions_after, collision_types,
        )
        agent_collision_ids = [
            aid for aid, name in collision_types.items()
            if name in ("SAME_CELL", "SWAP")
        ]
        fired = self.system.deadlock_detector.record_step(
            info["step"], positions_after, dict(dones), agent_collision_ids, actions=dict(actions)
        )
        self._deadlock_fired_last_step = bool(fired)

        # 6) Waiting-time bookkeeping for the rule-based priority policy
        for aid in self.env.agents:
            previous = self._prev_positions.get(aid)
            if previous is not None and previous == positions_after[aid]:
                self._waiting_times[aid] = self._waiting_times.get(aid, 0) + 1
            else:
                self._waiting_times[aid] = 0
        self._prev_positions = positions_after
        self._waiting_flags = {aid: actions.get(aid) == int(Action.STAY) for aid in self.env.agents}

        return LiveStep(
            step=info["step"],
            rewards={aid: float(v) for aid, v in rewards.items()},
            collisions=collision_entries,
            goals_reached=[aid for aid, done in dones.items() if done],
            deadlocks_fired=bool(fired),
            terminated=bool(terminated),
            truncated=bool(info.get("truncated", False)),
            actions=dict(actions),
            messages=int(messages),
            movements=movements,
        )

    def _build_step_events(
        self,
        actions: Dict[int, int],
        rewards: Dict[int, float],
        before: Dict[int, Any],
        after: Dict[int, Any],
        collision_types: Dict[int, str],
    ):
        """Per-agent movement records + enriched collision records for one step.

        The attempted ``target`` cell is recomputed from the chosen action (the
        same proposal the frozen ConflictArbiter received); collisions always
        bounce the agent back, so ``position`` equals ``from`` there. ``other``
        is resolved geometrically: for SWAP the partner that crossed, for
        SAME_CELL the lowest other id on the contested cell, else -1.
        """
        # Attempted cells, recomputed from the chosen actions exactly as the
        # frozen ConflictArbiter proposes them. This is what attribution must
        # compare: a SWAP/SAME_CELL bounces BOTH agents back, so their final
        # positions stay unchanged.
        targets: Dict[int, Any] = {}
        for aid in self.env.agents:
            action = int(actions.get(aid, int(Action.STAY)))
            targets[aid] = before[aid] if action == int(Action.STAY) else (
                before[aid] + ACTION_DELTAS[Action(action)]
            )

        movements: List[Dict[str, Any]] = []
        for aid in sorted(self.env.agents):
            movements.append(protocol.movement_entry(
                agent_id=aid,
                action=int(actions.get(aid, int(Action.STAY))),
                from_pos=before[aid],
                to_pos=after[aid],
                target=targets[aid],
                moved=(before[aid] != after[aid]),
                reward=float(rewards.get(aid, 0.0)),
            ))

        collisions: List[Dict[str, Any]] = []
        for aid, type_name in sorted(collision_types.items()):
            other = -1
            if type_name == "SWAP":
                # The partner proposed to cross into my cell while I proposed
                # to enter theirs.
                for other_id in self.env.agents:
                    if other_id == aid:
                        continue
                    if targets[other_id] == before[aid] and targets[aid] == before[other_id]:
                        other = other_id
                        break
            elif type_name == "SAME_CELL":
                sharers = [
                    other_id for other_id in self.env.agents
                    if other_id != aid and targets[other_id] == targets[aid]
                ]
                if sharers:
                    other = min(sharers)
            collisions.append(protocol.collision_entry(
                aid, type_name, position=after[aid], target=targets[aid], other=other,
            ))
        return movements, collisions

    # ------------------------------------------------------------------
    # Observations & state
    # ------------------------------------------------------------------

    def _active_ids(self) -> List[int]:
        """Agents that are not finished yet (sorted for deterministic output)."""
        return sorted(
            aid for aid in self.env.agents if not self.env.get_agent_state(aid)["done"]
        )

    def _observation(self, agent_id: int) -> List[int]:
        """Observation for one agent in the configured multi-agent mode."""
        from python_backend.multi_agent.multi_agent_system import MultiAgentSystem
        from python_backend.multi_agent.observations import build_observation

        mode = self.system.config.observation_mode
        vector = build_observation(self.env, agent_id, self.num_agents, mode, self._waiting_flags)
        if self.system.config.ctde:
            return MultiAgentSystem.execution_observation(vector)
        return vector

    def agent_observation(self, agent_id: int) -> List[int]:
        """Observation vector for one agent (frozen 12-vector when single-agent)."""
        if self.system is None:
            return self.env.get_agent_observation(agent_id)
        return self._observation(agent_id)

    def state_message(self) -> Dict[str, Any]:
        """Authoritative ``state`` message for the 3D frontend and the UI."""
        agents: List[Dict[str, Any]] = []
        for aid in sorted(self.env.agents):
            position = self.env.get_agent_position(aid)
            goal = self.env.get_goal_position(aid)
            done = bool(self.env.get_agent_state(aid)["done"])
            agents.append({
                "id": aid,
                "position": protocol.position_list(position.row, position.col),
                "goal": protocol.position_list(goal.row, goal.col),
                "observation": self.agent_observation(aid),
                "status": "GOAL_REACHED" if done else "NAVIGATING",
                "done": done,
                "reward": round(float(self.episode_rewards.get(aid, 0.0)), 4),
            })
        return protocol.state_message(
            episode=self.episode_index,
            step=self.step_index,
            algorithm=self.config.algorithm,
            agents=agents,
            stats=self.stats(),
            action_source=self.config.action_source,
            max_steps=self.env.max_steps,
        )

    def stats(self) -> Dict[str, Any]:
        """Running session statistics shown in the Unity UI (Phase 7)."""
        return {
            "algorithm_label": METHOD_LABELS.get(self.config.algorithm, self.config.algorithm),
            "episode": self.episode_index,
            "episode_limit": self.config.episode_limit,
            "step": self.step_index,
            "max_steps": self.env.max_steps,
            "agents": self.num_agents,
            "collisions": self.total_collisions,
            "episode_collisions": self.episode_collisions,
            "deadlocks": self.total_deadlocks,
            "episode_deadlocks": self.episode_deadlocks,
            "messages": self.total_messages,
            "episode_messages": self.episode_messages,
            "distance": self.episode_distance,
            "episodes_completed": self.episodes_completed,
            "successes": self.successes,
            "success_rate": round(self.successes / self.episodes_completed, 4)
            if self.episodes_completed
            else 0.0,
            "total_reward": round(sum(self.episode_rewards.values()), 4),
            "trained": self.trained,
            "action_source": self.config.action_source,
        }

    def step_result_message(self, step: LiveStep) -> Dict[str, Any]:
        """``step_result`` message for one completed step."""
        return protocol.step_result_message(
            step=step.step,
            rewards=step.rewards,
            collisions=step.collisions,
            goals_reached=step.goals_reached,
            deadlocks=1 if step.deadlocks_fired else 0,
            terminated=step.terminated,
            truncated=step.truncated,
            movements=step.movements,
        )

    # ------------------------------------------------------------------
    # Episode completion
    # ------------------------------------------------------------------

    def _finish_episode(self, step: LiveStep) -> None:
        """Records the episode summary and decides whether the session is over."""
        successes = {aid: bool(self.env.get_agent_state(aid)["done"]) for aid in self.env.agents}
        success = all(successes.values())
        self.episodes_completed += 1
        if success:
            self.successes += 1
        record = {
            "episode": self.episode_index,
            "steps": step.step,
            "reward": round(sum(self.episode_rewards.values()), 4),
            "per_agent_reward": {str(k): round(v, 4) for k, v in self.episode_rewards.items()},
            "success": success,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "collisions": self.episode_collisions,
            "deadlocks": self.episode_deadlocks,
            "messages": self.episode_messages,
            "distance": self.episode_distance,
            "goals_reached": sum(1 for v in successes.values() if v),
            "goal_completion_rate": round(sum(1 for v in successes.values() if v) / self.num_agents, 4),
        }
        self.history.append(record)
        if self.episodes_completed >= self.config.episode_limit:
            self.finished = True

    def episode_end_message(self) -> Dict[str, Any]:
        """``episode_end`` message for the episode that just finished."""
        if not self.history:
            raise RuntimeError("No completed episode to summarize.")
        record = self.history[-1]
        return protocol.episode_end_message(
            episode=record["episode"],
            steps=record["steps"],
            terminated=record["terminated"],
            truncated=record["truncated"],
            success=record["success"],
            rewards={int(k): float(v) for k, v in record["per_agent_reward"].items()},
            collisions=record["collisions"],
            deadlocks=record["deadlocks"],
            messages=record["messages"],
            goal_completion_rate=record["goal_completion_rate"],
        )

    def scenario_message(self) -> Dict[str, Any]:
        """``scenario`` message (the world Unity must build)."""
        return protocol.scenario_message(
            self.scenario.to_message(self.num_agents)
        )

    def welcome_message(self) -> Dict[str, Any]:
        """``welcome`` message for a newly connected client."""
        return protocol.welcome_message(
            algorithm=self.config.algorithm,
            mode="live-simulation",
            action_source=self.config.action_source,
            num_agents=self.num_agents,
            episode=self.episode_index,
            extra={
                "algorithm_label": METHOD_LABELS.get(self.config.algorithm, self.config.algorithm),
                "scenario": self.scenario.name,
                "speed_ms": self.config.speed_ms,
                "trained": self.trained,
                "episode_limit": self.config.episode_limit,
            },
        )

    def close(self) -> None:
        """Releases environment resources."""
        self.env.close()
