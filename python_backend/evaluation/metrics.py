"""Metric collection for GridMind experiments (Developer 4, Phase 8).

Everything here consumes ONLY the frozen Environment API v1.0 plus the
existing policy classes (RandomPolicy / QLearningAgent / DQNAgent) and
MultiAgentSystem. No environment internals are touched and no metric is
invented: every number is measured from a real episode.

Metric definitions (documented identically in ``docs/experiment_protocol.md``):

  success          The agent reached its goal (``done`` becomes True). For a
                   multi-agent episode ``success`` is reported both as
                   ``all_success`` (every agent reached its goal) and
                   ``any_success``.
  reward           Sum of TRUE environment rewards (never the cooperative/
                   learned reward used for training).
  steps            Number of ``env.step`` calls in the episode, including the
                   step that ended it.
  collisions       One count per (agent, step) entry reported by the
                   environment in ``info["collisions"]``: BOUNDARY, OBSTACLE,
                   SAME_CELL, SWAP. A same-cell conflict therefore counts once
                   per involved agent, exactly as the environment reports it.
  deadlocks        Deadlock events fired by ``DeadlockDetector`` (multi-agent
                   only; single-agent episodes always report 0 = not
                   applicable).
  distance         Manhattan distance travelled by an agent, summed over the
                   episode. Total distance is the sum over agents.
  messages         Communication messages sent (0 when communication is off).
  training time    Wall-clock seconds spent training a method, measured with
                   ``time.perf_counter`` around the training call only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from python_backend.environment.constants import Position

Policy = Callable[..., int]


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------


@dataclass
class EpisodeMetrics:
    """Metrics for one episode of one algorithm configuration."""

    episode: int = 0
    steps: int = 0
    reward: float = 0.0
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    collisions: int = 0
    collision_types: Dict[str, int] = field(default_factory=dict)
    deadlocks: int = 0
    recoveries: int = 0
    distance: int = 0
    messages: int = 0
    per_agent_reward: Dict[int, float] = field(default_factory=dict)
    per_agent_distance: Dict[int, int] = field(default_factory=dict)
    per_agent_success: Dict[int, bool] = field(default_factory=dict)
    per_agent_collisions: Dict[int, int] = field(default_factory=dict)
    goals_reached: int = 0
    num_agents: int = 1
    layout_seed: Optional[int] = None
    scenario: str = ""

    @property
    def goal_completion_rate(self) -> float:
        """Fraction of agents that reached their goal in this episode."""
        return self.goals_reached / self.num_agents if self.num_agents else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view (agent-keyed maps become string keys)."""
        return {
            "episode": self.episode,
            "steps": self.steps,
            "reward": round(self.reward, 4),
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "collisions": self.collisions,
            "collision_types": dict(self.collision_types),
            "deadlocks": self.deadlocks,
            "recoveries": self.recoveries,
            "distance": self.distance,
            "messages": self.messages,
            "goals_reached": self.goals_reached,
            "goal_completion_rate": round(self.goal_completion_rate, 4),
            "num_agents": self.num_agents,
            "layout_seed": self.layout_seed,
            "scenario": self.scenario,
            "per_agent_reward": {str(k): round(v, 4) for k, v in self.per_agent_reward.items()},
            "per_agent_distance": {str(k): v for k, v in self.per_agent_distance.items()},
            "per_agent_success": {str(k): v for k, v in self.per_agent_success.items()},
            "per_agent_collisions": {str(k): v for k, v in self.per_agent_collisions.items()},
        }


@dataclass
class AggregateMetrics:
    """Mean/std metrics over a set of episodes (one algorithm configuration)."""

    label: str
    episodes: int
    num_agents: int
    success_rate: float = 0.0
    success_rate_std: float = 0.0
    goal_completion_rate: float = 0.0
    avg_reward: float = 0.0
    reward_std: float = 0.0
    avg_steps: float = 0.0
    avg_collisions: float = 0.0
    avg_collision_rate: float = 0.0
    avg_collision_types: Dict[str, float] = field(default_factory=dict)
    avg_deadlocks: float = 0.0
    avg_distance: float = 0.0
    avg_messages: float = 0.0
    training_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view (rounded for stable diffs between runs)."""
        return {
            "label": self.label,
            "episodes": self.episodes,
            "num_agents": self.num_agents,
            "success_rate": round(self.success_rate, 4),
            "success_rate_std": round(self.success_rate_std, 4),
            "goal_completion_rate": round(self.goal_completion_rate, 4),
            "avg_reward": round(self.avg_reward, 4),
            "reward_std": round(self.reward_std, 4),
            "avg_steps": round(self.avg_steps, 4),
            "avg_collisions": round(self.avg_collisions, 4),
            "avg_collision_rate": round(self.avg_collision_rate, 4),
            "avg_collision_types": {k: round(v, 4) for k, v in self.avg_collision_types.items()},
            "avg_deadlocks": round(self.avg_deadlocks, 4),
            "avg_distance": round(self.avg_distance, 4),
            "avg_messages": round(self.avg_messages, 4),
            "training_seconds": round(self.training_seconds, 4),
            "evaluation_seconds": round(self.evaluation_seconds, 4),
            "computation_seconds": round(self.training_seconds + self.evaluation_seconds, 4),
            "conditions": dict(self.conditions),
        }


# ---------------------------------------------------------------------------
# Collector (Phase 8)
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Accumulates episode metrics for one algorithm configuration.

    Usage::

        collector = MetricsCollector(label="dqn", num_agents=1)
        with collector.measure_training():
            ...            # train the model
        collector.add(episode_metrics)
        summary = collector.aggregate()
    """

    def __init__(
        self,
        label: str,
        num_agents: int = 1,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.label = label
        self.num_agents = num_agents
        self.conditions: Dict[str, Any] = dict(conditions or {})
        self.episodes: List[EpisodeMetrics] = []
        self.training_seconds: float = 0.0

    # -- collection ---------------------------------------------------------

    def add(self, episode: EpisodeMetrics) -> EpisodeMetrics:
        """Records one episode."""
        self.episodes.append(episode)
        return episode

    def extend(self, episodes: Sequence[EpisodeMetrics]) -> None:
        """Records several episodes."""
        self.episodes.extend(episodes)

    class _TrainingTimer:
        def __init__(self, collector: "MetricsCollector") -> None:
            self.collector = collector
            self.started = 0.0

        def __enter__(self) -> "MetricsCollector._TrainingTimer":
            import time

            self.started = time.perf_counter()
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            import time

            self.collector.training_seconds += time.perf_counter() - self.started

    def measure_training(self) -> "MetricsCollector._TrainingTimer":
        """Context manager that adds wall-clock time to ``training_seconds``."""
        return MetricsCollector._TrainingTimer(self)

    # -- aggregation --------------------------------------------------------

    def aggregate(self) -> AggregateMetrics:
        """Computes mean/std metrics over all collected episodes."""
        n = len(self.episodes)
        summary = AggregateMetrics(
            label=self.label,
            episodes=n,
            num_agents=self.num_agents,
            training_seconds=self.training_seconds,
            evaluation_seconds=float(self.conditions.get("evaluation_seconds", 0.0)),
            conditions=dict(self.conditions),
        )
        if n == 0:
            return summary

        successes = [1.0 if e.success else 0.0 for e in self.episodes]
        rewards = [e.reward for e in self.episodes]
        completion = [e.goal_completion_rate for e in self.episodes]

        summary.success_rate = _mean(successes)
        summary.success_rate_std = _std(successes)
        summary.goal_completion_rate = _mean(completion)
        summary.avg_reward = _mean(rewards)
        summary.reward_std = _std(rewards)
        summary.avg_steps = _mean([float(e.steps) for e in self.episodes])
        summary.avg_collisions = _mean([float(e.collisions) for e in self.episodes])
        # Collision rate = collision events per agent per step.
        summary.avg_collision_rate = _mean(
            [e.collisions / max(1, e.num_agents * e.steps) for e in self.episodes]
        )
        summary.avg_deadlocks = _mean([float(e.deadlocks) for e in self.episodes])
        summary.avg_distance = _mean([float(e.distance) for e in self.episodes])
        summary.avg_messages = _mean([float(e.messages) for e in self.episodes])

        type_totals: Dict[str, float] = {}
        for episode in self.episodes:
            for name, count in episode.collision_types.items():
                type_totals[name] = type_totals.get(name, 0.0) + float(count)
        summary.avg_collision_types = {k: v / n for k, v in type_totals.items()}
        summary.conditions = {
            k: v for k, v in self.conditions.items() if k != "episodes"
        } | {"episodes": n}
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Full JSON-serializable view: aggregate plus per-episode records."""
        return {
            "aggregate": self.aggregate().to_dict(),
            "episodes": [e.to_dict() for e in self.episodes],
        }


# ---------------------------------------------------------------------------
# Recorded single-agent episode execution
# ---------------------------------------------------------------------------


def run_single_agent_episode(
    env,
    agent_id: int,
    policy: Policy,
    greedy: bool = True,
    max_steps: Optional[int] = None,
    episode: int = 0,
    scenario: str = "",
    layout_seed: Optional[int] = None,
) -> EpisodeMetrics:
    """Runs one episode with one policy-controlled agent and records metrics.

    Actions are restricted to ``env.get_valid_actions(agent_id)`` and the
    episode stops at the first per-agent termination (goal) or truncation
    (``max_steps``), matching ``algorithms/trainer.py`` accounting so numbers
    stay comparable with Developer 2's harness.

    The agent is expected to be the only *active* agent: other agents, if
    present, must be terminal (they are held in place by the environment).
    """
    valid_ids = set(env.agents.keys())
    if agent_id not in valid_ids:
        raise ValueError(f"agent_id {agent_id} is not in the environment: {sorted(valid_ids)}")

    limit = max_steps if max_steps is not None else env.max_steps
    obs = env.get_agent_observation(agent_id)

    metrics = EpisodeMetrics(
        episode=episode, num_agents=1, scenario=scenario, layout_seed=layout_seed
    )
    previous = env.get_agent_position(agent_id)
    total_reward = 0.0
    distance = 0

    for _ in range(limit):
        if env.get_agent_state(agent_id)["done"]:
            break

        valid = env.get_valid_actions(agent_id)
        action = policy(obs, valid, greedy=greedy)
        next_obs, rewards, dones, terminated, info = env.step({agent_id: action})

        total_reward += float(rewards[agent_id])
        metrics.steps += 1
        position = env.get_agent_position(agent_id)
        distance += _manhattan(previous, position)
        previous = position
        _record_collisions(metrics, info.get("collisions", {}))

        obs = next_obs[agent_id]
        if dones[agent_id]:
            metrics.success = True
            metrics.terminated = True
            break
        if info.get("truncated", False):
            metrics.truncated = True
            break

    metrics.reward = total_reward
    metrics.distance = distance
    metrics.per_agent_reward = {agent_id: total_reward}
    metrics.per_agent_distance = {agent_id: distance}
    metrics.per_agent_success = {agent_id: metrics.success}
    metrics.goals_reached = 1 if metrics.success else 0
    return metrics


def run_single_agent_evaluation(
    collector: MetricsCollector,
    env,
    agent_id: int,
    policy: Policy,
    episodes: int,
    random_reset=None,
    layout_seed_start: Optional[int] = None,
    scenario: str = "",
) -> MetricsCollector:
    """Runs ``episodes`` recorded episodes, optionally on seeded random layouts.

    Args:
        random_reset: When given, called with the episode index and must return
            ``env.reset`` kwargs for that episode (e.g. the frozen suite's
            seeded random layout). When None the environment is reset plainly
            (deterministic fixed scenario).
        layout_seed_start: Recorded in the metrics for reproducibility when
            ``random_reset`` is None.
    """
    for index in range(episodes):
        if random_reset is not None:
            kwargs = random_reset(index)
            env.reset(**kwargs)
            layout_seed = kwargs.get("seed")
        else:
            env.reset()
            layout_seed = (
                layout_seed_start + index if layout_seed_start is not None else None
            )
        collector.add(
            run_single_agent_episode(
                env,
                agent_id=agent_id,
                policy=policy,
                greedy=True,
                episode=index,
                scenario=scenario,
                layout_seed=layout_seed,
            )
        )
    return collector


# ---------------------------------------------------------------------------
# Multi-agent distance recording (delegating wrapper, never mutates the env)
# ---------------------------------------------------------------------------


class PositionRecorder:
    """Transparent read-only proxy around a GridWorld that records positions.

    ``MultiAgentSystem`` (Developer 3) owns the multi-agent episode loop and
    reports reward/steps/collisions/deadlocks/messages but not the geometric
    path length. Wrapping the environment lets the evaluation module measure
    distance travelled WITHOUT modifying or duplicating that module: every
    attribute and method call is delegated to the real environment, and the
    frozen Environment API v1.0 is untouched.
    """

    def __init__(self, env) -> None:
        self.__dict__["_env"] = env
        self.__dict__["frames"] = []

    # -- delegation ---------------------------------------------------------

    def __getattr__(self, item: str) -> Any:
        if item in ("_env", "frames"):
            raise AttributeError(item)
        return getattr(self.__dict__["_env"], item)

    # -- recording ----------------------------------------------------------

    def _snapshot(self) -> Dict[int, Position]:
        env = self.__dict__["_env"]
        return {aid: env.get_agent_position(aid) for aid in env.agents}

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Delegates reset and starts a new position trace."""
        result = self.__dict__["_env"].reset(*args, **kwargs)
        self.__dict__["frames"] = [self._snapshot()]
        return result

    def step(self, actions: Dict[int, int]) -> Any:
        """Delegates step and appends the resulting positions to the trace."""
        result = self.__dict__["_env"].step(actions)
        self.__dict__["frames"].append(self._snapshot())
        return result

    # -- metrics ------------------------------------------------------------

    def distances(self) -> Dict[int, int]:
        """Per-agent Manhattan distance travelled across the recorded trace."""
        frames: List[Dict[int, Position]] = self.__dict__["frames"]
        totals: Dict[int, int] = {}
        for before, after in zip(frames, frames[1:]):
            for aid, pos in after.items():
                previous = before.get(aid)
                if previous is None:
                    continue
                totals[aid] = totals.get(aid, 0) + _manhattan(previous, pos)
        return totals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_collisions(metrics: EpisodeMetrics, collisions: Dict[int, str]) -> None:
    """Adds environment collision reports to the episode metrics."""
    for _agent_id, type_name in collisions.items():
        metrics.collisions += 1
        metrics.collision_types[type_name] = metrics.collision_types.get(type_name, 0) + 1


def _manhattan(a: Position, b: Position) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
