"""Experiment management for GridMind (Developer 4, Phases 8-13).

Runs the comparisons required by the Master Context:

  Phase 9  methods:        random, q_learning, dqn, independent_dqn,
                           rule_based_coordination, coordinated_dqn,
                           ctde_inspired (optional)
  Phase 10 fairness:       one frozen suite (``experiments/scenarios/``) and
                           one frozen layout-seed sequence shared by EVERY
                           method; the same evaluation episode budget,
                           ``max_steps`` and greedy (deterministic) execution
                           for all of them.
  Phase 12 ablation:       independent vs rule-based vs coordinated (vs CTDE).
  Phase 13 scalability:    independent DQN at 1, 2, 3, 4 and 5 agents on the
                           same obstacle layout (agent prefix rule).

Reuse policy (no duplicated RL logic): training and multi-agent episode
execution are delegated to the existing modules —

  * ``algorithms/trainer.py``  -> train_q_learning / train_dqn  (Developer 2)
  * ``multi_agent/multi_agent_system.py`` -> MultiAgentSystem    (Developer 3)
  * ``evaluation/metrics.py``  -> recorded single-agent episodes + distance

Only the random baseline is implemented directly here (a uniform choice over
valid actions), matching ``algorithms/random_policy.py``.

No number in this module is hardcoded: every reported value comes from an
executed episode (Master Context 43/75).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from python_backend.evaluation import model_store
from python_backend.evaluation.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    MetricsCollector,
    PositionRecorder,
    run_single_agent_episode,
)
from python_backend.evaluation.scenarios import (
    DEMO_SCENARIO,
    EvaluationSuite,
    Scenario,
    load_scenario,
    load_suite,
    scenario_manifest,
    suite_manifest,
)

# ---------------------------------------------------------------------------
# Method catalogue
# ---------------------------------------------------------------------------

RANDOM = "random"
Q_LEARNING = "q_learning"
DQN = "dqn"
INDEPENDENT_DQN = "independent_dqn"
RULE_BASED = "rule_based_coordination"
COORDINATED_DQN = "coordinated_dqn"
CTDE_INSPIRED = "ctde_inspired"

SINGLE_AGENT_METHODS = (RANDOM, Q_LEARNING, DQN)
MULTI_AGENT_METHODS = (INDEPENDENT_DQN, RULE_BASED, COORDINATED_DQN, CTDE_INSPIRED)
ALL_METHODS = SINGLE_AGENT_METHODS + MULTI_AGENT_METHODS

#: Honest display labels (Master Context 28/33: never call a rule "learned").
METHOD_LABELS: Dict[str, str] = {
    RANDOM: "Random baseline",
    Q_LEARNING: "Tabular Q-learning",
    DQN: "DQN (single agent)",
    INDEPENDENT_DQN: "Independent DQN",
    RULE_BASED: "Rule-based coordination",
    COORDINATED_DQN: "Coordinated DQN (learned)",
    CTDE_INSPIRED: "CTDE-inspired DQN (optional)",
}

#: Conditions reported for every method (Phase 10).
FIXED_CONDITION = "fixed"
RANDOM_CONDITION = "random"


def method_family(method: str) -> str:
    """``"single_agent"`` or ``"multi_agent"`` for a method name."""
    if method in SINGLE_AGENT_METHODS:
        return "single_agent"
    if method in MULTI_AGENT_METHODS:
        return "multi_agent"
    raise ValueError(f"Unknown method '{method}'. Available: {list(ALL_METHODS)}")


def build_agent_dqn_config(seed: int, episodes: int, epsilon_min: float = 0.05):
    """DQN hyperparameters with a budget-independent epsilon schedule.

    ``epsilon_decay`` is derived so that epsilon reaches ``epsilon_min`` exactly
    after ``episodes`` episodes::

        decay = (epsilon_min / epsilon_start) ** (1 / episodes)

    The schedule is therefore identical across compared methods and run
    budgets, which keeps the softmax-free exploration budget fair (§43) instead
    of letting a different decay make one method look better.
    """
    from python_backend.algorithms.dqn import DQNConfig

    decay = (epsilon_min / 1.0) ** (1.0 / max(1, episodes))
    return DQNConfig(
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=decay,
        epsilon_min=epsilon_min,
        # 2e-4 (was 5e-4): at 5e-4 the TD target fed the positive-feedback
        # Q-value blow-up that REWARD_SCALE is sized for (measured max|Q|
        # 6e4 on the 3-agent demo); 2e-4 keeps Q bounded over long runs.
        learning_rate=2e-4,
        batch_size=64,
        buffer_capacity=50000,
        min_buffer_size=500,
        target_update_frequency=250,
        hidden_size=128,
        observation_size=12,
        seed=seed,
    )


def build_q_learning_config(seed: int, episodes: int, epsilon_min: float = 0.05):
    """Tabular Q-learning hyperparameters (same epsilon-schedule rule as DQN)."""
    from python_backend.algorithms.q_learning import QLearningConfig

    decay = (epsilon_min / 1.0) ** (1.0 / max(1, episodes))
    return QLearningConfig(
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=decay,
        epsilon_min=epsilon_min,
        encoder="relative",
        seed=seed,
    )


def multi_agent_config_kwargs(method: str, num_agents: int, dqn_configs, seed: int) -> Dict[str, Any]:
    """``MultiAgentConfig`` kwargs for a multi-agent method (Developer 3 API)."""
    from python_backend.multi_agent.observations import LOCAL_ONLY, LOCAL_PLUS_OTHER_AGENTS

    common: Dict[str, Any] = {
        "num_agents": num_agents,
        "policy_kind": "dqn",
        "dqn_configs": dqn_configs,
        "random_seed": seed,
    }
    if method == INDEPENDENT_DQN:
        return {**common, "coordination_mode": "none", "observation_mode": LOCAL_ONLY,
                "cooperative_weight": 0.0, "ctde": False}
    if method == RULE_BASED:
        # RULE-BASED COORDINATION: deterministic priority yields, NOT learned.
        return {**common, "coordination_mode": "rule_based", "observation_mode": LOCAL_ONLY,
                "cooperative_weight": 0.0, "ctde": False}
    if method == COORDINATED_DQN:
        # LEARNED coordination: enriched observations + cooperative reward term.
        return {**common, "coordination_mode": "none",
                "observation_mode": LOCAL_PLUS_OTHER_AGENTS,
                "cooperative_weight": 0.3, "ctde": False}
    if method == CTDE_INSPIRED:
        return {**common, "coordination_mode": "none",
                "observation_mode": LOCAL_PLUS_OTHER_AGENTS,
                "cooperative_weight": 0.3, "ctde": True}
    raise ValueError(f"'{method}' is not a multi-agent method.")


# ---------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Fully explicit experiment configuration (nothing hidden, §36/§56)."""

    suite: EvaluationSuite = field(default_factory=lambda: load_suite())
    scenario_name: str = DEMO_SCENARIO
    methods: Sequence[str] = ALL_METHODS
    num_agents: int = 3
    train_episodes: int = 400
    eval_episodes: Optional[int] = None  # None -> suite default (random block)
    fixed_eval_episodes: int = 1
    seed: int = 7
    save_models: bool = True
    experiment_name: str = "comparison"
    quick: bool = False

    def resolved_eval_episodes(self) -> int:
        """Evaluation episode count for the random condition."""
        if self.eval_episodes is not None:
            return self.eval_episodes
        if self.suite.random is not None:
            return self.suite.random.episodes
        return 10

    @property
    def scenario(self) -> Scenario:
        """The frozen fixed scenario for this run.

        Scenarios usually come from the evaluation suite. A run may also point
        at a frozen scenario file that the suite does not list (the Phase 13
        scalability layout), so a missing suite entry falls back to the file.
        """
        try:
            return self.suite.fixed(self.scenario_name)
        except KeyError:
            return load_scenario(self.scenario_name)

    def manifest(self) -> Dict[str, Any]:
        """Reproducibility block recorded with every result file."""
        return {
            "experiment": self.experiment_name,
            "methods": list(self.methods),
            "num_agents": self.num_agents,
            "train_episodes": self.train_episodes,
            "eval_episodes": self.resolved_eval_episodes(),
            "fixed_eval_episodes": self.fixed_eval_episodes,
            "seed": self.seed,
            "train_layout": "fixed (deterministic scenario)",
            "execution_policy": "greedy (epsilon-min for learning methods)",
            "rewards_used": "true environment rewards",
            "models_saved": self.save_models,
            "scenario": scenario_manifest(self.scenario),
            "suite": suite_manifest(self.suite),
        }


@dataclass
class TrainedMethod:
    """A trained (or untrained) method ready for evaluation."""

    method: str
    label: str
    num_agents: int
    family: str
    training_seconds: float
    payload: Any = None            # QLearningAgent / DQNAgent / MultiAgentSystem
    history: List[Dict[str, Any]] = field(default_factory=list)
    model_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentReport:
    """Rows plus their collectors, ready for tables, JSON and plots."""

    name: str
    manifest: Dict[str, Any]
    collectors: Dict[str, MetricsCollector] = field(default_factory=dict)
    rows: List[AggregateMetrics] = field(default_factory=list)
    history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def rows_for(self, condition: str) -> List[AggregateMetrics]:
        """Rows belonging to one condition (``fixed`` / ``random``)."""
        return [r for r in self.rows if r.conditions.get("condition") == condition]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable report."""
        return {
            "name": self.name,
            "manifest": self.manifest,
            "rows": [r.to_dict() for r in self.rows],
            "episodes": {
                key: collector.to_dict() for key, collector in self.collectors.items()
            },
        }


# ---------------------------------------------------------------------------
# Training (Phases 2-9): delegate to Developer 2 / Developer 3 loops
# ---------------------------------------------------------------------------


def train_method(config: ExperimentConfig, method: str) -> TrainedMethod:
    """Trains (or prepares) one method on the frozen fixed scenario.

    Every method trains on the SAME deterministic scenario with the same episode
    budget (fair conditions, §43). Random has nothing to learn.
    """
    family = method_family(method)
    scenarios = config.scenario
    num_agents = 1 if family == "single_agent" else config.num_agents
    scenario = scenarios.with_num_agents(num_agents)
    env = scenario.build_env()

    if method == RANDOM:
        return TrainedMethod(
            method=method, label=METHOD_LABELS[method], num_agents=1, family=family,
            training_seconds=0.0,
            metadata={
                "note": "no training: uniform over valid actions",
                "seed": config.seed,
            },
        )

    if method == Q_LEARNING:
        from python_backend.algorithms.trainer import train_q_learning

        agent = _new_q_learning_agent(config)
        started = time.perf_counter()
        history = train_q_learning(env, agent, episodes=config.train_episodes, seed=config.seed)
        elapsed = time.perf_counter() - started
        paths = _save_q_learning(agent, config) if config.save_models else []
        return TrainedMethod(
            method=method, label=METHOD_LABELS[method], num_agents=1, family=family,
            training_seconds=elapsed, payload=agent, history=history, model_paths=paths,
            metadata={"encoder": agent.config.encoder, "q_states_seen": agent.num_states_seen},
        )

    if method == DQN:
        from python_backend.algorithms.trainer import train_dqn

        agent = _new_dqn_agent(config)
        started = time.perf_counter()
        history = train_dqn(env, agent, episodes=config.train_episodes, seed=config.seed)
        elapsed = time.perf_counter() - started
        paths = _save_dqn(agent, config) if config.save_models else []
        return TrainedMethod(
            method=method, label=METHOD_LABELS[method], num_agents=1, family=family,
            training_seconds=elapsed, payload=agent, history=history, model_paths=paths,
            metadata={"epsilon_decay": round(agent.epsilon_decay, 6), "epsilon": agent.epsilon},
        )

    # ---- multi-agent methods: Developer 3's system owns the episode loop ----
    from python_backend.multi_agent.multi_agent_system import MultiAgentConfig, MultiAgentSystem

    dqn_configs = {
        aid: build_agent_dqn_config(seed=config.seed + aid, episodes=config.train_episodes)
        for aid in range(config.num_agents)
    }
    system = MultiAgentSystem(
        env,
        MultiAgentConfig(**multi_agent_config_kwargs(method, config.num_agents, dqn_configs, config.seed)),
    )
    started = time.perf_counter()
    results = system.train(config.train_episodes)
    elapsed = time.perf_counter() - started

    history = [
        {
            "episode": r.episode,
            "reward": sum(r.reward.values()),
            "steps": r.steps,
            "success": r.all_success,
            "collisions": r.collision_totals.obstacle + r.collision_totals.agent,
            "deadlocks": r.deadlocks,
            "messages": r.messages_per_episode,
        }
        for r in results
    ]
    paths = _save_multi_agent(system, config) if config.save_models else []
    return TrainedMethod(
        method=method, label=METHOD_LABELS[method], num_agents=config.num_agents,
        family=family, training_seconds=elapsed, payload=system, history=history,
        model_paths=paths,
        metadata={
            "system_label": system.system_label(),
            "coordination_mode": system.config.coordination_mode,
            "observation_mode": system.config.observation_mode,
            "cooperative_weight": system.config.cooperative_weight,
            "ctde": system.config.ctde,
        },
    )


def _new_q_learning_agent(config: ExperimentConfig):
    from python_backend.algorithms.q_learning import QLearningAgent

    return QLearningAgent(build_q_learning_config(seed=config.seed, episodes=config.train_episodes))


def _new_dqn_agent(config: ExperimentConfig):
    from python_backend.algorithms.dqn import DQNAgent

    return DQNAgent(
        agent_id=0,
        config=build_agent_dqn_config(seed=config.seed, episodes=config.train_episodes),
    )


def _save_q_learning(agent, config: ExperimentConfig) -> List[str]:
    path = model_store.save_q_learning(
        agent, f"{config.scenario_name}_q_learning",
        metadata={
            "scenario": config.scenario_name, "seed": config.seed,
            "train_episodes": config.train_episodes, "encoder": agent.config.encoder,
        },
    )
    return [path]


def _save_dqn(agent, config: ExperimentConfig) -> List[str]:
    path = model_store.save_dqn(
        agent, f"{config.scenario_name}_dqn",
        metadata={
            "scenario": config.scenario_name, "seed": config.seed,
            "train_episodes": config.train_episodes,
        },
    )
    return [path]


def _save_multi_agent(system, config: ExperimentConfig) -> List[str]:
    tag = f"{config.scenario_name}_{system.system_label()}_{config.num_agents}agents"
    return model_store.save_multi_agent(system, tag)


# ---------------------------------------------------------------------------
# Evaluation (Phases 8, 10)
# ---------------------------------------------------------------------------


def evaluate_method(
    config: ExperimentConfig,
    trained: TrainedMethod,
    conditions: Sequence[str] = (FIXED_CONDITION, RANDOM_CONDITION),
) -> Dict[str, MetricsCollector]:
    """Evaluates one trained method under each requested condition.

    Conditions:
        ``fixed``  — the frozen deterministic scenario, run with plain resets.
        ``random`` — the suite's seeded random layouts (identical seed sequence
                     for every method and repeated for each condition).

    The method's measured training time is attached to every collector so the
    comparison tables report real computation cost (Phase 13).
    """
    collectors: Dict[str, MetricsCollector] = {}
    for condition in conditions:
        collector = _evaluate_condition(config, trained, condition)
        collector.training_seconds = trained.training_seconds
        collectors[condition] = collector
    return collectors


def _evaluate_condition(config: ExperimentConfig, trained: TrainedMethod, condition: str) -> MetricsCollector:
    label = f"{trained.method}@{condition}"
    base_conditions = {
        "condition": condition,
        "method": trained.method,
        "method_label": trained.label,
        "family": trained.family,
        "train_episodes": config.train_episodes,
        "seed": config.seed,
    }

    if condition == FIXED_CONDITION:
        scenario = config.scenario.with_num_agents(trained.num_agents)
        env = scenario.build_env()
        episodes = max(1, config.fixed_eval_episodes)
        collector = MetricsCollector(label, num_agents=trained.num_agents, conditions={
            **base_conditions, "scenario": scenario.name, "episodes": episodes,
        })
        started = time.perf_counter()
        _run_condition(config, trained, env, collector, episodes, random_reset=None,
                       scenario_name=scenario.name)
    else:
        if config.suite.random is None:
            raise ValueError("Suite has no random block; cannot evaluate the 'random' condition.")
        block = config.suite.random
        if block.grid_size != config.scenario.height or block.grid_size != config.scenario.width:
            raise ValueError(
                f"Random suite is {block.grid_size}x{block.grid_size} but scenario "
                f"'{config.scenario.name}' is {config.scenario.height}x{config.scenario.width}; "
                "they must match for a fair comparison."
            )
        # Layouts are fully re-generated by reset(randomize=True); the initial
        # agent count must match the system being evaluated.
        env = _build_random_env(config, trained.num_agents)
        episodes = config.resolved_eval_episodes()
        collector = MetricsCollector(label, num_agents=trained.num_agents, conditions={
            **base_conditions, "scenario": f"random:{config.suite.name}",
            "episodes": episodes, "layout_seed_start": block.seed_start,
            "num_obstacles": block.num_obstacles,
        })
        started = time.perf_counter()
        _run_condition(
            config, trained, env, collector, episodes,
            random_reset=lambda index: block.reset_kwargs(index, trained.num_agents),
            scenario_name=f"random:{config.suite.name}",
        )
    collector.conditions["evaluation_seconds"] = round(time.perf_counter() - started, 4)
    return collector


def _build_random_env(config: ExperimentConfig, num_agents: int):
    from python_backend.environment.constants import Position
    from python_backend.environment.grid_world import GridWorld
    from python_backend.environment.reward import RewardConfig

    block = config.suite.random
    assert block is not None
    # Placeholder cells: the first reset(randomize=True) replaces every agent
    # and obstacle with a seeded layout, but the environment needs a valid
    # construction-time configuration (agent count must match the system).
    placeholder = [
        (Position(0, min(col, block.grid_size - 1)), Position(block.grid_size - 1, min(col, block.grid_size - 1)))
        for col in range(num_agents)
    ]
    return GridWorld(
        height=block.grid_size,
        width=block.grid_size,
        agents_config=placeholder,
        obstacles=set(),
        reward_config=RewardConfig(),
        max_steps=block.max_steps,
        seed=block.seed_start,
    )


def _run_condition(
    config: ExperimentConfig,
    trained: TrainedMethod,
    env,
    collector: MetricsCollector,
    episodes: int,
    random_reset,
    scenario_name: str,
) -> None:
    """Runs the episodes of one condition with the right episode loop."""
    if trained.family == "single_agent":
        policy = _single_agent_policy(trained)
        for index in range(episodes):
            if random_reset is not None:
                kwargs = random_reset(index)
                env.reset(**kwargs)
                layout_seed = kwargs.get("seed")
            else:
                env.reset()
                layout_seed = None
            collector.add(
                run_single_agent_episode(
                    env, agent_id=0, policy=policy, greedy=True, episode=index,
                    scenario=scenario_name, layout_seed=layout_seed,
                )
            )
        return

    # Multi-agent: MultiAgentSystem owns the loop; positions are recorded with a
    # delegating proxy so distance travelled is measured without touching it.
    # The proxy forwards every attribute/method call to the real environment, so
    # swapping the reference is behaviour-identical (only positions are traced).
    system = trained.payload
    recorder = PositionRecorder(env)
    system.env = recorder
    for index in range(episodes):
        reset_kwargs = random_reset(index) if random_reset is not None else None
        result = system.run_episode(train=False, episode=index, reset_kwargs=reset_kwargs)
        collector.add(
            episode_metrics_from_multi_agent(
                result, recorder, scenario_name,
                layout_seed=(reset_kwargs or {}).get("seed"),
            )
        )


def _single_agent_policy(trained: TrainedMethod):
    """Greedy action callable for a single-agent method."""
    if trained.method == RANDOM:
        from python_backend.algorithms.random_policy import RandomPolicy

        policy = RandomPolicy(seed=trained.metadata.get("seed"))

        def random_choice(observation, valid_actions, greedy=True):
            """Uniform choice over valid actions (``greedy`` is ignored)."""
            return policy.choose_action(observation, valid_actions)

        return random_choice
    if trained.payload is None:
        raise ValueError(f"Method '{trained.method}' has no trained payload to evaluate.")
    return trained.payload.choose_action


def episode_metrics_from_multi_agent(
    result,
    recorder: PositionRecorder,
    scenario_name: str,
    layout_seed: Optional[int] = None,
) -> EpisodeMetrics:
    """Converts Developer 3's ``EpisodeResult`` into evaluation metrics.

    TRUE environment rewards are used (``result.reward``); the learned reward
    that included the cooperative term is deliberately ignored for reporting.
    """
    distances = recorder.distances()
    totals = result.collision_totals
    per_agent_collisions = {
        aid: sum(types.values()) for aid, types in totals.by_agent.items()
    }
    metrics = EpisodeMetrics(
        episode=result.episode,
        steps=result.steps,
        reward=float(sum(result.reward.values())),
        success=bool(result.all_success),
        terminated=bool(result.terminated),
        truncated=bool(result.truncated),
        collisions=int(totals.obstacle + totals.agent),
        collision_types={
            "obstacle": totals.obstacle,
            "agent": totals.agent,
            "same_cell": totals.same_cell,
            "swap": totals.swap,
        },
        deadlocks=int(result.deadlocks),
        recoveries=int(result.recovery_count),
        distance=int(sum(distances.values())),
        messages=int(result.messages_per_episode),
        per_agent_reward={aid: float(v) for aid, v in result.reward.items()},
        per_agent_distance={aid: int(v) for aid, v in distances.items()},
        per_agent_success={aid: bool(v) for aid, v in result.successes.items()},
        goals_reached=int(sum(1 for v in result.successes.values() if v)),
        per_agent_collisions=per_agent_collisions,
        num_agents=len(result.successes),
        layout_seed=layout_seed,
        scenario=scenario_name,
    )
    return metrics


# ---------------------------------------------------------------------------
# Top-level suites
# ---------------------------------------------------------------------------


def run_comparison(
    config: Optional[ExperimentConfig] = None,
    methods: Optional[Sequence[str]] = None,
) -> ExperimentReport:
    """Phase 9/10: train and evaluate the requested methods on one frozen suite."""
    config = config or ExperimentConfig()
    if methods is not None:
        config.methods = tuple(methods)
    report = ExperimentReport(name=config.experiment_name, manifest=config.manifest())

    for method in config.methods:
        trained = train_method(config, method)
        collectors = evaluate_method(config, trained)
        for condition, collector in collectors.items():
            report.collectors[f"{method}@{condition}"] = collector
            report.rows.append(collector.aggregate())
        report.history[method] = trained.history
        if trained.model_paths:
            report.manifest.setdefault("models", {})[method] = trained.model_paths
    if config.save_models:
        model_store.write_manifest(config.experiment_name, report.to_dict()["manifest"])
    return report


def run_ablation(
    config: Optional[ExperimentConfig] = None,
    methods: Sequence[str] = (INDEPENDENT_DQN, RULE_BASED, COORDINATED_DQN),
) -> ExperimentReport:
    """Phase 12: which multi-agent component actually contributes?"""
    base = config or ExperimentConfig()
    ablation_config = ExperimentConfig(
        suite=base.suite, scenario_name=base.scenario_name, methods=tuple(methods),
        num_agents=base.num_agents, train_episodes=base.train_episodes,
        eval_episodes=base.eval_episodes, fixed_eval_episodes=base.fixed_eval_episodes,
        seed=base.seed, save_models=base.save_models, experiment_name="ablation",
    )
    return run_comparison(ablation_config)


def run_scalability(
    config: Optional[ExperimentConfig] = None,
    agent_counts: Sequence[int] = (1, 2, 3, 4, 5),
    method: str = INDEPENDENT_DQN,
) -> ExperimentReport:
    """Phase 13: how does performance change with the number of agents?

    The same obstacle layout and the same agent prefix are used for every agent
    count, so the only changing variable is how many agents share the grid.
    N=1 uses the single-agent DQN path (``MultiAgentSystem`` requires 2-5).
    Sweeps longer than the demo layout define use the dedicated five-agent
    scalability layout (see :func:`scalability_scenario`).
    """
    from python_backend.evaluation.scenarios import scalability_scenario

    base = config or ExperimentConfig()
    if method not in (INDEPENDENT_DQN, DQN):
        raise ValueError(f"Scalability supports '{INDEPENDENT_DQN}' or '{DQN}', got '{method}'.")
    if not agent_counts:
        raise ValueError("Scalability needs at least one agent count.")

    report_config = ExperimentConfig(
        suite=base.suite, scenario_name=scalability_scenario(agent_counts).name, methods=(),
        num_agents=base.num_agents, train_episodes=base.train_episodes,
        eval_episodes=base.eval_episodes, fixed_eval_episodes=base.fixed_eval_episodes,
        seed=base.seed, save_models=base.save_models, experiment_name="scalability",
    )
    report = ExperimentReport(name="scalability", manifest=report_config.manifest())
    report.manifest["agent_counts"] = list(agent_counts)
    report.manifest["method"] = method

    for count in agent_counts:
        count_config = ExperimentConfig(
            suite=report_config.suite, scenario_name=report_config.scenario_name,
            methods=(DQN if count == 1 else INDEPENDENT_DQN,), num_agents=count,
            train_episodes=report_config.train_episodes,
            eval_episodes=report_config.eval_episodes,
            fixed_eval_episodes=report_config.fixed_eval_episodes,
            seed=report_config.seed, save_models=report_config.save_models,
            experiment_name=f"scalability_{count}agents",
        )
        method_name = count_config.methods[0]
        trained = train_method(count_config, method_name)
        collectors = evaluate_method(count_config, trained)
        for condition, collector in collectors.items():
            collector.conditions["num_agents"] = count
            key = f"{count}agents@{condition}"
            report.collectors[key] = collector
            report.rows.append(collector.aggregate())
        report.history[f"{count}agents"] = trained.history
        report.manifest.setdefault("training_seconds", {})[f"{count}agents"] = round(
            trained.training_seconds, 4
        )
        if trained.model_paths:
            report.manifest.setdefault("models", {})[f"{count}agents"] = trained.model_paths
    if report_config.save_models:
        model_store.write_manifest("scalability", report.to_dict()["manifest"])
    return report
