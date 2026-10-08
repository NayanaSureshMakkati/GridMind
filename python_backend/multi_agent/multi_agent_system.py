"""Multi-agent DQN system with configurable coordination (Developer 3).

******************************************************************************
*  ARCHITECTURE                                                              *
******************************************************************************

MultiAgentSystem manages 2-5 independently controllable DQN agents sharing one
frozen GridWorld (Environment API v1.0 — black box; this module only consumes
reset/step/get_valid_actions/get_agent_observation/get_all_states).

Per step:
    1. build observations per configured mode        (Phase 5)
    2. agents choose actions (epsilon-greedy, valid actions only;
       policy_kind="random" uses RandomPolicy instead — Phase 13 baseline)
    3. RuleBasedArbiter.resolve()                    (optional, Phase 4)
    4. deadlock recovery override                    (optional, Phase 8)
    5. CommunicationManager.broadcast_intents()      (optional, Phase 9;
       after action selection so messages carry the final intended action)
    6. env.step(all actions) — simultaneous, env-owned conflict resolution
    7. CollisionAnalyzer tallies by type             (Phase 3)
    8. per-agent transitions stored; optional cooperative team term added
       to the LEARNED reward only (Phases 6/10)
    9. train_step() per agent (own networks, own buffer)
    10. DeadlockDetector.record_step()               (Phase 7)

Modes (all are this system; nothing below touches the env or DQN code):
    coordination_mode:
        "none"       — Independent DQN (baseline)
        "rule_based" — RULE-BASED COORDINATION (deterministic priority yields;
                       see coordination.py — explicitly NOT learned)
    observation_mode:
        LOCAL_ONLY (frozen 12-vector) or LOCAL_PLUS_OTHER_AGENTS. Enriched
        observations plus an optional cooperative reward term are the
        LEARNED COORDINATION mechanism (Phase 6): agents learn to avoid
        conflicts from what they see and what they are rewarded for.
    cooperative_weight (Phase 10): weight of the team term added to the
        LEARNED reward only. The team term is the MEAN of the other active
        agents' environment rewards for the step, so it scales independently
        of agent count and (with weight <= 1) can never dominate the
        individual goal reward (+100). Episode results record BOTH the true
        environment rewards and the learned rewards separately — metrics and
        comparisons always use the TRUE rewards.
    ctde (optional, Phase 11): CTDE-INSPIRED approximation — train with
        LOCAL_PLUS_OTHER_AGENTS observations, execute with the other-agent
        blocks zeroed (local-only information). Same network/input shape at
        train and execution time. This is NOT MAPPO/QMIX/MADDPG: no shared
        critic, no parameter sharing, no published algorithm is claimed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from python_backend.algorithms.dqn import DQNAgent, DQNConfig
from python_backend.algorithms.random_policy import RandomPolicy
from python_backend.environment.constants import Action
from python_backend.environment.collision import CollisionType
from python_backend.environment.grid_world import GridWorld
from python_backend.multi_agent.communication import CommunicationManager
from python_backend.multi_agent.coordination import RuleBasedArbiter
from python_backend.multi_agent.deadlock import DeadlockDetector, RecoveryManager
from python_backend.multi_agent.observations import (
    BASE_OBSERVATION_SIZE,
    LOCAL_ONLY,
    LOCAL_PLUS_OTHER_AGENTS,
    OBSERVATION_MODES,
    build_observation,
    observation_size,
)

COORDINATION_MODES = ("none", "rule_based")
POLICY_KINDS = ("dqn", "random")

# info["collisions"] type names bucketed per Phase 3 (never hidden).
OBSTACLE_COLLISION_TYPES = ("BOUNDARY", "OBSTACLE")
AGENT_COLLISION_TYPES = ("SAME_CELL", "SWAP")


@dataclass
class MultiAgentConfig:
    """Configuration for MultiAgentSystem (everything explicit, nothing hardcoded)."""

    num_agents: int
    coordination_mode: str = "none"
    observation_mode: str = LOCAL_ONLY
    policy_kind: str = "dqn"  # "dqn" (learned) or "random" (uniform baseline, Phase 13)
    random_seed: Optional[int] = None  # seed for the random policy's RNG
    ctde: bool = False
    cooperative_weight: float = 0.0
    deadlock_stagnation_steps: int = 8
    deadlock_cycle_window: int = 12
    recovery_kind: str = "none"
    max_recoveries_per_episode: int = 1
    communication_enabled: bool = False
    dqn_config: Optional[DQNConfig] = None
    dqn_configs: Optional[Dict[int, DQNConfig]] = None

    def __post_init__(self) -> None:
        if not 2 <= self.num_agents <= 5:
            raise ValueError(f"num_agents must be in 2..5, got {self.num_agents}.")
        if self.coordination_mode not in COORDINATION_MODES:
            raise ValueError(
                f"Unknown coordination mode '{self.coordination_mode}'. "
                f"Available: {list(COORDINATION_MODES)}"
            )
        if self.policy_kind not in POLICY_KINDS:
            raise ValueError(
                f"Unknown policy kind '{self.policy_kind}'. "
                f"Available: {list(POLICY_KINDS)}"
            )
        if self.observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"Unknown observation mode '{self.observation_mode}'. "
                f"Available: {list(OBSERVATION_MODES)}"
            )
        if not 0.0 <= self.cooperative_weight <= 1.0:
            raise ValueError("cooperative_weight must be in [0, 1] so the team "
                             "term can never dominate the individual goal reward.")
        if self.recovery_kind not in RecoveryManager.KINDS:
            raise ValueError(
                f"Unknown recovery kind '{self.recovery_kind}'. "
                f"Available: {list(RecoveryManager.KINDS)}"
            )
        if self.ctde and self.observation_mode != LOCAL_PLUS_OTHER_AGENTS:
            raise ValueError("ctde=True requires observation_mode=LOCAL_PLUS_OTHER_AGENTS "
                             "(train global-ish, execute local).")


@dataclass
class CollisionTotals:
    """Per-type collision totals for one episode (Phase 3: never hidden)."""

    obstacle: int = 0    # BOUNDARY + OBSTACLE (env's obstacle_collision penalty)
    agent: int = 0       # SAME_CELL + SWAP (env's agent_collision penalty)
    same_cell: int = 0
    swap: int = 0
    by_agent: Dict[int, Dict[str, int]] = field(default_factory=dict)

    def record(self, agent_id: int, type_name: str) -> None:
        """Records one collision event of the given CollisionType name."""
        if type_name in OBSTACLE_COLLISION_TYPES:
            self.obstacle += 1
        elif type_name == "SAME_CELL":
            self.same_cell += 1
            self.agent += 1
        elif type_name == "SWAP":
            self.swap += 1
            self.agent += 1
        else:
            return  # NONE / unknown: nothing to record
        per_agent = self.by_agent.setdefault(agent_id, {})
        per_agent[type_name] = per_agent.get(type_name, 0) + 1


@dataclass
class EpisodeResult:
    """Per-episode result packet (results format for experiments)."""

    episode: int = 0
    steps: int = 0
    terminated: bool = False
    truncated: bool = False
    reward: Dict[int, float] = field(default_factory=dict)          # TRUE env rewards (totals)
    learned_reward: Dict[int, float] = field(default_factory=dict)  # incl. cooperative term
    step_rewards: List[Dict[int, float]] = field(default_factory=list)
    learned_step_rewards: List[Dict[int, float]] = field(default_factory=list)
    collision_totals: CollisionTotals = field(default_factory=CollisionTotals)
    deadlocks: int = 0
    deadlock_events: List[Any] = field(default_factory=list)
    recovery_count: int = 0
    coordination_events: int = 0
    messages_per_episode: int = 0
    successes: Dict[int, bool] = field(default_factory=dict)
    all_success: bool = False
    mean_loss: float = 0.0


@dataclass
class MultiAgentEvaluationResult:
    """Aggregated greedy evaluation across episodes (results format)."""

    num_agents: int
    episodes: int
    coordination_mode: str
    observation_mode: str
    ctde: bool
    all_success_rate: float
    any_success_rate: float
    avg_reward: Dict[int, float] = field(default_factory=dict)   # TRUE env rewards
    avg_steps: float = 0.0
    avg_collisions: Dict[int, float] = field(default_factory=dict)
    avg_collision_types: Dict[str, float] = field(default_factory=dict)  # obstacle/agent/same_cell/swap per episode
    deadlocks_per_episode: float = 0.0
    recoveries_per_episode: float = 0.0
    messages_per_episode: float = 0.0
    messages_per_step: float = 0.0
    messages_per_agent_step: float = 0.0  # messages / (num_agents * steps)


class MultiAgentSystem:
    """N independently controllable DQN agents in one shared frozen GridWorld.

    Phases 1-11. Coordination mode, observation mode, cooperative reward,
    deadlock detection/recovery, and communication are all configured via
    MultiAgentConfig; with the defaults this system IS the Independent DQN
    baseline.
    """

    def __init__(self, env: GridWorld, config: MultiAgentConfig) -> None:
        if len(env.agents) != config.num_agents:
            raise ValueError(
                f"Environment has {len(env.agents)} agents but config.num_agents="
                f"{config.num_agents}. They must match (see observations.py)."
            )
        self.env = env
        self.config = config
        n = config.num_agents
        self.num_agents = n
        obs_size = observation_size(n, config.observation_mode)

        self.agents: Dict[int, DQNAgent] = {}
        self.random_policies: Dict[int, RandomPolicy] = {}
        for aid in range(n):
            base = config.dqn_configs.get(aid, config.dqn_config) if config.dqn_configs else config.dqn_config
            self.agents[aid] = DQNAgent(agent_id=aid, config=self._agent_config(base, obs_size))
            if config.policy_kind == "random":
                seed = None if config.random_seed is None else config.random_seed + aid
                self.random_policies[aid] = RandomPolicy(seed=seed)

        self.waiting_times: Dict[int, int] = {aid: 0 for aid in self.agents}
        self._prev_positions: Dict[int, Tuple[int, int]] = {}
        self.communication = CommunicationManager(enabled=config.communication_enabled)
        self.deadlock_detector = DeadlockDetector(
            stagnation_steps=config.deadlock_stagnation_steps,
            cycle_window=config.deadlock_cycle_window,
        )
        self.recovery_manager = RecoveryManager(
            kind=config.recovery_kind,
            max_recoveries_per_episode=config.max_recoveries_per_episode,
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_config(base: Optional[DQNConfig], obs_size: int) -> DQNConfig:
        """Clones hyperparameters from `base`, forcing the correct observation size."""
        b = base or DQNConfig()
        return DQNConfig(
            gamma=b.gamma,
            epsilon=b.epsilon,
            epsilon_decay=b.epsilon_decay,
            epsilon_min=b.epsilon_min,
            learning_rate=b.learning_rate,
            batch_size=b.batch_size,
            buffer_capacity=b.buffer_capacity,
            min_buffer_size=b.min_buffer_size,
            target_update_frequency=b.target_update_frequency,
            hidden_size=b.hidden_size,
            observation_size=obs_size,
            seed=b.seed,
        )

    def system_label(self) -> str:
        """Honest label for experiment metadata (Master Context §28/§33)."""
        if self.config.policy_kind == "random":
            return "random"
        if self.config.coordination_mode == "rule_based":
            return "rule_based_coordination"
        if self.config.ctde:
            return "ctde_inspired_dqn"
        if self.config.observation_mode == LOCAL_PLUS_OTHER_AGENTS or self.config.cooperative_weight > 0:
            return "coordinated_dqn"
        return "independent_dqn"

    @staticmethod
    def execution_observation(observation: List[int]) -> List[int]:
        """CTDE execution view: local part kept, other-agent blocks zeroed.

        Input shape is preserved so the same network runs at train and
        execution time; only the information content changes (Phase 11).
        """
        return list(observation[:BASE_OBSERVATION_SIZE]) + [0] * (len(observation) - BASE_OBSERVATION_SIZE)

    # ------------------------------------------------------------------
    # Episode execution (Phases 1, 2 and the wiring of 3-10)
    # ------------------------------------------------------------------

    def run_episode(
        self,
        train: bool = True,
        episode: int = 0,
        action_overrides: Optional[Dict[int, int]] = None,
        reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> EpisodeResult:
        """Runs one episode: communicate -> observe -> act -> arbitrate -> step -> learn.

        Args:
            train: False runs greedy execution (evaluation/demo); no learning.
            episode: Episode index for bookkeeping.
            action_overrides: Optional {agent_id: action} forcing scripted
                actions (deterministic tests/debugging, Master Context §18).
                Every override must be a valid action for that agent.
            reset_kwargs: Optional kwargs forwarded to env.reset() for the
                single reset performed here (e.g. seeded random layouts).
                None -> plain deterministic reset (Master Context §18).
        """
        cfg = self.config
        mode = cfg.observation_mode
        self.communication.reset()
        self.deadlock_detector.reset()
        self.recovery_manager.reset()

        result = EpisodeResult(episode=episode)
        result.successes = {aid: False for aid in self.agents}
        losses: List[float] = []
        waiting_flags = {aid: False for aid in self.agents}
        self.waiting_times = {aid: 0 for aid in self.agents}
        self._prev_positions = {}
        deadlock_fired_last_step = False
        active: Set[int] = set(self.agents.keys())
        self.env.reset(**(reset_kwargs or {}))

        while True:
            step = self.env.current_step + 1
            active_ids = sorted(active)

            # 1) Observations (Phase 5; CTDE execution view when evaluating, Phase 11)
            current_obs: Dict[int, List[int]] = {}
            for aid in active_ids:
                vec = build_observation(self.env, aid, self.num_agents, mode, waiting_flags)
                current_obs[aid] = vec if (train or not cfg.ctde) else self.execution_observation(vec)

            # 2) Simultaneous action collection (Phase 2)
            actions: Dict[int, int] = {}
            for aid in active_ids:
                if action_overrides and aid in action_overrides:
                    act = int(action_overrides[aid])
                    if act not in self.env.get_valid_actions(aid):
                        raise ValueError(
                            f"action_overrides[{aid}]={act} is not a valid action."
                        )
                else:
                    valid = self.env.get_valid_actions(aid)
                    if cfg.policy_kind == "random":
                        act = self.random_policies[aid].choose_action(current_obs[aid], valid)
                    else:
                        act = self.agents[aid].choose_action(current_obs[aid], valid, greedy=not train)
                actions[aid] = act

            # 3) RULE-BASED COORDINATION pre-step yields (Phase 4)
            if cfg.coordination_mode == "rule_based" and len(actions) >= 2:
                arbiter = RuleBasedArbiter(self.env, self.waiting_times)
                actions, coord_events = arbiter.resolve(step, actions)
                result.coordination_events += len(coord_events)

            # 4) Deadlock recovery override (Phase 8; takes precedence over 3)
            if deadlock_fired_last_step and self.recovery_manager.should_recover():
                actions = self.recovery_manager.recovery_actions(self.env, active_ids)
                result.recovery_count += 1
            deadlock_fired_last_step = False

            # 5) Optional communication (Phase 9): broadcast AFTER action
            #    selection/arbitration so each message carries the final
            #    intended action the agent will submit this step.
            positions_now = {
                aid: (p.row, p.col)
                for aid, p in ((aid, self.env.get_agent_position(aid)) for aid in active_ids)
            }
            self.communication.broadcast_intents(
                step,
                active_ids,
                positions_now,
                actions,
                goal_reached={
                    aid: bool(self.env.get_agent_state(aid)["done"])
                    for aid in active_ids
                },
            )

            # 6) Simultaneous environment step (frozen API v1.0 does all
            #    physical conflict resolution — this system never moves agents)
            next_obs_env, rewards, dones, terminated, info = self.env.step(actions)

            # 7) Collision analysis (Phase 3 — recorded, never hidden)
            for aid, type_name in info.get("collisions", {}).items():
                result.collision_totals.record(aid, type_name)

            # 8) Cooperative term on the LEARNED reward only (Phases 6/10)
            true_rewards = {aid: float(rewards[aid]) for aid in self.agents}
            learned_rewards = dict(true_rewards)
            if cfg.cooperative_weight > 0.0:
                w = cfg.cooperative_weight
                for aid in active_ids:
                    others = [true_rewards[o] for o in active_ids if o != aid]
                    if others:
                        learned_rewards[aid] = true_rewards[aid] + w * (sum(others) / len(others))

            # Next-step waiting flags reflect the FINAL actions sent to env.
            waiting_next = {aid: actions.get(aid) == int(Action.STAY) for aid in self.agents}
            next_obs: Dict[int, List[int]] = {}
            for aid in active_ids:
                vec = build_observation(self.env, aid, self.num_agents, mode, waiting_next)
                next_obs[aid] = vec if (train or not cfg.ctde) else self.execution_observation(vec)

            # 9) Per-agent transition storage + learning (Phase 2)
            for aid in active_ids:
                if train:
                    self.agents[aid].remember(
                        current_obs[aid], actions[aid], learned_rewards[aid],
                        next_obs[aid], dones[aid],
                    )
                    loss = self.agents[aid].train_step()
                    if loss is not None:
                        losses.append(loss)
                result.reward[aid] = result.reward.get(aid, 0.0) + true_rewards[aid]
                result.learned_reward[aid] = result.learned_reward.get(aid, 0.0) + learned_rewards[aid]
                if dones[aid]:
                    result.successes[aid] = True

            result.step_rewards.append(dict(true_rewards))
            result.learned_step_rewards.append(dict(learned_rewards))

            # 10) Deadlock detection (Phase 7 — recorded, env never silently reset)
            positions_after = {
                aid: (p.row, p.col)
                for aid, p in ((aid, self.env.get_agent_position(aid)) for aid in self.agents)
            }
            agent_collision_ids = [
                aid for aid, nm in info.get("collisions", {}).items()
                if nm in AGENT_COLLISION_TYPES
            ]
            if self.deadlock_detector.record_step(
                info["step"], positions_after, dict(dones), agent_collision_ids,
                actions=dict(actions),
            ):
                deadlock_fired_last_step = True

            # Waiting-time bookkeeping for the rule-based priority policy.
            for aid in self.agents:
                prev = self._prev_positions.get(aid)
                if prev is not None and prev == positions_after[aid]:
                    self.waiting_times[aid] = self.waiting_times.get(aid, 0) + 1
                else:
                    self.waiting_times[aid] = 0
            self._prev_positions = positions_after
            waiting_flags = waiting_next

            # Episode end conditions (terminated/truncated kept separate)
            truncated = bool(info.get("truncated", False))
            if terminated or truncated:
                result.terminated = terminated
                result.truncated = truncated
                break

            # Retire finished agents (they STAY; no further learning)
            for aid in list(active):
                if dones[aid]:
                    active.discard(aid)
            if not active:
                break

        # Finalize the result packet
        result.steps = self.env.current_step
        result.deadlocks = self.deadlock_detector.deadlock_count
        result.deadlock_events = list(self.deadlock_detector.events)
        result.messages_per_episode = self.communication.messages_per_episode
        result.all_success = all(result.successes.get(aid, False) for aid in self.agents)
        result.mean_loss = (sum(losses) / len(losses)) if losses else 0.0
        return result

    # ------------------------------------------------------------------
    # Training / evaluation loops
    # ------------------------------------------------------------------

    def train(
        self,
        episodes: int,
        reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[EpisodeResult]:
        """Trains all agents for `episodes` episodes with per-episode epsilon decay.

        Default (reset_kwargs=None) trains on the deterministic fixed layout
        per Master Context §18; pass seeded randomize kwargs to generalize.
        """
        history: List[EpisodeResult] = []
        for episode in range(episodes):
            result = self.run_episode(train=True, episode=episode, reset_kwargs=reset_kwargs)
            for agent in self.agents.values():
                agent.decay_epsilon()
            history.append(result)
        return history

    def evaluate(
        self,
        episodes: int = 50,
        seed: int = 1000,
        randomize: bool = True,
        num_obstacles: int = 4,
    ) -> MultiAgentEvaluationResult:
        """Greedy evaluation; seeded random layouts when randomize=True.

        With ctde=True the execution view (other-agent blocks zeroed) is used,
        matching the decentralized-execution claim (Phase 11).
        """
        totals = {
            "all_success": 0,
            "any_success": 0,
            "reward": {aid: 0.0 for aid in self.agents},
            "collisions": {aid: 0 for aid in self.agents},
            "collision_types": {"obstacle": 0, "agent": 0, "same_cell": 0, "swap": 0},
            "deadlocks": 0,
            "recoveries": 0,
            "messages": 0,
            "steps": 0,
        }
        for i in range(episodes):
            result = self.run_episode(
                train=False,
                episode=i,
                reset_kwargs={
                    "seed": seed + i,
                    "randomize": randomize,
                    "num_agents": self.num_agents,
                    "num_obstacles": num_obstacles,
                },
            )
            totals["all_success"] += int(result.all_success)
            totals["any_success"] += int(any(result.successes.values()))
            totals["deadlocks"] += result.deadlocks
            totals["recoveries"] += result.recovery_count
            totals["messages"] += result.messages_per_episode
            totals["steps"] += result.steps
            for aid in self.agents:
                totals["reward"][aid] += result.reward[aid]
                totals["collisions"][aid] += sum(result.collision_totals.by_agent.get(aid, {}).values())
            ct = result.collision_totals
            totals["collision_types"]["obstacle"] += ct.obstacle
            totals["collision_types"]["agent"] += ct.agent
            totals["collision_types"]["same_cell"] += ct.same_cell
            totals["collision_types"]["swap"] += ct.swap

        n_eps = max(1, episodes)
        total_steps = max(1, totals["steps"])
        return MultiAgentEvaluationResult(
            num_agents=self.num_agents,
            episodes=episodes,
            coordination_mode=self.config.coordination_mode,
            observation_mode=self.config.observation_mode,
            ctde=self.config.ctde,
            all_success_rate=totals["all_success"] / n_eps,
            any_success_rate=totals["any_success"] / n_eps,
            avg_reward={aid: r / n_eps for aid, r in totals["reward"].items()},
            avg_steps=totals["steps"] / n_eps,
            avg_collisions={aid: c / n_eps for aid, c in totals["collisions"].items()},
            avg_collision_types={k: v / n_eps for k, v in totals["collision_types"].items()},
            deadlocks_per_episode=totals["deadlocks"] / n_eps,
            recoveries_per_episode=totals["recoveries"] / n_eps,
            messages_per_episode=totals["messages"] / n_eps,
            messages_per_step=totals["messages"] / total_steps,
            messages_per_agent_step=totals["messages"] / (self.num_agents * total_steps),
        )

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def save_models(self, directory: str, tag: Optional[str] = None) -> None:
        """Saves each agent's checkpoint as <dir>/<tag>_agent<aid>.pt."""
        import os

        os.makedirs(directory, exist_ok=True)
        tag = tag or self.system_label()
        metadata = {
            "system": self.system_label(),
            "num_agents": self.num_agents,
            "policy_kind": self.config.policy_kind,
            "coordination_mode": self.config.coordination_mode,
            "observation_mode": self.config.observation_mode,
            "ctde": self.config.ctde,
            "cooperative_weight": self.config.cooperative_weight,
        }
        for aid, agent in self.agents.items():
            agent.save_model(os.path.join(directory, f"{tag}_agent{aid}.pt"), metadata=metadata)

    def load_models(self, directory: str, tag: Optional[str] = None) -> None:
        """Loads each agent's checkpoint from <dir>/<tag>_agent<aid>.pt."""
        import os

        tag = tag or self.system_label()
        for aid, agent in self.agents.items():
            path = os.path.join(directory, f"{tag}_agent{aid}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing checkpoint for agent {aid}: {path}")
            agent.load_model(path)
