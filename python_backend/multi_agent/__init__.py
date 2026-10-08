"""Multi-agent RL engine (Developer 3, Phases 1-11).

Contents:
    IndependentDQNSystem — Independent DQN baseline (Phase 2)
    MultiAgentSystem     — the full multi-agent system with configurable
                           coordination (none / RULE-BASED), observation modes,
                           cooperative reward, deadlock detection/recovery,
                           and optional compact communication.
"""

from python_backend.multi_agent.communication import CommunicationManager
from python_backend.multi_agent.coordination import CoordinationEvent, PriorityRule, RuleBasedArbiter
from python_backend.multi_agent.deadlock import DeadlockDetector, DeadlockEvent, RecoveryManager
from python_backend.multi_agent.independent_dqn import IndependentDQNSystem
from python_backend.multi_agent.multi_agent_system import (
    COORDINATION_MODES,
    CollisionTotals,
    EpisodeResult,
    MultiAgentConfig,
    MultiAgentEvaluationResult,
    MultiAgentSystem,
)
from python_backend.multi_agent.observations import (
    LOCAL_ONLY,
    LOCAL_PLUS_OTHER_AGENTS,
    OBSERVATION_MODES,
    build_observation,
    observation_size,
)

__all__ = [
    "IndependentDQNSystem",
    "MultiAgentSystem",
    "MultiAgentConfig",
    "MultiAgentEvaluationResult",
    "EpisodeResult",
    "CollisionTotals",
    "COORDINATION_MODES",
    "RuleBasedArbiter",
    "PriorityRule",
    "CoordinationEvent",
    "DeadlockDetector",
    "DeadlockEvent",
    "RecoveryManager",
    "CommunicationManager",
    "LOCAL_ONLY",
    "LOCAL_PLUS_OTHER_AGENTS",
    "OBSERVATION_MODES",
    "build_observation",
    "observation_size",
]
