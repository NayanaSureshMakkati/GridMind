"""Model management for GridMind experiments (Developer 4, Phase 8 / §35).

Checkpoints are grouped by algorithm so ``models/`` stays navigable, and every
save also writes a JSON manifest next to the results:

    models/q_learning/<name>.pkl        tabular Q-table (Developer 2 format)
    models/dqn/<name>.pt                single-agent DQN (Developer 2 format)
    models/multi_agent/<tag>_agent<i>.pt  one file per agent (Developer 3 format)

Manifests (``experiments/results/manifest_<name>.json``) record algorithm name,
seed, scenario, suite, hyperparameters and the measured metrics so a checkpoint
can be reproduced and audited later (Master Context 35/56).

This module never re-implements an algorithm: it delegates to
``QLearningAgent.save/load``, ``DQNAgent.save_model/load_model`` and
``MultiAgentSystem.save_models/load_models``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")

#: Group directory per algorithm family (Master Context §35).
GROUPS: Dict[str, str] = {
    "q_learning": "q_learning",
    "dqn": "dqn",
    "multi_agent": "multi_agent",
}


def group_dir(group: str) -> str:
    """Absolute path of a model group directory, created if missing."""
    if group not in GROUPS:
        raise ValueError(f"Unknown model group '{group}'. Available: {sorted(GROUPS)}")
    path = os.path.join(MODELS_DIR, GROUPS[group])
    os.makedirs(path, exist_ok=True)
    return path


def results_dir() -> str:
    """Absolute path of the results directory, created if missing."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def resolve(group: str, name: str, extension: str) -> str:
    """Resolves a checkpoint path inside a group (``name`` may be a full path)."""
    if os.path.isabs(name) or os.sep in name or "/" in name:
        return name
    return os.path.join(group_dir(group), f"{name}{extension}")


# ---------------------------------------------------------------------------
# Single-agent checkpoints
# ---------------------------------------------------------------------------


def save_q_learning(agent, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Saves a tabular Q-learning agent (Developer 2's ``.pkl`` format)."""
    path = resolve("q_learning", name, ".pkl")
    agent.save(path, metadata=metadata)
    return path


def load_q_learning(name_or_path: str):
    """Loads a tabular Q-learning agent from a group name or explicit path."""
    from python_backend.algorithms.q_learning import QLearningAgent

    path = resolve("q_learning", name_or_path, ".pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Q-learning checkpoint not found: {path}")
    return QLearningAgent.load(path)


def save_dqn(agent, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Saves a single-agent DQN checkpoint (Developer 2's ``.pt`` format)."""
    path = resolve("dqn", name, ".pt")
    agent.save_model(path, metadata=metadata)
    return path


def load_dqn(name_or_path: str, agent_id: int = 0, seed: Optional[int] = None):
    """Loads a single-agent DQN checkpoint, rebuilding the network shape.

    The checkpoint's stored ``observation_size`` / ``hidden_size`` are read from
    the file's documented hyperparameter block so the network is constructed
    with the right shape before weights are restored. A mismatched file raises
    a clear error instead of a cryptic ``load_state_dict`` failure.
    """
    import torch

    from python_backend.algorithms.dqn import DQNAgent, DQNConfig

    path = resolve("dqn", name_or_path, ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"DQN checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "gridmind.dqn.v1":
        raise ValueError(f"Not a GridMind DQN checkpoint: {path}")
    hyper = checkpoint.get("hyperparameters", {})

    agent = DQNAgent(
        agent_id=agent_id,
        config=DQNConfig(
            observation_size=int(hyper.get("observation_size", 12)),
            hidden_size=int(hyper.get("hidden_size", 128)),
            seed=seed,
        ),
    )
    agent.load_model(path)
    return agent


# ---------------------------------------------------------------------------
# Multi-agent checkpoints
# ---------------------------------------------------------------------------


def save_multi_agent(system, tag: str) -> List[str]:
    """Saves one checkpoint per agent of a MultiAgentSystem.

    Returns the list of written paths (``<tag>_agent<i>.pt``).
    """
    directory = group_dir("multi_agent")
    system.save_models(directory, tag=tag)
    return [
        os.path.join(directory, f"{tag}_agent{aid}.pt") for aid in sorted(system.agents)
    ]


def multi_agent_paths(tag: str, num_agents: int) -> List[str]:
    """Expected checkpoint paths for a multi-agent tag (existence-checked)."""
    directory = group_dir("multi_agent")
    paths = [os.path.join(directory, f"{tag}_agent{aid}.pt") for aid in range(num_agents)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing multi-agent checkpoint(s): " + ", ".join(missing)
        )
    return paths


def load_multi_agent(system, tag: str) -> None:
    """Loads all per-agent checkpoints of a multi-agent system in place."""
    directory = group_dir("multi_agent")
    multi_agent_paths(tag, system.num_agents)  # raises a clear error if incomplete
    system.load_models(directory, tag=tag)


# ---------------------------------------------------------------------------
# Manifests (reproducibility)
# ---------------------------------------------------------------------------


def write_manifest(name: str, payload: Dict[str, Any]) -> str:
    """Writes ``experiments/results/manifest_<name>.json`` and returns its path."""
    path = os.path.join(results_dir(), f"manifest_{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def read_manifest(name: str) -> Dict[str, Any]:
    """Reads a manifest written by :func:`write_manifest`."""
    path = os.path.join(results_dir(), f"manifest_{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
