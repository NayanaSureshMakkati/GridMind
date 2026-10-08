"""Frozen scenario set for GridMind experiments and the Unity demo (Developer 4).

Single source of truth for layout data. Both the Python backend and the Unity
frontend consume these files:

  * Python loads them here (``load_scenario`` / ``load_suite``).
  * Unity receives the loaded scenario over the wire in the ``scenario``
    message (see ``docs/communication_protocol.md``), so the 3D world can never
    drift from the logical world.

Layout files live in ``experiments/scenarios/`` as plain JSON:

    {"name": ..., "height": ..., "width": ..., "max_steps": ..., "seed": ...,
     "obstacles": [[row, col], ...],
     "agents": [{"id": 0, "start": [row, col], "goal": [row, col]}, ...]}

Scenarios are FROZEN (Master Context 43): every method compared in an
experiment runs on the same scenarios, so results stay comparable. Editing a
scenario file invalidates previously reported numbers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from python_backend.environment.constants import Position
from python_backend.environment.grid_world import GridWorld
from python_backend.environment.reward import RewardConfig

# Repository root (this file lives at <root>/python_backend/evaluation/).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENARIOS_DIR = os.path.join(ROOT, "experiments", "scenarios")
DEFAULT_SUITE = "eval_suite_v1"
DEMO_SCENARIO = "demo_10x10_3agents"
#: Layout backing the Phase 13 scalability sweep. It defines five agents on the
#: demo grid, with the demo obstacles and the first three agents unchanged, so
#: prefixes 1..5 share one layout (and 1..3 reproduce ``demo_10x10_3agents``).
SCALABILITY_SCENARIO = "scalability_10x10_5agents"


@dataclass(frozen=True)
class Scenario:
    """One frozen logical layout (grid size, obstacles, agent starts/goals)."""

    name: str
    height: int
    width: int
    max_steps: int
    obstacles: tuple
    agents: tuple  # tuple of (start: Position, goal: Position) ordered by agent id
    seed: Optional[int] = None
    description: str = ""

    @property
    def num_agents(self) -> int:
        """Number of agents defined in this scenario."""
        return len(self.agents)

    def with_num_agents(self, num_agents: int) -> "Scenario":
        """Returns a copy using the first ``num_agents`` agents (scalability rule).

        Taking a prefix keeps the obstacle layout and the remaining agents'
        starts/goals identical, so agent-count comparisons stay fair
        (Master Context 43/67).
        """
        if not 1 <= num_agents <= self.num_agents:
            raise ValueError(
                f"num_agents must be in 1..{self.num_agents}, got {num_agents}."
            )
        return Scenario(
            name=self.name,
            height=self.height,
            width=self.width,
            max_steps=self.max_steps,
            obstacles=self.obstacles,
            agents=self.agents[:num_agents],
            seed=self.seed,
            description=self.description,
        )

    def build_env(self, num_agents: Optional[int] = None) -> GridWorld:
        """Builds a fresh :class:`GridWorld` from this scenario.

        Args:
            num_agents: Optional prefix length (see :meth:`with_num_agents`).

        Returns:
            A new GridWorld with the scenario's obstacles, starts, goals and
            step limit. Only the frozen Environment API v1.0 is used.
        """
        scenario = self.with_num_agents(num_agents) if num_agents else self
        return GridWorld(
            height=scenario.height,
            width=scenario.width,
            agents_config=list(scenario.agents),
            obstacles=set(scenario.obstacles),
            reward_config=RewardConfig(),
            max_steps=scenario.max_steps,
            seed=scenario.seed,
        )

    def to_message(self, num_agents: Optional[int] = None) -> Dict[str, Any]:
        """Serializes the scenario for the Python->Unity ``scenario`` message.

        Agent start/goal positions are ``[row, col]`` arrays, while obstacles use
        named ``{"row": r, "col": c}`` entries: Unity parses messages with
        ``JsonUtility``, which supports one-dimensional arrays but not jagged
        ones (``int[][]``).
        """
        scenario = self.with_num_agents(num_agents) if num_agents else self
        return {
            "name": scenario.name,
            "height": scenario.height,
            "width": scenario.width,
            "max_steps": scenario.max_steps,
            "obstacles": [{"row": p.row, "col": p.col} for p in scenario.obstacles],
            "agents": [
                {"id": idx, "start": [start.row, start.col], "goal": [goal.row, goal.col]}
                for idx, (start, goal) in enumerate(scenario.agents)
            ],
        }


@dataclass(frozen=True)
class RandomSuite:
    """Seeded random-layout evaluation block (identical for all methods)."""

    grid_size: int
    num_obstacles: int
    max_steps: int
    episodes: int
    seed_start: int

    def reset_kwargs(self, episode_index: int, num_agents: int = 1) -> Dict[str, Any]:
        """``env.reset`` kwargs for one evaluation episode.

        The layout seed is ``seed_start + episode_index`` for every method, so
        each method faces byte-for-byte identical layouts (fairness, §43).
        """
        return {
            "seed": self.seed_start + episode_index,
            "randomize": True,
            "num_agents": num_agents,
            "num_obstacles": self.num_obstacles,
        }


@dataclass(frozen=True)
class EvaluationSuite:
    """A frozen suite: fixed scenarios plus one shared random-layout block."""

    name: str
    fixed_scenarios: tuple = field(default_factory=tuple)  # tuple[Scenario, ...]
    random: Optional[RandomSuite] = None
    description: str = ""

    def fixed(self, name: str) -> Scenario:
        """Returns the named fixed scenario (raises KeyError if absent)."""
        for scenario in self.fixed_scenarios:
            if scenario.name == name:
                return scenario
        raise KeyError(
            f"Scenario '{name}' is not part of suite '{self.name}'. "
            f"Available: {[s.name for s in self.fixed_scenarios]}"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def scenario_path(name: str) -> str:
    """Absolute path of a scenario file (``name`` without ``.json``)."""
    filename = name if name.endswith(".json") else f"{name}.json"
    return os.path.join(SCENARIOS_DIR, filename)


def load_scenario(name: str = DEMO_SCENARIO) -> Scenario:
    """Loads and validates a frozen scenario file.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: if the JSON structure or a placement is invalid.
    """
    path = scenario_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scenario file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return scenario_from_dict(data, source=path)


def scenario_from_dict(data: Dict[str, Any], source: str = "<dict>") -> Scenario:
    """Parses a scenario dict, validating structure and grid placements."""
    required = ("name", "height", "width", "obstacles", "agents")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Scenario {source} is missing keys: {missing}")

    height, width = int(data["height"]), int(data["width"])
    if height <= 0 or width <= 0:
        raise ValueError(f"Scenario {source} has a non-positive grid size.")

    # Obstacles are accepted either as ``[row, col]`` pairs (the form written by
    # hand in the scenario files) or as ``{"row": r, "col": c}`` objects (the
    # form sent to Unity, whose JsonUtility cannot parse jagged arrays).
    obstacles_list = []
    for entry in data["obstacles"]:
        if isinstance(entry, dict):
            obstacles_list.append(Position(int(entry["row"]), int(entry["col"])))
        else:
            obstacles_list.append(Position(int(entry[0]), int(entry[1])))
    obstacles = tuple(obstacles_list)
    agents: List = []
    for entry in data["agents"]:
        start = Position(int(entry["start"][0]), int(entry["start"][1]))
        goal = Position(int(entry["goal"][0]), int(entry["goal"][1]))
        agents.append((start, goal))

    if not agents:
        raise ValueError(f"Scenario {source} defines no agents.")

    scenario = Scenario(
        name=str(data["name"]),
        height=height,
        width=width,
        max_steps=int(data.get("max_steps", height * width * 2)),
        obstacles=obstacles,
        agents=tuple(agents),
        seed=data.get("seed"),
        description=str(data.get("description", "")),
    )

    # Structural validation without constructing the environment: reuse the
    # environment's own placement rules by building it once (frozen API v1.0).
    scenario.build_env()
    for obstacle in obstacles:
        if not (0 <= obstacle.row < height and 0 <= obstacle.col < width):
            raise ValueError(f"Scenario {source} has obstacle {obstacle} out of bounds.")
    return scenario


def scalability_scenario(agent_counts: Sequence[int]) -> Scenario:
    """Frozen layout for a Phase 13 sweep over ``agent_counts``.

    The sweep compares the same obstacles with an agent prefix, so the layout
    must define at least ``max(agent_counts)`` agents. Sweeps that fit inside
    the demo layout keep using it; longer sweeps switch to the dedicated
    five-agent layout, whose prefixes reproduce the demo layout exactly.
    """
    if not agent_counts:
        raise ValueError("scalability_scenario needs at least one agent count.")
    needed = max(int(count) for count in agent_counts)
    return load_scenario(DEMO_SCENARIO if needed <= 3 else SCALABILITY_SCENARIO)


def load_suite(name: str = DEFAULT_SUITE) -> EvaluationSuite:
    """Loads a frozen evaluation suite (fixed scenario files + random block)."""
    path = os.path.join(SCENARIOS_DIR, f"{name}.json" if not name.endswith(".json") else name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation suite not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    fixed = tuple(load_scenario(s) for s in data.get("fixed_scenarios", []))
    random_block = data.get("random")
    random_suite = None
    if random_block and random_block.get("enabled", True):
        random_suite = RandomSuite(
            grid_size=int(random_block["grid_size"]),
            num_obstacles=int(random_block.get("num_obstacles", 4)),
            max_steps=int(random_block.get("max_steps", 60)),
            episodes=int(random_block.get("episodes", 30)),
            seed_start=int(random_block.get("seed_start", 1000)),
        )
    if not fixed and random_suite is None:
        raise ValueError(f"Suite '{name}' defines neither fixed scenarios nor a random block.")
    return EvaluationSuite(
        name=str(data.get("name", name)),
        fixed_scenarios=fixed,
        random=random_suite,
        description=str(data.get("description", "")),
    )


def list_scenarios() -> List[str]:
    """Names of all scenario files present in ``experiments/scenarios/``."""
    if not os.path.isdir(SCENARIOS_DIR):
        return []
    return sorted(
        f[:-5] for f in os.listdir(SCENARIOS_DIR)
        if f.endswith(".json") and not f.startswith("eval_suite")
    )


def scenario_manifest(scenario: Scenario) -> Dict[str, Any]:
    """Reproducibility metadata for a scenario (stored with every experiment)."""
    return {
        "scenario": scenario.name,
        "grid": f"{scenario.height}x{scenario.width}",
        "num_agents": scenario.num_agents,
        "max_steps": scenario.max_steps,
        "obstacles": [[p.row, p.col] for p in scenario.obstacles],
        "starts": [[s.row, s.col] for s, _ in scenario.agents],
        "goals": [[g.row, g.col] for _, g in scenario.agents],
    }


def suite_manifest(suite: EvaluationSuite) -> Dict[str, Any]:
    """Reproducibility metadata for a suite, recorded in result files."""
    return {
        "suite": suite.name,
        "fixed_scenarios": [scenario_manifest(s) for s in suite.fixed_scenarios],
        "random": (
            {
                "grid_size": suite.random.grid_size,
                "num_obstacles": suite.random.num_obstacles,
                "max_steps": suite.random.max_steps,
                "episodes": suite.random.episodes,
                "seed_start": suite.random.seed_start,
            }
            if suite.random
            else None
        ),
    }
