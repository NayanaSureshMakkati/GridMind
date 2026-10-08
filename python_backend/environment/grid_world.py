"""Master GridWorld Environment Engine for GridMind (Environment API v1.0)."""

from collections import deque
import random
from typing import Any, Dict, List, Set, Tuple
from python_backend.environment.agent import Agent
from python_backend.environment.collision import CollisionReport, CollisionType, ConflictArbiter
from python_backend.environment.constants import Action, ACTION_DELTAS, Position
from python_backend.environment.observation import ObservationBuilder
from python_backend.environment.reward import RewardConfig


class GridWorld:
    """Discrete Multi-Agent Navigation Environment."""

    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        agents_config: List[Tuple[Position, Position]] | None = None,
        obstacles: Set[Position] | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 100,
        seed: int | None = None,
    ) -> None:
        self.height = height
        self.width = width
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_config = reward_config or RewardConfig()
        self.rng = random.Random(seed)

        self.obstacles: Set[Position] = set(obstacles) if obstacles else set()
        self.agents: Dict[int, Agent] = {}

        if agents_config:
            for idx, (start, goal) in enumerate(agents_config):
                self._validate_placement(start, f"Agent {idx} start")
                self._validate_placement(goal, f"Agent {idx} goal")
                self.agents[idx] = Agent(idx, start, goal)

        self._last_collision_reports: Dict[int, CollisionReport] = {}

    def _validate_placement(self, pos: Position, label: str) -> None:
        if not self.is_valid_position(pos):
            raise ValueError(f"{label} position {pos} is out of bounds.")
        if pos in self.obstacles:
            raise ValueError(f"{label} position {pos} conflicts with an obstacle.")

    # ================= Public Contract API =================

    def reset(
        self,
        seed: int | None = None,
        randomize: bool = False,
        num_agents: int = 1,
        num_obstacles: int = 0,
    ) -> Dict[int, List[int]]:
        """Resets the environment and returns the initial observation for all agents."""
        if seed is not None:
            self.rng = random.Random(seed)

        self.current_step = 0

        if randomize:
            self._generate_random_layout(num_agents, num_obstacles)
        else:
            for agent in self.agents.values():
                agent.reset()

        self._last_collision_reports = {aid: CollisionReport() for aid in self.agents}
        return self.get_all_observations()

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, List[int]], Dict[int, float], Dict[int, bool], bool, Dict[str, Any]]:
        """Simultaneously processes all agent actions.

        Returns:
            observations: Dict[agent_id, List[int]]
            rewards: Dict[agent_id, float]
            dones: Dict[agent_id, bool] (per-agent goal attainment)
            terminated: bool (True if all agents reached goals)
            info: Dict[str, Any] (diagnostic metadata, truncated flag, collisions)
        """
        self.current_step += 1

        # 1. Collect current and proposed positions
        curr_positions: Dict[int, Position] = {
            aid: agent.current_position for aid, agent in self.agents.items()
        }
        proposed_positions: Dict[int, Position] = {}

        for aid, agent in self.agents.items():
            if agent.done:
                # Terminal agents hold position
                proposed_positions[aid] = agent.current_position
            else:
                act = Action(actions.get(aid, Action.STAY))
                proposed_positions[aid] = agent.current_position + ACTION_DELTAS[act]

        # 2. Arbitrate conflicts simultaneously
        final_positions, reports = ConflictArbiter.resolve_simultaneous_moves(
            current_positions=curr_positions,
            proposed_positions=proposed_positions,
            obstacles=self.obstacles,
            grid_height=self.height,
            grid_width=self.width,
        )
        self._last_collision_reports = reports

        # 3. Calculate rewards and update state
        rewards: Dict[int, float] = {}
        for aid, agent in self.agents.items():
            act = Action(actions.get(aid, Action.STAY))
            reward = self._compute_agent_reward(aid, act, reports[aid], final_positions[aid])
            reached_goal = final_positions[aid] == agent.goal_position
            agent.record_step(final_positions[aid], reward, reached_goal)
            rewards[aid] = reward

        # 4. Determine termination and truncation
        all_goals_reached = all(agent.done for agent in self.agents.values())
        truncated = self.current_step >= self.max_steps
        terminated = all_goals_reached

        observations = self.get_all_observations()
        dones = {aid: agent.done for aid, agent in self.agents.items()}

        info = {
            "truncated": truncated,
            "step": self.current_step,
            "collisions": {
                aid: rep.collision_type.name for aid, rep in reports.items() if rep.has_collision
            },
        }

        return observations, rewards, dones, terminated, info

    def get_agent_state(self, agent_id: int) -> Dict[str, Any]:
        """Returns the full internal state packet for an agent."""
        agent = self.agents[agent_id]
        return {
            "agent_id": agent.agent_id,
            "current_position": agent.current_position,
            "goal_position": agent.goal_position,
            "steps": agent.steps,
            "total_reward": agent.total_reward,
            "done": agent.done,
        }

    def get_all_states(self) -> Dict[int, Dict[str, Any]]:
        """Returns internal state packets for all agents."""
        return {aid: self.get_agent_state(aid) for aid in self.agents}

    def get_agent_observation(self, agent_id: int) -> List[int]:
        """Returns the standard 12-element observation vector for agent_id."""
        return ObservationBuilder.build_vector(
            agent_id=agent_id,
            agent_positions={aid: a.current_position for aid, a in self.agents.items()},
            goal_positions={aid: a.goal_position for aid, a in self.agents.items()},
            obstacles=self.obstacles,
            grid_height=self.height,
            grid_width=self.width,
        )

    def get_all_observations(self) -> Dict[int, List[int]]:
        """Returns standard observation vectors for all agents."""
        return {aid: self.get_agent_observation(aid) for aid in self.agents}

    def get_valid_actions(self, agent_id: int) -> List[int]:
        """Identifies actions that avoid physical obstacles and boundaries."""
        agent = self.agents[agent_id]
        if agent.done:
            return [Action.STAY]

        valid: List[int] = []
        for act in Action:
            target = agent.current_position + ACTION_DELTAS[act]
            if self.is_valid_position(target) and not self.is_obstacle(target):
                valid.append(act)
        return valid

    def get_agent_position(self, agent_id: int) -> Position:
        return self.agents[agent_id].current_position

    def get_goal_position(self, agent_id: int) -> Position:
        return self.agents[agent_id].goal_position

    def is_obstacle(self, position: Position) -> bool:
        return position in self.obstacles

    def is_valid_position(self, position: Position) -> bool:
        return 0 <= position.row < self.height and 0 <= position.col < self.width

    def is_goal_reached(self, agent_id: int) -> bool:
        return self.agents[agent_id].done

    def is_collision(self, agent_id: int) -> bool:
        return self._last_collision_reports.get(agent_id, CollisionReport()).has_collision

    def render(self) -> str:
        """Renders an ASCII text depiction of the logical environment."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for obs in self.obstacles:
            grid[obs.row][obs.col] = "#"
        for aid, agent in self.agents.items():
            g = agent.goal_position
            grid[g.row][g.col] = f"G{aid}"
        for aid, agent in self.agents.items():
            p = agent.current_position
            grid[p.row][p.col] = f"A{aid}"

        lines = [" ".join(f"{cell:>3}" for cell in row) for row in grid]
        representation = "\n".join(lines)
        return representation

    def close(self) -> None:
        """Tears down resources."""
        self.agents.clear()
        self.obstacles.clear()

    # ================= Private Internal Helpers =================

    def _compute_agent_reward(
        self, agent_id: int, action: Action, report: CollisionReport, final_pos: Position
    ) -> float:
        agent = self.agents[agent_id]
        if agent.done:
            return 0.0

        if final_pos == agent.goal_position:
            return self.reward_config.goal

        if report.collision_type in (CollisionType.BOUNDARY, CollisionType.OBSTACLE):
            return self.reward_config.obstacle_collision
        if report.collision_type in (CollisionType.SAME_CELL, CollisionType.SWAP):
            return self.reward_config.agent_collision

        if action == Action.STAY:
            return self.reward_config.stay

        return self.reward_config.step

    def _generate_random_layout(self, num_agents: int, num_obstacles: int) -> None:
        all_cells = [
            Position(r, c) for r in range(self.height) for c in range(self.width)
        ]
        self.rng.shuffle(all_cells)

        needed = (num_agents * 2) + num_obstacles
        if needed > len(all_cells):
            raise ValueError("Requested items exceed total grid cell capacity.")

        self.obstacles = set(all_cells[:num_obstacles])
        remaining = all_cells[num_obstacles:]

        self.agents.clear()
        for i in range(num_agents):
            start = remaining[i * 2]
            goal = remaining[(i * 2) + 1]

            # Enforce reachable path guarantee via BFS validation
            attempts = 0
            while (
                not self.is_reachable(start, goal, self.obstacles, self.height, self.width)
                and attempts < 100
            ):
                self.rng.shuffle(remaining)
                start = remaining[i * 2]
                goal = remaining[(i * 2) + 1]
                attempts += 1

            if not self.is_reachable(start, goal, self.obstacles, self.height, self.width):
                raise RuntimeError("Failed to generate a reachable configuration via BFS.")

            self.agents[i] = Agent(i, start, goal)

    @staticmethod
    def is_reachable(
        start: Position,
        goal: Position,
        obstacles: Set[Position],
        height: int,
        width: int,
    ) -> bool:
        """Evaluates path existence using Breadth-First Search (BFS validation only).

        BFS is used strictly for validation (e.g. random layout feasibility).
        It is never exposed as an agent navigation algorithm.

        Args:
            start: Starting cell (need not be free of obstacles for the search).
            goal: Target cell; if it lies outside the grid or in an obstacle the
                function returns False.
            obstacles: Set of blocked cells.
            height: Number of grid rows (search is bounded to [0, height)).
            width: Number of grid columns (search is bounded to [0, width)).

        Returns:
            True if a 4-connected path exists from start to goal inside the grid.
        """
        if not (0 <= goal.row < height and 0 <= goal.col < width):
            return False
        if goal in obstacles:
            return False
        if start == goal:
            return True

        queue: deque[Position] = deque([start])
        visited: Set[Position] = {start} | obstacles

        while queue:
            curr = queue.popleft()
            if curr == goal:
                return True
            for act in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
                neighbor = curr + ACTION_DELTAS[act]
                if (
                    0 <= neighbor.row < height
                    and 0 <= neighbor.col < width
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False