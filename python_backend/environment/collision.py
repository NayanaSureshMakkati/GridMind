"""Simultaneous conflict resolution and collision arbitration engine."""

from enum import Enum, auto
from typing import Dict, List, Set, Tuple
from python_backend.environment.constants import Position


class CollisionType(Enum):
    """Categorization of physical grid conflicts."""

    NONE = auto()
    BOUNDARY = auto()
    OBSTACLE = auto()
    SAME_CELL = auto()
    SWAP = auto()


class CollisionReport:
    """Records collision outcomes for an individual agent during a step."""

    def __init__(self, collision_type: CollisionType = CollisionType.NONE, involved_agent_id: int | None = None) -> None:
        self.collision_type = collision_type
        self.involved_agent_id = involved_agent_id

    @property
    def has_collision(self) -> bool:
        return self.collision_type != CollisionType.NONE


class ConflictArbiter:
    """Resolves simultaneous multi-agent proposals without sequential bias."""

    @staticmethod
    def resolve_simultaneous_moves(
        current_positions: Dict[int, Position],
        proposed_positions: Dict[int, Position],
        obstacles: Set[Position],
        grid_height: int,
        grid_width: int,
    ) -> Tuple[Dict[int, Position], Dict[int, CollisionReport]]:
        """Arbitrates proposed moves simultaneously and bounces colliding agents.

        Resolution Algorithm:
        1. Flag static boundary and obstacle breaches.
        2. Detect multi-agent swap conflicts (A -> B's pos and B -> A's pos).
        3. Detect multi-agent same-cell conflicts (A and B -> same destination).
        4. Cascade bounce-backs: Agents colliding with stationary or previously bounced agents
           must revert to their original positions.
        """
        final_positions: Dict[int, Position] = {}
        reports: Dict[int, CollisionReport] = {
            agent_id: CollisionReport() for agent_id in current_positions
        }

        # Step 1: Detect static collisions (Boundary & Obstacle)
        bounced_agents: Set[int] = set()
        for agent_id, proposed in proposed_positions.items():
            if not (0 <= proposed.row < grid_height and 0 <= proposed.col < grid_width):
                bounced_agents.add(agent_id)
                reports[agent_id] = CollisionReport(CollisionType.BOUNDARY)
            elif proposed in obstacles:
                bounced_agents.add(agent_id)
                reports[agent_id] = CollisionReport(CollisionType.OBSTACLE)

        # Step 2: Detect swap conflicts (Edge crossings)
        agent_ids = list(current_positions.keys())
        for i in range(len(agent_ids)):
            a_id = agent_ids[i]
            for j in range(i + 1, len(agent_ids)):
                b_id = agent_ids[j]
                if (
                    proposed_positions[a_id] == current_positions[b_id]
                    and proposed_positions[b_id] == current_positions[a_id]
                    and current_positions[a_id] != current_positions[b_id]
                ):
                    bounced_agents.add(a_id)
                    bounced_agents.add(b_id)
                    reports[a_id] = CollisionReport(CollisionType.SWAP, b_id)
                    reports[b_id] = CollisionReport(CollisionType.SWAP, a_id)

        # Step 3: Detect same-cell conflicts (Vertex collisions)
        dest_counts: Dict[Position, List[int]] = {}
        for agent_id, pos in proposed_positions.items():
            if agent_id not in bounced_agents:
                dest_counts.setdefault(pos, []).append(agent_id)

        for pos, claimants in dest_counts.items():
            if len(claimants) > 1:
                for a_id in claimants:
                    bounced_agents.add(a_id)
                    reports[a_id] = CollisionReport(CollisionType.SAME_CELL)

        # Step 4: Iterative cascade resolution
        # If an agent targets a cell that remains occupied by a bounced/stationary agent, it must bounce.
        changed = True
        while changed:
            changed = False
            # Tentative occupied spaces
            occupied: Set[Position] = {
                current_positions[aid] if aid in bounced_agents else proposed_positions[aid]
                for aid in agent_ids
            }
            for a_id in agent_ids:
                if a_id not in bounced_agents:
                    dest = proposed_positions[a_id]
                    # Check if destination overlaps an agent whose final spot is stationary
                    for other_id in agent_ids:
                        if a_id != other_id:
                            other_target = (
                                current_positions[other_id]
                                if other_id in bounced_agents
                                else proposed_positions[other_id]
                            )
                            if dest == other_target:
                                bounced_agents.add(a_id)
                                reports[a_id] = CollisionReport(CollisionType.SAME_CELL, other_id)
                                changed = True
                                break

        # Calculate resolved positions
        for a_id in agent_ids:
            if a_id in bounced_agents:
                final_positions[a_id] = current_positions[a_id]
            else:
                final_positions[a_id] = proposed_positions[a_id]

        return final_positions, reports