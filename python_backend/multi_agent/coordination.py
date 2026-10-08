"""Rule-based coordination primitives (Developer 3, Phase 4).

******************************************************************************
*  RULE-BASED COORDINATION — explicitly NOT learned coordination.            *
*  Nothing in this module trains, learns, or uses a neural network.          *
******************************************************************************

The arbiter runs BEFORE env.step(): it rewrites proposed actions so that
lower-priority agents yield (action becomes STAY) to higher-priority
claimants. The frozen environment still performs all physical conflict
resolution exactly as specified in Environment API v1.0 — this layer only
prevents requests, it never moves an agent itself.

Priority policy (deterministic, lower tuple = higher priority):
    1. shorter Manhattan distance to the agent's goal,
    2. then longer waiting time (steps since last position change),
    3. then lower agent id.

Resolution rules:
    - SWAP: two agents proposing to cross each other both wait (a one-sided
      sidestep would still collide with the now-stationary peer inside the
      environment's simultaneous arbiter).
    - OCCUPIED: a mover whose target cell is (or remains) held by a
      stationary/yielded agent waits.
    - SAME_CELL: multiple movers targeting one cell — the highest-priority
      claimant proceeds, the rest wait.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from python_backend.environment.constants import ACTION_DELTAS, Action, Position

PRIORITY_LABEL = "rule_based"  # identification label for experiment metadata


def manhattan_distance(a: Position, b: Position) -> int:
    """Manhattan distance between two grid positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class CoordinationEvent:
    """One arbitration decision, recorded for collision/coordination analysis."""

    step: int
    kind: str  # "swap" | "occupied" | "same_cell" | "cascade_limit"
    agent_id: int
    other_agent_id: Optional[int] = None
    cell: Optional[Position] = None


class PriorityRule:
    """Deterministic priority ordering over agents (see module docstring)."""

    def __init__(self, env) -> None:
        self.env = env

    def priority_key(self, agent_id: int, waiting_time: int) -> Tuple[int, int, int]:
        """Lower tuple = higher priority (sorted ascending)."""
        pos = self.env.get_agent_position(agent_id)
        goal = self.env.get_goal_position(agent_id)
        return (manhattan_distance(pos, goal), -waiting_time, agent_id)


class RuleBasedArbiter:
    """Pre-step yield arbiter implementing RULE-BASED COORDINATION."""

    def __init__(
        self,
        env,
        waiting_times: Dict[int, int],
        allow_swap_wait: bool = True,
    ) -> None:
        """Args:
        env: GridWorld (frozen API).
        waiting_times: {agent_id: steps since last position change} snapshot
            for the current step (mutated copies are fine).
        allow_swap_wait: If False, swap conflicts are left to the environment
            (both agents bounce; counts as agent collisions).
        """
        self.env = env
        self.priority_rule = PriorityRule(env)
        self.waiting_times = dict(waiting_times)
        self.allow_swap_wait = allow_swap_wait

    def resolve(
        self, step: int, proposed_actions: Dict[int, int]
    ) -> Tuple[Dict[int, int], List[CoordinationEvent]]:
        """Rewrites proposed actions so lower-priority agents yield.

        Args:
            step: Current environment step (for event bookkeeping).
            proposed_actions: {agent_id: action} — must already contain only
                actions valid for each agent (env.get_valid_actions()).

        Returns:
            (adjusted_actions, events): actions safe to pass to env.step()
            together with the arbitration decisions taken.
        """
        actions = dict(proposed_actions)
        events: List[CoordinationEvent] = []
        if len(actions) < 2:
            return actions, events

        cur = {aid: self.env.get_agent_position(aid) for aid in actions}
        prop = {aid: cur[aid] + ACTION_DELTAS[Action(actions[aid])] for aid in actions}

        rounds = 0
        changed = True
        while changed and rounds <= len(actions):
            changed = False
            rounds += 1

            # --- Rule 1: head-on swaps -> both wait -------------------------
            if self.allow_swap_wait:
                ids = sorted(actions)
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        a, b = ids[i], ids[j]
                        if (
                            prop[a] == cur[b]
                            and prop[b] == cur[a]
                            and cur[a] != cur[b]
                            and Action(actions[a]) != Action.STAY
                        ):
                            actions[a] = int(Action.STAY)
                            actions[b] = int(Action.STAY)
                            prop[a], prop[b] = cur[a], cur[b]
                            events.append(CoordinationEvent(step, "swap", a, b, cur[b]))
                            events.append(CoordinationEvent(step, "swap", b, a, cur[a]))
                            changed = True

            # --- Rule 2: mover into stationary/yielded agent's cell ---------
            for aid in sorted(actions):
                if Action(actions[aid]) == Action.STAY:
                    continue
                for other in sorted(actions):
                    if other == aid:
                        continue
                    other_moves = Action(actions[other]) != Action.STAY
                    if not other_moves and prop[aid] == cur[other]:
                        actions[aid] = int(Action.STAY)
                        prop[aid] = cur[aid]
                        events.append(CoordinationEvent(step, "occupied", aid, other, cur[other]))
                        changed = True
                        break

            # --- Rule 3: same-cell contest -> priority winner proceeds ------
            claims: Dict[Position, List[int]] = {}
            for aid in sorted(actions):
                if Action(actions[aid]) != Action.STAY:
                    claims.setdefault(prop[aid], []).append(aid)
            for cell, claimants in claims.items():
                if len(claimants) < 2:
                    continue
                ranked = sorted(
                    claimants,
                    key=lambda a: self.priority_rule.priority_key(a, self.waiting_times.get(a, 0)),
                )
                for loser in ranked[1:]:
                    actions[loser] = int(Action.STAY)
                    prop[loser] = cur[loser]
                    events.append(
                        CoordinationEvent(step, "same_cell", loser, ranked[0], cell)
                    )
                    changed = True

        if rounds > len(actions):
            # Fail-safe: unresolved cascade -> everyone waits (recorded).
            for aid in sorted(actions):
                if Action(actions[aid]) != Action.STAY:
                    actions[aid] = int(Action.STAY)
                    events.append(CoordinationEvent(step, "cascade_limit", aid))

        return actions, events
