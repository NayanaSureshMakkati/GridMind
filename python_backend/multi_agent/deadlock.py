"""Deadlock detection and recovery (Developer 3, Phases 7-8).

Detection signals (any one fires a deadlock):
    - stagnation: no agent changed position for `stagnation_steps` consecutive
      steps while at least one agent had not reached its goal;
    - position cycle: an agent's exact full position signature repeated within
      a `cycle_window`-step history (classic A->B->A->... loops);
    - action pattern: the exact per-agent action signature (sorted
      (agent_id, action) pairs, Phase 7 "repeated action patterns") repeated
      at least `action_pattern_repeats` times (default 2) within a
      `cycle_window`-step history while agents remain active. A single prior
      occurrence is NOT enough: with N agents and 5 actions, coincidental
      signature repeats are common in healthy behavior, so the threshold is
      2+ to keep the metric meaningful without losing true loops;
    - mutual blocking: >= `mutual_block_collision_threshold` agent-vs-agent
      collisions within a `collision_window`-step sliding window.

Deadlocks are RECORDED, never hidden, and the environment is never silently
reset. Recovery is configurable and disabled by default so measurements stay
clean (Master Context §30: do not silently hide deadlock events).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from python_backend.environment.constants import Action


@dataclass
class DeadlockEvent:
    """A detected deadlock, recorded per episode."""

    step: int
    reasons: List[str] = field(default_factory=list)


class DeadlockDetector:
    """Online deadlock detector fed one step at a time."""

    def __init__(
        self,
        stagnation_steps: int = 8,
        cycle_window: int = 12,
        collision_window: int = 10,
        mutual_block_collision_threshold: int = 4,
        action_pattern_repeats: int = 2,
    ) -> None:
        if stagnation_steps < 1 or cycle_window < 2 or collision_window < 1:
            raise ValueError("Deadlock windows must be positive (cycle_window >= 2).")
        if action_pattern_repeats < 1:
            raise ValueError("action_pattern_repeats must be >= 1.")
        self.stagnation_steps = stagnation_steps
        self.cycle_window = cycle_window
        self.collision_window = collision_window
        self.mutual_block_collision_threshold = mutual_block_collision_threshold
        self.action_pattern_repeats = action_pattern_repeats

        self.reset()

    def reset(self) -> None:
        """Clears all per-episode detection state."""
        self._stagnation_counter = 0
        self._last_positions: Optional[Dict[int, Tuple[int, int]]] = None
        self._position_history: List[Tuple[int, Tuple[Tuple[int, int], ...]]] = []
        self._action_history: List[Tuple[int, Tuple[Tuple[int, int], ...]]] = []
        self._recent_agent_collisions: List[int] = []
        self.events: List[DeadlockEvent] = []

    def record_step(
        self,
        step: int,
        positions: Dict[int, Tuple[int, int]],
        done_flags: Dict[int, bool],
        agent_collision_ids: List[int],
        actions: Optional[Dict[int, int]] = None,
    ) -> bool:
        """Feeds one step; returns True iff a NEW deadlock fired this step.

        Args:
            step: Current environment step counter.
            positions: {agent_id: (row, col)} after the step.
            done_flags: {agent_id: goal reached}.
            agent_collision_ids: ids of agents that collided WITH AGENTS
                (same-cell/swap) this step.
            actions: Optional {agent_id: action} actually sent this step, used
                for the Phase 7 repeated-action-pattern signal. Pass None (or
                omit) to skip that signal.
        """
        active = [aid for aid, d in done_flags.items() if not d]

        # --- stagnation: nobody among ALL agents moved ------------------
        if self._last_positions is not None and positions == self._last_positions and active:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0
        self._last_positions = dict(positions)

        # --- position cycle ---------------------------------------------
        signature = tuple(positions[aid] for aid in sorted(positions))
        self._position_history.append((step, signature))
        if len(self._position_history) > self.cycle_window:
            self._position_history.pop(0)

        # --- action pattern (Phase 7: repeated action patterns) ----------
        if actions:
            action_sig = tuple(
                (aid, actions[aid]) for aid in sorted(actions)
            )
            self._action_history.append((step, action_sig))
            if len(self._action_history) > self.cycle_window:
                self._action_history.pop(0)
        else:
            self._action_history.clear()

        # --- mutual blocking --------------------------------------------
        self._recent_agent_collisions.extend(agent_collision_ids)
        if len(self._recent_agent_collisions) > self.collision_window * max(1, len(positions)):
            del self._recent_agent_collisions[: -self.collision_window * len(positions)]

        return self._fire_if_deadlocked(step, active)

    def _fire_if_deadlocked(self, step: int, active_ids: List[int]) -> bool:
        reasons: List[str] = []
        if active_ids and self._stagnation_counter >= self.stagnation_steps:
            reasons.append("stagnation")

        current = self._position_history[-1][1] if self._position_history else None
        if (
            active_ids
            and current is not None
            and len(self._position_history) >= 4
            and sum(1 for _, sig in self._position_history[:-1] if sig == current) >= 1
        ):
            reasons.append("position_cycle")

        current_action_sig = self._action_history[-1][1] if self._action_history else None
        # Minimum history length scales with the threshold: the firing step
        # itself plus `action_pattern_repeats` prior occurrences, with one
        # extra entry of margin (threshold 2 -> >= 4, the calibrated default).
        min_action_history = self.action_pattern_repeats + 2
        if (
            active_ids
            and current_action_sig is not None
            and len(self._action_history) >= min_action_history
            and sum(
                1 for _, sig in self._action_history[:-1] if sig == current_action_sig
            ) >= self.action_pattern_repeats
        ):
            reasons.append("action_pattern")

        if len(self._recent_agent_collisions) >= self.mutual_block_collision_threshold:
            reasons.append("mutual_blocking")

        if reasons:
            self.events.append(DeadlockEvent(step=step, reasons=reasons))
            # Reset the counters that fired so one deadlock is counted once.
            if "stagnation" in reasons:
                self._stagnation_counter = 0
            if "position_cycle" in reasons:
                self._position_history.clear()
            if "action_pattern" in reasons:
                self._action_history.clear()
            if "mutual_blocking" in reasons:
                self._recent_agent_collisions.clear()
            return True
        return False

    @property
    def deadlock_count(self) -> int:
        """Number of deadlock events detected this episode."""
        return len(self.events)


class RecoveryManager:
    """Configurable deadlock recovery (Phase 8).

    Kinds:
        "none"             — detection only (default; clean measurement).
        "alternate_action" — when a deadlock fires, every still-active agent
                             takes its lowest-id valid non-STAY action once
                             per episode (deterministic nudge out of cycles).
    """

    KINDS = ("none", "alternate_action")

    def __init__(self, kind: str = "none", max_recoveries_per_episode: int = 1) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"Unknown recovery kind '{kind}'. Available: {self.KINDS}")
        self.kind = kind
        self.max_recoveries_per_episode = max_recoveries_per_episode
        self.recovery_count = 0

    def reset(self) -> None:
        """Resets the per-episode recovery budget."""
        self.recovery_count = 0

    def should_recover(self) -> bool:
        """True iff a recovery action override should be applied this step."""
        if self.kind == "none":
            return False
        if self.recovery_count >= self.max_recoveries_per_episode:
            return False
        self.recovery_count += 1
        return True

    def recovery_actions(self, env, active_ids: List[int]) -> Dict[int, int]:
        """Builds the one-step override action map (deterministic)."""
        overrides: Dict[int, int] = {}
        for aid in active_ids:
            valid = env.get_valid_actions(aid)
            non_stay = [a for a in valid if a != int(Action.STAY)]
            overrides[aid] = non_stay[0] if non_stay else int(Action.STAY)
        return overrides
