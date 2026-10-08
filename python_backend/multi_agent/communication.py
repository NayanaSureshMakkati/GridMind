"""Optional compact inter-agent communication (Developer 3, Phase 9).

Communication is OPTIONAL and disabled by default. When enabled, each active
(non-terminal) agent broadcasts one compact fixed-width message per step,
collected AFTER action selection so the intended action is known:

    message := (sender_id, row, col, intended_action, goal_reached)

Information carried (Phase 9 list):
    position        -> (row, col) of the sender at broadcast time
    intended action -> the action the sender will submit this step
    goal status     -> goal_reached flag (False for senders: terminal agents
                       go silent — they have nothing useful to communicate)

The CommunicationManager collects messages at the start of the step pipeline's
action phase, exposes them to any consumer that asks (e.g. an observation
builder or arbiter extension), and accounts overhead:

    messages_per_step    — collected this step (0 when disabled)
    messages_per_episode — cumulative for the current episode
    total_messages       — cumulative across all episodes

There is no natural-language content and no learned protocol. The manager
never talks to the environment; it only routes what agents already know.
"""

from typing import Dict, List, Optional, Tuple

# (sender_id, row, col, intended_action, goal_reached)
MESSAGE_SIZE = 5


class CommunicationManager:
    """Collects and routes compact per-step agent messages."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.reset(full=True)

    def reset(self, full: bool = False) -> None:
        """Clears per-episode counters (and totals when full=True)."""
        self._current_step_messages: List[Tuple] = []
        self._episode_messages = 0
        if full:
            self.total_messages = 0

    def broadcast_intents(
        self,
        step: int,
        active_ids: List[int],
        positions: Dict[int, Tuple[int, int]],
        actions: Dict[int, int],
        goal_reached: Optional[Dict[int, bool]] = None,
    ) -> int:
        """Collects one message per active agent for this step.

        Args:
            step: Current environment step.
            active_ids: Agents that will act this step (terminal agents stay
                silent — they have nothing useful to communicate).
            positions: {agent_id: (row, col)} at broadcast time.
            actions: {agent_id: intended action} chosen for this step.
            goal_reached: Optional {agent_id: bool} goal status; missing or
                None means "not reached" (senders are active, non-terminal).

        Returns:
            Number of messages collected this step (0 when disabled).
        """
        self._current_step_messages = []
        if not self.enabled:
            return 0
        goal_reached = goal_reached or {}
        for aid in active_ids:
            self._current_step_messages.append(
                (
                    aid,
                    positions[aid][0],
                    positions[aid][1],
                    int(actions[aid]),
                    1 if goal_reached.get(aid, False) else 0,
                )
            )
        self._episode_messages += len(self._current_step_messages)
        self.total_messages += len(self._current_step_messages)
        return len(self._current_step_messages)

    def get_messages(self) -> List[Tuple[int, int, int, int, int]]:
        """Messages collected for the current step.

        Each message is (sender_id, row, col, intended_action, goal_reached).
        """
        return list(self._current_step_messages)

    @property
    def messages_per_step(self) -> int:
        """Messages collected in the most recent step (0 when disabled)."""
        return len(self._current_step_messages)

    @property
    def messages_per_episode(self) -> int:
        """Cumulative messages so far in the current episode."""
        return self._episode_messages
