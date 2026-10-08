"""JSONL event log for live sessions (visualization telemetry).

When a visualization runs with ``--event-log PATH`` the server appends one JSON
object per line so every agent movement, collision, goal and deadlock can be
read back afterwards (spreadsheet, notebook, diff, replay tooling):

    {"ts": "...", "type": "scenario", "name": "demo_10x10_3agents", ...}
    {"ts": "...", "type": "step_result", "step": 12,
     "movements": [{"agent_id": 0, "action_name": "RIGHT", "from": [2, 3],
                    "to": [2, 4], "target": [2, 4], "moved": true,
                    "reward": -1.0}, ...],
     "collisions": [{"agent_id": 1, "type": "OBSTACLE", "position": [4, 5],
                     "target": [4, 4], "other": -1}], ...}
    {"ts": "...", "type": "episode_end", "episode": 1, "success": true, ...}

The wire protocol is unchanged; this file simply mirrors the messages the
server already streams. Records are flushed per line so a run that is stopped
with Ctrl+C still leaves a readable log.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


class StepEventLogger:
    """Append-only JSONL writer for live-session messages."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append (not truncate): consecutive runs keep their history, and the
        # session_start record marks where each run begins.
        self._file = self.path.open("a", encoding="utf-8")
        self.log({"type": "session_start"})

    def log(self, record: Dict[str, Any]) -> None:
        """Appends one JSON object as a single line and flushes it."""
        payload = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **record}
        self._file.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
        self._file.flush()

    def log_message(self, message: Dict[str, Any]) -> None:
        """Mirrors a protocol message into the log (adds the timestamp only)."""
        self.log(message)

    def close(self) -> None:
        """Closes the underlying file (safe to call repeatedly)."""
        if self._file is not None and not self._file.closed:
            self._file.close()

    def __enter__(self) -> "StepEventLogger":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def open_event_log(path: Optional[Union[str, Path]]) -> Optional[StepEventLogger]:
    """Returns a logger for ``path``, or None when no path is configured."""
    if not path:
        return None
    return StepEventLogger(path)
