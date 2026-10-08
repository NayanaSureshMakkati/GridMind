"""Python <-> Unity JSON protocol (Developer 4, Phase 1).

Transport
---------
Newline-delimited JSON over a localhost TCP socket. One message per line, UTF-8,
terminated by ``\\n``. Unity is the client, Python is the server (Python owns
the logical simulation, so it listens and Unity connects).

  Unity  --hello/control/actions-->  Python      (client -> server)
  Unity  <--welcome/scenario/state/step_result--  Python (server -> client)

Message catalogue
-----------------
Client -> server
    hello    {"type":"hello","client":"unity","protocol":1}
    control  {"type":"control","command":"start"|"pause"|"resume"|"reset"|
              "step"|"set_speed"|"quit", "speed_ms": 250}
    actions  {"type":"actions","actions":{"0":3,"1":2,"2":4}}
    ping     {"type":"ping"}
    quit     {"type":"quit"}

Server -> client
    welcome      {"type":"welcome","protocol":1,"algorithm":...,"mode":...}
    scenario     {"type":"scenario","height":10,...,"obstacles":[[r,c]],"agents":[...]}
    state        {"type":"state","episode":..,"step":..,"algorithm":..,"agents":[...],
                  "stats":{...}}
    step_result  {"type":"step_result","step":..,"rewards":{..},"collisions":[...],
                  "goals_reached":[..],"deadlocks":..}
    episode_end  {"type":"episode_end","episode":..,"terminated":..,"truncated":.., ...}
    ack          {"type":"ack","command":"start","ok":true}
    error        {"type":"error","code":"BAD_MESSAGE","detail":"..."}
    pong         {"type":"pong"}

Coordinate rule (authoritative)
-------------------------------
Every position on the wire is a LOGICAL ``[row, col]`` pair (Environment API v1.0
convention). Unity performs the single documented conversion to 3D. The backend
never sends Unity coordinates, so the two conventions can never mix.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

PROTOCOL_VERSION = 1
NEWLINE = "\n"
ENCODING = "utf-8"
#: Guard against a runaway/corrupt client filling memory with one giant "line".
MAX_LINE_BYTES = 1 << 20  # 1 MiB

CLIENT_MESSAGE_TYPES = ("hello", "control", "actions", "ping", "quit")
SERVER_MESSAGE_TYPES = (
    "welcome", "scenario", "state", "step_result", "episode_end", "ack", "error", "pong",
)
CONTROL_COMMANDS = ("start", "pause", "resume", "reset", "step", "set_speed", "quit")

#: Action id -> name (fixed mapping, Environment API v1.0 section 2).
ACTION_NAMES: Dict[int, str] = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}


class ProtocolError(ValueError):
    """Raised when a message is malformed or violates the protocol contract."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def encode(message: Dict[str, Any]) -> bytes:
    """Serializes a message dict to a newline-terminated UTF-8 byte line."""
    if not isinstance(message, dict) or "type" not in message:
        raise ProtocolError("BAD_MESSAGE", "every message needs a 'type' field")
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False)
    return (payload + NEWLINE).encode(ENCODING)


def decode(line: bytes | str) -> Dict[str, Any]:
    """Parses one wire line into a message dict (server-validated)."""
    if isinstance(line, bytes):
        if len(line) > MAX_LINE_BYTES:
            raise ProtocolError("TOO_LARGE", f"message exceeds {MAX_LINE_BYTES} bytes")
        text = line.decode(ENCODING, errors="strict")
    else:
        text = line
    text = text.strip()
    if not text:
        raise ProtocolError("BAD_MESSAGE", "empty message")
    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError("BAD_JSON", f"could not parse JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("BAD_MESSAGE", "message must be a JSON object")
    if "type" not in message or not isinstance(message["type"], str):
        raise ProtocolError("BAD_MESSAGE", "message is missing a string 'type' field")
    return message


# ---------------------------------------------------------------------------
# Client -> server validation
# ---------------------------------------------------------------------------


def validate_client_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Validates a client message and returns it (raises ProtocolError otherwise)."""
    kind = message["type"]
    if kind not in CLIENT_MESSAGE_TYPES:
        raise ProtocolError("UNKNOWN_TYPE", f"'{kind}' is not a client message type")

    if kind == "control":
        command = message.get("command")
        if command not in CONTROL_COMMANDS:
            raise ProtocolError(
                "BAD_COMMAND",
                f"'command' must be one of {list(CONTROL_COMMANDS)}, got {command!r}",
            )
        if command == "set_speed":
            speed = message.get("speed_ms")
            if not isinstance(speed, int) or speed < 0:
                raise ProtocolError("BAD_COMMAND", "'set_speed' requires integer speed_ms >= 0")
    elif kind == "actions":
        parse_actions(message)  # raises on malformed action maps
    elif kind == "hello":
        protocol = message.get("protocol", PROTOCOL_VERSION)
        if not isinstance(protocol, int):
            raise ProtocolError("BAD_MESSAGE", "'protocol' must be an integer")
        if protocol != PROTOCOL_VERSION:
            raise ProtocolError(
                "PROTOCOL_MISMATCH",
                f"client protocol {protocol} != server protocol {PROTOCOL_VERSION}",
            )
    return message


def parse_actions(message: Dict[str, Any]) -> Dict[int, int]:
    """Extracts ``{agent_id: action_id}`` from an ``actions`` message.

    Accepts both ``{"0": 3}`` (JSON object, as sent by Unity) and
    ``{"agents":[{"id":0,"action":3}]}`` (array form), because JSON object keys
    are strings while the logical agent ids are integers.
    """
    raw = message.get("actions")
    if raw is None and "agents" in message:
        entries = message["agents"]
        if not isinstance(entries, list):
            raise ProtocolError("BAD_ACTIONS", "'agents' must be a list")
        actions: Dict[int, int] = {}
        for entry in entries:
            if not isinstance(entry, dict) or "id" not in entry or "action" not in entry:
                raise ProtocolError("BAD_ACTIONS", "each entry needs 'id' and 'action'")
            actions[_as_int(entry["id"], "id")] = _as_int(entry["action"], "action")
        return actions

    if not isinstance(raw, dict):
        raise ProtocolError("BAD_ACTIONS", "'actions' must be an object mapping agent id to action")
    actions = {}
    for key, value in raw.items():
        agent_id = _as_int(key, "agent id")
        action = _as_int(value, "action")
        if action not in ACTION_NAMES:
            raise ProtocolError(
                "BAD_ACTION_ID", f"action {action} for agent {agent_id} is not in 0..4"
            )
        actions[agent_id] = action
    return actions


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("BAD_ACTIONS", f"{label} {value!r} is not an integer") from exc


# ---------------------------------------------------------------------------
# Server -> client builders
# ---------------------------------------------------------------------------


def welcome_message(
    algorithm: str,
    mode: str,
    action_source: str,
    num_agents: int,
    episode: int = 1,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """First message after a client connects (protocol negotiation + identity)."""
    message: Dict[str, Any] = {
        "type": "welcome",
        "protocol": PROTOCOL_VERSION,
        "server": "gridmind-python",
        "algorithm": algorithm,
        "mode": mode,
        "action_source": action_source,
        "num_agents": num_agents,
        "episode": episode,
    }
    if extra:
        message.update(extra)
    return message


def scenario_message(scenario_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Frozen scenario description (obstacles + starts + goals) for Unity."""
    return {"type": "scenario", **scenario_dict}


def state_message(
    episode: int,
    step: int,
    algorithm: str,
    agents: List[Dict[str, Any]],
    stats: Dict[str, Any],
    action_source: str = "policy",
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Authoritative logical state of every agent plus running session stats.

    Each agent entry is
    ``{"id", "position":[row,col], "goal":[row,col], "observation":[...],
       "status":"NAVIGATING"|"GOAL_REACHED", "reward": <episode total>, "done": bool}``.
    """
    message: Dict[str, Any] = {
        "type": "state",
        "episode": episode,
        "step": step,
        "algorithm": algorithm,
        "action_source": action_source,
        "agents": agents,
        "stats": stats,
    }
    if max_steps is not None:
        message["max_steps"] = max_steps
    return message


def movement_entry(
    agent_id: int,
    action: int,
    from_pos: Any,
    to_pos: Any,
    target: Any,
    moved: bool,
    reward: float,
) -> Dict[str, Any]:
    """One per-agent movement record for ``step_result.movements``.

    ``from_pos``/``to_pos``/``target`` are objects with ``row``/``col`` (the
    environment's ``Position``). ``target`` is the cell the agent tried to enter
    before conflict resolution (== ``from`` when it stayed); ``moved`` is False
    whenever the arbiter bounced the agent back or it held position.
    """
    return {
        "agent_id": int(agent_id),
        "action": int(action),
        "action_name": ACTION_NAMES.get(int(action), f"UNKNOWN({action})"),
        "from": position_list(from_pos.row, from_pos.col),
        "to": position_list(to_pos.row, to_pos.col),
        "target": position_list(target.row, target.col),
        "moved": bool(moved),
        "reward": round(float(reward), 4),
    }


def step_result_message(
    step: int,
    rewards: Dict[int, float],
    collisions: List[Dict[str, Any]],
    goals_reached: List[int],
    deadlocks: int = 0,
    terminated: bool = False,
    truncated: bool = False,
    movements: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Outcome of one simultaneous environment step (Phase 1 example format).

    ``movements`` carries one record per agent (from/to/target/reward) so a
    frontend can render or log every agent's motion; ``collisions`` entries may
    carry ``position``/``target``/``other`` when the session can attribute them.
    """
    message: Dict[str, Any] = {
        "type": "step_result",
        "step": step,
        "rewards": {str(k): round(float(v), 4) for k, v in rewards.items()},
        "collisions": collisions,
        "goals_reached": sorted(goals_reached),
        "deadlocks": int(deadlocks),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if movements is not None:
        message["movements"] = movements
    return message


def episode_end_message(
    episode: int,
    steps: int,
    terminated: bool,
    truncated: bool,
    success: bool,
    rewards: Dict[int, float],
    collisions: int,
    deadlocks: int,
    messages: int,
    goal_completion_rate: float,
) -> Dict[str, Any]:
    """End-of-episode summary (success vs time limit kept separate)."""
    return {
        "type": "episode_end",
        "episode": episode,
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(success),
        "rewards": {str(k): round(float(v), 4) for k, v in rewards.items()},
        "collisions": int(collisions),
        "deadlocks": int(deadlocks),
        "messages": int(messages),
        "goal_completion_rate": round(float(goal_completion_rate), 4),
    }


def ack_message(command: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Acknowledgement of a control command."""
    message: Dict[str, Any] = {"type": "ack", "command": command, "ok": True}
    if extra:
        message.update(extra)
    return message


def error_message(code: str, detail: str, command: Optional[str] = None) -> Dict[str, Any]:
    """Explicit error report (never silently swallowed, Master Context 58)."""
    message: Dict[str, Any] = {"type": "error", "code": code, "detail": detail}
    if command:
        message["command"] = command
    return message


def pong_message() -> Dict[str, Any]:
    """Reply to ``ping`` (connection liveness check)."""
    return {"type": "pong"}


def action_name(action_id: int) -> str:
    """Human-readable action name (for logs and debugging)."""
    return ACTION_NAMES.get(int(action_id), f"UNKNOWN({action_id})")


def position_list(row: int, col: int) -> List[int]:
    """Logical position as ``[row, col]`` (the only position form on the wire)."""
    return [int(row), int(col)]


def collision_entry(
    agent_id: int,
    type_name: str,
    position: Optional[Any] = None,
    target: Optional[Any] = None,
    other: Optional[int] = None,
) -> Dict[str, Any]:
    """One collision record for ``step_result.collisions``.

    The extra fields are optional and additive (protocol stays v1):
    ``position`` is where the agent ended up (always bounced back for a
    collision), ``target`` is the cell it tried to enter, and ``other`` is the
    other agent involved for agent-vs-agent types (``-1`` when none/unknown).
    """
    entry: Dict[str, Any] = {"agent_id": int(agent_id), "type": type_name}
    if position is not None:
        entry["position"] = position_list(position.row, position.col)
    if target is not None:
        entry["target"] = position_list(target.row, target.col)
    if other is not None:
        entry["other"] = int(other)
    return entry


def split_lines(buffer: str) -> Tuple[List[str], str]:
    """Splits a receive buffer into complete lines plus the leftover partial line."""
    if NEWLINE not in buffer:
        if len(buffer.encode(ENCODING)) > MAX_LINE_BYTES:
            raise ProtocolError("TOO_LARGE", f"partial message exceeds {MAX_LINE_BYTES} bytes")
        return [], buffer
    parts = buffer.split(NEWLINE)
    leftover = parts.pop()
    return [p for p in parts if p.strip()], leftover
