"""Reference Python client for the Python <-> Unity protocol (Developer 4).

Two uses:

1. **Test/diagnostic client** — :class:`GridMindClient` speaks the exact same
   protocol as ``NetworkManager.cs`` (newline-delimited JSON over localhost
   TCP). The end-to-end test drives a real server with it, and the
   troubleshooting guide tells developers to use it when Unity misbehaves, so
   the protocol can be verified without Unity in the loop.

2. **Headless viewer** — ``python -m python_backend.integration.client`` connects
   to a running server and prints the ASCII grid plus session statistics, which
   makes the full loop observable on a machine without Unity.

Unity's implementation must stay behaviourally identical to this client.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import Any, Dict, List, Optional

from python_backend.integration import protocol
from python_backend.integration.protocol import ProtocolError

DEFAULT_TIMEOUT = 5.0


class GridMindClient:
    """Blocking localhost client for the GridMind protocol."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        timeout: float = DEFAULT_TIMEOUT,
        verbose: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.verbose = verbose
        self.socket: Optional[socket.socket] = None
        self._buffer = ""            # partial (unterminated) line
        self._lines: List[str] = []  # complete lines not yet consumed
        self.received: List[Dict[str, Any]] = []

    # -- connection ---------------------------------------------------------

    def connect(self) -> "GridMindClient":
        """Connects and sends the ``hello`` handshake."""
        self.socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.socket.settimeout(self.timeout)
        self.send({"type": "hello", "client": "python-reference-client",
                   "protocol": protocol.PROTOCOL_VERSION})
        return self

    def close(self) -> None:
        """Closes the socket (idempotent)."""
        if self.socket is not None:
            try:
                self.socket.close()
            finally:
                self.socket = None

    def __enter__(self) -> "GridMindClient":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- send / receive -----------------------------------------------------

    def send(self, message: Dict[str, Any]) -> None:
        """Sends one message (raises RuntimeError when not connected)."""
        if self.socket is None:
            raise RuntimeError("Client is not connected: call connect() first.")
        payload = protocol.encode(message)
        if self.verbose:
            print(f"[client] -> {message.get('type')} {message}", flush=True)
        self.socket.sendall(payload)

    def send_control(self, command: str, **extra: Any) -> None:
        """Sends a control command (``start``/``pause``/``resume``/``reset``/...)."""
        self.send({"type": "control", "command": command, **extra})

    def send_actions(self, actions: Dict[int, int]) -> None:
        """Sends client-supplied actions (external/manual mode)."""
        self.send({"type": "actions", "actions": {str(k): int(v) for k, v in actions.items()}})

    def receive(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Receives the next message, or None on timeout/EOF."""
        if self.socket is None:
            raise RuntimeError("Client is not connected: call connect() first.")
        effective = self.timeout if timeout is None else timeout
        deadline = time.perf_counter() + effective
        while True:
            if self._lines:
                message = protocol.decode(self._lines.pop(0))
                self._record(message)
                return message

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            self.socket.settimeout(min(remaining, self.timeout))
            try:
                chunk = self.socket.recv(protocol.MAX_LINE_BYTES)
            except socket.timeout:
                return None
            if not chunk:
                return None
            self._buffer += chunk.decode(protocol.ENCODING)
            new_lines, self._buffer = protocol.split_lines(self._buffer)
            self._lines.extend(new_lines)

    def _record(self, message: Dict[str, Any]) -> None:
        self.received.append(message)
        if self.verbose:
            print(f"[client] <- {message.get('type')}", flush=True)

    def wait_for(self, message_type: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Receives messages until one of ``message_type`` arrives.

        Raises:
            TimeoutError: if no such message arrives in time.
        """
        deadline = time.perf_counter() + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"No '{message_type}' message within {timeout}s.")
            message = self.receive(timeout=remaining)
            if message is None:
                continue
            if message.get("type") == message_type:
                return message
            if message.get("type") == "error":
                raise RuntimeError(f"Server error: {message.get('code')}: {message.get('detail')}")

    def wait_for_ack(self, command: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Waits for the ``ack`` of a specific control command.

        Several acks can be in flight (a ``hello`` handshake is acknowledged
        too), so the command name is matched explicitly.
        """
        deadline = time.perf_counter() + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"No ack for '{command}' within {timeout}s.")
            message = self.receive(timeout=remaining)
            if message is None:
                continue
            if message.get("type") == "ack" and message.get("command") == command:
                return message
            if message.get("type") == "error":
                raise RuntimeError(
                    f"Server error for '{command}': {message.get('code')}: {message.get('detail')}"
                )

    def drain(self, timeout: float = 0.2) -> List[Dict[str, Any]]:
        """Collects messages available within ``timeout`` (non-blocking-ish)."""
        collected: List[Dict[str, Any]] = []
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            message = self.receive(timeout=max(0.0, deadline - time.perf_counter()))
            if message is None:
                break
            collected.append(message)
        return collected


# ---------------------------------------------------------------------------
# Headless ASCII viewer (no Unity required)
# ---------------------------------------------------------------------------


def render_grid(scenario: Dict[str, Any], state: Dict[str, Any]) -> str:
    """ASCII render of the logical grid from ``scenario`` + ``state`` messages.

    ``.`` free, ``#`` obstacle, ``g`` goal, ``A<i>`` agent, ``@`` agent on goal.
    """
    height, width = int(scenario["height"]), int(scenario["width"])
    grid = [["." for _ in range(width)] for _ in range(height)]
    for obstacle in scenario["obstacles"]:
        grid[int(obstacle["row"])][int(obstacle["col"])] = "#"
    for agent in scenario["agents"]:
        row, col = agent["goal"]
        grid[row][col] = "g"
    for agent in state["agents"]:
        row, col = agent["position"]
        grid[row][col] = "@" if grid[row][col] == "g" else f"A{agent['id']}"
    return "\n".join(" ".join(f"{cell:>2}" for cell in row) for row in grid)


def format_status(state: Dict[str, Any], step_result: Optional[Dict[str, Any]] = None) -> str:
    """One-line status summary for the headless viewer."""
    stats = state.get("stats", {})
    parts = [
        f"episode {state.get('episode')}/{stats.get('episode_limit')}",
        f"step {state.get('step')}/{state.get('max_steps')}",
        f"algorithm={state.get('algorithm')}",
        f"collisions={stats.get('collisions')}",
        f"deadlocks={stats.get('deadlocks')}",
        f"reward={stats.get('total_reward')}",
    ]
    if step_result is not None:
        parts.append(f"step_reward={step_result.get('rewards')}")
        if step_result.get("collisions"):
            parts.append(f"collisions_now={step_result['collisions']}")
    return "  ".join(str(p) for p in parts)


def run_headless_viewer(
    host: str,
    port: int,
    episodes: int,
    show_grid: bool = False,
) -> int:
    """Connects, starts the session and prints status until ``episodes`` end."""
    with GridMindClient(host=host, port=port, verbose=False) as client:
        welcome = client.wait_for("welcome")
        scenario = client.wait_for("scenario")
        print(f"connected: algorithm={welcome.get('algorithm')} "
              f"trained={welcome.get('trained')} agents={welcome.get('num_agents')}")
        print(f"scenario: {scenario.get('name')} "
              f"{scenario.get('height')}x{scenario.get('width')}")
        client.send_control("start")
        client.wait_for_ack("start")

        finished = 0
        last_state: Optional[Dict[str, Any]] = None
        while finished < episodes:
            message = client.receive(timeout=30.0)
            if message is None:
                print("timeout waiting for the server", file=sys.stderr)
                return 1
            kind = message["type"]
            if kind == "state":
                last_state = message
                if show_grid:
                    print(f"\n{render_grid(scenario, message)}")
                print(format_status(message))
            elif kind == "step_result":
                if show_grid and last_state is not None:
                    print(f"  -> {format_status(last_state, message)}")
            elif kind == "episode_end":
                finished += 1
                print(
                    f"EPISODE {message['episode']} END: steps={message['steps']} "
                    f"success={message['success']} terminated={message['terminated']} "
                    f"truncated={message['truncated']} collisions={message['collisions']} "
                    f"deadlocks={message['deadlocks']} messages={message['messages']} "
                    f"goals={message['goal_completion_rate']}"
                )
            elif kind == "error":
                print(f"server error {message['code']}: {message['detail']}", file=sys.stderr)
                return 1
        client.send_control("quit")
        client.wait_for_ack("quit")
        client.drain(timeout=0.3)
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI parser for the reference client / headless viewer."""
    parser = argparse.ArgumentParser(
        description="GridMind reference client (headless ASCII viewer for the Unity protocol).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=1, help="episodes to watch before exiting")
    parser.add_argument("--grid", action="store_true",
                        help="print the ASCII grid on every state message")
    parser.add_argument("--verbose", action="store_true", help="print every message")
    return parser


def main(argv: Optional[list] = None) -> int:
    """Entry point for ``python -m python_backend.integration.client``."""
    args = build_argument_parser().parse_args(argv)
    try:
        return run_headless_viewer(args.host, args.port, args.episodes, show_grid=args.grid)
    except (ConnectionRefusedError, OSError) as exc:
        print(f"could not connect to {args.host}:{args.port} ({exc}).", file=sys.stderr)
        print("Start the server first: python -m python_backend.integration.server", file=sys.stderr)
        return 2
    except (RuntimeError, TimeoutError, ProtocolError) as exc:
        print(f"protocol failure: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
