"""Python side of the Python <-> Unity link (Developer 4, Phase 2).

A small localhost TCP server (newline-delimited JSON, see
``python_backend/integration/protocol.py``) that:

  1. accepts ONE Unity client (it may reconnect between runs),
  2. sends ``welcome`` + ``scenario`` (the world Unity must build),
  3. steps the authoritative :class:`~python_backend.integration.session.LiveSession`
     at a configurable pace and pushes ``step_result`` + ``state`` every step,
  4. honours control commands (start/pause/resume/reset/step/set_speed/quit),
  5. accepts client-supplied ``actions`` when the session runs in
     ``action_source="external"`` mode (manual debugging — the RL policy is
     still Python's).

No cloud service, no extra dependency: ``socket`` + ``json`` from the standard
library. Every protocol error is answered with an explicit ``error`` message and
logged; nothing is silently swallowed (Master Context §58).

CLI::

    python -m python_backend.integration.server --algorithm independent_dqn
    python -m python_backend.integration.server --port 8765 --speed-ms 150
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

from python_backend.integration.event_log import StepEventLogger
from python_backend.integration import protocol
from python_backend.integration.protocol import ProtocolError
from python_backend.integration.session import LiveSession, SessionConfig

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
POLL_INTERVAL = 0.05
#: How long to wait for a client before giving up when a time limit is set.
ACCEPT_TIMEOUT = 1.0


class GridMindServer:
    """Localhost JSON server driving one live GridMind session.

    Args:
        session: The live session to drive (owns environment + policy).
        host: Interface to bind (localhost by default — never expose this).
        port: TCP port (0 selects a free port; the real one lands in ``self.port``).
        verbose: Print connection/step/command events (troubleshooting aid).
        exit_on_disconnect: Stop serving when the client disconnects.
        exit_when_finished: Stop serving when the session finished all episodes.
        on_message: Optional callback ``(client_id, message)`` for tests/tools.
        event_log: Optional JSONL recorder; when given, every streamed
            ``scenario``/``step_result``/``episode_end`` is mirrored to disk.
    """

    def __init__(
        self,
        session: LiveSession,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        verbose: bool = True,
        exit_on_disconnect: bool = False,
        exit_when_finished: bool = False,
        on_message: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        event_log: Optional[StepEventLogger] = None,
    ) -> None:
        self.session = session
        self.host = host
        self.port = port
        self.verbose = verbose
        self.exit_on_disconnect = exit_on_disconnect
        self.exit_when_finished = exit_when_finished
        self.on_message = on_message
        self.event_log = event_log
        self.running = False          # auto-stepping enabled
        self.stopped = False          # server must shut down
        #: Set once the socket is bound and listening (used by tests and by
        #: ``main.py --mode visualize --headless`` to avoid a connect race).
        self.ready = threading.Event()
        self._conn: Optional[socket.socket] = None
        self._last_step_time = 0.0
        self._listener: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, message: str) -> None:
        """Prints a timestamped log line when verbose."""
        if self.verbose:
            print(f"[gridmind-server] {message}", flush=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def serve_forever(self, max_seconds: Optional[float] = None) -> None:
        """Serves clients until stopped, disconnected (optionally) or timed out.

        Args:
            max_seconds: Optional wall-clock limit; when reached the server
                returns (used by automated end-to-end tests).
        """
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        self.port = listener.getsockname()[1]
        listener.listen(1)
        listener.settimeout(ACCEPT_TIMEOUT)
        self._listener = listener
        self.ready.set()
        self.log(f"listening on {self.host}:{self.port}")

        deadline = time.perf_counter() + max_seconds if max_seconds else None
        try:
            while not self.stopped:
                if deadline is not None and time.perf_counter() >= deadline:
                    self.log("time limit reached; shutting down")
                    break
                try:
                    conn, address = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                self.log(f"client connected from {address[0]}:{address[1]}")
                try:
                    self._serve_client(conn, deadline)
                finally:
                    conn.close()
                    self._conn = None
                    self.log("client disconnected")
                    if self.exit_on_disconnect and not self.stopped:
                        # A clean exit is a normal end of session, not a crash.
                        self.stopped = True
        finally:
            listener.close()
            self._listener = None
            self.ready.clear()
            self.log("server stopped")

    def stop(self) -> None:
        """Requests shutdown (safe to call from another thread)."""
        self.stopped = True

    # ------------------------------------------------------------------
    # Per-client loop
    # ------------------------------------------------------------------

    def _serve_client(self, conn: socket.socket, deadline: Optional[float]) -> None:
        """Handles one connected client until it disconnects or the session ends."""
        self._conn = conn
        conn.settimeout(POLL_INTERVAL)
        buffer = ""

        self._send(conn, self.session.welcome_message())
        self._send(conn, self.session.scenario_message())
        self._send(conn, self.session.state_message())
        if self.event_log is not None:
            self.event_log.log_message(self.session.scenario_message())
        self._last_step_time = time.perf_counter()

        while not self.stopped:
            try:
                chunk = conn.recv(protocol.MAX_LINE_BYTES)
            except socket.timeout:
                chunk = b""
            except OSError as exc:
                self.log(f"receive failed: {exc}")
                return

            if chunk:
                try:
                    lines, buffer = protocol.split_lines(buffer + chunk.decode(protocol.ENCODING))
                except (ProtocolError, UnicodeDecodeError) as exc:
                    self._send(conn, protocol.error_message("BAD_MESSAGE", str(exc)))
                    buffer = ""
                    lines = []
                for line in lines:
                    self._handle_line(conn, line)
                    if self.stopped:
                        return

            if self.running:
                self._maybe_step(conn)

            if deadline is not None and time.perf_counter() >= deadline:
                return
            if self.exit_when_finished and self.session.finished:
                self.log("session finished; shutting down")
                self.stopped = True
                return

    def _handle_line(self, conn: socket.socket, line: str) -> None:
        """Parses and dispatches one client message."""
        try:
            message = protocol.decode(line)
            protocol.validate_client_message(message)
        except ProtocolError as exc:
            self.log(f"protocol error: {exc}")
            self._send(conn, protocol.error_message(exc.code, exc.detail))
            return

        if self.on_message is not None:
            self.on_message("client", message)

        kind = message["type"]
        if kind == "control":
            self._handle_control(conn, message)
        elif kind == "actions":
            self._handle_actions(conn, message)
        elif kind == "ping":
            self._send(conn, protocol.pong_message())
        elif kind == "hello":
            self._send(conn, protocol.ack_message("hello"))
            self._send(conn, self.session.scenario_message())
            self._send(conn, self.session.state_message())
        elif kind == "quit":
            self._send(conn, protocol.ack_message("quit", {"detail": "closing session"}))
            self.log("client requested quit")
            self.stopped = True

    # -- control commands ---------------------------------------------------

    def _handle_control(self, conn: socket.socket, message: Dict[str, Any]) -> None:
        command = message["command"]
        detail: Dict[str, Any] = {}

        if command == "start":
            self.running = True
            self._last_step_time = time.perf_counter()
            detail["running"] = True
        elif command == "pause":
            self.running = False
            detail["running"] = False
        elif command == "resume":
            self.running = True
            self._last_step_time = time.perf_counter()
            detail["running"] = True
        elif command == "reset":
            self.running = False
            self.session.finished = False
            self.session.reset_episode(increment=self.session.episodes_completed > 0)
            detail["episode"] = self.session.episode_index
            self._send(conn, self.session.state_message())
        elif command == "step":
            self.running = False
            self._do_step(conn)
        elif command == "set_speed":
            self.session.config.speed_ms = int(message.get("speed_ms", self.session.config.speed_ms))
            detail["speed_ms"] = self.session.config.speed_ms
        elif command == "quit":
            self.running = False
            detail["detail"] = "closing session"
            self.log("client requested quit")
            self.stopped = True

        self._send(conn, protocol.ack_message(command, detail))
        self.log(f"control '{command}' handled")

    # -- external actions ---------------------------------------------------

    def _handle_actions(self, conn: socket.socket, message: Dict[str, Any]) -> None:
        """Applies client-supplied actions (external/manual mode only)."""
        try:
            actions = protocol.parse_actions(message)
        except ProtocolError as exc:
            self._send(conn, protocol.error_message(exc.code, exc.detail, command="actions"))
            return

        if self.session.config.action_source != "external":
            self._send(conn, protocol.error_message(
                "NOT_EXTERNAL",
                "This session runs in 'policy' mode: actions are decided by the "
                "Python policy. Start the server with --action-source external for "
                "manual stepping.",
                command="actions",
            ))
            return

        try:
            step = self.session.step_once(actions)
        except ValueError as exc:
            self._send(conn, protocol.error_message("BAD_ACTIONS", str(exc), command="actions"))
            return
        self._emit_step(conn, step)

    # ------------------------------------------------------------------
    # Stepping and sending
    # ------------------------------------------------------------------

    def _maybe_step(self, conn: socket.socket) -> None:
        """Steps when enough wall-clock time has passed (paced streaming)."""
        now = time.perf_counter()
        if (now - self._last_step_time) * 1000.0 < self.session.config.speed_ms:
            return
        self._last_step_time = now
        self._do_step(conn)

    def _do_step(self, conn: socket.socket) -> None:
        """Performs one step and streams the resulting messages."""
        if self.session.last_step is not None and (
            self.session.last_step.terminated or self.session.last_step.truncated
        ):
            if self.session.finished:
                self._send(conn, protocol.error_message(
                    "SESSION_FINISHED",
                    f"All {self.session.config.episode_limit} episode(s) are complete. "
                    "Send control 'reset' to start a new run.",
                ))
                self.running = False
            else:
                self.session.reset_episode(increment=True)
                self._send(conn, self.session.state_message())
            return

        try:
            step = self.session.step_once()
        except (RuntimeError, ValueError) as exc:
            self._send(conn, protocol.error_message("STEP_FAILED", str(exc)))
            self.running = False
            return
        self._emit_step(conn, step)

    def _emit_step(self, conn: socket.socket, step) -> None:
        """Sends ``step_result`` then ``state`` (and ``episode_end`` if finished)."""
        step_result = self.session.step_result_message(step)
        self._send(conn, step_result)
        self._send(conn, self.session.state_message())
        if self.event_log is not None:
            self.event_log.log_message(step_result)
        if step.terminated or step.truncated:
            episode_end = self.session.episode_end_message()
            self._send(conn, episode_end)
            if self.event_log is not None:
                self.event_log.log_message(episode_end)
            record = self.session.history[-1]
            self.log(
                f"episode {record['episode']} finished: steps={record['steps']} "
                f"success={record['success']} terminated={record['terminated']} "
                f"truncated={record['truncated']} collisions={record['collisions']} "
                f"deadlocks={record['deadlocks']}"
            )

    def _send(self, conn: socket.socket, message: Dict[str, Any]) -> None:
        """Sends one message; a broken pipe simply ends this client's session."""
        try:
            conn.sendall(protocol.encode(message))
        except OSError as exc:
            self.log(f"send failed ({exc}); client gone")
            self._conn = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI parser for the standalone server."""
    parser = argparse.ArgumentParser(
        description="GridMind Python<->Unity server: streams authoritative logical state.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port (default 8765)")
    parser.add_argument("--algorithm", default="independent_dqn",
                        help="random | q_learning | dqn | independent_dqn | "
                             "rule_based_coordination | coordinated_dqn | ctde_inspired")
    parser.add_argument("--scenario", default="demo_10x10_3agents", help="scenario file name")
    parser.add_argument("--agents", type=int, default=3, help="number of agents (prefix rule)")
    parser.add_argument("--seed", type=int, default=7, help="seed for policies and layouts")
    parser.add_argument("--speed-ms", type=int, default=250, help="milliseconds per step")
    parser.add_argument("--episodes", type=int, default=3, help="episodes before the session ends")
    parser.add_argument("--action-source", choices=("policy", "external"), default="policy",
                        help="who picks the actions: Python policy or the client's actions message")
    parser.add_argument("--model", default=None, help="single-agent checkpoint (.pt/.pkl)")
    parser.add_argument("--model-tag", default=None, help="multi-agent checkpoint tag")
    parser.add_argument("--communication", action="store_true", help="enable message counting")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="stop after this many seconds (useful for smoke tests)")
    parser.add_argument("--event-log", default=None,
                        help="append every step (movements/collisions/goals) as JSON "
                             "lines to this file (e.g. experiments/logs/events.jsonl)")
    parser.add_argument("--quiet", action="store_true", help="suppress server logs")
    parser.add_argument("--exit-on-disconnect", action="store_true",
                        help="stop when Unity disconnects")
    return parser


def main(argv: Optional[list] = None) -> int:
    """Entry point for ``python -m python_backend.integration.server``."""
    args = build_argument_parser().parse_args(argv)
    config = SessionConfig(
        algorithm=args.algorithm,
        scenario_name=args.scenario,
        num_agents=args.agents,
        seed=args.seed,
        action_source=args.action_source,
        speed_ms=args.speed_ms,
        episode_limit=args.episodes,
        model_path=args.model,
        model_tag=args.model_tag,
        communication_enabled=args.communication,
    )
    try:
        session = LiveSession(config)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[gridmind-server] configuration error: {exc}", file=sys.stderr)
        return 2

    event_log = open_event_log(args.event_log)
    if event_log is not None:
        print(f"[gridmind-server] event log: {args.event_log}", flush=True)
    server = GridMindServer(
        session,
        host=args.host,
        port=args.port,
        verbose=not args.quiet,
        exit_on_disconnect=args.exit_on_disconnect,
        event_log=event_log,
    )
    try:
        server.serve_forever(max_seconds=args.max_seconds)
    except KeyboardInterrupt:
        server.log("interrupted by user")
    finally:
        session.close()
        if event_log is not None:
            event_log.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
