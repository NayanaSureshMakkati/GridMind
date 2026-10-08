"""Integration tests: protocol contract, live session, server<->client loop.

Phase 16 requirements covered here:

* the JSON protocol round-trips and rejects malformed messages explicitly,
* a live session steps single-agent AND multi-agent methods,
* the live multi-agent pipeline produces byte-identical episode outcomes to
  ``MultiAgentSystem.run_episode`` (drift guard for the reused pipeline),
* a real localhost server + reference client complete the full loop:
  connect -> scenario -> state -> actions/step -> reward -> goal -> episode_end.

No test simulates a clock: streaming is driven with ``speed_ms=0`` and explicit
control commands, so the suite stays fast and deterministic.
"""

import json
import threading
import time

import pytest

from python_backend.environment.constants import Position
from python_backend.evaluation.scenarios import Scenario, load_scenario
from python_backend.integration import protocol
from python_backend.integration.event_log import StepEventLogger, open_event_log
from python_backend.integration.client import GridMindClient, render_grid
from python_backend.integration.protocol import ProtocolError
from python_backend.integration.server import GridMindServer
from python_backend.integration.session import LiveSession, SessionConfig

DEMO = "demo_10x10_3agents"


@pytest.fixture(scope="module")
def small_scenario() -> Scenario:
    """Tiny 5x5 deterministic scenario (fast, but a real environment)."""
    return Scenario(
        name="test_5x5",
        height=5,
        width=5,
        max_steps=8,
        obstacles=(Position(2, 2),),
        agents=((Position(0, 0), Position(4, 4)), (Position(4, 0), Position(0, 4))),
        seed=3,
    )


def make_session(algorithm: str, scenario: Scenario, **overrides) -> LiveSession:
    """Builds a live session with test-friendly defaults."""
    config = SessionConfig(
        algorithm=algorithm,
        scenario_name=scenario.name,
        num_agents=max(1, scenario.num_agents),
        seed=overrides.pop("seed", 5),
        episode_limit=overrides.pop("episode_limit", 1),
        speed_ms=0,
        **overrides,
    )
    return LiveSession(config, scenario=scenario)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_encode_decode_round_trip(self):
        message = protocol.pong_message()
        assert protocol.decode(protocol.encode(message)) == message

    def test_decode_rejects_bad_json(self):
        with pytest.raises(ProtocolError) as excinfo:
            protocol.decode("{not json")
        assert excinfo.value.code == "BAD_JSON"

    def test_decode_rejects_missing_type(self):
        with pytest.raises(ProtocolError) as excinfo:
            protocol.decode('{"step": 1}')
        assert excinfo.value.code == "BAD_MESSAGE"

    def test_decode_rejects_non_object(self):
        with pytest.raises(ProtocolError):
            protocol.decode("[1, 2, 3]")

    def test_unknown_client_type_rejected(self):
        with pytest.raises(ProtocolError) as excinfo:
            protocol.validate_client_message({"type": "teleport"})
        assert excinfo.value.code == "UNKNOWN_TYPE"

    def test_control_command_validation(self):
        protocol.validate_client_message({"type": "control", "command": "pause"})
        with pytest.raises(ProtocolError):
            protocol.validate_client_message({"type": "control", "command": "fly"})
        with pytest.raises(ProtocolError):
            protocol.validate_client_message({"type": "control", "command": "set_speed",
                                              "speed_ms": -1})

    def test_protocol_version_mismatch(self):
        with pytest.raises(ProtocolError) as excinfo:
            protocol.validate_client_message({"type": "hello", "protocol": 99})
        assert excinfo.value.code == "PROTOCOL_MISMATCH"

    def test_actions_object_form(self):
        actions = protocol.parse_actions({"type": "actions", "actions": {"0": 3, "2": 4}})
        assert actions == {0: 3, 2: 4}

    def test_actions_array_form(self):
        actions = protocol.parse_actions(
            {"type": "actions", "agents": [{"id": 1, "action": 2}]}
        )
        assert actions == {1: 2}

    def test_actions_reject_invalid_action_id(self):
        with pytest.raises(ProtocolError) as excinfo:
            protocol.parse_actions({"type": "actions", "actions": {"0": 9}})
        assert excinfo.value.code == "BAD_ACTION_ID"

    def test_actions_reject_non_integer(self):
        with pytest.raises(ProtocolError):
            protocol.parse_actions({"type": "actions", "actions": {"zero": 1}})

    def test_split_lines_keeps_partial(self):
        lines, leftover = protocol.split_lines('{"a":1}\n{"b":')
        assert lines == ['{"a":1}']
        assert leftover == '{"b":'

    def test_error_message_is_explicit(self):
        message = protocol.error_message("BAD_ACTIONS", "detail here")
        assert message["code"] == "BAD_ACTIONS" and message["detail"] == "detail here"


# ---------------------------------------------------------------------------
# Live session
# ---------------------------------------------------------------------------


class TestLiveSession:
    def test_single_agent_session_steps(self, small_scenario):
        session = make_session("random", small_scenario)
        assert session.num_agents == 1
        state = session.state_message()
        assert state["type"] == "state"
        assert state["episode"] == 1
        assert len(state["agents"]) == 1
        assert len(state["agents"][0]["observation"]) == 12

        step = session.step_once()
        assert step.step == 1
        assert set(step.rewards) == {0}
        assert session.step_index == 1

    def test_single_agent_session_runs_to_episode_end(self, small_scenario):
        session = make_session("random", small_scenario)
        for _ in range(small_scenario.max_steps):
            step = session.step_once()
            if step.terminated or step.truncated:
                break
        assert step.truncated or step.terminated
        message = session.episode_end_message()
        assert message["type"] == "episode_end"
        assert message["steps"] >= 1
        assert isinstance(message["terminated"], bool)
        assert isinstance(message["truncated"], bool)
        assert message["goal_completion_rate"] in (0.0, 1.0)
        assert session.episodes_completed == 1

    def test_multi_agent_session_steps_all_agents(self, small_scenario):
        session = make_session("independent_dqn", small_scenario)
        assert session.num_agents == 2
        step = session.step_once()
        assert set(step.actions) == {0, 1}
        assert set(step.rewards) == {0, 1}

    def test_stepping_after_episode_end_raises(self, small_scenario):
        session = make_session("random", small_scenario)
        for _ in range(small_scenario.max_steps):
            step = session.step_once()
            if step.terminated or step.truncated:
                break
        with pytest.raises(RuntimeError):
            session.step_once()

    def test_external_mode_requires_actions(self, small_scenario):
        session = make_session("random", small_scenario, action_source="external")
        with pytest.raises(ValueError):
            session.step_once()

    def test_external_mode_rejects_invalid_action(self, small_scenario):
        session = make_session("random", small_scenario, action_source="external")
        # Agent 0 starts at (0, 0): UP (0) leaves the grid.
        with pytest.raises(ValueError):
            session.step_once({0: 0})

    def test_external_mode_executes_client_action(self, small_scenario):
        session = make_session("random", small_scenario, action_source="external")
        before = session.env.get_agent_position(0)
        step = session.step_once({0: 1})  # DOWN
        after = session.env.get_agent_position(0)
        assert after.row == before.row + 1
        assert step.actions[0] == 1

    def test_reset_episode_increments_and_restores(self, small_scenario):
        session = make_session("random", small_scenario, episode_limit=2)
        session.step_once()
        session.reset_episode(increment=True)
        assert session.episode_index == 2
        assert session.step_index == 0
        assert session.env.get_agent_position(0) == small_scenario.agents[0][0]

    def test_scenario_message_shape(self):
        session = make_session("independent_dqn", load_scenario(DEMO))
        message = session.scenario_message()
        assert message["type"] == "scenario"
        assert message["height"] == 10 and message["width"] == 10
        assert len(message["agents"]) == 3
        assert {"row", "col"} == set(message["obstacles"][0])
        assert message["agents"][0]["start"] == [0, 0]

    def test_state_positions_are_logical_row_col(self, small_scenario):
        session = make_session("random", small_scenario)
        agent = session.state_message()["agents"][0]
        assert agent["position"] == [0, 0]
        assert agent["goal"] == [4, 4]
        assert agent["status"] == "NAVIGATING"

    def test_stats_track_cumulative_counters(self, small_scenario):
        session = make_session("independent_dqn", small_scenario, communication_enabled=True)
        session.step_once()
        stats = session.stats()
        assert stats["episodes_completed"] in (0, 1)
        assert stats["collisions"] == session.total_collisions
        assert stats["messages"] >= 0

    def test_episode_limit_marks_session_finished(self, small_scenario):
        session = make_session("random", small_scenario, episode_limit=1)
        for _ in range(small_scenario.max_steps):
            step = session.step_once()
            if step.terminated or step.truncated:
                break
        assert session.finished is True


class TestLivePipelineEquivalence:
    """The live step pipeline must match MultiAgentSystem.run_episode exactly."""

    @pytest.mark.parametrize("method", ["independent_dqn", "rule_based_coordination"])
    def test_live_episode_matches_batch_episode(self, method):
        from python_backend.evaluation.experiments import (
            build_agent_dqn_config,
            multi_agent_config_kwargs,
        )
        from python_backend.multi_agent.multi_agent_system import (
            MultiAgentConfig,
            MultiAgentSystem,
        )

        scenario = load_scenario(DEMO)
        num_agents = 3
        seed = 11
        budget = 400

        session = make_session(method, scenario, seed=seed)
        steps = 0
        while True:
            step = session.step_once()
            steps += 1
            if step.terminated or step.truncated:
                break
            assert steps < 200, "live episode failed to terminate"

        dqn_configs = {
            aid: build_agent_dqn_config(seed=seed + aid, episodes=budget)
            for aid in range(num_agents)
        }
        env = scenario.with_num_agents(num_agents).build_env()
        system = MultiAgentSystem(
            env,
            MultiAgentConfig(**multi_agent_config_kwargs(method, num_agents, dqn_configs, seed)),
        )
        reference = system.run_episode(train=False, episode=0)

        assert steps == reference.steps
        assert session.total_deadlocks == reference.deadlocks
        assert session.episode_collisions == (
            reference.collision_totals.obstacle + reference.collision_totals.agent
        )
        for aid in range(num_agents):
            assert session.env.get_agent_position(aid) == env.get_agent_position(aid)
        assert session.history[-1]["success"] == reference.all_success


# ---------------------------------------------------------------------------
# Per-step event detail (movements + attributed collisions)
# ---------------------------------------------------------------------------


class TestStepEventDetails:
    """``step_result`` must expose every agent's movement and collision detail."""

    def test_step_result_has_one_movement_per_agent(self, small_scenario):
        session = make_session("independent_dqn", small_scenario)
        message = session.step_result_message(session.step_once())
        assert message["type"] == "step_result"
        assert len(message["movements"]) == session.num_agents
        assert {m["agent_id"] for m in message["movements"]} == set(session.env.agents)
        for movement in message["movements"]:
            assert movement["action_name"] in protocol.ACTION_NAMES.values()
            assert len(movement["from"]) == 2 and len(movement["to"]) == 2
            assert len(movement["target"]) == 2
            assert movement["moved"] == (movement["from"] != movement["to"])
            assert movement["reward"] == message["rewards"][str(movement["agent_id"])]

    def test_movement_records_real_motion(self, small_scenario):
        session = make_session("random", small_scenario, action_source="external")
        movement = session.step_result_message(session.step_once({0: 1}))["movements"][0]
        assert movement["from"] == [0, 0]
        assert movement["to"] == [1, 0]
        assert movement["target"] == [1, 0]
        assert movement["action_name"] == "DOWN"
        assert movement["moved"] is True

    def test_obstacle_collision_carries_position_and_target(self, small_scenario):
        # Clients cannot *submit* obstacle moves (LiveSession rejects invalid
        # actions), so the attribution is exercised on the event builder
        # directly: agent on (2,1) tried RIGHT into the obstacle at (2,2) and
        # was bounced back by the arbiter.
        session = make_session("random", small_scenario)
        movements, collisions = session._build_step_events(
            {0: 3}, {0: -20.0}, {0: Position(2, 1)}, {0: Position(2, 1)},
            {0: "OBSTACLE"},
        )
        assert collisions == [{
            "agent_id": 0, "type": "OBSTACLE",
            "position": [2, 1], "target": [2, 2], "other": -1,
        }]
        assert movements[0]["moved"] is False
        assert movements[0]["target"] == [2, 2]

    def test_boundary_collision_targets_the_cell_outside(self, small_scenario):
        session = make_session("random", small_scenario)
        _movements, collisions = session._build_step_events(
            {0: 0}, {0: -20.0}, {0: Position(0, 0)}, {0: Position(0, 0)},
            {0: "BOUNDARY"},
        )
        assert collisions[0]["type"] == "BOUNDARY"
        assert collisions[0]["position"] == [0, 0]
        assert collisions[0]["target"] == [-1, 0]

    def test_swap_collision_names_the_partner(self, small_scenario):
        session = make_session("independent_dqn", small_scenario)
        # Two agents proposed to cross (each aimed at the other's cell) and the
        # arbiter bounced BOTH back, so their positions did not change.
        before = {0: Position(2, 2), 1: Position(2, 3)}
        movements, collisions = session._build_step_events(
            {0: 3, 1: 2}, {0: -20.0, 1: -20.0}, before, dict(before),
            {0: "SWAP", 1: "SWAP"},
        )
        assert {c["agent_id"]: c["other"] for c in collisions} == {0: 1, 1: 0}
        assert all(c["type"] == "SWAP" for c in collisions)
        assert [c["target"] for c in collisions] == [[2, 3], [2, 2]]
        assert {m["agent_id"]: m["moved"] for m in movements} == {0: False, 1: False}

    def test_same_cell_collision_names_the_other_agent(self, small_scenario):
        session = make_session("independent_dqn", small_scenario)
        # Both agents proposed the SAME cell (2, 2) from either side -> bounced.
        before = {0: Position(2, 1), 1: Position(2, 3)}
        _movements, collisions = session._build_step_events(
            {0: 3, 1: 2}, {0: -20.0, 1: -20.0}, before, dict(before),
            {0: "SAME_CELL", 1: "SAME_CELL"},
        )
        assert all(c["type"] == "SAME_CELL" for c in collisions)
        assert {c["agent_id"]: c["other"] for c in collisions} == {0: 1, 1: 0}
        assert all(c["target"] == [2, 2] for c in collisions)
        assert all(c["position"] != [2, 2] for c in collisions)  # bounced back


# ---------------------------------------------------------------------------
# JSONL event log
# ---------------------------------------------------------------------------


class TestEventLog:
    def test_open_event_log_without_path_is_none(self):
        assert open_event_log(None) is None

    def test_logger_writes_parseable_jsonl(self, tmp_path):
        path = tmp_path / "nested" / "events.jsonl"
        logger = StepEventLogger(path)
        logger.log_message({"type": "step_result", "step": 1})
        logger.close()

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert records[0]["type"] == "session_start" and records[0]["ts"]
        assert records[-1]["type"] == "step_result"

    def test_server_mirrors_events_to_the_log(self, small_scenario, tmp_path):
        session = make_session("random", small_scenario, action_source="external")
        path = tmp_path / "events.jsonl"
        logger = StepEventLogger(path)
        server = GridMindServer(session, host="127.0.0.1", port=0, verbose=False,
                                event_log=logger)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"max_seconds": 30.0}, daemon=True
        )
        thread.start()
        assert server.ready.wait(timeout=5.0), "server did not start listening"
        try:
            with GridMindClient(host="127.0.0.1", port=server.port, timeout=10.0) as client:
                client.wait_for("welcome")
                client.wait_for("scenario")
                client.wait_for("state")
                client.send_actions({0: 1})  # agent 0 moves DOWN
                client.wait_for("step_result")
                client.send_control("quit")
        finally:
            server.stop()
            thread.join(timeout=5.0)
            logger.close()
            session.close()

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        types = [record["type"] for record in records]
        assert "scenario" in types
        step_records = [r for r in records if r["type"] == "step_result"]
        assert step_records, "no step_result was logged"
        movement = step_records[0]["movements"][0]
        assert movement["action_name"] == "DOWN"
        assert movement["from"] == [0, 0] and movement["to"] == [1, 0]


# ---------------------------------------------------------------------------
# Server <-> client (Phase 16 end-to-end)
# ---------------------------------------------------------------------------


class TestServerClientLoop:
    def _start_server(self, session, **kwargs):
        server = GridMindServer(session, host="127.0.0.1", port=0, verbose=False, **kwargs)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"max_seconds": 30.0}, daemon=True
        )
        thread.start()
        assert server.ready.wait(timeout=5.0), "server did not start listening"
        return server, thread

    def test_full_loop_reaches_episode_end(self, small_scenario):
        session = make_session("random", small_scenario)
        server, thread = self._start_server(session)
        try:
            with GridMindClient(host="127.0.0.1", port=server.port, timeout=10.0) as client:
                welcome = client.wait_for("welcome")
                assert welcome["protocol"] == protocol.PROTOCOL_VERSION
                assert welcome["num_agents"] == session.num_agents

                scenario = client.wait_for("scenario")
                assert scenario["height"] == small_scenario.height

                state = client.wait_for("state")
                assert state["episode"] == 1
                assert state["agents"][0]["position"] == [0, 0]
                assert render_grid(scenario, state).count("A0") == 1

                client.send_control("pause")
                reply = client.wait_for_ack("pause")
                assert reply["ok"] is True

                client.send_control("step")
                stepped = client.wait_for("state")
                assert stepped["step"] == 1

                client.send_control("start")
                deadline = time.perf_counter() + 20.0
                step_results = 0
                while time.perf_counter() < deadline:
                    message = client.receive(timeout=5.0)
                    if message is None:
                        break
                    if message["type"] == "step_result":
                        step_results += 1
                        assert set(message["rewards"]) == {str(a) for a in session.env.agents}
                    if message["type"] == "episode_end":
                        assert message["steps"] >= 1
                        assert message["success"] in (True, False)
                        assert message["collisions"] >= 0
                        break
                assert step_results > 0

                client.send_control("quit")
        finally:
            server.stop()
            thread.join(timeout=5.0)
            session.close()

    def test_ping_pong_and_unknown_client_message(self, small_scenario):
        session = make_session("random", small_scenario)
        server, thread = self._start_server(session)
        try:
            with GridMindClient(host="127.0.0.1", port=server.port, timeout=10.0) as client:
                client.wait_for("welcome")
                client.send({"type": "ping"})
                assert client.wait_for("pong")["type"] == "pong"

                client.send({"type": "warp"})
                error = client.wait_for("error")
                assert error["code"] == "UNKNOWN_TYPE"

                client.send({"type": "actions", "actions": {"0": 3}})
                error = client.wait_for("error")
                assert error["code"] == "NOT_EXTERNAL"
                client.send_control("quit")
        finally:
            server.stop()
            thread.join(timeout=5.0)
            session.close()

    def test_external_action_mode_executes_client_actions(self, small_scenario):
        session = make_session("random", small_scenario, action_source="external")
        server, thread = self._start_server(session)
        try:
            with GridMindClient(host="127.0.0.1", port=server.port, timeout=10.0) as client:
                client.wait_for("welcome")
                client.wait_for("scenario")
                client.wait_for("state")

                client.send_actions({0: 1})  # agent 0 moves DOWN
                result = client.wait_for("step_result")
                assert result["step"] == 1

                state = client.wait_for("state")
                assert state["agents"][0]["position"] == [1, 0]

                client.send_actions({0: 1})
                state = client.wait_for("state")
                assert state["agents"][0]["position"] == [2, 0]
                client.send_control("quit")
        finally:
            server.stop()
            thread.join(timeout=5.0)
            session.close()

    def test_reset_command_restarts_the_episode(self, small_scenario):
        session = make_session("random", small_scenario, episode_limit=3)
        server, thread = self._start_server(session)
        try:
            with GridMindClient(host="127.0.0.1", port=server.port, timeout=10.0) as client:
                client.wait_for("welcome")
                client.wait_for("scenario")
                client.wait_for("state")
                client.send_control("step")
                client.wait_for("step_result")
                client.send_control("reset")
                message = client.wait_for("state")
                assert message["episode"] == 1
                assert message["agents"][0]["position"] == [0, 0]
                client.send_control("quit")
        finally:
            server.stop()
            thread.join(timeout=5.0)
            session.close()
