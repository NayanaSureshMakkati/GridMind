# GridMind — Python ↔ Unity Communication Protocol v1

> **Owner:** Developer 4. **Status:** implemented and covered by
> `python_backend/tests/test_integration.py` (protocol round-trip, malformed
> messages, live session, real server↔client end-to-end loop).
>
> Python is the source of truth for state, actions, rewards, collisions,
> deadlocks and episode termination. Unity renders and controls; it never
> simulates and never runs RL logic (Master Context §47).

## 1. Transport

| Property | Value |
|----------|-------|
| Transport | TCP over localhost (no cloud service, no extra dependency) |
| Framing | newline-delimited JSON, **one message per line** (`\n`), UTF-8 |
| Direction | Unity = client, Python = server (Python listens) |
| Default endpoint | `127.0.0.1:8765` |
| Max message size | 1 MiB per line (larger input is rejected, never buffered forever) |
| Reconnect | Unity retries every `reconnectDelaySeconds` (default 2 s) and re-sends `hello` |

Python side: `python_backend/integration/{protocol,session,server,client}.py`
Unity side: `Assets/Scripts/{NetworkManager,CommunicationManager}.cs`

## 2. Coordinate rule (authoritative)

Every position on the wire is a **logical `[row, col]`** pair, exactly as
defined in `docs/environment_api.md` §1: origin top-left, rows increase
downwards, columns increase rightwards.

Unity converts to 3D in one place only (`CoordinateConverter.cs`):

```
Unity X =  col * CELL_SIZE
Unity Y =  AGENT_HEIGHT
Unity Z = -row * CELL_SIZE
```

The backend never sends Unity coordinates, so the two conventions cannot mix.

> **Obstacles are the one exception in shape, not in meaning:** they are sent as
> `{"row": r, "col": c}` objects because Unity parses messages with
> `JsonUtility`, which cannot deserialize jagged arrays (`int[][]`). Agent
> positions stay plain `[row, col]` arrays.

## 3. Connection lifecycle

```
Unity                                Python
  |-- connect -------------------------->|
  |-- hello {protocol:1} --------------->|
  |<------------- ack("hello")           |
  |<------------- scenario               |   world to build (obstacles/goals/agents)
  |<------------- state                  |   episode 1, step 0
  |-- control {"command":"start"} ------>|
  |<------------- ack("start")           |
  |<------------- step_result            |   rewards, collisions, goals reached
  |<------------- state                  |   authoritative positions + stats
  |             ... repeated at speed_ms ...
  |<------------- episode_end            |   terminated/truncated + summary
  |-- control {"command":"quit"} ------->|
  |<------------- ack("quit")            |
```

## 4. Client → server messages

### 4.1 `hello`
```json
{"type": "hello", "client": "unity", "protocol": 1}
```
A different `protocol` value is answered with `PROTOCOL_MISMATCH` (the version
this build speaks is `NetworkManager.ProtocolVersion` = 1).

### 4.2 `control`
```json
{"type": "control", "command": "start"}
{"type": "control", "command": "set_speed", "speed_ms": 150}
```

| Command | Effect | Notes |
|---------|--------|-------|
| `start` | begins/resumes paced auto-stepping | each step waits `speed_ms` |
| `pause` | stops auto-stepping (state retained) | `step` still works while paused |
| `resume` | same as `start` | explicit for the UI |
| `step` | performs exactly one step | pauses auto-stepping first |
| `reset` | restarts the episode at step 0 | positions/goals restored from the scenario |
| `set_speed` | sets `speed_ms` (integer ≥ 0) | `0` = as fast as possible |
| `quit` | closes the session and stops the server | ack is sent first |

Every command is acknowledged with `ack`.

### 4.3 `actions` (only in external/manual mode)
```json
{"type": "actions", "actions": {"0": 3, "1": 2, "2": 4}}
```
or the equivalent array form:
```json
{"type": "actions", "agents": [{"id": 0, "action": 3}]}
```

* Action ids are the frozen mapping: `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY`.
* Actions are executed by the **Python** environment using its own simultaneous
  conflict resolution; Unity supplies intent, never movement.
* Honoured only when the server runs with `--action-source external`
  (`SessionConfig.action_source = "external"`). In `policy` mode the server
  answers `NOT_EXTERNAL`, because the RL policy decides actions.
* Invalid or out-of-grid actions are rejected with `BAD_ACTIONS` and the step is
  **not** performed.

### 4.4 `ping` / `quit`
`ping` is answered with `pong` (liveness check). `quit` ends the session.

## 5. Server → client messages

### 5.1 `welcome`
```json
{"type": "welcome", "protocol": 1, "server": "gridmind-python",
 "algorithm": "independent_dqn", "algorithm_label": "Independent DQN",
 "mode": "live-simulation", "action_source": "policy",
 "num_agents": 3, "episode": 1, "scenario": "demo_10x10_3agents",
 "speed_ms": 250, "trained": false, "episode_limit": 3}
```
`trained=false` means no checkpoint was supplied — the UI shows this so an
untrained run is never mistaken for a trained one.

### 5.2 `scenario`
```json
{"type": "scenario", "name": "demo_10x10_3agents",
 "height": 10, "width": 10, "max_steps": 60,
 "obstacles": [{"row": 4, "col": 1}, {"row": 4, "col": 2}],
 "agents": [{"id": 0, "start": [0, 0], "goal": [9, 9]},
            {"id": 1, "start": [9, 0], "goal": [0, 9]}]}
```
Loaded from `experiments/scenarios/*.json` — the same file Python simulates, so
the 3D world cannot drift from the logical world.

### 5.3 `state`
```json
{"type": "state", "episode": 1, "step": 10, "algorithm": "independent_dqn",
 "action_source": "policy", "max_steps": 60,
 "agents": [
   {"id": 0, "position": [2, 3], "goal": [9, 9],
    "observation": [2,3,9,9,0,0,1,0,0,0,0,1],
    "status": "NAVIGATING", "done": false, "reward": -12.0}
 ],
 "stats": {"algorithm_label": "Independent DQN", "episode": 1, "episode_limit": 3,
           "step": 10, "max_steps": 60, "agents": 3,
           "collisions": 0, "episode_collisions": 0,
           "deadlocks": 0, "episode_deadlocks": 0,
           "messages": 0, "episode_messages": 0, "distance": 12,
           "episodes_completed": 0, "successes": 0, "success_rate": 0.0,
           "total_reward": -14.0, "trained": false, "action_source": "policy"}}
```
* `status` is `NAVIGATING` or `GOAL_REACHED` and is the only status source the
  UI needs.
* `observation` is the agent's observation vector in the configured mode
  (12 elements for local-only observations, longer when the coordinated system
  is enriched).
* `stats` carries cumulative counters for the UI panel (Phase 7).

### 5.4 `step_result`
```json
{"type": "step_result", "step": 10,
 "rewards": {"0": -1.0, "1": -20.0, "2": 100.0},
 "movements": [
   {"agent_id": 0, "action": 3, "action_name": "RIGHT",
    "from": [2, 3], "to": [2, 4], "target": [2, 4],
    "moved": true, "reward": -1.0},
   {"agent_id": 1, "action": 2, "action_name": "LEFT",
    "from": [4, 5], "to": [4, 5], "target": [4, 4],
    "moved": false, "reward": -20.0}
 ],
 "collisions": [{"agent_id": 1, "type": "OBSTACLE",
                 "position": [4, 5], "target": [4, 4], "other": -1}],
 "goals_reached": [2], "deadlocks": 0,
 "terminated": false, "truncated": false}
```
`collisions` uses the frozen taxonomy (`BOUNDARY`, `OBSTACLE`, `SAME_CELL`,
`SWAP`); `deadlocks` is 1 when a deadlock fired on this step (cumulative counts
live in `state.stats`).

`movements` carries **one record per agent per step** — the exhaustive motion
record for the 3D view and the event log:

| Field | Meaning |
|-------|---------|
| `action` / `action_name` | the frozen action id and its name (`UP`/`DOWN`/`LEFT`/`RIGHT`/`STAY`) |
| `from` / `to` | logical `[row, col]` before and after the step |
| `target` | the cell the agent tried to enter *before* conflict resolution (`from` when it stayed) |
| `moved` | `false` when the agent stayed or the arbiter bounced it back |
| `reward` | that agent's reward for this step (same value as `rewards[str(id)]`) |

Collision entries are the same frozen taxonomy, enriched (additively, still
protocol v1) so every collision can be attributed and drawn:

| Field | Meaning |
|-------|---------|
| `position` | where the agent ended up (always its pre-step cell for a collision) |
| `target` | the cell it tried to enter (an out-of-grid cell for `BOUNDARY`) |
| `other` | the other agent involved for `SAME_CELL`/`SWAP` (`-1` when none or ambiguous) |

> Collisions always bounce the agent back, so `position == from` on a collision
> step. `BOUNDARY`/`OBSTACLE` moves are rejected by the server when the client
> submits them (`BAD_ACTIONS`), so those entries only arise from the
> environment's own arbitration path.

#### Recording the whole session to disk

`python main.py --mode visualize ... --event-log PATH` (and the equivalent
`--event-log` flag on `python -m python_backend.integration.server`) mirrors
every streamed `scenario`, `step_result` and `episode_end` into a JSON-Lines
file — one JSON object per line with a UTC `ts`, so all movements, collisions,
goals and deadlocks of a run stay readable after the window is closed. The
wire protocol itself is unchanged.

### 5.5 `episode_end`
```json
{"type": "episode_end", "episode": 1, "steps": 54,
 "terminated": true, "truncated": false, "success": true,
 "rewards": {"0": 100.0, "1": 100.0, "2": 100.0},
 "collisions": 2, "deadlocks": 0, "messages": 0,
 "goal_completion_rate": 1.0}
```
`terminated` (all goals reached) and `truncated` (step limit) are separate
flags and must never be merged by a consumer (§17).

### 5.6 `ack`, `error`, `pong`
```json
{"type": "ack", "command": "start", "ok": true}
{"type": "error", "code": "BAD_ACTIONS", "detail": "action 9 for agent 0 is not in 0..4"}
{"type": "pong"}
```

| Error code | Meaning |
|------------|---------|
| `BAD_JSON` | line is not valid JSON |
| `BAD_MESSAGE` | no `type` field / not a JSON object / oversize line |
| `UNKNOWN_TYPE` | message type not in the client catalogue |
| `BAD_COMMAND` | unknown control command or invalid `set_speed` value |
| `BAD_ACTIONS` | malformed action map or an action outside `0..4` |
| `NOT_EXTERNAL` | `actions` sent while the session is in `policy` mode |
| `PROTOCOL_MISMATCH` | client `hello` protocol ≠ server protocol |
| `STEP_FAILED` | step rejected (e.g. episode already finished) |
| `SESSION_FINISHED` | all configured episodes are complete; send `reset` |
| `TOO_LARGE` | a single line exceeded the size limit |

Errors are explicit and always reported; nothing is silently ignored (§58).

## 6. Unity component mapping

| Script | Responsibility | Never does |
|--------|----------------|------------|
| `NetworkManager.cs` | socket, background read thread, message queue, send | parse semantics, decide actions |
| `CommunicationManager.cs` | interpret messages, update scene + UI + event log | simulate movement/rewards |
| `EnvironmentManager.cs` | build the 3D world: tiles, obstacle blocks, coloured agents/goals, collision flashes | pick actions |
| `AgentController.cs` | interpolate an agent to its logical position, face the movement direction, flash on collision | pick actions |
| `UIManager.cs` | stats panel + Start/Pause/Resume/Step/Reset/Exit | hold authoritative state |
| `EventLogUI.cs` | scrollable movement/collision/goal/deadlock log (toggle `L`) | infer state not streamed |
| `CameraController.cs` | elevated isometric view, optional follow | affect the simulation |

`CommunicationManager` rebuilds the world from the `scenario` message
(`ClearWorld` → `InitializeGrid` → `SpawnAgent`), so Unity never needs a second,
hardcoded copy of the layout.

## 7. Manual verification without Unity

```bash
# terminal 1 — Python server (authoritative simulation)
python -m python_backend.integration.server --algorithm independent_dqn --agents 3 --speed-ms 250

# terminal 2 — reference client (prints the ASCII grid + stats)
python -m python_backend.integration.client --grid --episodes 1
```
Any protocol mismatch with Unity can be reproduced here, because this client
speaks the same wire format as `NetworkManager.cs`.
