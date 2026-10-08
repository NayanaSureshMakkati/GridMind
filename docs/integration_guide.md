# GridMind — Unity Integration, Demo Guide and Troubleshooting

> **Owner:** Developer 4. Protocol details: `docs/communication_protocol.md`.
> Experiment methods and metrics: `docs/experiment_protocol.md`.

## 1. What runs where

```
UNITY (3D frontend)                       PYTHON (logical backend = truth)
  CameraController                          environment/  GridWorld (API v1.0)
  EnvironmentManager  (builds the world)    algorithms/   random, Q-learning, DQN
  AgentController     (interpolates)        multi_agent/  independent, rule-based,
  UIManager           (stats + controls)                  coordinated, CTDE-inspired
  CommunicationManager (message -> scene)   evaluation/   metrics, experiments, plots
  NetworkManager      (TCP client)          integration/  protocol, session, server
        |                                          ^
        +------------ newline JSON over localhost --+
```

Unity does not decide actions, does not compute rewards and does not resolve
collisions. Everything shown is what Python decided.

## 2. Setup

### Python
```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
pip install -r requirements.txt
python -m pytest python_backend/tests/ -q      # expect: 197 passed
```

### Unity
1. Install **Unity 2021.3 LTS or newer** (any render pipeline).
2. Unity Hub → **Add** → select the `unity_frontend/` folder.
3. Open `Assets/Scenes/GridMindSimulation.unity`.
4. Press **Play** (the scene already contains `EnvironmentManager`, the camera,
   the light, and a `GridMindNetwork` object carrying `NetworkManager`,
   `CommunicationManager`, `UIManager`; `CameraController` sits on the camera).

The scene works standalone (it builds the demo world from primitives). When a
Python server is connectable, `NetworkManager` connects automatically and the
world is rebuilt from the `scenario` message Python sends, so the 3D layout and
the logical layout are the same by construction.

## 3. Running the final demo (10×10, 3 agents, deterministic)

```bash
# 1. train the system you want to demonstrate (do this once)
python main.py --mode train --algorithm coordinated --agents 3 --train-episodes 400

# 2. start the authoritative simulation and wait for Unity
python main.py --mode visualize --algorithm coordinated --agents 3 \
               --model-tag demo_10x10_3agents_coordinated_dqn_3agents \
               --episodes 3 --speed-ms 250

# 3. in Unity: press Play, then click "Start" in the panel
```

The 400-episode training run finishes in a few minutes and writes
`models/multi_agent/demo_10x10_3agents_coordinated_dqn_3agents_agent{0,1,2}.pt`.
On the deterministic layout the trained greedy policy takes all three agents to
their goals (`success 1.00`, `goal% 1.00`, ~22 steps, reward ≈ 245, 0
collisions), so the demo shows real goal-reaching rather than idle agents.
Without `--model-tag` (or with a tag that has no checkpoint) Python reports
`trained=false` in `welcome` and the agents act from an untrained policy.

Panel controls: **Start** (run), **Pause**, **Resume**, **Step** (one logical
step), **Reset** (new episode from the scenario), **Exit**, plus a speed slider
(0–1000 ms per step).

The scene renders the world in 3D: solid checkerboard floor tiles, full-height
obstacle blocks, and colour-coded agent capsules with a heading "nose" that
turns towards the movement direction and flashes red when Python reports a
collision for that agent. Every movement, obstacle/agent collision, goal and
deadlock is listed in the **event log panel** on the right (`L` toggles it,
`Clear` empties it, auto-scroll pins to the newest line); episode summaries are
appended as they arrive.

Camera: right-drag = orbit, wheel = zoom, `F` = follow the selected agent,
`1`–`5` = select agent, `R` = reframe.

Useful options:

```bash
# verify the whole loop without Unity (server + reference client in one process)
python main.py --mode visualize --headless --algorithm independent_dqn --episodes 1

# record the whole session: every scenario/step/episode_end as JSON lines
python main.py --mode visualize --algorithm coordinated --agents 3 \
               --event-log experiments/logs/live_events.jsonl

# manual stepping: Unity sends the actions, Python executes them
python main.py --mode visualize --algorithm independent_dqn --action-source external

# compare two runs side by side: run independent first, note the metrics, then coordinated
python main.py --mode visualize --algorithm multiagent --agents 3 --episodes 3
```

## 4. Final commands (Phase 15)

| Command | Mode |
|---------|------|
| `python main.py --mode train --algorithm qlearning` | train single-agent Q-learning |
| `python main.py --mode train --algorithm dqn` | train single-agent DQN |
| `python main.py --mode train --algorithm multiagent --agents 3` | train Independent DQN (3 agents) |
| `python main.py --mode train --algorithm coordinated --agents 3` | train Coordinated DQN (3 agents) |
| `python main.py --mode evaluate --algorithm multiagent --agents 3` | evaluate a saved checkpoint |
| `python main.py --mode visualize --algorithm coordinated --agents 3` | stream to Unity |
| `python main.py --mode compare` | Phase 9/10 full comparison + plots |
| `python main.py --mode ablation` | Phase 12 independent vs rule-based vs coordinated |
| `python main.py --mode scalability` | Phase 13: 1–5 agents |
| `python main.py --mode demo` | deterministic ASCII example (no RL, no Unity) |

Server-only and client-only entry points:

```bash
python -m python_backend.integration.server --help
python -m python_backend.integration.client --help
```

## 5. End-to-end checklist (Phase 16)

Run this once before the viva:

1. `python -m pytest python_backend/tests/ -q` → all tests pass.
2. `python main.py --mode demo` → ASCII world, agents move, episode ends.
3. `python main.py --mode train --algorithm dqn --quick` → model saved under `models/dqn/`.
4. `python main.py --mode evaluate --algorithm dqn --quick` → metrics reported from the saved model.
5. `python main.py --mode visualize --headless --algorithm independent_dqn --episodes 1` → full loop
   (state → actions → step → reward → goals → episode_end) printed without Unity.
6. `python main.py --mode visualize --algorithm independent_dqn --episodes 3` + Unity Play → 3D agents
   move, the event log lists every movement/collision/goal, and the panel shows
   algorithm/agents/episode/step/status/collisions/deadlocks. Add
   `--event-log experiments/logs/live_events.jsonl` to keep a JSON-Lines copy.
7. `python main.py --mode compare` → tables, JSON/CSV results and plots generated.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Panel says `OFFLINE` | Python server not running, or Unity started first | start the server first; the client retries every 2 s, no restart needed |
| `connect failed: Connection refused` in the Console | wrong port | match `--port` with `NetworkManager.port` (both default 8765) |
| Unity connects but nothing moves | session not started | click **Start** (or send `control start`); the server streams only while running |
| `NOT_EXTERNAL` error | `actions` sent while the server runs `--action-source policy` | restart the server with `--action-source external` for manual stepping |
| `PROTOCOL_MISMATCH` | Unity and Python builds disagree | update both: protocol version is 1 in `protocol.py` and `NetworkManager.ProtocolVersion` |
| 3D world shows the wrong layout | an old `scenario` message | press **Reset** or reconnect; the world is rebuilt from the scenario message every time |
| Agents stay still in a live demo | no checkpoint was supplied (`trained=false`) | train first, then pass `--model-tag` (multi-agent) or `--model` (single-agent) |
| `Missing checkpoint for agent 1` | tag mismatch | the default tag is `<scenario>_<system_label>_<N>agents`; list `models/multi_agent/` and pass the exact `--model-tag` |
| `FileNotFoundError: DQN checkpoint not found` | models not trained yet | `python main.py --mode train --algorithm dqn` |
| Episode ends immediately | `max_steps` too small for the scenario | raise `max_steps` in the scenario file (and re-run experiments afterwards) |
| Everything runs but the panel is empty | `UIManager` missing from the scene | add it to the `GridMindNetwork` object (scene already ships with it wired) |
| CPU pegged / demo too slow | `speed_ms = 0` | raise the speed slider or `--speed-ms` |
| Plots not created | `matplotlib` missing | `pip install -r requirements.txt`; plotting uses the headless `Agg` backend, no display needed |
| Tests fail after changing a scenario | metrics depend on the frozen layouts | scenarios are frozen on purpose; regenerate results after any change |

Escalation path: reproduce any Unity symptom without Unity by pointing the
reference client at the same server —

```bash
python -m python_backend.integration.client --port 8765 --grid --episodes 1
```

If the reference client behaves correctly, the fault is in the Unity scene
(component missing, wrong port, Play not pressed); if it also fails, the fault is
in the Python session and the server logs show which message was rejected.

## 7. Known limitations

* The Unity link is localhost-only by design (no authentication, not for
  networks or the internet).
* Unity streams one step at a time at `speed_ms`; there is no lock-step
  acknowledgement protocol, so at `speed_ms = 0` the client may skip rendered
  frames while still receiving every state message.
* Agent status changes are shown in the UI panel, not by recolouring the 3D
  agent (`AgentController.cs` is Developer 1's module and was left untouched).
* The demo scene's prefabs are optional; primitives are used when none are
  assigned.
* Only the frozen grid-based, discrete-action environment is visualized —
  no continuous 3D movement, by design (§8/§68).
