# GridMind — Intelligent Multi-Agent Navigation System using Reinforcement Learning in a 3D Simulation Environment

Multiple agents learn to navigate a shared discrete grid world while avoiding static obstacles and each other. Python is
the authoritative logical simulation + RL backend; Unity visualizes what Python
decides.

> **Current status:**
> - Environment foundation **complete** (Developer 1): frozen Environment API
>   v1.0, simultaneous multi-agent stepping, collision arbitration, seeded random
>   layouts, BFS validation, Unity demo world.
> - Single-agent RL **complete** (Developer 2): random baseline, tabular
>   Q-learning, DQN, training/evaluation harness, experiment driver.
> - Independent DQN and multi-agent intelligence **complete** (Developer 3):
>   2–5 agents, Independent DQN, rule-based coordination, learned coordination,
>   CTDE-inspired variant, collision analysis, deadlock detection/recovery,
>   optional communication, scalability + ablation drivers.
> - **Integration + evaluation complete (Developer 4):** JSON protocol, live
>   Python↔Unity session/server/client, Unity network + UI + camera scripts,
>   scenario-driven 3D world, metrics/experiments/comparison/plots, single
>   `main.py` CLI, integration and evaluation test suites, documentation.
>
> Test status: **206/206 passing** (30 environment + 32 RL + 55 multi-agent +
> 49 evaluation + 40 integration), verified with
> `python -m pytest python_backend/tests/ -q`.

## Documentation

| Document | Contents |
|----------|----------|
| `docs/environment_api.md` | Environment API v1.0 (FROZEN) + Unity coordinate convention |
| `docs/multi_agent.md` | Developer 3: multi-agent architecture, coordination, deadlock, metrics |
| `docs/communication_protocol.md` | Python↔Unity wire protocol v1: messages, controls, error codes |
| `docs/experiment_protocol.md` | Metric definitions, fairness rules, commands, reproducibility |
| `docs/integration_guide.md` | Unity setup, demo instructions, final commands, troubleshooting |

## Project structure

```
GridMind/
├── main.py                        # single CLI: train/evaluate/visualize/compare/ablation/scalability/demo
├── requirements.txt               # numpy, pytest, torch, matplotlib
├── docs/                          # environment API, multi-agent, protocol, experiments, integration guide
├── python_backend/
│   ├── environment/               # Developer 1: constants, agent, grid_world, reward, collision, observation
│   ├── algorithms/                # Developer 2: random_policy, state_encoding, q_learning, network,
│   │                              #              replay_buffer, dqn, trainer
│   ├── multi_agent/               # Developer 3: multi_agent_system, independent_dqn, coordination,
│   │                              #              observations, deadlock, communication
│   ├── integration/               # Developer 4: protocol, session, server, client
│   ├── evaluation/                # Developer 4: scenarios, metrics, experiments, comparison, plots,
│   │                              #              model_store
│   └── tests/                     # test_environment, test_rl, test_multi_agent, test_evaluation,
│                                  # test_integration
├── experiments/
│   ├── scenarios/                 # FROZEN scenario + suite files (shared by Python and Unity)
│   ├── logs/ · results/ · plots/  # generated output
├── models/                        # checkpoints: q_learning/ dqn/ multi_agent/ (generated)
└── unity_frontend/Assets/
    ├── Scenes/GridMindSimulation.unity
    └── Scripts/                   # CoordinateConverter, AgentController, EnvironmentManager (Dev 1)
                                   # NetworkManager, CommunicationManager, UIManager, CameraController, EventLogUI (Dev 4)
```

## System architecture

Python is the single source of truth; Unity only renders what Python decides.
Layers call downward: `integration/` and `evaluation/` orchestrate the
multi-agent system, which drives the algorithms against the frozen environment.

```
┌──────────────────────────────────────────────────────────────────────┐
│             UNITY 3D FRONTEND · renders, never simulates             │
│                                                                      │
│  NetworkManager    background TCP thread, queued JSON lines          │
│        └─► CommunicationManager · the ONLY protocol interpreter      │
│              ├─► EnvironmentManager   builds grid, obstacles,        │
│              │                        agents and goals in 3D         │
│              ├─► UIManager            live metrics + run controls    │
│              └─► CameraController     orbit / zoom / follow          │
└──────────▲───────────────────────────────────────────────────────────┘

           │        JSON lines over localhost TCP (default :8765)

           │ Python → Unity  (server streams):
           │ welcome · scenario · state · step_result ·
           │ episode_end · ack · error

           │ Unity → Python  (client sends):
           │ control (start / pause / resume / step / reset /
           │ set_speed / quit) · actions (external mode)

┌──────────┴───────────────────────────────────────────────────────────┐
│                                                                      │
│        PYTHON BACKEND · authoritative logical simulation + RL        │
│                                                                      │
│   main.py - one CLI: train · evaluate · visualize · compare ·        │
│                        ablation · scalability · demo                 │
│   ────────────────────────────────────────────────────────────────   │
│   integration/    GridMindServer (TCP) · LiveSession ·               │
│                    reference client (headless viewer)                │
│   ────────────────────────────────────────────────────────────────   │
│   multi_agent/    MultiAgentSystem · rule-based priority yields ·    │
│                    deadlock detection/recovery · optional comms      │
│   ────────────────────────────────────────────────────────────────   │
│   algorithms/     Random baseline · tabular Q-learning · DQN         │
│                    (replay buffer, target net, Double DQN,           │
│                     Huber loss, action masking)                      │
│   ────────────────────────────────────────────────────────────────   │
│   environment/    GridWorld · FROZEN API v1.0 · simultaneous         │
│                    stepping, conflict arbitration, rewards,          │
│                    BFS validation (checks only, never navigates)     │
│   ────────────────────────────────────────────────────────────────   │
│   evaluation/     metrics · frozen scenarios · experiments ·         │
│                    fairness-checked comparison · plots · model_store │
└──────────────────────────────────────────────────────────────────────┘

    checkpoints (.pt / .pkl)                frozen inputs (JSON)
    models/q_learning/ · dqn/ ·             experiments/scenarios/*.json
    multi_agent/                            shared by Python AND Unity

    generated output: experiments/results/ · experiments/plots/
```

The same protocol is exercised without Unity by the headless reference client
(`--mode visualize --headless`), so the frontend and backend can be verified
independently.

## Setup

**Prerequisites:** Python 3.10+ (Unity 2021.3 LTS+ is optional — only needed to
watch a run in 3D).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt  # numpy, pytest, torch, matplotlib
```

Verify the install by running the test suite (`## Running the tests` below).

Every experiment is runnable from a clean checkout with no Unity: checkpoints,
reports and plots are all written to git-ignored directories (see `.gitignore`),
so running experiments never dirties the repository. To see a run in 3D, do the
optional Unity steps in `docs/integration_guide.md` (add `unity_frontend/` in
Unity Hub, open `Assets/Scenes/GridMindSimulation.unity`, press **Play**).

## Running the tests

```bash
python -m pytest python_backend/tests/ -q      # 206 tests
```

## Quick start

```bash
# deterministic example environment (no RL, no Unity)
python main.py --mode demo

# train + evaluate one algorithm end-to-end
python main.py --mode train --algorithm dqn
python main.py --mode train --algorithm coordinated --agents 3

# full algorithm comparison with plots (long: real training)
python main.py --mode compare
python main.py --mode compare --quick          # smoke pass with tiny budgets

# Phase 12 / Phase 13
python main.py --mode ablation
python main.py --mode scalability --agent-counts 1,2,3,4,5

# Python <-> Unity live simulation
python main.py --mode visualize --algorithm independent_dqn --agents 3 --episodes 3
python main.py --mode visualize --headless --algorithm independent_dqn --episodes 1   # no Unity needed

# same run, with every movement/collision/goal recorded as JSON lines
python main.py --mode visualize --algorithm coordinated --agents 3 \
    --event-log experiments/logs/live_events.jsonl
```

Every mode and flag is documented in `docs/integration_guide.md`; experiment
methodology and metric definitions are in `docs/experiment_protocol.md`.

## Running the experiments

All experiment code is reachable two ways: the single `main.py` CLI, and the
standalone driver scripts in `scripts/`. Every number these produce comes from an
executed episode or a measured training run — nothing is pre-filled. Methodology,
fair-comparison rules and metric definitions live in
`docs/experiment_protocol.md`.

### Methods and modes

Methods (CLI aliases): `random`, `qlearning`, `dqn`, `multiagent` (Independent
DQN), `rule_based`, `coordinated` (Coordinated DQN), `ctde` (CTDE-inspired).

| Command | What it does | Output |
|---------|--------------|--------|
| `python main.py --mode demo` | deterministic ASCII world, random policy (no RL, no Unity) | stdout |
| `python main.py --mode train --algorithm <a>` | trains one method, saves the checkpoint, writes report + plots | `models/`, `experiments/results/`, `experiments/plots/` |
| `python main.py --mode evaluate --algorithm <a>` | evaluates a saved checkpoint (no retraining) | `experiments/results/evaluate_<a>.{json,csv}` |
| `python main.py --mode compare` | Phase 9/10 full comparison across methods | `comparison.{json,csv}`, `comparison_episodes.csv`, `comparison_training_history.csv`, plots |
| `python main.py --mode ablation` | Phase 12: independent vs rule-based vs coordinated (vs CTDE) | `ablation.{json,csv}`, `ablation_episodes.csv`, plots |
| `python main.py --mode scalability` | Phase 13: same layout at increasing agent counts | `scalability.{json,csv}`, `scalability_episodes.csv`, plots |
| `python main.py --mode visualize --algorithm <a>` | streams authoritative state to Unity (or a headless client) | live TCP on `:8765`, optional JSON-Lines event log |

Key flags: `--train-episodes`, `--eval-episodes`, `--fixed-eval-episodes`,
`--seed`, `--scenario`, `--suite`, `--methods`, `--agents`, `--agent-counts`,
`--quick`, `--no-plots`, `--no-save-models`, `--model`, `--model-tag`.

### Workflow examples

Activate the virtualenv first (see `## Setup`).

```bash
# 0) fast smoke pass — real runs with tiny budgets; numbers are NOT results
python main.py --mode compare --quick

# 1) one method end-to-end: train -> save checkpoint -> report -> plots
python main.py --mode train --algorithm dqn
python main.py --mode train --algorithm coordinated --agents 3

# 2) evaluate a saved checkpoint only (no retraining)
python main.py --mode evaluate --algorithm multiagent --agents 3

# 3) headline comparison: random / Q-learning / DQN / independent /
#    rule-based / coordinated / CTDE  (long: this is real training)
python main.py --mode compare

# 4) Phase 12 ablation and Phase 13 scalability
python main.py --mode ablation
python main.py --mode scalability --agent-counts 1,2,3,4,5

# 5) restrict to the headline pair only
python main.py --mode compare --methods multiagent,coordinated
```

Runtime scales with episodes × agents × steps — that is the real cost of real
runs, and it is measured and reported (`training_seconds`, `evaluation_seconds`,
`computation_seconds`) rather than hidden.

### Standalone driver scripts

The original per-developer drivers are still runnable and are handy for chunked
or resumable runs:

```bash
# single-agent: random vs Q-learning vs DQN (Phases 4, 5, 9, 10, 11)
python scripts/run_rl_experiments.py                 # full suite
python scripts/run_rl_experiments.py --quick         # reduced budgets
python scripts/run_rl_experiments.py --suite single  # 5x5 + 10x10 only

# multi-agent: ablation + scalability (Phases 12, 13; the full suite is long)
python scripts/run_marl_experiments.py --quick       # smoke test
python scripts/run_marl_experiments.py --suite ablation
python scripts/run_marl_experiments.py --suite scalability --agents 2,3,4,5

# run a subset of families; --name keeps chunked runs from overwriting each other
python scripts/run_marl_experiments.py --suite ablation \
    --kinds rule_based,ctde --name marl_ablation_extra_full

# one-off DQN tuning diagnostics (probes, not deliverables)
python scripts/diag_dqn_convergence.py
```

`run_marl_experiments.py` writes its `marl_*.{json,csv}` output after every
finished configuration, so an interrupted multi-hour suite keeps the
configurations it already completed.

### Interactive visualization sweep

```bash
bash scripts/visualize_all_algorithms.sh              # every algorithm, 45 s each
bash scripts/visualize_all_algorithms.sh 90           # slower pace
bash scripts/visualize_all_algorithms.sh 60 coordinated_dqn,ctde_inspired   # subset
```

### Outputs

| Path | Contents |
|------|----------|
| `experiments/results/<name>.json` | full report: manifest + aggregate rows + per-episode metrics |
| `experiments/results/<name>.csv` | one row per method/condition (aggregates) |
| `experiments/results/manifest_<name>.json` | reproducibility manifest (seeds, scenario, suite, budgets, counts) |
| `experiments/results/comparison_episodes.csv` · `comparison_training_history.csv` | per-episode evaluation metrics and training curves |
| `experiments/plots/*.png` | reward/success/collisions/steps curves, comparison bars, scalability and ablation figures |
| `models/q_learning/`, `models/dqn/`, `models/multi_agent/` | saved checkpoints (`--model` / `--model-tag` pick them up) |
| `experiments/logs/*.jsonl` | optional JSON-Lines event log from `--event-log` |

All of the above are git-ignored.

### Reproducibility

* Scenarios are frozen JSON files in `experiments/scenarios/`, shared by Python
  and Unity, and never edited between compared runs.
* Layout seeds are explicit everywhere: policy seeds via `--seed`, random-layout
  seeds as `seed_start + episode_index` (identical sequence for every method),
  and per-agent DQN seeds as `seed + agent_id`.
* Every method in a comparison run uses the same `--train-episodes` budget; the
  epsilon schedule lands exactly on `epsilon_min` at the end of the budget, so
  changing the budget does not advantage any method.
* Reports refuse to print rows produced under different conditions
  (`comparison.assert_fair`). Re-run with `--quick` for a fast smoke check, but
  report only full-suite numbers.

## Python ↔ Unity

1. Start the authoritative simulation:
   `python main.py --mode visualize --algorithm coordinated --agents 3 --model-tag <tag>`
2. In Unity: open the project, open `Assets/Scenes/GridMindSimulation.unity`, press **Play**.
3. Click **Start** in the panel (Start / Pause / Resume / Step / Reset / Exit).

Unity receives `welcome`, `scenario` (the world to build) and one `state` +
`step_result` pair per logical step, and returns `control` commands (and
`actions` when the server runs `--action-source external`). The protocol is
newline-delimited JSON over localhost TCP — see `docs/communication_protocol.md`.

The 3D world is built entirely from that stream: solid floor tiles, full-height
obstacle blocks and colour-coded agent capsules that turn towards their movement
direction and flash red on a collision. Every movement, obstacle/agent
collision, goal and deadlock is also listed line-by-line in the in-game event
log panel (toggle `L`), and `--event-log PATH` records the same events as
JSON-Lines for analysis after the run.

Verify the link without Unity using the reference client:

```bash
python -m python_backend.integration.server --algorithm independent_dqn --agents 3
python -m python_backend.integration.client --grid --episodes 1
```

## Python → Unity coordinate conversion

```
Unity X =  col * CELL_SIZE     (CELL_SIZE = 1.0)
Unity Y =  AGENT_HEIGHT        (0.5 for agents; tiles at 0)
Unity Z = -row * CELL_SIZE     (negated so row 0 renders at the back/top)
```

Example: Python `(row=2, col=3)` → Unity `(3.0, 0.5, -2.0)`.

## Environment API (v1.0, frozen)

```python
from python_backend.environment.grid_world import GridWorld
from python_backend.environment.constants import Position

env = GridWorld(
    height=10, width=10,
    agents_config=[(Position(0, 0), Position(9, 9))],
    obstacles={Position(4, 4)},
    max_steps=100,
    seed=42,
)
obs = env.reset()
actions = {0: 3}  # agent 0 -> RIGHT
obs, rewards, dones, terminated, info = env.step(actions)
```

Action space (fixed): `0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY`.

## Key design rules

- Simultaneous multi-agent stepping — no sequential movement, no ordering bias.
- Collisions (boundary / obstacle / same-cell / swap) bounce agents back with
  penalties from `RewardConfig`; `terminated` (goals) and `truncated` (step
  limit) stay separate.
- Scenarios are frozen JSON files shared by Python and Unity, so the 3D world
  cannot drift from the logical world.
- Comparison tables refuse to print rows that were not produced under identical
  conditions (`comparison.assert_fair`).
- Rewards are always the true environment rewards; the cooperative training term
  is never reported as a result.
- Rule-based coordination is never described as learned; the CTDE variant is
  labelled "CTDE-inspired" and is not MAPPO/QMIX/MADDPG.
- All reported numbers come from executed episodes — no fabricated results.
