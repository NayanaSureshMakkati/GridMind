# GridMind — Training State (checkpoint)

> **STATUS: ALL TRAINING COMPLETE.** The chain finished 2026-10-07 19:59:27 with
> every stage at exit 0 and no Python processes left running. This file is kept as
> the record of what was run, the fixes that were needed, and the known caveats.
>
> **Purpose:** durable record of exactly how far training got, so a later session
> can resume instead of restarting. Written 2026-10-07 19:10; completed 20:00.

---

## 1. How the run is driven

One background chain script executes every remaining stage in order and appends
to a single log:

| Item | Path |
| --- | --- |
| Chain script | `experiments/logs/run_all_training.sh` |
| Chain log (source of truth) | `experiments/logs/training_complete.log` |
| Env | `.venv/Scripts/python.exe` (Python 3.10.11) |

Check progress with:

```bash
grep -E "##########|EXIT_" experiments/logs/training_complete.log   # stage markers
sed -n '/MARL_DRIVER START/,$p' experiments/logs/training_complete.log | tail -30
```

A stage is finished only when its `EXIT_<NAME>=` line exists.
**If the last marker is `MARL_DRIVER START` with no `EXIT_MARL_DRIVER`, it is
still running** — check `tasklist //FI "IMAGENAME eq python.exe"`.

---

## 2. Stage status

| # | Stage | Command | Status | Artifacts |
| --- | --- | --- | --- | --- |
| 1 | Full comparison (7 methods, 400 eps) | `main.py --mode compare` | ✅ done (2026-10-06 20:09) | `results/comparison.{json,csv}`, `comparison_episodes.csv`, `manifest_comparison.json` |
| 2 | Ablation (independent / rule-based / coordinated, 3 agents) | `main.py --mode ablation` | ✅ done (2026-10-06 20:17) | `results/ablation.{json,csv}`, `manifest_ablation.json` |
| 3 | **Scalability sweep 1–5 agents** | `main.py --mode scalability --agent-counts 1,2,3,4,5` | ✅ results written 2026-10-07 18:30 | `results/scalability.{json,csv}`, `scalability_episodes.csv`, `manifest_scalability.json` |
| 4 | Single-agent RL driver | `scripts/run_rl_experiments.py` | ✅ done 2026-10-07 18:34, exit 0 | `results/training_{q,dqn}_{5x5,10x10}.csv`, `models/{q_5x5.pkl,q_10x10.pkl,dqn_5x5.pt,dqn_10x10.pt}` |
| 5 | **MARL driver (600 eps/config)** | `scripts/run_marl_experiments.py` | ✅ **done 2026-10-07 19:59, exit 0** | `results/marl_full.{json,csv}` — 13 configurations, 5 systems (random, independent, rule_based, coordinated, ctde), agent counts 2–5 |

> This run used the **pre-patch** script, which writes its result file only once,
> at the very end. The incremental per-configuration save added in this session
> (§4) applies to any **future** run, so re-running the command in §3 will
> checkpoint itself as it goes.

Stage 3 note: the sweep computed correctly but the **plot step crashed**
(exit 1). Code fixed; the four figures were regenerated from the saved JSON.
No retraining was needed (the real 400-episode results were never overwritten).

**Verified after the fix:** re-running the same command path
(`main.py --mode scalability --quick --no-save-models`, which exercises the
exact N=1 `dqn` / N≥2 `independent_dqn` label mix that crashed) now prints
`plots: 4 file(s)` and exits **0**. The full-run results and figures were then
restored and regenerated from `scalability.json`.

### Stage 5 results (`results/marl_full.json`)

Ablation at 3 agents, evaluated on 30 seeded random layouts:

| system | all-success | any-success | avg reward/ep | collisions | deadlocks/ep |
| --- | --- | --- | --- | --- | --- |
| random | 0.000 | 0.533 | −58.8 | 3.4 | 5.57 |
| independent_dqn | 0.000 | 0.133 | −219.7 | 21.9 | 26.20 |
| rule_based_coordination | 0.000 | 0.167 | −95.4 | **0.0** | 28.43 |
| coordinated_dqn | 0.000 | 0.167 | −253.4 | 27.2 | 27.27 |
| ctde_inspired_dqn | 0.000 | 0.033 | −132.0 | 7.9 | 23.90 |

Scalability (independent vs coordinated), success and collisions:

| agents | independent any-success / collisions | coordinated any-success / collisions |
| --- | --- | --- |
| 2 | 0.200 / 7.0 | 0.167 / 77.0 |
| 3 | 0.133 / 21.9 | 0.167 / 27.2 |
| 4 | **0.300** / 52.1 | 0.200 / 68.2 |
| 5 | **0.433** / 34.5 | 0.267 / 26.0 |

Note the honest reading: on **randomly generated** 10×10 layouts with 4 obstacles,
no system reaches all-success; every reported `all_success_rate` is 0.0. Only the
*fixed* demo layout is solved (coordinated, success 1.00, ~22 steps) — that is the
configuration the Unity demo uses.

---

## 3. Exact resume commands

Run from the repo root. Nothing here is destructive; each command overwrites its
own result file.

**Resume the MARL driver (the only unfinished stage):**

```bash
# everything (ablation + scalability 2..5), saves after every configuration
.venv/Scripts/python.exe scripts/run_marl_experiments.py --name marl_full

# or only the remaining scalability chunk (faster, keeps ablation separate)
.venv/Scripts/python.exe scripts/run_marl_experiments.py \
    --suite scalability --agents 2,3,4,5 --kinds independent,coordinated \
    --name marl_scalability_2345_full
```

**Re-run the whole chain from the top (only if a stage looks corrupted):**

```bash
bash experiments/logs/run_all_training.sh
```

**Rebuild only the scalability figures (no retraining):**

```bash
.venv/Scripts/python.exe main.py --mode scalability --agent-counts 1,2,3,4,5
```

**Confirm nothing is broken afterwards:**

```bash
.venv/Scripts/python.exe -m pytest python_backend/tests/ -q     # expect: 206 passed
```

---

## 4. Code changes made in this session (do not revert)

| File | Change | Why |
| --- | --- | --- |
| `experiments/scenarios/scalability_10x10_5agents.json` | **new** frozen layout: same 10×10 grid and obstacles as `demo_10x10_3agents`, with 5 agents whose first three are identical to the demo's | The frozen suite only defined 3 agents, so `--agent-counts 1,2,3,4,5` crashed with `ValueError: num_agents must be in 1..3, got 4`. Prefixes 1–3 still reproduce the demo layout exactly. |
| `python_backend/evaluation/scenarios.py` | added `SCALABILITY_SCENARIO` and `scalability_scenario(agent_counts)` | Picks the demo layout for short sweeps and the 5-agent layout when the sweep needs more agents. |
| `python_backend/evaluation/experiments.py` | `ExperimentConfig.scenario` falls back to `load_scenario()` for names outside the suite; `run_scalability` selects the layout via `scalability_scenario()` | Lets the scalability run target a frozen scenario file the suite does not list. |
| `python_backend/evaluation/plots.py` | `plot_scalability` now plots each label against **its own** agent counts | N=1 uses the single-agent path so it is labelled `dqn@…`, N≥2 `independent_dqn@…`; the old code plotted a 1-point series against a 5-point axis and raised `ValueError: x and y must have same first dimension`. |
| `scripts/run_marl_experiments.py` | `_save_progress()` writes the result file after **every** configuration (`--name` only) | The driver previously held an hour of compute in memory and wrote only at the very end, so any interruption lost everything. |

`python_backend/tests/ -q` → **206 passed** after all of the above.

---

## 5. Known caveat — single-agent DQN on 10×10 does not converge

Measured today (stage 4, full budget):

| Run | Result |
| --- | --- |
| Q-learning 5×5 (500 eps) | success **1.00** |
| DQN 5×5 (300 eps) | success **1.00** |
| Q-learning 10×10 (6000 eps) | success **1.00** |
| **DQN 10×10 (1200 eps)** | **success 0.00**, reward −50 (never reaches the goal) |

Also 0.00 in `comparison.json` for `dqn@fixed`, and previously confirmed by
`scripts/diag_dqn_{convergence,configs,budget}.py` (three learning-rate /
epsilon-decay schedules all failed). This is a **learning/tuning problem, not a
plumbing problem** — the training loop, masking and checkpointing all work.

It does **not** affect the Unity demo, which uses the *coordinated multi-agent*
model (success 1.00 on the demo layout).

---

## 6. Unity run command (verified working)

Everything Python-side is verified headless; Unity Editor was not available on
this machine, so the editor step itself is the only unverified part.

Terminal 1 — start the authoritative simulation (leave it running):

```bash
.venv/Scripts/python.exe main.py --mode visualize --algorithm coordinated --agents 3 \
    --model-tag demo_10x10_3agents_coordinated_dqn_3agents \
    --episodes 3 --speed-ms 250
```

`--model-tag` is **required**: without it the session loads no checkpoint, reports
`trained=false`, and the agents wander (107 collisions, success False). With the
tag it reports `trained=True` and the three agents reach their goals with 0
collisions (≈22 steps, reward ≈ 245).

Terminal 2 — Unity Editor:
1. Open the `unity_frontend/` project (Unity 2021.3 LTS or newer).
2. Open `Assets/Scenes/GridMindSimulation.unity`.
3. Press **Play**, then click **Start** in the panel.

Server listens on `127.0.0.1:8765` (`--host`, `--port`, `--speed-ms` to change).

Quick Python-only check of the same path (no Unity needed):

```bash
.venv/Scripts/python.exe main.py --mode visualize --headless \
    --algorithm coordinated --agents 3 \
    --model-tag demo_10x10_3agents_coordinated_dqn_3agents \
    --episodes 1 --no-plots --max-seconds 40
```

---

## 7. Checkpoints on disk (today's training)

```
models/scalability_10x10_5agents_*          # from the scalability sweep (N=1..5)
models/multi_agent/scalability_10x10_5agents_independent_dqn_{2,3,4,5}agents_agent*.pt
models/q_5x5.pkl  models/q_10x10.pkl        # tabular Q-learning (all success 1.00)
models/dqn_5x5.pt  models/dqn_10x10.pt      # single-agent DQN (10x10 does not converge)
models/multi_agent/demo_10x10_3agents_coordinated_dqn_3agents_agent*.pt   # Unity demo model
```

All load through `python_backend/evaluation/model_store.py`. The MARL driver does
**not** save model checkpoints — it only writes metrics to `experiments/results/`.
