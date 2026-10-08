# GridMind — Multi-Agent Coordination Layer (Developer 3)

> Status: **Phases 1–14 complete.** The frozen Environment API v1.0
> (`docs/environment_api.md`) is consumed as a black box and was **not modified**:
> no changes to GridWorld, DQN, reward definitions, coordinate system, or action mapping.

## Architecture

```
                    ┌────────────────────────────────────────────────┐
                    │                MultiAgentSystem                 │
                    │  (config: MultiAgentConfig — everything explicit)│
                    └────────────────────────────────────────────────┘
                                       │ per step
     ┌──────────┬──────────┬───────────┼────────────┬──────────────┬──────────┐
     │          │          │           ▼            │              │          │
 Observations  Action    RuleBased   Communication  Deadlock      Collision  Cooperative
 (Phase 5)     selection Arbiter     Manager        recovery      analysis   reward term
 LOCAL_ONLY or per agent (Phase 4,   (Phase 9,      (Phase 8,     (Phase 3,  (Phases 6/10,
 LOCAL_PLUS    (Phase 2)  optional)   optional)      optional)     never      learned reward
 OTHER_AGENTS  DQN or                                              hidden)    only)
               Random
     │          │          │           │              │              │          │
     └──────────┴──────────┴───────────┴──────┬───────┴──────────────┴──────────┘
                                              ▼
                                   env.step(all actions)   ← FROZEN API v1.0
                                   (env owns ALL physical
                                    conflict resolution)
```

### Modules

| File | Phase(s) | Responsibility |
|---|---|---|
| `python_backend/multi_agent/multi_agent_system.py` | 1, 2, 3, 6, 10, 11, 12 | `MultiAgentSystem` + `MultiAgentConfig`: 2–5 independently controllable agents, episode loop, training/evaluation, collision totals, CTDE execution view |
| `python_backend/multi_agent/independent_dqn.py` | 2 | `IndependentDQNSystem`: pure baseline (no coordination hooks) |
| `python_backend/multi_agent/coordination.py` | 4 | `RuleBasedArbiter` — **RULE-BASED COORDINATION**, explicitly NOT learned |
| `python_backend/multi_agent/observations.py` | 5 | LOCAL_ONLY (frozen 12-vector) vs LOCAL_PLUS_OTHER_AGENTS (12 + 4·(N−1)) |
| `python_backend/multi_agent/deadlock.py` | 7, 8 | `DeadlockDetector` (stagnation / position cycle / action pattern / mutual blocking) + configurable `RecoveryManager` |
| `python_backend/multi_agent/communication.py` | 9 | `CommunicationManager` — optional compact messages, overhead accounting |
| `scripts/run_marl_experiments.py` | 12, 13 | Scalability (2–5 agents) and ablation drivers; CSV+JSON results |
| `python_backend/tests/test_multi_agent.py` | 14 | 50 tests over the full phase list |

### Per-step pipeline (inside `MultiAgentSystem.run_episode`)

1. Build observations per configured mode (Phase 5; CTDE execution view when evaluating).
2. Each agent chooses an action simultaneously — epsilon-greedy over **valid actions only**, or `RandomPolicy` when `policy_kind="random"` (Phase 13 baseline).
3. `RuleBasedArbiter.resolve()` rewrites actions so lower-priority claimants yield (optional, Phase 4).
4. Deadlock recovery override, if a deadlock fired last step (optional, Phase 8; takes precedence over 3).
5. `CommunicationManager.broadcast_intents()` — after action selection so each message carries the **final** intended action (optional, Phase 9).
6. `env.step(all actions)` — the frozen environment performs **all** physical conflict resolution (swap, same-cell, cascade); this layer never moves agents itself.
7. Collisions tallied by type from `info["collisions"]` (Phase 3; recorded, never hidden).
8. Transitions stored per agent; optional cooperative team term added to the **learned** reward only (Phases 6/10).
9. `train_step()` per agent (own networks, own replay buffer).
10. `DeadlockDetector.record_step()` (Phase 7; recorded, the environment is **never silently reset**).

## Coordination explanation

**RULE-BASED COORDINATION (Phase 4 — deterministic, NOT learned).**
`RuleBasedArbiter` runs *before* `env.step()` and rewrites proposed actions.
Priority is a deterministic tuple — lower wins:

1. shorter Manhattan distance to the agent's own goal,
2. then longer waiting time (steps since last position change),
3. then lower agent id.

Resolution rules: head-on swaps → both agents wait; a mover into a stationary/yielded
agent's cell → waits; same-cell contest → highest-priority claimant proceeds, the rest
wait. All decisions are recorded as `CoordinationEvent`s. This layer only *prevents
requests* — the frozen environment still performs all physical resolution, so the
arbiter can never create a collision (verified by test).

**LEARNED COORDINATION (Phase 6).** Coordinated DQN = Independent DQN plus:
`LOCAL_PLUS_OTHER_AGENTS` observations (each agent sees every other agent's
goal delta, wait flag, done flag) and an optional cooperative reward term.
Agents learn to avoid conflicts from what they observe and what they are
rewarded for — there is no explicit arbitration rule involved.

**Cooperative reward (Phase 10).** `learned_reward = true_reward + w · mean(other
active agents' true rewards)`, with `w ∈ [0, 1]` enforced. Since the individual
goal reward is +100, the team term mathematically cannot dominate it. The
environment's own reward stream is untouched; metrics always use TRUE rewards,
and both streams are recorded separately.

**CTDE-inspired (Phase 11 — honest scope).** Training uses the enriched
observations (global-ish information); execution zeroes the other-agent blocks
(local-only information) while keeping the identical network input shape.
This is a simplified CTDE-*inspired* approximation. It is **not** MAPPO, QMIX,
MADDPG, or any other published algorithm: there is no centralized critic, no
parameter sharing, no mixing network. `system_label()` reports
`ctde_inspired_dqn` accordingly.

**Communication (Phase 9).** Optional (`communication_enabled=False` by default).
Each active agent broadcasts one fixed-width message per step:
`(sender_id, row, col, intended_action, goal_reached)`. Terminal agents stay
silent. Overhead is tracked per step, per episode, and per agent-step.

## Training / evaluation commands

```bash
# Full test suite (Phase 14) — 115 tests
python -m pytest python_backend/tests/ -v

# Phases 12–13 experiments (full suite: 600 train episodes, 30 eval episodes)
python scripts/run_marl_experiments.py                    # all suites
python scripts/run_marl_experiments.py --suite scalability
python scripts/run_marl_experiments.py --suite ablation
python scripts/run_marl_experiments.py --suite scalability --agents 2,3,4,5

# Chunked execution (same seeds → chunks are mergeable/comparable)
python scripts/run_marl_experiments.py --suite ablation --kinds random,independent --name marl_ablation_chunk1
python scripts/run_marl_experiments.py --suite ablation --kinds rule_based,coordinated,ctde --name marl_ablation_chunk2

# Smoke test (fast; numbers are NOT reportable)
python scripts/run_marl_experiments.py --quick
```

Programmatic use:

```python
from python_backend.environment.grid_world import GridWorld
from python_backend.multi_agent.multi_agent_system import MultiAgentConfig, MultiAgentSystem

env = GridWorld(height=10, width=10, agents_config=..., max_steps=60, seed=42)
system = MultiAgentSystem(env, MultiAgentConfig(num_agents=3, observation_mode="local_plus_other_agents",
                                                cooperative_weight=0.3))
system.train(episodes=600)
system.save_models("models", tag="coordinated_3ag")
result = system.evaluate(episodes=30, seed=1000)
```

## Metrics (results format)

`MultiAgentEvaluationResult` (and one CSV/JSON row per configuration):

| Field | Meaning |
|---|---|
| `all_success_rate` / `any_success_rate` | fraction of eval episodes where all / at least one agent reaches its goal |
| `avg_reward` | mean TRUE environment reward per agent (cooperative term excluded) |
| `avg_steps` | mean episode length |
| `avg_collisions` | mean total collisions per agent |
| `avg_collision_types` | obstacle / agent / same_cell / swap breakdown per episode (Phase 12) |
| `deadlocks_per_episode` | deadlock events detected (stagnation, position_cycle, action_pattern, mutual_blocking) |
| `recoveries_per_episode` | recovery overrides applied |
| `messages_per_episode` / `messages_per_step` / `messages_per_agent_step` | communication overhead |
| `train_seconds` | wall-clock training time |

Results are written to `experiments/results/marl_<suite>_<suffix>.{csv,json}`.
Fair comparison: all compared systems train and evaluate on the **same seeded
scenario sequence**; DQN agents use fixed per-agent seeds, so identical configs
reproduce identical results.

## Limitations

- **Action masking:** TD bootstrap maximizes Q over all 5 actions; selection is
  always masked to valid actions, but targets may include invalid-action Q-values.
- **Rule-based priority is myopic:** it resolves one step of intent overlap; it
  cannot anticipate multi-step congestion (no MAPF-style lookahead).
- **Deadlock detector is heuristic:** thresholds (stagnation length, window size,
  collision count) are tunable and may under- or over-fire on atypical layouts.
- **Recovery is deliberately simple** (`alternate_action`, once per episode by
  default) to keep measurements clean; it is not a planning-based solver.
- **Communication is not used for decisions** by default: messages are collected
  and accounted, but agents act on observations; a consumer extension could feed
  them into the arbiter or observations.
- **CTDE is an approximation:** information asymmetry is simulated by zeroing
  blocks; it is not a published CTDE algorithm.
- **Cooperative reward uses the mean of others' rewards** (count-invariant) but
  still creates non-stationarity from each agent's perspective — inherent to
  independent learning in shared environments.
- **Scalability measured up to 5 agents** per the phase spec; nothing in the
  design prevents more, but observation size grows linearly in enriched mode.

## Test coverage (Phase 14)

50 multi-agent tests (115 total with Developers 1–2) covering: multiple agents,
simultaneous actions, collision detection (all four types, never hidden),
rule-based coordination (priority, swap, same-cell, no-collision guarantee),
observation modes, deadlock detection (all four signals), recovery (budget,
valid actions, determinism), communication (accounting, optional, content),
cooperative reward (true/learned separation, non-domination), scalability
(2–5 agents), evaluation determinism, and model persistence.
