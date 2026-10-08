# GridMind — Experiment Protocol and Metrics

> **Owner:** Developer 4. Covers Master Context §37–§43, §56, §65–§67.
> Every number reported by these tools comes from an executed episode or a
> measured training run. Nothing is pre-filled, estimated or fabricated.

## 1. Frozen inputs (fair comparison, §43)

| Input | File | Rule |
|-------|------|------|
| Fixed scenarios | `experiments/scenarios/*.json` | never edited between compared runs |
| Evaluation suite | `experiments/scenarios/eval_suite_v1.json` | lists the fixed scenarios + the seeded random block |
| Random-layout seeds | `seed_start + episode_index` | **identical sequence for every method** |
| Obstacle count | suite `random.num_obstacles` | same for every method |
| `max_steps` | 60 (scenario / suite) | same for every method |
| Agent prefix | first *N* agents of the scenario | scalability compares the same layout with more agents |

Two methods are only comparable when they ran with the same scenario, the same
episode budget, the same layout seeds and the same obstacle count. That rule is
enforced in code: `comparison.assert_fair(rows)` raises a `ValueError` if rows
being compared differ in any of those fields, so an unfair table cannot be
printed or saved.

## 2. Compared methods

| CLI alias | Method | Family | What it is |
|-----------|--------|--------|------------|
| `random` | Random baseline | single-agent | uniform choice over valid actions |
| `qlearning` | Tabular Q-learning | single-agent | `agents/q_learning.py` (Developer 2) |
| `dqn` | DQN (single agent) | single-agent | `agents/dqn.py` (Developer 2) |
| `multiagent` | **Independent DQN** | multi-agent | `MultiAgentSystem` with `coordination_mode="none"` (Developer 3) |
| `rule_based` | **Rule-based coordination** | multi-agent | deterministic priority yields — explicitly *not* learned |
| `coordinated` | **Coordinated DQN** | multi-agent | enriched observations + cooperative reward term (learned) |
| `ctde` | CTDE-inspired DQN | multi-agent | trains with the richer view, executes local-only. **Not** MAPPO/QMIX/MADDPG |

The headline comparison (§65) is `multiagent` (independent) vs `coordinated`;
`rule_based` and `ctde` are the ablation variants (§42) and must keep their
honest labels in every table and figure.

## 3. Training conditions

* Training uses the **deterministic fixed scenario** (first *N* agents of
  `demo_10x10_3agents.json`), i.e. the same layout for every method, as required
  by §18 for reproducibility.
* One shared episode budget: `--train-episodes` (default 400, `--quick` = 20).
  The same number is used for every method in a run, so no method gets a
  training advantage.
* `epsilon_decay` is derived as `(epsilon_min / 1.0) ** (1 / episodes)`, so
  every learning method starts at `epsilon = 1.0` and *lands exactly on*
  `epsilon_min = 0.05` at the end of the budget. Budget-independent exploration
  keeps the comparison honest when the budget changes.
* DQN keeps its bootstrap target in a bounded regime: each stored reward is
  rescaled by `REWARD_SCALE = 1.0` (`python_backend/algorithms/dqn.py`) and the
  DQN learning rate is `2e-4` (`build_agent_dqn_config`). The rescaling is
  monotonic, so it never changes the argmax of the true Q-function; it only
  controls the absolute size of the TD target. Measured on the 3-agent demo
  (60-step episodes, `gamma = 0.99`), the previous `scale = 10.0` / `5e-4`
  pair drove a positive-feedback Q-value blow-up (max`|Q|` 0.6 → 1e9 over a
  3000-episode run) that collapsed the greedy policy to all-`STAY`
  (fixed-layout success 0.00), while `1.0` / `2e-4` plateaus at max`|Q|` ≈ 1e2
  and the greedy policy keeps all 3 agents on their goals (success 1.00,
  reward 245).
* Execution/evaluation is **greedy** (no exploration), for every method.
* Random / rule-based / CTDE distinctions are configuration only; the learning
  code is not duplicated anywhere in the evaluation module.

### Reuse policy (no duplicated RL logic)

| Concern | Reused from |
|---------|-------------|
| Q-learning / DQN training loops | `python_backend/algorithms/trainer.py` |
| Multi-agent episode loop, coordination, deadlock, communication | `python_backend/multi_agent/multi_agent_system.py` |
| Random baseline | `python_backend/algorithms/random_policy.py` |
| Environment | frozen Environment API v1.0 (`GridWorld`) |

The evaluation module only *observes*: it records metrics around those loops.

## 4. Evaluation conditions

| Condition | Layout | Episodes |
|-----------|--------|----------|
| `fixed` | the frozen deterministic scenario | `--fixed-eval-episodes` (default 1: the scenario is deterministic, so repeating it measures nothing new) |
| `random` | seeded random layouts, same seed sequence for all methods | `--eval-episodes` (default: suite value, 30) |

The random condition is the statistical one. With one deterministic scenario and
a deterministic greedy policy, every episode returns the same outcome, so a
success *rate* over a single fixed scenario is 0 or 1 by construction — that is
why the headline rates come from the random condition, and why the fixed
condition is reported as a reproducibility check instead.

## 5. Metric definitions

| Metric | Definition |
|--------|------------|
| `success` | the agent reached its goal (`done == True`) |
| `success_rate` | mean of `success` over evaluated episodes (+ std) |
| `goal_completion_rate` | mean fraction of agents that reached their goal |
| `avg_reward` | mean **true** environment reward per episode (+ std). The cooperative reward term used for *training* is never reported |
| `avg_steps` | mean number of `env.step` calls, including the terminating step |
| `collisions` | one count per `(agent, step)` entry in `info["collisions"]` (a same-cell conflict counts once per involved agent, as the environment reports it) |
| `avg_collision_rate` | collisions per agent per step |
| `avg_collision_types` | same split by `BOUNDARY` / `OBSTACLE` / `SAME_CELL` / `SWAP` |
| `deadlocks` | deadlock events fired by `DeadlockDetector` (single-agent: not applicable, reported 0) |
| `distance` | Manhattan distance travelled by an agent, summed over the episode (total = sum over agents) |
| `messages` | communication messages sent (0 when communication is disabled) |
| `training_seconds` | wall-clock seconds measured with `perf_counter` around the training call only |
| `evaluation_seconds` | wall-clock seconds for the evaluation episodes |
| `computation_seconds` | `training_seconds + evaluation_seconds` (scalability cost, §67) |

Single-agent methods control **agent 0 only** in the scenario (the demo
scenario's first agent). Cross-family rows therefore differ in `agents` (1 vs
3): the algorithm comparison is valid within a family, and the *agent-count
matched* comparison is the multi-agent ablation table (§65), where every row
controls the same number of agents.

## 6. Outputs

| Path | Content |
|------|---------|
| `experiments/results/<name>.json` | full report: manifest + aggregate rows + per-episode metrics |
| `experiments/results/<name>.csv` | one row per method/condition (aggregates) |
| `experiments/results/comparison_episodes.csv` | per-episode metrics for every method |
| `experiments/results/comparison_training_history.csv` | per-episode training curves |
| `experiments/results/manifest_<name>.json` | reproducibility manifest (seeds, scenario, suite, budget, counts) |
| `experiments/plots/*.png` | Phase 11/12/13 figures |
| `models/q_learning/`, `models/dqn/`, `models/multi_agent/` | checkpoints (Developer 2/3 formats) |

Figures produced: reward vs episode, success rate vs episode (rolling mean over
the plotted window — stated in the axis label), collisions vs episode, average
steps bar chart, algorithm comparison (success/reward/steps/collisions),
deadlock comparison, communication overhead, scalability curves (success,
reward, collisions, deadlocks vs agent count), and ablation figures.

## 7. Commands

```bash
# fast smoke pass (small budgets; numbers are NOT meaningful results)
python main.py --mode compare --quick

# full algorithm comparison: random / Q-learning / DQN / independent /
# rule-based / coordinated / CTDE
python main.py --mode compare

# one method end-to-end (train -> save -> report -> plots)
python main.py --mode train --algorithm dqn
python main.py --mode train --algorithm multiagent --agents 3
python main.py --mode train --algorithm coordinated --agents 3

# evaluate saved checkpoints only (no training)
python main.py --mode evaluate --algorithm multiagent --agents 3

# Phase 12 ablation: independent vs rule-based vs coordinated (vs CTDE)
python main.py --mode ablation

# Phase 13 scalability: 1..5 agents on the same layout
python main.py --mode scalability --agent-counts 1,2,3,4,5
```

Useful flags: `--train-episodes`, `--eval-episodes`, `--fixed-eval-episodes`,
`--seed`, `--scenario`, `--suite`, `--methods`, `--no-plots`,
`--no-save-models`, `--model`, `--model-tag`.

Runtime scales with episodes × agents × steps; that is the real cost of real
runs, and it is measured and reported rather than hidden.

## 8. Reproducibility

The report manifest records: experiment name, methods, agent count, training and
evaluation episode counts, seed, training layout, execution policy, reward
source, the full scenario description and the suite description. Seeds are
explicit everywhere: policy seeds (`--seed`), layout seeds (suite
`seed_start + i`), and per-agent DQN seeds (`seed + agent_id`).

Verification (measured, not assumed): training `independent_dqn` for 20 episodes
on 2 agents and then evaluating the **saved checkpoint** in a separate process
reproduced the identical random-condition metrics
(`success=0.00, reward=-860.33, steps=60.00, collisions=38.00, deadlocks=33.33`)
— a same-process/separate-process sanity check of the frozen seeds.

Tests: `python -m pytest python_backend/tests/ -q` → **197 passed**
(30 environment + 32 RL + 55 multi-agent + 49 evaluation + 31 integration).

## 9. Academic integrity rules applied

* No expected/illustrative number is written into any table or figure.
* Smoke budgets (`--quick`) produce real but non-conclusive numbers; reports
  must state the budget they were produced with.
* Rule-based coordination is always labelled rule-based, never "learned".
* The CTDE variant is labelled "CTDE-inspired" and is not claimed to be MAPPO,
  QMIX or MADDPG.
* Untrained models are reported as `trained=false` (live sessions) or with the
  budget they were trained with (reports).
