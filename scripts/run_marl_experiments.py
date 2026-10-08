"""Developer 3 experiment driver — SCALABILITY (Phase 12) and ABLATION (Phase 13).

All numbers printed/saved come from actual runs (Master Context §43/§75).
Fair-comparison rules (§43): every compared configuration trains and evaluates
on the SAME seeded scenario sequence; DQN agents use fixed per-agent seeds
(DQNConfig(seed=...)) so identical configs reproduce identical results.

Usage:
    python scripts/run_marl_experiments.py            # full suite (long!)
    python scripts/run_marl_experiments.py --quick    # smoke test
    python scripts/run_marl_experiments.py --suite scalability
    python scripts/run_marl_experiments.py --suite ablation
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_backend.environment.grid_world import GridWorld
from python_backend.environment.reward import RewardConfig
from python_backend.multi_agent.multi_agent_system import (
    MultiAgentConfig,
    MultiAgentSystem,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")

TRAIN_EPISODES = 600          # exactly completes the epsilon schedule
                              # 1.0 -> 0.05 at decay 0.995 (0.995^600 ~= 0.049).
                              # Uniform across ALL compared configs
                              # (fairness, §43). Calibrated against Developer
                              # 2's single-agent DQN budget (300 eps @ 5x5,
                              # 1200 eps @ 10x10) in scripts/run_rl_experiments.py.
EVAL_EPISODES = 30
MAX_STEPS = 60
NUM_OBSTACLES = 4             # seeded per-episode layouts (identical across systems)
DQN_SEED = 77
QUICK_TRAIN_EPISODES = 6
QUICK_EVAL_EPISODES = 3


def build_env(num_agents: int, max_steps: int = MAX_STEPS) -> GridWorld:
    """10x10 environment with a deterministic fixed layout (§18 training rule).

    Evaluation re-generates seeded random layouts per episode via
    reset(randomize=True, seed=...), identical across compared systems (§43).
    """
    from python_backend.environment.constants import Position

    obstacles = {Position(4, 4), Position(4, 5), Position(5, 4)}
    agents_config = [
        (Position(i, 0), Position(9 - i, 9)) for i in range(num_agents)
    ]
    return GridWorld(
        height=10,
        width=10,
        agents_config=agents_config,
        obstacles=obstacles,
        reward_config=RewardConfig(),
        max_steps=max_steps,
        seed=42,
    )


def dqn_hp():
    """Imported lazily so --help stays fast."""
    from python_backend.algorithms.dqn import DQNConfig
    return DQNConfig(
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        learning_rate=5e-4,
        batch_size=64,
        buffer_capacity=50000,
        min_buffer_size=500,
        target_update_frequency=250,
        hidden_size=128,
        observation_size=12,  # MultiAgentSystem overrides per observation mode
        seed=DQN_SEED,
    )


def build_system(env: GridWorld, num_agents: int, kind: str) -> MultiAgentSystem:
    """Builds one compared system. Kinds:
        random        — RandomPolicy (uniform over valid actions, Phase 13 baseline;
                        no networks, no training, honestly labeled "random")
        independent   — Independent DQN (coordination none, local obs)
        rule_based    — RULE-BASED COORDINATION (deterministic priority yields)
        coordinated   — LEARNED COORDINATION (enriched obs + cooperative reward)
        ctde          — CTDE-INSPIRED (train enriched, execute local-only view)
    """
    from python_backend.algorithms.dqn import DQNConfig

    base = dqn_hp()
    if kind == "random":
        # True uniform-random baseline over valid actions (Developer 2's
        # RandomPolicy). No networks are built or trained.
        cfg = MultiAgentConfig(num_agents=num_agents, policy_kind="random", random_seed=DQN_SEED)
        return MultiAgentSystem(env, cfg)

    cfg = MultiAgentConfig(num_agents=num_agents, dqn_config=DQNConfig(**vars(base)))
    if kind == "independent":
        pass
    elif kind == "rule_based":
        cfg.coordination_mode = "rule_based"
    elif kind == "coordinated":
        cfg.observation_mode = "local_plus_other_agents"
        cfg.cooperative_weight = 0.3
    elif kind == "ctde":
        cfg.observation_mode = "local_plus_other_agents"
        cfg.ctde = True
    else:
        raise ValueError(f"Unknown kind '{kind}'")
    return MultiAgentSystem(env, cfg)


def run_config(kind: str, num_agents: int, train_episodes: int, eval_episodes: int) -> dict:
    """Trains (except 'random') and evaluates one configuration; returns metrics."""
    env = build_env(num_agents)
    system = build_system(env, num_agents, kind)
    label = system.system_label()

    t0 = time.time()
    if kind != "random":
        system.train(train_episodes)  # deterministic fixed-layout training (§18)
    train_seconds = time.time() - t0

    result = system.evaluate(
        episodes=eval_episodes, seed=1000, randomize=True, num_obstacles=NUM_OBSTACLES
    )
    ct = result.avg_collision_types
    row = {
        "system": label,
        "requested_kind": kind,
        "num_agents": num_agents,
        "train_episodes": train_episodes if kind != "random" else 0,
        "eval_episodes": eval_episodes,
        "all_success_rate": round(result.all_success_rate, 4),
        "any_success_rate": round(result.any_success_rate, 4),
        "avg_reward_per_agent": json.dumps({k: round(v, 2) for k, v in result.avg_reward.items()}),
        "avg_reward_mean": round(sum(result.avg_reward.values()) / max(1, len(result.avg_reward)), 2),
        "avg_steps": round(result.avg_steps, 2),
        "avg_collisions_total": round(sum(result.avg_collisions.values()), 3),
        "avg_obstacle_collisions": round(ct["obstacle"], 3),
        "avg_agent_collisions": round(ct["agent"], 3),
        "avg_same_cell_collisions": round(ct["same_cell"], 3),
        "avg_swap_collisions": round(ct["swap"], 3),
        "deadlocks_per_episode": round(result.deadlocks_per_episode, 3),
        "recoveries_per_episode": round(result.recoveries_per_episode, 3),
        "messages_per_episode": round(result.messages_per_episode, 2),
        "messages_per_agent_step": round(result.messages_per_agent_step, 4),
        "train_seconds": round(train_seconds, 1),
    }
    return row


def save_results(rows: list, name: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {csv_path}\nSaved: {json_path}")


def print_table(rows: list) -> None:
    cols = ["system", "num_agents", "all_success_rate", "any_success_rate",
            "avg_reward_mean", "avg_steps", "avg_collisions_total",
            "deadlocks_per_episode", "messages_per_agent_step", "train_seconds"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-|-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def _save_progress(rows: list, progress_name: str | None) -> None:
    """Persists the rows finished so far.

    A full suite is an hour of compute, so an interrupted run must keep the
    configurations it already finished instead of losing everything: the
    caller passes ``--name`` and the file grows after every configuration
    (re-running the same command simply overwrites it with a full run).
    """
    if not progress_name or not rows:
        return
    save_results(rows, progress_name)
    print(f"progress saved: {len(rows)} configuration(s) -> {progress_name}.json", flush=True)


def suite_scalability(train_episodes: int, eval_episodes: int, agent_counts, kinds=None,
                      progress_name: str | None = None) -> list:
    """PHASE 12: 2, 3, 4, 5 agents for the requested compared systems."""
    kinds = kinds or ("independent", "coordinated")
    rows = []
    for num_agents in agent_counts:
        for kind in kinds:
            print(f"=== scalability: {kind}, {num_agents} agents ===", flush=True)
            rows.append(run_config(kind, num_agents, train_episodes, eval_episodes))
            print_table(rows[-1:])
            _save_progress(rows, progress_name)
    return rows


def suite_ablation(train_episodes: int, eval_episodes: int, kinds=None,
                   progress_name: str | None = None) -> list:
    """PHASE 13: random / independent / rule-based / coordinated / CTDE at 3 agents."""
    kinds = kinds or ("random", "independent", "rule_based", "coordinated", "ctde")
    rows = []
    for kind in kinds:
        print(f"=== ablation: {kind} (3 agents) ===", flush=True)
        rows.append(run_config(kind, 3, train_episodes, eval_episodes))
        print_table(rows[-1:])
        _save_progress(rows, progress_name)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="GridMind MARL experiments (Phases 12-13)")
    parser.add_argument("--quick", action="store_true", help="reduced episodes (smoke test)")
    parser.add_argument("--suite", choices=["all", "scalability", "ablation"], default="all")
    parser.add_argument("--agents", default="2,3,4,5",
                        help="comma-separated agent counts for the scalability suite")
    parser.add_argument("--train-episodes", type=int, default=None,
                        help="override the default training episode count")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="override the default evaluation episode count")
    parser.add_argument("--kinds", default=None,
                        help="comma-separated subset of: random,independent,"
                             "rule_based,coordinated,ctde (chunked execution; "
                             "same seeds make chunks mergeable/comparable)")
    parser.add_argument("--name", default=None,
                        help="override the results file name (chunked runs "
                             "would otherwise overwrite each other)")
    args = parser.parse_args()

    train_episodes = QUICK_TRAIN_EPISODES if args.quick else TRAIN_EPISODES
    eval_episodes = QUICK_EVAL_EPISODES if args.quick else EVAL_EPISODES
    if args.train_episodes is not None and not args.quick:
        train_episodes = args.train_episodes
    if args.eval_episodes is not None and not args.quick:
        eval_episodes = args.eval_episodes
    agent_counts = [int(a) for a in args.agents.split(",") if a.strip()]
    valid_kinds = ("random", "independent", "rule_based", "coordinated", "ctde")
    kinds = None
    if args.kinds:
        kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
        unknown = [k for k in kinds if k not in valid_kinds]
        if unknown:
            parser.error(f"unknown kind(s) {unknown}; valid: {list(valid_kinds)}")
        if not kinds:
            parser.error("--kinds produced an empty kind list")

    # The result name is resolved before the suites run so each finished
    # configuration can be written to disk immediately (resumable runs).
    suffix = "quick" if args.quick else "full"
    if args.name:
        name = args.name
    elif args.suite == "all":
        name = f"marl_{suffix}"
    else:
        agents_tag = "_" + "".join(str(a) for a in agent_counts) if args.suite == "scalability" else ""
        name = f"marl_{args.suite}{agents_tag}_{suffix}"
    progress_name = None if args.quick else name

    rows = []
    if args.suite in ("all", "ablation"):
        rows += suite_ablation(train_episodes, eval_episodes, kinds, progress_name)
    if args.suite in ("all", "scalability"):
        rows += suite_scalability(train_episodes, eval_episodes, agent_counts, kinds, progress_name)

    save_results(rows, name)
    print("\n=== FINAL RESULTS ===")
    print_table(rows)
    print(f"\nSettings: train_episodes={train_episodes} eval_episodes={eval_episodes} "
          f"agents={agent_counts} max_steps={MAX_STEPS} obstacles={NUM_OBSTACLES}")
    print("NOTE: with --quick these numbers are smoke tests only; "
          "report only full-suite numbers in the report (§75).")


if __name__ == "__main__":
    main()
