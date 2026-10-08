"""Developer 2 experiment driver (Phases 4, 5, 9, 10, 11).

Runs the real trainings and the Random vs Q-learning vs DQN comparison, then
saves models (models/), metrics (experiments/results/), and plots
(experiments/plots/). All numbers printed/saved come from actual runs.

Usage:
    python scripts/run_rl_experiments.py                 # full suite
    python scripts/run_rl_experiments.py --quick         # reduced episodes
    python scripts/run_rl_experiments.py --suite single  # 5x5 + 10x10 only
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_backend.environment.grid_world import GridWorld
from python_backend.environment.constants import Position
from python_backend.algorithms.q_learning import QLearningAgent, QLearningConfig
from python_backend.algorithms.dqn import DQNAgent, DQNConfig
from python_backend.algorithms.trainer import (
    train_q_learning,
    train_dqn,
    evaluate_random,
    evaluate_q_learning,
    evaluate_dqn,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")
PLOTS_DIR = os.path.join(ROOT, "experiments", "plots")

# Per-grid epsilon decay: bigger grids need slower decay so epsilon-greedy
# random exploration actually reaches the goal before exploitation kicks in.
Q_HP_5 = {
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.05,
}
Q_HP_10 = {
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.997,
    "epsilon_min": 0.05,
}
Q_HP = Q_HP_5  # default; grid-specific configs selected at runtime
DQN_HP = {
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,
    "learning_rate": 5e-4,
    "batch_size": 64,
    "buffer_capacity": 50000,
    "min_buffer_size": 500,
    "target_update_frequency": 250,
    "hidden_size": 128,
}

# Per Master Context §18, TRAINING uses deterministic fixed scenarios.
# Generalization is measured separately by zero-shot evaluation on seeded
# random layouts.
TRAIN_RESET: dict = {}  # fixed scenario
EVAL_RANDOM_RESET = {"randomize": True, "num_obstacles": 4}


def fixed_obstacles(grid_size: int) -> set:
    """Deterministic obstacle pattern per grid size (BFS-validated below)."""
    if grid_size == 5:
        return {(1, 1), (1, 3), (3, 1), (3, 3)}  # four pillars
    if grid_size == 10:
        wall = {(r, 4) for r in range(1, 8)} - {(4, 4)}  # wall with a gap
        return wall | {(6, 7), (7, 7)}
    raise ValueError(f"No fixed scenario defined for {grid_size}x{grid_size}")


def save_csv(path: str, rows: list) -> None:
    if not rows:
        return
    # Union of keys across rows (optional keys like eval_success_rate may
    # appear only on some episodes).
    fieldnames: list = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def moving_average(values: list, window: int = 50) -> list:
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_training(history: list, title: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = [h["episode"] for h in history]
    rewards = [h["reward"] for h in history]
    steps = [h["steps"] for h in history]
    successes = [1.0 if h["success"] else 0.0 for h in history]
    epsilons = [h["epsilon"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    axes[0, 0].plot(episodes, rewards, alpha=0.3, color="gray", label="per episode")
    axes[0, 0].plot(episodes, moving_average(rewards), color="blue", label="MA(50)")
    axes[0, 0].set_title("Episode reward")
    axes[0, 0].legend()

    axes[0, 1].plot(episodes, moving_average(successes), color="green")
    axes[0, 1].set_title("Success rate MA(50)")
    axes[0, 1].set_ylim(-0.05, 1.05)

    axes[1, 0].plot(episodes, steps, alpha=0.3, color="gray")
    axes[0, 0].set_ylabel("reward")
    axes[1, 0].set_title("Steps per episode")
    axes[1, 0].set_xlabel("episode")

    axes[1, 1].plot(episodes, epsilons, color="red")
    axes[1, 1].set_title("Epsilon schedule")
    axes[1, 1].set_xlabel("episode")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_comparison(results: dict, title: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algorithms = list(results.keys())
    metrics = [
        ("success_rate", "Success rate", (-0.05, 1.05)),
        ("avg_reward", "Average reward", None),
        ("avg_steps", "Average steps", None),
        ("avg_collisions", "Average collisions", None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    fig.suptitle(title)

    for ax, (key, label, ylim) in zip(axes, metrics):
        values = [results[a][key] for a in algorithms]
        bars = ax.bar(algorithms, values, color=["#999", "#4c8", "#48c"][: len(algorithms)])
        ax.set_title(label)
        if ylim:
            ax.set_ylim(*ylim)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment suites
# ---------------------------------------------------------------------------


def make_env(grid_size: int) -> GridWorld:
    """Fixed one-agent scenario: (0,0) -> (n-1,n-1) with a fixed obstacle set."""
    obstacles = {(Position(r, c)) for (r, c) in fixed_obstacles(grid_size)}
    start, goal = Position(0, 0), Position(grid_size - 1, grid_size - 1)
    assert GridWorld.is_reachable(start, goal, obstacles, grid_size, grid_size), \
        "fixed scenario must be reachable"
    return GridWorld(
        grid_size, grid_size,
        agents_config=[(start, goal)],
        obstacles=obstacles,
        max_steps=grid_size * 5,
        seed=42,
    )


def run_single_agent_q(grid_size: int, episodes: int, eval_episodes: int) -> dict:
    """Phases 4-5: train + evaluate Q-learning with one agent."""
    print(f"\n=== Q-learning {grid_size}x{grid_size}, {episodes} episodes ===")
    env = make_env(grid_size)
    q_hp = Q_HP_5 if grid_size == 5 else Q_HP_10
    agent = QLearningAgent(QLearningConfig(seed=0, **q_hp))

    t0 = time.time()
    history = train_q_learning(
        env, agent, episodes=episodes, seed=0,
        eval_every=max(1, episodes // 10),
        reset_kwargs=TRAIN_RESET,
    )
    train_time = time.time() - t0

    # Greedy evaluation on the FIXED training scenario (Phase 5) ...
    metrics = evaluate_q_learning(env, agent, episodes=eval_episodes, reset_kwargs=TRAIN_RESET)
    # ... and zero-shot on seeded random layouts (generalization probe).
    gen = evaluate_q_learning(env, agent, episodes=eval_episodes, reset_kwargs=EVAL_RANDOM_RESET)
    metrics["random_layout_success_rate"] = gen["success_rate"]
    metrics["train_time_sec"] = round(train_time, 1)
    metrics["q_states_seen"] = agent.num_states_seen()

    tag = f"q_5x5" if grid_size == 5 else f"q_{grid_size}x{grid_size}"
    os.makedirs(MODELS_DIR, exist_ok=True)
    agent.save(
        os.path.join(MODELS_DIR, f"{tag}.pkl"),
        metadata={"grid": f"{grid_size}x{grid_size}", "scenario": "fixed", "hyperparameters": q_hp},
    )
    save_csv(
        os.path.join(RESULTS_DIR, f"training_{tag}.csv"),
        [{k: (int(v) if isinstance(v, bool) else v) for k, v in h.items()} for h in history],
    )
    plot_training(history, f"Q-learning training ({grid_size}x{grid_size})", os.path.join(PLOTS_DIR, f"training_{tag}.png"))

    print(f"  train: {train_time:.1f}s | eval: success={metrics['success_rate']:.2f} "
          f"reward={metrics['avg_reward']:.1f} steps={metrics['avg_steps']:.1f} "
          f"collisions={metrics['avg_collisions']:.2f} | states={metrics['q_states_seen']}")
    return metrics


def run_single_agent_dqn(grid_size: int, episodes: int, eval_episodes: int) -> dict:
    """Phases 9-11: train + evaluate DQN with one agent."""
    print(f"\n=== DQN {grid_size}x{grid_size}, {episodes} episodes ===")
    env = make_env(grid_size)
    agent = DQNAgent(agent_id=0, config=DQNConfig(seed=0, **DQN_HP))

    t0 = time.time()
    history = train_dqn(
        env, agent, episodes=episodes, seed=0,
        eval_every=max(1, episodes // 10),
        reset_kwargs=TRAIN_RESET,
    )
    train_time = time.time() - t0

    metrics = evaluate_dqn(env, agent, episodes=eval_episodes, reset_kwargs=TRAIN_RESET)
    gen = evaluate_dqn(env, agent, episodes=eval_episodes, reset_kwargs=EVAL_RANDOM_RESET)
    metrics["random_layout_success_rate"] = gen["success_rate"]
    metrics["train_time_sec"] = round(train_time, 1)
    metrics["train_steps"] = agent.train_step_count

    tag = f"dqn_5x5" if grid_size == 5 else f"dqn_{grid_size}x{grid_size}"
    os.makedirs(MODELS_DIR, exist_ok=True)
    agent.save_model(
        os.path.join(MODELS_DIR, f"{tag}.pt"),
        metadata={"grid": f"{grid_size}x{grid_size}", "scenario": "fixed", "hyperparameters": DQN_HP},
    )
    save_csv(
        os.path.join(RESULTS_DIR, f"training_{tag}.csv"),
        [{k: (int(v) if isinstance(v, bool) else v) for k, v in h.items()} for h in history],
    )
    plot_training(history, f"DQN training ({grid_size}x{grid_size})", os.path.join(PLOTS_DIR, f"training_{tag}.png"))

    print(f"  train: {train_time:.1f}s | eval: success={metrics['success_rate']:.2f} "
          f"reward={metrics['avg_reward']:.1f} steps={metrics['avg_steps']:.1f} "
          f"collisions={metrics['avg_collisions']:.2f} | grad steps={metrics['train_steps']}")
    return metrics


def run_comparison_experiment(eval_episodes: int) -> dict:
    """Phase 11: Random vs Q-learning vs DQN on identical environments and seeds."""
    print("\n=== Comparison: Random vs Q-learning vs DQN (10x10, identical seeds) ===")
    env = make_env(10)

    q_agent = QLearningAgent.load(os.path.join(MODELS_DIR, "q_10x10.pkl"))
    dqn_agent = DQNAgent(agent_id=0, config=DQNConfig(seed=0, **DQN_HP))
    dqn_agent.load_model(os.path.join(MODELS_DIR, "dqn_10x10.pt"))

    # Both condition sets are identical for all three algorithms.
    conditions = {
        "fixed_scenario": TRAIN_RESET,
        "random_layouts": EVAL_RANDOM_RESET,
    }
    results: dict = {}
    for cond_name, reset_kwargs in conditions.items():
        results[cond_name] = {
            "random": evaluate_random(env, episodes=eval_episodes, reset_kwargs=reset_kwargs),
            "q_learning": evaluate_q_learning(env, q_agent, episodes=eval_episodes, reset_kwargs=reset_kwargs),
            "dqn": evaluate_dqn(env, dqn_agent, episodes=eval_episodes, reset_kwargs=reset_kwargs),
        }
        print(f"  [{cond_name}]")
        for name, m in results[cond_name].items():
            print(f"    {name:12s} success={m['success_rate']:.2f} reward={m['avg_reward']:8.1f} "
                  f"steps={m['avg_steps']:6.1f} collisions={m['avg_collisions']:.2f}")

    save_json(os.path.join(RESULTS_DIR, "comparison_random_q_dqn.json"), results)
    plot_comparison(
        results["random_layouts"],
        "Random vs Q-learning vs DQN (10x10 random layouts, identical seeds)",
        os.path.join(PLOTS_DIR, "comparison_random_q_dqn.png"),
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="GridMind RL experiments (Developer 2)")
    parser.add_argument("--quick", action="store_true", help="reduced episode counts")
    parser.add_argument("--suite", choices=["all", "single", "comparison"], default="all")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if args.quick:
        plan = {
            "q_5x5_eps": 300, "q_10x10_eps": 3000, "dqn_5x5_eps": 400,
            "dqn_10x10_eps": 600, "eval_episodes": 30,
        }
    else:
        plan = {
            "q_5x5_eps": 500, "q_10x10_eps": 6000, "dqn_5x5_eps": 300,
            "dqn_10x10_eps": 1200, "eval_episodes": 100,
        }

    if args.suite in ("all", "single"):
        run_single_agent_q(5, plan["q_5x5_eps"], plan["eval_episodes"])
        run_single_agent_dqn(5, plan["dqn_5x5_eps"], plan["eval_episodes"])
        run_single_agent_q(10, plan["q_10x10_eps"], plan["eval_episodes"])
        run_single_agent_dqn(10, plan["dqn_10x10_eps"], plan["eval_episodes"])

    if args.suite in ("all", "comparison"):
        run_comparison_experiment(plan["eval_episodes"])

    print("\nDone. Models -> models/ | Results -> experiments/results/ | Plots -> experiments/plots/")


if __name__ == "__main__":
    main()
