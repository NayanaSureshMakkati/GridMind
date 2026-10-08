"""Plots for GridMind experiments (Developer 4, Phase 11).

All figures are written to ``experiments/plots/`` as PNG files. Every plot is
built from measured data only (episode histories, aggregate rows); nothing is
smoothed away without saying so ("rolling mean over N episodes" is in the axis
label and the filename).

matplotlib is used with the non-interactive ``Agg`` backend, so plotting never
requires a display and never blocks a headless experiment run.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from python_backend.evaluation.metrics import AggregateMetrics
from python_backend.evaluation.model_store import ROOT

PLOTS_DIR = os.path.join(ROOT, "experiments", "plots")

ROLLING_WINDOW = 20


def _pyplot():
    """Imports matplotlib with a non-interactive backend (headless-safe)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plots_dir(directory: Optional[str] = None) -> str:
    path = directory or PLOTS_DIR
    os.makedirs(path, exist_ok=True)
    return path


def _rolling(values: Sequence[float], window: int = ROLLING_WINDOW) -> List[float]:
    """Centered-free trailing rolling mean (documented in the axis label)."""
    window = max(1, min(window, len(values)))
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


# ---------------------------------------------------------------------------
# Training curves (reward / success / collisions vs episode)
# ---------------------------------------------------------------------------


def plot_training_curves(
    histories: Dict[str, List[Dict[str, Any]]],
    metric: str = "reward",
    filename: Optional[str] = None,
    window: int = ROLLING_WINDOW,
    directory: Optional[str] = None,
) -> str:
    """Plots one training metric against episode for every method.

    ``metric`` must be a key of the training-history records (``reward``,
    ``success``, ``collisions``, ``deadlocks``, ``steps``, ``epsilon``).
    """
    if not histories:
        raise ValueError("No training histories to plot.")
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, 5))
    for method, history in histories.items():
        series = [float(record.get(metric, 0.0)) for record in history]
        if not series:
            continue
        axis.plot(range(1, len(series) + 1), series, alpha=0.25, linewidth=1)
        axis.plot(
            range(1, len(series) + 1),
            _rolling(series, window),
            linewidth=2,
            label=f"{method} (rolling mean {min(window, len(series))})",
        )
    axis.set_xlabel("training episode")
    axis.set_ylabel(metric)
    axis.set_title(f"Training {metric} vs episode (faint = raw, bold = rolling mean)")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = os.path.join(_plots_dir(directory), filename or f"training_{metric}_vs_episode.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_metric_bars(
    rows: Sequence[AggregateMetrics],
    metric: str,
    title: str,
    filename: str,
    ylabel: Optional[str] = None,
    directory: Optional[str] = None,
) -> str:
    """Bar chart of one aggregate metric across methods."""
    if not rows:
        raise ValueError(f"No rows to plot for '{metric}'.")
    plt = _pyplot()
    labels = [r.label for r in rows]
    values = [float(getattr(r, metric)) for r in rows]

    figure, axis = plt.subplots(figsize=(max(7, 1.6 * len(labels)), 5))
    axis.bar(range(len(labels)), values, color="#4c72b0")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axis.set_ylabel(ylabel or metric)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.3)
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    path = os.path.join(_plots_dir(directory), filename)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_algorithm_comparison(rows: Sequence[AggregateMetrics], condition: str = "random",
                              directory: Optional[str] = None) -> List[str]:
    """Phase 11 'algorithm comparison': success rate, reward and collisions."""
    return [
        plot_metric_bars(
            rows, "success_rate",
            f"Success rate by algorithm ({condition} condition)",
            f"comparison_success_rate_{condition}.png",
            ylabel="success rate (0-1)", directory=directory,
        ),
        plot_metric_bars(
            rows, "avg_reward",
            f"Average cumulative reward by algorithm ({condition} condition)",
            f"comparison_avg_reward_{condition}.png",
            ylabel="true environment reward", directory=directory,
        ),
        plot_metric_bars(
            rows, "avg_steps",
            f"Average episode length by algorithm ({condition} condition)",
            f"comparison_avg_steps_{condition}.png",
            ylabel="steps per episode", directory=directory,
        ),
        plot_metric_bars(
            rows, "avg_collisions",
            f"Collisions per episode by algorithm ({condition} condition)",
            f"comparison_collisions_{condition}.png",
            ylabel="collisions per episode", directory=directory,
        ),
    ]


def plot_ablation(rows: Sequence[AggregateMetrics], condition: str = "random",
                  directory: Optional[str] = None) -> List[str]:
    """Phase 12 ablation figures: deadlocks, collisions, reward."""
    return [
        plot_metric_bars(
            rows, "avg_deadlocks",
            f"Deadlock events per episode ({condition} condition)",
            f"ablation_deadlocks_{condition}.png",
            ylabel="deadlocks per episode", directory=directory,
        ),
        plot_metric_bars(
            rows, "avg_collisions",
            f"Collision events per episode ({condition} condition)",
            f"ablation_collisions_{condition}.png",
            ylabel="collisions per episode", directory=directory,
        ),
        plot_metric_bars(
            rows, "avg_reward",
            f"Average reward per episode ({condition} condition)",
            f"ablation_reward_{condition}.png",
            ylabel="true environment reward", directory=directory,
        ),
    ]


def plot_scalability(
    grouped_rows: Dict[int, List[AggregateMetrics]],
    metrics: Sequence[str] = ("success_rate", "avg_reward", "avg_collisions", "avg_deadlocks"),
    condition: str = "random",
    directory: Optional[str] = None,
) -> List[str]:
    """Phase 13 scalability figures: metric vs number of agents."""
    plt = _pyplot()
    paths: List[str] = []
    counts = sorted(grouped_rows.keys())

    for metric in metrics:
        # Each label keeps its own agent counts: N=1 runs through the
        # single-agent DQN path, so it carries a different label from the
        # multi-agent runs and must not be plotted against the full count axis.
        values: Dict[str, Dict[int, float]] = {}
        for count in counts:
            for row in grouped_rows[count]:
                if row.conditions.get("condition") != condition:
                    continue
                values.setdefault(row.label, {})[count] = float(getattr(row, metric))
        if not values:
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        for label, points in values.items():
            series_counts = sorted(points)
            axis.plot(series_counts, [points[c] for c in series_counts],
                      marker="o", label=f"{label} ({condition})")
        axis.set_xlabel("number of agents")
        axis.set_ylabel(metric)
        axis.set_xticks(counts)
        axis.set_title(f"Scalability: {metric} vs number of agents")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        path = os.path.join(_plots_dir(directory), f"scalability_{metric}.png")
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_communication_overhead(rows: Sequence[AggregateMetrics], condition: str = "random",
                                directory: Optional[str] = None) -> str:
    """Phase 11 communication overhead: messages per episode per method."""
    return plot_metric_bars(
        rows, "avg_messages",
        f"Communication overhead ({condition} condition; 0 = communication disabled)",
        f"communication_overhead_{condition}.png",
        ylabel="messages per episode", directory=directory,
    )


def plot_report(report, conditions: Sequence[str] = ("fixed", "random"),
                directory: Optional[str] = None) -> List[str]:
    """Generates every applicable figure for an experiment report."""
    written: List[str] = []
    for metric in ("reward", "success", "collisions"):
        if any(report.history.get(method) for method in report.history):
            written.append(plot_training_curves(report.history, metric, directory=directory))
    for condition in conditions:
        rows = report.rows_for(condition)
        if not rows:
            continue
        written.extend(plot_algorithm_comparison(rows, condition, directory=directory))
        written.append(plot_communication_overhead(rows, condition, directory=directory))
        written.append(
            plot_metric_bars(
                rows, "avg_deadlocks",
                f"Deadlock events per episode ({condition} condition)",
                f"deadlocks_{condition}.png", ylabel="deadlocks per episode",
                directory=directory,
            )
        )
    return written


def plot_scalability_report(report, condition: str = "random",
                            directory: Optional[str] = None) -> List[str]:
    """Scalability figures for a scalability report."""
    from python_backend.evaluation.comparison import scalability_rows

    grouped = scalability_rows(report)
    if not grouped:
        return []
    return plot_scalability(grouped, condition=condition, directory=directory)


def plot_ablation_report(report, condition: str = "random",
                         directory: Optional[str] = None) -> List[str]:
    """Ablation figures for an ablation report."""
    rows = report.rows_for(condition)
    if not rows:
        return []
    return plot_ablation(rows, condition=condition, directory=directory)


def plot_live_session(history: List[Dict[str, Any]], filename: str = "live_demo_rewards.png",
                      directory: Optional[str] = None) -> str:
    """Reward-per-episode figure for a live (Unity) demonstration session.

    The session history is produced by
    ``integration.session.LiveSession.history``.
    """
    if not history:
        raise ValueError("The live session recorded no completed episode to plot.")
    plt = _pyplot()
    episodes = [int(record.get("episode", i + 1)) for i, record in enumerate(history)]
    rewards = [float(record.get("reward", 0.0)) for record in history]

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(episodes, rewards, marker="o")
    axis.set_xlabel("episode")
    axis.set_ylabel("total true reward")
    axis.set_title("Live demonstration: reward per episode")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = os.path.join(_plots_dir(directory), filename)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
