"""Comparison, tables and persistence for GridMind experiments (Developer 4).

Consumes :class:`~python_backend.evaluation.experiments.ExperimentReport`
objects and produces:

  * console tables (one row per method, one column per metric),
  * machine-readable results (``experiments/results/*.json`` and ``*.csv``),
  * explicit deltas against a baseline row.

Fairness guard (§43): :func:`assert_fair` refuses to compare rows whose
conditions differ (different episode counts, layout seeds or scenarios), so an
invalid comparison cannot silently end up in the report.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from python_backend.evaluation.metrics import AggregateMetrics, MetricsCollector
from python_backend.evaluation.model_store import results_dir

#: (header, attribute) columns used by every comparison table.
DEFAULT_COLUMNS: Sequence[tuple] = (
    ("method", "label"),
    ("agents", "num_agents"),
    ("eps", "episodes"),
    ("success", "success_rate"),
    ("goal%", "goal_completion_rate"),
    ("reward", "avg_reward"),
    ("steps", "avg_steps"),
    ("collisions", "avg_collisions"),
    ("coll/step", "avg_collision_rate"),
    ("deadlocks", "avg_deadlocks"),
    ("distance", "avg_distance"),
    ("messages", "avg_messages"),
    ("train_s", "training_seconds"),
)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if value == int(value) and abs(value) >= 100:
            return f"{value:.0f}"
        return f"{value:.2f}"
    return str(value)


def format_table(rows: Sequence[AggregateMetrics], columns=DEFAULT_COLUMNS) -> str:
    """Renders rows as a fixed-width ASCII table."""
    if not rows:
        return "(no rows)"
    header = [c[0] for c in columns]
    cells = [
        [_format_value(getattr(row, attribute)) for _, attribute in columns]
        for row in rows
    ]
    widths = [
        max(len(header[i]), *(len(cell[i]) for cell in cells))
        for i in range(len(header))
    ]
    lines = ["  ".join(header[i].ljust(widths[i]) for i in range(len(header)))]
    lines.append("  ".join("-" * widths[i] for i in range(len(header))))
    for cell in cells:
        lines.append("  ".join(cell[i].ljust(widths[i]) for i in range(len(header))))
    return "\n".join(lines)


def assert_fair(rows: Sequence[AggregateMetrics]) -> None:
    """Raises ValueError if rows to be compared do not share conditions.

    Checks the fields that must be identical for a valid comparison: the
    evaluation episode count, the layout seed sequence start, the obstacle
    count and the scenario identifier.
    """
    if len(rows) < 2:
        return
    keys = ("condition", "episodes", "layout_seed_start", "num_obstacles", "scenario", "train_episodes")
    first = rows[0]
    for row in rows[1:]:
        for key in keys:
            a = first.conditions.get(key)
            b = row.conditions.get(key)
            if a != b:
                raise ValueError(
                    f"Unfair comparison: '{first.label}' has {key}={a!r} but "
                    f"'{row.label}' has {key}={b!r}. Both rows must be produced "
                    "under identical conditions (Master Context 43)."
                )


def table_for(report, condition: str, fair: bool = True) -> str:
    """Comparison table for one condition of a report."""
    rows = report.rows_for(condition)
    if fair:
        assert_fair(rows)
    return f"--- condition: {condition} ---\n{format_table(rows)}"


def delta_vs_baseline(
    rows: Sequence[AggregateMetrics],
    baseline_label: str,
    metrics: Sequence[str] = ("success_rate", "avg_reward", "avg_steps", "avg_collisions", "avg_deadlocks"),
) -> List[Dict[str, Any]]:
    """Absolute and relative change of every row against a baseline row.

    Returns one entry per row: ``{"label", "baseline", "deltas": {metric:
    {"absolute": x, "relative_percent": y}}}``. Relative change is omitted when
    the baseline value is 0 (division by zero is never hidden as 0%).
    """
    baseline = next((r for r in rows if r.label == baseline_label), None)
    if baseline is None:
        available = [r.label for r in rows]
        raise ValueError(f"Baseline '{baseline_label}' not found. Available: {available}")

    output: List[Dict[str, Any]] = []
    for row in rows:
        entry: Dict[str, Any] = {
            "label": row.label,
            "baseline": baseline.label,
            "deltas": {},
        }
        for metric in metrics:
            base_value = float(getattr(baseline, metric))
            value = float(getattr(row, metric))
            delta: Dict[str, Any] = {"absolute": round(value - base_value, 4)}
            if base_value != 0:
                delta["relative_percent"] = round((value - base_value) / abs(base_value) * 100.0, 2)
            entry["deltas"][metric] = delta
        output.append(entry)
    return output


def scalability_rows(report) -> Dict[int, List[AggregateMetrics]]:
    """Groups scalability rows by agent count (ascending)."""
    grouped: Dict[int, List[AggregateMetrics]] = {}
    for row in report.rows:
        count = int(row.conditions.get("num_agents", row.num_agents))
        grouped.setdefault(count, []).append(row)
    return dict(sorted(grouped.items()))


def scalability_table(report) -> str:
    """Table with one row per (agent count, condition) pair."""
    grouped = scalability_rows(report)
    flat: List[AggregateMetrics] = [row for _, rows in grouped.items() for row in rows]
    return format_table(flat)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_report(report, name: Optional[str] = None) -> Dict[str, str]:
    """Writes the report as JSON (full detail) and CSV (one row per metric row).

    Returns the written paths keyed by format.
    """
    base = name or report.name
    directory = results_dir()
    json_path = os.path.join(directory, f"{base}.json")
    csv_path = os.path.join(directory, f"{base}.csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
    save_rows_csv(report.rows, csv_path)
    return {"json": json_path, "csv": csv_path}


def save_rows_csv(rows: Iterable[AggregateMetrics], path: str) -> str:
    """Writes aggregate rows to CSV (one row per method/condition)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "label", "episodes", "num_agents", "success_rate", "success_rate_std",
        "goal_completion_rate", "avg_reward", "reward_std", "avg_steps",
        "avg_collisions", "avg_collision_rate", "avg_deadlocks", "avg_distance",
        "avg_messages", "training_seconds", "evaluation_seconds",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["conditions"])
        writer.writeheader()
        for row in rows:
            payload = {field: getattr(row, field) for field in fields}
            payload["conditions"] = json.dumps(row.conditions, sort_keys=True)
            writer.writerow(payload)
    return path


def save_episodes_csv(collectors: Dict[str, MetricsCollector], path: str) -> str:
    """Writes per-episode metrics for every collector (one row per episode)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "collector", "episode", "steps", "reward", "success", "terminated",
        "truncated", "collisions", "deadlocks", "recoveries", "distance",
        "messages", "goals_reached", "goal_completion_rate", "num_agents",
        "layout_seed", "scenario",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, collector in collectors.items():
            for episode in collector.episodes:
                record = episode.to_dict()
                record["collector"] = key
                writer.writerow({field: record.get(field) for field in fields})
    return path


def save_history_csv(histories: Dict[str, List[Dict[str, Any]]], path: str) -> str:
    """Writes per-episode TRAINING history (reward vs episode curves)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows: List[Dict[str, Any]] = []
    fields = ["method", "episode", "reward", "steps", "success", "collisions", "deadlocks", "messages", "epsilon", "mean_loss"]
    for method, history in histories.items():
        for record in history:
            row = {"method": method}
            row.update({field: record.get(field) for field in fields if field != "method"})
            rows.append(row)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return path


def print_report(report, conditions: Sequence[str] = ("fixed", "random")) -> None:
    """Prints every condition table plus deltas against the first row."""
    print(f"\n=== {report.name} ===")
    print(f"scenario/suite: {report.manifest.get('scenario', {}).get('scenario', '?')} / "
          f"{report.manifest.get('suite', {}).get('suite', '?')}")
    print(f"train episodes: {report.manifest.get('train_episodes')} | "
          f"eval episodes: {report.manifest.get('eval_episodes')} | "
          f"seed: {report.manifest.get('seed')}")
    for condition in conditions:
        rows = report.rows_for(condition)
        if not rows:
            continue
        assert_fair(rows)
        print()
        print(table_for(report, condition, fair=False))
