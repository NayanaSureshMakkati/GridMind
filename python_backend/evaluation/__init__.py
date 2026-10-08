"""Evaluation, metrics, experiments, comparisons and plots (Developer 4).

* :mod:`python_backend.evaluation.scenarios`   — frozen layout/suite files (shared with Unity).
* :mod:`python_backend.evaluation.metrics`     — metric definitions + recording.
* :mod:`python_backend.evaluation.experiments` — training/evaluation drivers, scalability, ablation.
* :mod:`python_backend.evaluation.comparison`  — tables, fairness guard, JSON/CSV output.
* :mod:`python_backend.evaluation.plots`       — figures for ``experiments/plots/``.
* :mod:`python_backend.evaluation.model_store` — checkpoints + reproducibility manifests.

Protocol for experiments: ``docs/experiment_protocol.md``.
"""

from python_backend.evaluation.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    MetricsCollector,
    PositionRecorder,
    run_single_agent_episode,
)
from python_backend.evaluation.scenarios import (
    DEFAULT_SUITE,
    DEMO_SCENARIO,
    EvaluationSuite,
    RandomSuite,
    Scenario,
    load_scenario,
    load_suite,
)

__all__ = [
    "AggregateMetrics",
    "EpisodeMetrics",
    "MetricsCollector",
    "PositionRecorder",
    "run_single_agent_episode",
    "DEFAULT_SUITE",
    "DEMO_SCENARIO",
    "EvaluationSuite",
    "RandomSuite",
    "Scenario",
    "load_scenario",
    "load_suite",
]
