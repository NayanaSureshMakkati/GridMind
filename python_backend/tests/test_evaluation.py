"""Evaluation tests: metrics, scenarios, experiments, comparison, plots.

Everything asserted here is measured from real (tiny) episodes. The experiment
tests deliberately use a 2-3 episode budget: they verify WIRING and metric
definitions, not learning quality (no test asserts a success rate).
"""

import os

import pytest

from python_backend.environment.constants import Position
from python_backend.evaluation import comparison, plots
from python_backend.evaluation.experiments import (
    COORDINATED_DQN,
    DQN,
    FIXED_CONDITION,
    INDEPENDENT_DQN,
    Q_LEARNING,
    RANDOM,
    RANDOM_CONDITION,
    RULE_BASED,
    ExperimentConfig,
    build_agent_dqn_config,
    evaluate_method,
    method_family,
    run_comparison,
    train_method,
)
from python_backend.evaluation.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    MetricsCollector,
    PositionRecorder,
    run_single_agent_episode,
)
from python_backend.evaluation import model_store, scenarios
from python_backend.evaluation.scenarios import Scenario, load_scenario, load_suite
from python_backend.environment.grid_world import GridWorld


def tiny_scenario() -> Scenario:
    """Deterministic 4x4 scenario with a clear path for agent 0."""
    return Scenario(
        name="test_4x4",
        height=4,
        width=4,
        max_steps=10,
        obstacles=(Position(2, 0),),
        agents=((Position(0, 0), Position(3, 3)),),
        seed=1,
    )


def quick_config(methods, scenario_name="demo_10x10_3agents", num_agents=2, **overrides):
    """Experiment config with a very small budget (wiring tests only)."""
    return ExperimentConfig(
        suite=load_suite(),
        scenario_name=scenario_name,
        methods=tuple(methods),
        num_agents=num_agents,
        train_episodes=overrides.pop("train_episodes", 2),
        eval_episodes=overrides.pop("eval_episodes", 2),
        fixed_eval_episodes=overrides.pop("fixed_eval_episodes", 1),
        seed=overrides.pop("seed", 3),
        save_models=overrides.pop("save_models", False),
        experiment_name=overrides.pop("experiment_name", "pytest_evaluation"),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_load_demo_scenario(self):
        scenario = load_scenario("demo_10x10_3agents")
        assert (scenario.height, scenario.width) == (10, 10)
        assert scenario.num_agents == 3
        assert scenario.agents[0] == (Position(0, 0), Position(9, 9))
        assert Position(4, 1) in scenario.obstacles

    def test_load_suite_has_fixed_and_random_blocks(self):
        suite = load_suite()
        assert suite.name == "eval_suite_v1"
        assert suite.fixed_scenarios
        assert suite.random is not None
        assert suite.random.seed_start == 1000
        assert suite.random.episodes >= 1

    def test_with_num_agents_uses_prefix_and_keeps_layout(self):
        scenario = load_scenario("demo_10x10_3agents")
        two = scenario.with_num_agents(2)
        assert two.num_agents == 2
        assert two.obstacles == scenario.obstacles
        assert two.agents == scenario.agents[:2]
        with pytest.raises(ValueError):
            scenario.with_num_agents(4)

    def test_build_env_matches_scenario(self):
        scenario = load_scenario("demo_10x10_3agents")
        env = scenario.build_env()
        assert len(env.agents) == 3
        assert env.max_steps == scenario.max_steps
        assert env.obstacles == set(scenario.obstacles)
        assert env.get_agent_position(1) == Position(9, 0)

    def test_to_message_uses_named_obstacles_and_position_arrays(self):
        message = load_scenario("demo_10x10_3agents").to_message(2)
        assert set(message["obstacles"][0]) == {"row", "col"}
        assert message["agents"][0] == {"id": 0, "start": [0, 0], "goal": [9, 9]}
        assert len(message["agents"]) == 2

    def test_random_reset_kwargs_are_seeded_per_episode(self):
        block = load_suite().random
        first = block.reset_kwargs(0, num_agents=3)
        second = block.reset_kwargs(1, num_agents=3)
        assert first["seed"] == block.seed_start
        assert second["seed"] == block.seed_start + 1
        assert first["randomize"] is True and first["num_agents"] == 3

    def test_scenario_validation_rejects_missing_keys(self):
        with pytest.raises(ValueError):
            scenarios.scenario_from_dict({"name": "broken"})

    def test_scenario_validation_rejects_obstacle_on_start(self):
        with pytest.raises(ValueError):
            scenarios.scenario_from_dict({
                "name": "clash", "height": 3, "width": 3,
                "obstacles": [[0, 0]], "agents": [{"id": 0, "start": [0, 0], "goal": [2, 2]}],
            })

    def test_load_missing_scenario_raises(self):
        with pytest.raises(FileNotFoundError):
            load_scenario("does_not_exist")

    def test_manifests_record_layout_and_suite(self):
        scenario = load_scenario("demo_10x10_3agents")
        manifest = scenarios.scenario_manifest(scenario)
        assert manifest["grid"] == "10x10" and manifest["num_agents"] == 3
        suite_manifest = scenarios.suite_manifest(load_suite())
        assert suite_manifest["suite"] == "eval_suite_v1"
        assert suite_manifest["random"]["episodes"] >= 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_single_agent_episode_records_distance_and_reward(self):
        scenario = tiny_scenario()
        env = scenario.build_env()

        def policy(observation, valid_actions, greedy=True):
            """Greedy hand-written route: prefer DOWN, else RIGHT (avoids the obstacle)."""
            for action in (1, 3):  # DOWN, RIGHT
                if action in valid_actions:
                    return action
            return 4  # STAY

        metrics = run_single_agent_episode(env, 0, policy, episode=0)
        # Route: (0,0)->(1,0)->(1,1)->(2,1)->(3,1)->(3,2)->(3,3): 6 cells, 6 steps.
        assert metrics.steps == 6
        assert metrics.distance == 6
        assert metrics.success is True
        assert metrics.terminated is True and metrics.truncated is False
        assert metrics.goals_reached == 1
        assert metrics.reward > 0  # ends on the +100 goal reward

    def test_single_agent_episode_reports_truncation(self):
        scenario = Scenario(name="t", height=5, width=5, max_steps=2,
                            obstacles=(), agents=((Position(0, 0), Position(4, 4)),))

        def policy(observation, valid_actions, greedy=True):
            return 4  # STAY forever

        metrics = run_single_agent_episode(scenario.build_env(), 0, policy)
        assert metrics.steps == 2
        assert metrics.truncated is True
        assert metrics.success is False
        assert metrics.distance == 0

    def test_episode_metrics_to_dict_is_json_ready(self):
        import json

        metrics = EpisodeMetrics(episode=1, steps=3, reward=-4.0, success=True,
                                 per_agent_reward={0: -4.0}, collision_types={"SWAP": 1})
        payload = json.loads(json.dumps(metrics.to_dict()))
        assert payload["per_agent_reward"] == {"0": -4.0}
        assert payload["collision_types"] == {"SWAP": 1}

    def test_collector_aggregates_means_and_std(self):
        collector = MetricsCollector(label="unit", num_agents=1)
        collector.add(EpisodeMetrics(episode=0, steps=10, reward=100.0, success=True, collisions=0))
        collector.add(EpisodeMetrics(episode=1, steps=20, reward=0.0, success=False, collisions=2))
        summary = collector.aggregate()
        assert summary.episodes == 2
        assert summary.success_rate == pytest.approx(0.5)
        assert summary.avg_reward == pytest.approx(50.0)
        assert summary.avg_steps == pytest.approx(15.0)
        assert summary.avg_collisions == pytest.approx(1.0)
        assert summary.success_rate_std > 0

    def test_collector_aggregate_of_nothing_is_zeroed(self):
        summary = MetricsCollector(label="empty").aggregate()
        assert summary.episodes == 0 and summary.success_rate == 0.0

    def test_training_timer_records_seconds(self):
        collector = MetricsCollector(label="timer")
        with collector.measure_training():
            pass
        assert collector.aggregate().training_seconds >= 0.0

    def test_position_recorder_measures_manhattan_path(self):
        scenario = tiny_scenario()
        env = PositionRecorder(scenario.build_env())
        env.reset()
        env.step({0: 1})   # DOWN
        env.step({0: 3})   # RIGHT
        assert env.distances() == {0: 2}
        assert env.current_step == 2  # delegation still works

    def test_goal_completion_rate_reflects_agents(self):
        metrics = EpisodeMetrics(num_agents=3, goals_reached=1)
        assert metrics.goal_completion_rate == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class TestExperiments:
    def test_method_family_classification(self):
        assert method_family(RANDOM) == "single_agent"
        assert method_family(DQN) == "single_agent"
        assert method_family(INDEPENDENT_DQN) == "multi_agent"
        with pytest.raises(ValueError):
            method_family("nope")

    def test_epsilon_schedule_reaches_floor_within_budget(self):
        config = build_agent_dqn_config(seed=0, episodes=50)
        assert config.epsilon_decay < 1.0
        assert config.epsilon * (config.epsilon_decay ** 50) == pytest.approx(0.05, abs=0.01)

    @pytest.mark.parametrize("method", [RANDOM, Q_LEARNING, DQN])
    def test_single_agent_methods_train_and_evaluate(self, method):
        config = quick_config((method,))
        trained = train_method(config, method)
        assert trained.family == "single_agent"
        assert trained.num_agents == 1

        collectors = evaluate_method(config, trained)
        for condition in (FIXED_CONDITION, RANDOM_CONDITION):
            summary = collectors[condition].aggregate()
            assert summary.episodes >= 1
            assert 0.0 <= summary.success_rate <= 1.0
            assert summary.avg_steps >= 0.0
            assert summary.conditions["method"] == method
            assert summary.conditions["condition"] == condition

    @pytest.mark.parametrize("method", [INDEPENDENT_DQN, RULE_BASED, COORDINATED_DQN])
    def test_multi_agent_methods_train_and_evaluate(self, method):
        config = quick_config((method,), num_agents=2)
        trained = train_method(config, method)
        assert trained.family == "multi_agent"
        assert trained.num_agents == 2
        assert trained.history, "training history must be recorded for plots"

        collectors = evaluate_method(config, trained)
        fixed = collectors[FIXED_CONDITION].aggregate()
        assert fixed.num_agents == 2
        random_summary = collectors[RANDOM_CONDITION].aggregate()
        assert random_summary.episodes == config.resolved_eval_episodes()
        # Distance travelled is measured through the delegating recorder.
        assert random_summary.avg_distance >= 0.0

    def test_train_of_random_method_needs_no_training(self):
        config = quick_config((RANDOM,))
        trained = train_method(config, RANDOM)
        assert trained.training_seconds == 0.0
        assert trained.payload is None

    def test_run_comparison_produces_rows_and_manifest(self):
        config = quick_config((RANDOM, Q_LEARNING), train_episodes=1, eval_episodes=1)
        report = run_comparison(config)
        assert len(report.rows_for(FIXED_CONDITION)) == 2
        assert len(report.rows_for(RANDOM_CONDITION)) == 2
        manifest = report.manifest
        assert manifest["train_episodes"] == 1
        assert manifest["scenario"]["scenario"] == config.scenario_name
        assert manifest["suite"]["suite"] == config.suite.name

    def test_conditions_are_identical_across_methods(self):
        """Fairness (§43): same episodes/seeds/scenario for every method."""
        config = quick_config((RANDOM, Q_LEARNING), train_episodes=1, eval_episodes=1)
        report = run_comparison(config)
        for condition in (FIXED_CONDITION, RANDOM_CONDITION):
            keys = {"episodes", "scenario", "layout_seed_start", "num_obstacles"}
            values = [
                {k: row.conditions.get(k) for k in keys}
                for row in report.rows_for(condition)
            ]
            assert all(value == values[0] for value in values)

    def test_unknown_method_rejected(self):
        config = quick_config(("does_not_exist",))
        with pytest.raises(ValueError):
            train_method(config, "does_not_exist")


# ---------------------------------------------------------------------------
# Comparison, persistence and plots
# ---------------------------------------------------------------------------


def row(label, **overrides) -> AggregateMetrics:
    base = dict(label=label, episodes=2, num_agents=1, success_rate=0.5,
                avg_reward=10.0, avg_steps=5.0, avg_collisions=1.0,
                conditions={"condition": "random", "episodes": 2, "scenario": "s"})
    base.update(overrides)
    return AggregateMetrics(**base)


class TestComparison:
    def test_format_table_contains_labels_and_metric_values(self):
        table = comparison.format_table([row("a"), row("b", success_rate=1.0)])
        assert "method" in table and "a" in table and "b" in table
        assert "1.00" in table

    def test_assert_fair_accepts_matching_conditions(self):
        comparison.assert_fair([row("a"), row("b")])

    def test_assert_fair_rejects_different_conditions(self):
        unfair = row("b")
        unfair.conditions = {"condition": "random", "episodes": 5, "scenario": "s"}
        with pytest.raises(ValueError):
            comparison.assert_fair([row("a"), unfair])

    def test_delta_vs_baseline_computes_absolute_and_relative_change(self):
        rows = [row("base", avg_reward=10.0), row("better", avg_reward=15.0)]
        deltas = comparison.delta_vs_baseline(rows, "base", metrics=("avg_reward",))
        assert deltas[1]["deltas"]["avg_reward"]["absolute"] == pytest.approx(5.0)
        assert deltas[1]["deltas"]["avg_reward"]["relative_percent"] == pytest.approx(50.0)

    def test_delta_against_zero_baseline_omits_relative_change(self):
        rows = [row("base", avg_reward=0.0), row("other", avg_reward=5.0)]
        deltas = comparison.delta_vs_baseline(rows, "base", metrics=("avg_reward",))
        assert "relative_percent" not in deltas[1]["deltas"]["avg_reward"]

    def test_delta_requires_existing_baseline(self):
        with pytest.raises(ValueError):
            comparison.delta_vs_baseline([row("a")], "missing")

    def test_save_rows_and_episodes_csv(self, tmp_path):
        rows = [row("a"), row("b")]
        csv_path = comparison.save_rows_csv(rows, str(tmp_path / "rows.csv"))
        assert os.path.exists(csv_path)
        assert "avg_reward" in open(csv_path, encoding="utf-8").read()

        collector = MetricsCollector(label="a")
        collector.add(EpisodeMetrics(episode=0, steps=3, reward=1.0, success=True))
        episodes_path = comparison.save_episodes_csv({"a@random": collector},
                                                     str(tmp_path / "episodes.csv"))
        assert "collector" in open(episodes_path, encoding="utf-8").read()

    def test_save_history_csv_and_report(self, tmp_path):
        history_path = comparison.save_history_csv(
            {"dqn": [{"episode": 0, "reward": -5.0, "steps": 3, "success": False}]},
            str(tmp_path / "history.csv"),
        )
        assert "dqn" in open(history_path, encoding="utf-8").read()

        config = quick_config((RANDOM,), train_episodes=1, eval_episodes=1)
        report = run_comparison(config)
        paths = comparison.save_report(report, name="pytest_evaluation")
        try:
            assert os.path.exists(paths["json"]) and os.path.exists(paths["csv"])
        finally:
            for path in paths.values():
                if os.path.exists(path):
                    os.remove(path)


class TestPlots:
    def test_training_curve_and_bar_plots_are_written(self, tmp_path):
        path = plots.plot_training_curves(
            {"dqn": [{"episode": i, "reward": -i} for i in range(5)]},
            metric="reward", directory=str(tmp_path),
        )
        assert os.path.exists(path)

        bar = plots.plot_metric_bars([row("a"), row("b")], "avg_reward", "title",
                                     "bars.png", directory=str(tmp_path))
        assert os.path.exists(bar)

    def test_plot_report_writes_multiple_figures(self, tmp_path):
        config = quick_config((RANDOM, Q_LEARNING), train_episodes=1, eval_episodes=1)
        report = run_comparison(config)
        written = plots.plot_report(report, conditions=(RANDOM_CONDITION,), directory=str(tmp_path))
        assert len(written) >= 3
        assert all(os.path.exists(path) for path in written)

    def test_plot_of_empty_history_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            plots.plot_training_curves({}, directory=str(tmp_path))

    def test_scalability_and_ablation_plots_accept_grouped_rows(self, tmp_path):
        grouped = {1: [row("a", num_agents=1)], 2: [row("a", num_agents=2)]}
        written = plots.plot_scalability(grouped, condition="random", directory=str(tmp_path))
        assert len(written) >= 1

        ablation = plots.plot_ablation([row("a"), row("b")], condition="random",
                                       directory=str(tmp_path))
        assert all(os.path.exists(path) for path in ablation)

    def test_live_session_plot(self, tmp_path):
        path = plots.plot_live_session(
            [{"episode": 1, "reward": -10.0}, {"episode": 2, "reward": 5.0}],
            directory=str(tmp_path),
        )
        assert os.path.exists(path)
        with pytest.raises(ValueError):
            plots.plot_live_session([], directory=str(tmp_path))


# ---------------------------------------------------------------------------
# Model store
# ---------------------------------------------------------------------------


class TestModelStore:
    def test_q_learning_round_trip(self, tmp_path):
        from python_backend.algorithms.q_learning import QLearningAgent, QLearningConfig

        agent = QLearningAgent(QLearningConfig(seed=1))
        path = model_store.save_q_learning(agent, str(tmp_path / "q_agent"),
                                           metadata={"scenario": "unit"})
        assert os.path.exists(path)
        loaded = model_store.load_q_learning(str(tmp_path / "q_agent"))
        assert isinstance(loaded, QLearningAgent)

    def test_dqn_round_trip_rebuilds_network_shape(self, tmp_path):
        from python_backend.algorithms.dqn import DQNAgent, DQNConfig

        agent = DQNAgent(agent_id=0, config=DQNConfig(seed=2, observation_size=12, hidden_size=16))
        path = model_store.save_dqn(agent, str(tmp_path / "dqn_agent"))
        loaded = model_store.load_dqn(str(tmp_path / "dqn_agent"), agent_id=0, seed=2)
        assert loaded.config.hidden_size == 16
        assert os.path.exists(path)

    def test_missing_checkpoint_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            model_store.load_dqn(str(tmp_path / "nope"))
        with pytest.raises(FileNotFoundError):
            model_store.load_q_learning(str(tmp_path / "nope"))

    def test_multi_agent_paths_report_missing_agents(self):
        with pytest.raises(FileNotFoundError):
            model_store.multi_agent_paths("definitely_missing_tag", num_agents=2)

    def test_multi_agent_save_and_load_round_trip(self):
        from python_backend.multi_agent.multi_agent_system import MultiAgentConfig, MultiAgentSystem

        scenario = load_scenario("demo_10x10_3agents").with_num_agents(2)
        tag = "pytest_multi_agent"
        system = MultiAgentSystem(
            scenario.build_env(),
            MultiAgentConfig(num_agents=2, dqn_config=build_agent_dqn_config(seed=4, episodes=2)),
        )
        paths = model_store.save_multi_agent(system, tag)
        try:
            assert len(paths) == 2 and all(os.path.exists(p) for p in paths)
            model_store.load_multi_agent(system, tag)
        finally:
            for path in paths:
                if os.path.exists(path):
                    os.remove(path)

    def test_manifest_written_to_experiment_results(self):
        path = model_store.write_manifest("pytest_manifest", {"seed": 1})
        try:
            assert model_store.read_manifest("pytest_manifest")["seed"] == 1
        finally:
            if os.path.exists(path):
                os.remove(path)
