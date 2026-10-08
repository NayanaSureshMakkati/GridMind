"""GridMind — single entry point (Developer 4, Phase 15).

Usage
-----
    python main.py --mode train     --algorithm qlearning
    python main.py --mode train     --algorithm dqn
    python main.py --mode train     --algorithm multiagent
    python main.py --mode evaluate  --algorithm multiagent
    python main.py --mode visualize --algorithm coordinated --agents 3
    python main.py --mode compare
    python main.py --mode ablation
    python main.py --mode scalability
    python main.py --mode demo          # Developer 1's deterministic random-policy run

Every number printed here comes from an executed episode or a measured training
run; nothing is pre-filled (Master Context §43/§75). Use ``--quick`` for fast
smoke runs and ``--no-plots`` to skip figure generation.

Roles: Python owns the logical simulation and the learning; ``--mode visualize``
streams that state to Unity, which only renders it.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from typing import Dict, List, Optional, Sequence

from python_backend.evaluation import comparison, plots
from python_backend.evaluation.experiments import (
    ALL_METHODS,
    COORDINATED_DQN,
    CTDE_INSPIRED,
    DQN,
    INDEPENDENT_DQN,
    Q_LEARNING,
    RANDOM,
    RULE_BASED,
    ExperimentConfig,
    build_agent_dqn_config,
    evaluate_method,
    method_family,
    multi_agent_config_kwargs,
    run_ablation,
    run_comparison,
    run_scalability,
)
from python_backend.evaluation.model_store import load_dqn, load_q_learning, load_multi_agent
from python_backend.evaluation.scenarios import DEMO_SCENARIO, list_scenarios, load_scenario, load_suite

#: CLI aliases -> canonical method names. "multiagent" means Independent DQN
#: (the multi-agent baseline), matching the Master Context's phase names.
ALGORITHM_ALIASES: Dict[str, str] = {
    "random": RANDOM,
    "qlearning": Q_LEARNING,
    "q_learning": Q_LEARNING,
    "q-learning": Q_LEARNING,
    "dqn": DQN,
    "multiagent": INDEPENDENT_DQN,
    "multi-agent": INDEPENDENT_DQN,
    "independent": INDEPENDENT_DQN,
    "independent_dqn": INDEPENDENT_DQN,
    "rule_based": RULE_BASED,
    "rule-based": RULE_BASED,
    "rule_based_coordination": RULE_BASED,
    "coordinated": COORDINATED_DQN,
    "coordinated_dqn": COORDINATED_DQN,
    "ctde": CTDE_INSPIRED,
    "ctde_inspired": CTDE_INSPIRED,
}

QUICK_TRAIN_EPISODES = 20
QUICK_EVAL_EPISODES = 3
DEFAULT_TRAIN_EPISODES = 400


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI definition for every GridMind mode."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="GridMind: multi-agent RL navigation with a Unity 3D frontend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "algorithms: random, qlearning, dqn, multiagent, rule_based, "
            "coordinated, ctde\n"
            "scenarios:  " + ", ".join(list_scenarios() or [DEMO_SCENARIO])
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate", "visualize", "compare", "ablation", "scalability", "demo"),
        default="demo",
        help="what to run (default: demo)",
    )
    parser.add_argument("--algorithm", default="multiagent",
                        help="algorithm to train/evaluate/visualize (see aliases below)")
    parser.add_argument("--methods", default=None,
                        help="comma-separated method list for --mode compare "
                             "(default: every method)")
    parser.add_argument("--scenario", default=DEMO_SCENARIO, help="frozen scenario file name")
    parser.add_argument("--suite", default=None, help="evaluation suite file name (default eval_suite_v1)")
    parser.add_argument("--agents", type=int, default=3, help="agents (prefix of the scenario)")
    parser.add_argument("--agent-counts", default="1,2,3,4,5",
                        help="comma-separated agent counts for --mode scalability")
    parser.add_argument("--train-episodes", type=int, default=None,
                        help=f"training episodes per method (default {DEFAULT_TRAIN_EPISODES})")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="episodes for the seeded random evaluation condition")
    parser.add_argument("--fixed-eval-episodes", type=int, default=1,
                        help="deterministic-scenario evaluation episodes (default 1)")
    parser.add_argument("--seed", type=int, default=7, help="training/policy seed")
    parser.add_argument("--quick", action="store_true", help="fast smoke budget (few episodes)")
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    parser.add_argument("--no-save-models", action="store_true", help="skip checkpoint writing")
    parser.add_argument("--model", default=None, help="checkpoint path for evaluate/visualize")
    parser.add_argument("--model-tag", default=None, help="multi-agent checkpoint tag")
    parser.add_argument("--port", type=int, default=8765, help="TCP port for --mode visualize")
    parser.add_argument("--host", default="127.0.0.1", help="bind address for --mode visualize")
    parser.add_argument("--speed-ms", type=int, default=250, help="milliseconds per step in Unity")
    parser.add_argument("--episodes", type=int, default=3,
                        help="episodes per session for visualize/demo")
    parser.add_argument("--action-source", choices=("policy", "external"), default="policy",
                        help="who supplies actions in visualize mode")
    parser.add_argument("--communication", action="store_true",
                        help="enable message counting in the session/experiment")
    parser.add_argument("--headless", action="store_true",
                        help="visualize without Unity: run the reference client against the server")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="stop a visualize session after N seconds")
    parser.add_argument("--event-log", default=None,
                        help="append every step (movements/collisions/goals) as JSON "
                             "lines to this file (e.g. experiments/logs/events.jsonl)")
    return parser


def resolve_algorithm(name: str) -> str:
    """Maps a CLI alias to a canonical method name."""
    key = name.strip().lower()
    if key not in ALGORITHM_ALIASES:
        raise SystemExit(
            f"Unknown algorithm '{name}'. Try one of: {', '.join(sorted(ALGORITHM_ALIASES))}"
        )
    return ALGORITHM_ALIASES[key]


def build_config(args: argparse.Namespace, methods: Sequence[str], name: str) -> ExperimentConfig:
    """Builds an ExperimentConfig from CLI arguments."""
    suite = load_suite(args.suite) if args.suite else load_suite()
    train_episodes = args.train_episodes
    if train_episodes is None:
        train_episodes = QUICK_TRAIN_EPISODES if args.quick else DEFAULT_TRAIN_EPISODES
    eval_episodes = args.eval_episodes
    if eval_episodes is None and args.quick:
        eval_episodes = QUICK_EVAL_EPISODES
    return ExperimentConfig(
        suite=suite,
        scenario_name=args.scenario,
        methods=tuple(methods),
        num_agents=args.agents,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        fixed_eval_episodes=args.fixed_eval_episodes,
        seed=args.seed,
        save_models=not args.no_save_models,
        experiment_name=name,
    )


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_train(args: argparse.Namespace) -> int:
    """Trains one algorithm, saves the checkpoint and reports measured metrics."""
    method = resolve_algorithm(args.algorithm)
    config = build_config(args, (method,), name=f"train_{method}")
    print(f"Training '{method}' on scenario '{args.scenario}' "
          f"({config.train_episodes} episodes, seed {args.seed})...")
    report = run_comparison(config, methods=(method,))
    comparison.print_report(report)
    paths = comparison.save_report(report)
    print(f"\nresults: {paths['json']}\n         {paths['csv']}")
    if report.manifest.get("models", {}).get(method):
        print(f"model:   {', '.join(report.manifest['models'][method])}")
    if not args.no_plots:
        written = plots.plot_report(report)
        print(f"plots:   {len(written)} file(s) in {plots.PLOTS_DIR}")
    return 0


def _load_trained(config: ExperimentConfig, method: str, args: argparse.Namespace):
    """Rebuilds a TrainedMethod from saved checkpoints (no retraining)."""
    from python_backend.evaluation.experiments import TrainedMethod

    family = method_family(method)
    scenario = config.scenario.with_num_agents(1 if family == "single_agent" else config.num_agents)

    if method == RANDOM:
        return TrainedMethod(
            method=method, label="Random baseline", num_agents=1, family=family,
            training_seconds=0.0, metadata={"seed": args.seed},
        )
    if method == Q_LEARNING:
        agent = load_q_learning(args.model or f"{config.scenario_name}_q_learning")
        return TrainedMethod(method=method, label="Tabular Q-learning", num_agents=1,
                             family=family, training_seconds=0.0, payload=agent)
    if method == DQN:
        agent = load_dqn(args.model or f"{config.scenario_name}_dqn", agent_id=0, seed=args.seed)
        return TrainedMethod(method=method, label="DQN (single agent)", num_agents=1,
                             family=family, training_seconds=0.0, payload=agent)

    from python_backend.multi_agent.multi_agent_system import MultiAgentConfig, MultiAgentSystem

    dqn_configs = {
        aid: build_agent_dqn_config(seed=args.seed + aid, episodes=config.train_episodes)
        for aid in range(config.num_agents)
    }
    system = MultiAgentSystem(
        scenario.build_env(),
        MultiAgentConfig(**multi_agent_config_kwargs(method, config.num_agents, dqn_configs, args.seed)),
    )
    tag = args.model_tag or f"{config.scenario_name}_{system.system_label()}_{config.num_agents}agents"
    load_multi_agent(system, tag)
    return TrainedMethod(method=method, label=method, num_agents=config.num_agents,
                         family=family, training_seconds=0.0, payload=system,
                         metadata={"system_label": system.system_label(), "model_tag": tag})


def mode_evaluate(args: argparse.Namespace) -> int:
    """Evaluates saved checkpoints on the frozen suite (no training)."""
    method = resolve_algorithm(args.algorithm)
    config = build_config(args, (method,), name=f"evaluate_{method}")
    print(f"Evaluating '{method}' on suite '{config.suite.name}' "
          f"(fixed + {config.resolved_eval_episodes()} seeded random episodes)...")
    try:
        trained = _load_trained(config, method, args)
    except FileNotFoundError as exc:
        print(f"\n{exc}\nTrain first, e.g.: python main.py --mode train "
              f"--algorithm {args.algorithm}", file=sys.stderr)
        return 2

    from python_backend.evaluation.experiments import ExperimentReport
    from python_backend.evaluation.model_store import write_manifest

    report = ExperimentReport(name=config.experiment_name, manifest=config.manifest())
    collectors = evaluate_method(config, trained)
    for condition, collector in collectors.items():
        report.collectors[f"{method}@{condition}"] = collector
        report.rows.append(collector.aggregate())
    report.manifest["loaded_checkpoint"] = args.model or args.model_tag or "default tag"
    comparison.print_report(report)
    paths = comparison.save_report(report)
    write_manifest(config.experiment_name, report.manifest)
    print(f"\nresults: {paths['json']}\n         {paths['csv']}")
    if not args.no_plots:
        written = plots.plot_report(report)
        print(f"plots:   {len(written)} file(s) in {plots.PLOTS_DIR}")
    return 0


def _methods_from_args(args: argparse.Namespace, default: Sequence[str]) -> List[str]:
    if not args.methods:
        return list(default)
    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    resolved = []
    for name in requested:
        key = name.strip().lower()
        canonical = ALGORITHM_ALIASES.get(key, key)
        if canonical not in ALL_METHODS:
            raise SystemExit(
                f"Unknown method '{name}'. Available: {', '.join(ALL_METHODS)}"
            )
        resolved.append(canonical)
    return resolved


def mode_compare(args: argparse.Namespace) -> int:
    """Runs the full Phase 9/10 comparison (random ... coordinated DQN)."""
    methods = _methods_from_args(args, ALL_METHODS)
    config = build_config(args, methods, name="comparison")
    print(f"Comparing {len(methods)} methods on suite '{config.suite.name}': {', '.join(methods)}")
    print(f"train episodes = {config.train_episodes} per method (identical for all), "
          f"eval = {config.fixed_eval_episodes} fixed + {config.resolved_eval_episodes()} seeded random")
    report = run_comparison(config, methods=methods)
    comparison.print_report(report)

    for condition in ("random", "fixed"):
        rows = report.rows_for(condition)
        if len(rows) >= 2:
            print(f"\nDeltas vs '{rows[0].label}' ({condition} condition):")
            for entry in comparison.delta_vs_baseline(rows, rows[0].label):
                deltas = ", ".join(
                    f"{metric}={data['absolute']:+.2f}"
                    + (f" ({data['relative_percent']:+.1f}%)" if "relative_percent" in data else "")
                    for metric, data in entry["deltas"].items()
                )
                print(f"  {entry['label']:32s} {deltas}")

    paths = comparison.save_report(report)
    comparison.save_episodes_csv(report.collectors, os.path.join(
        comparison.results_dir(), "comparison_episodes.csv"))
    comparison.save_history_csv(report.history, os.path.join(
        comparison.results_dir(), "comparison_training_history.csv"))
    print(f"\nresults: {paths['json']}\n         {paths['csv']}")
    if not args.no_plots:
        written = plots.plot_report(report)
        print(f"plots:   {len(written)} file(s) in {plots.PLOTS_DIR}")
    return 0


def mode_ablation(args: argparse.Namespace) -> int:
    """Phase 12: independent vs rule-based vs coordinated (vs CTDE)."""
    methods = _methods_from_args(
        args, (INDEPENDENT_DQN, RULE_BASED, COORDINATED_DQN)
    )
    config = build_config(args, methods, name="ablation")
    print(f"Ablation: {', '.join(methods)} ({config.num_agents} agents, "
          f"{config.train_episodes} train episodes)")
    report = run_ablation(config, methods=methods)
    comparison.print_report(report)

    for condition in ("random", "fixed"):
        rows = report.rows_for(condition)
        if len(rows) >= 2:
            print(f"\nDeltas vs '{rows[0].label}' ({condition} condition):")
            for entry in comparison.delta_vs_baseline(
                rows, rows[0].label,
                metrics=("success_rate", "avg_reward", "avg_collisions", "avg_deadlocks", "avg_steps"),
            ):
                deltas = ", ".join(
                    f"{metric}={data['absolute']:+.2f}"
                    for metric, data in entry["deltas"].items()
                )
                print(f"  {entry['label']:32s} {deltas}")

    paths = comparison.save_report(report)
    comparison.save_episodes_csv(report.collectors, os.path.join(
        comparison.results_dir(), "ablation_episodes.csv"))
    print(f"\nresults: {paths['json']}\n         {paths['csv']}")
    if not args.no_plots:
        written = []
        for condition in ("random", "fixed"):
            rows = report.rows_for(condition)
            if rows:
                written.extend(plots.plot_ablation(rows, condition))
        print(f"plots:   {len(written)} file(s) in {plots.PLOTS_DIR}")
    return 0


def mode_scalability(args: argparse.Namespace) -> int:
    """Phase 13: 1..5 agents on the same layout, measuring cost and quality."""
    counts = [int(c) for c in args.agent_counts.split(",") if c.strip()]
    config = build_config(args, (), name="scalability")
    print(f"Scalability: independent DQN at agent counts {counts} "
          f"({config.train_episodes} train episodes each)")
    report = run_scalability(config, agent_counts=counts)
    print()
    print(comparison.scalability_table(report))

    paths = comparison.save_report(report)
    comparison.save_episodes_csv(report.collectors, os.path.join(
        comparison.results_dir(), "scalability_episodes.csv"))
    print(f"\nresults: {paths['json']}\n         {paths['csv']}")
    if not args.no_plots:
        written = plots.plot_scalability_report(report)
        print(f"plots:   {len(written)} file(s) in {plots.PLOTS_DIR}")
    return 0


def mode_visualize(args: argparse.Namespace) -> int:
    """Streams authoritative state to Unity (or to the reference client)."""
    from python_backend.integration.client import run_headless_viewer
    from python_backend.integration.event_log import open_event_log
    from python_backend.integration.server import GridMindServer
    from python_backend.integration.session import LiveSession, SessionConfig

    method = resolve_algorithm(args.algorithm)
    config = SessionConfig(
        algorithm=method,
        scenario_name=args.scenario,
        num_agents=args.agents,
        seed=args.seed,
        action_source=args.action_source,
        speed_ms=args.speed_ms,
        episode_limit=args.episodes,
        model_path=args.model,
        model_tag=args.model_tag,
        communication_enabled=args.communication,
    )
    try:
        session = LiveSession(config)
    except (ValueError, FileNotFoundError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if not session.trained:
        hint = "--model-tag <tag>" if session.family == "multi_agent" else "--model <path>"
        print("NOTE: no trained checkpoint was supplied -> the agents act from their "
              f"initial (untrained) policy. Train first, then pass {hint}.",
              file=sys.stderr)

    server = GridMindServer(session, host=args.host, port=args.port,
                            exit_on_disconnect=False,
                            event_log=open_event_log(args.event_log))
    if args.event_log:
        print(f"event log: {args.event_log}")
    print(f"GridMind server: {args.host}:{args.port} | algorithm={method} | "
          f"agents={session.num_agents} | episodes={args.episodes} | "
          f"speed={args.speed_ms}ms | action_source={args.action_source}")
    print("In Unity: press Play (NetworkManager connects automatically), then Start.")

    if not args.headless:
        print("Waiting for the Unity client... (Ctrl+C to stop)")
        try:
            server.serve_forever(max_seconds=args.max_seconds)
        except KeyboardInterrupt:
            print("\nstopping")
        finally:
            session.close()
            if server.event_log is not None:
                server.event_log.close()
        return 0

    # Headless verification: run the server and the reference client together so
    # the whole loop (state -> actions -> step -> reward -> goals) is exercised.
    print("Headless mode: driving the session with the reference client "
          "(use this to verify the link without Unity).")
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"max_seconds": args.max_seconds or 120.0},
        daemon=True,
    )
    thread.start()
    if not server.ready.wait(timeout=10.0):
        print("server failed to start listening", file=sys.stderr)
        server.stop()
        session.close()
        return 1
    try:
        code = run_headless_viewer(args.host, server.port, args.episodes, show_grid=True)
    finally:
        server.stop()
        thread.join(timeout=5.0)
        session.close()
        if server.event_log is not None:
            server.event_log.close()
    if session.history and not args.no_plots:
        path = plots.plot_live_session(session.history)
        print(f"plot:    {path}")
    return code


def mode_demo(args: argparse.Namespace) -> int:
    """Deterministic example run (Developer 1's scenario, random policy, ASCII)."""
    from python_backend.environment.reward import RewardConfig
    from python_backend.environment.grid_world import GridWorld

    scenario = load_scenario(args.scenario)
    env = GridWorld(
        height=scenario.height,
        width=scenario.width,
        agents_config=list(scenario.agents),
        obstacles=set(scenario.obstacles),
        reward_config=RewardConfig(),
        max_steps=scenario.max_steps,
        seed=scenario.seed,
    )
    observations = env.reset()
    print("=== GridMind example environment (random policy, no RL) ===")
    print(f"scenario: {scenario.name} | grid: {env.height}x{env.width} | "
          f"agents: {len(env.agents)} | max steps: {env.max_steps}")
    print(f"observation size: {len(observations[0])} per agent\n")
    print("initial state:")
    print(env.render())

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < args.episodes * env.max_steps:
        actions = {aid: env.rng.choice(env.get_valid_actions(aid)) for aid in env.agents}
        _, _, _, terminated, info = env.step(actions)
        truncated = info["truncated"]
        steps += 1
    print("\nfinal state:")
    print(env.render())
    print(f"\nfinished after {info['step']} steps | truncated={info['truncated']} "
          f"terminated={terminated}")
    for aid in env.agents:
        state = env.get_agent_state(aid)
        status = "GOAL REACHED" if state["done"] else "NOT reached"
        print(f"  Agent {aid}: steps={state['steps']:3d}  "
              f"total_reward={state['total_reward']:8.1f}  {status}")
    env.close()
    return 0


MODES = {
    "train": mode_train,
    "evaluate": mode_evaluate,
    "compare": mode_compare,
    "ablation": mode_ablation,
    "scalability": mode_scalability,
    "visualize": mode_visualize,
    "demo": mode_demo,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI dispatcher."""
    args = build_argument_parser().parse_args(argv)
    return MODES[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
