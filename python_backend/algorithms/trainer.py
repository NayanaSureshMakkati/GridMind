"""Training and evaluation harness (Phases 1, 4, 5, 9, 11).

Single-agent training loops for Random / Q-learning / DQN plus a shared
evaluation harness. Everything consumes ONLY the frozen Environment API v1.0:
reset / step / get_valid_actions / get_agent_observation. GridWorld internals
are never touched.

Metric accounting (fairness rules):
  - `steps` and `collisions` are counted from reset until the FIRST
    termination signal, inclusive of the step that ends the episode.
  - After a per-agent `done=True`, the agent is removed from action selection
    (terminal agents receive 0 reward; counting stops for them).
  - Truncation never masks bootstraps; the trainer simply stops the episode.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from python_backend.algorithms.dqn import DQNAgent, DQNConfig
from python_backend.algorithms.q_learning import QLearningAgent, QLearningConfig
from python_backend.algorithms.random_policy import RandomPolicy

EpisodeResult = Dict[str, Any]
Metrics = Dict[str, Any]


# ---------------------------------------------------------------------------
# Generic episode runner
# ---------------------------------------------------------------------------


def run_episode(
    env,
    policy: Callable[[List[int], Optional[List[int]], bool], int],
    agent_id: int = 0,
    max_steps: Optional[int] = None,
    train: bool = False,
) -> EpisodeResult:
    """Runs one episode with a policy callable (obs, valid_actions, greedy) -> action.

    If `train` is True, the callable is expected to be a bound method of a
    learning agent and the trainer also feeds transitions back through the
    optional `learner` hooks (see train_q_learning / train_dqn below, which
    manage updates themselves and use run_episode only for evaluation paths).
    """
    obs_dict = env.reset()
    obs = obs_dict[agent_id]

    total_reward = 0.0
    steps = 0
    collisions = 0
    success = False
    limit = max_steps if max_steps is not None else env.max_steps
    # Assigned before the loop so the return below cannot hit an
    # UnboundLocalError when the loop body never runs (agent already done
    # at reset, or limit == 0).
    info: Dict[str, Any] = {}

    for _ in range(limit):
        if env.get_agent_state(agent_id)["done"]:
            break

        valid = env.get_valid_actions(agent_id)
        action = policy(obs, valid, greedy=not train)
        next_obs, rewards, dones, terminated, info = env.step({agent_id: action})

        reward = rewards[agent_id]
        done = dones[agent_id]
        total_reward += reward
        steps += 1
        if agent_id in info.get("collisions", {}):
            collisions += 1

        if hasattr(policy, "__self__") and hasattr(policy.__self__, "update"):
            pass  # learning handled by dedicated train_* functions

        obs = next_obs[agent_id]

        if done:
            success = True
            break
        if info.get("truncated", False):
            break

    return {
        "reward": total_reward,
        "steps": steps,
        "success": success,
        "collisions": collisions,
        "truncated": info.get("truncated", False),
    }


# ---------------------------------------------------------------------------
# Q-learning training (Phase 4)
# ---------------------------------------------------------------------------


def train_q_learning(
    env,
    agent: QLearningAgent,
    episodes: int,
    seed: int = 0,
    eval_every: int = 0,
    callback: Optional[Callable[[int, Metrics], None]] = None,
    reset_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Metrics]:
    """Trains tabular Q-learning. Records per-episode metrics (Phase 4 spec).

    `reset_kwargs` (e.g. {"randomize": True, "num_obstacles": 4, ...}) is
    forwarded to env.reset(); pass {"seed": <int>} inside it for per-episode
    deterministic layouts. The `seed` argument seeds the harness's own RNG.
    """
    rng = random.Random(seed)
    history: List[Metrics] = []

    for episode in range(episodes):
        obs_dict = env.reset(**(reset_kwargs or {}))
        obs = obs_dict[0]

        total_reward = 0.0
        steps = 0
        collisions = 0
        success = False

        for _ in range(env.max_steps):
            if env.get_agent_state(0)["done"]:
                break

            valid = env.get_valid_actions(0)
            action = agent.choose_action(obs, valid)
            next_obs, rewards, dones, _, info = env.step({0: action})

            reward = rewards[0]
            done = dones[0]
            agent.update(obs, action, reward, next_obs[0], done)

            total_reward += reward
            steps += 1
            if 0 in info.get("collisions", {}):
                collisions += 1
            obs = next_obs[0]

            if done:
                success = True
                break
            if info.get("truncated", False):
                break

        agent.decay_epsilon()

        record = {
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "success": success,
            "epsilon": agent.epsilon,
            "collisions": collisions,
        }
        history.append(record)

        if eval_every > 0 and (episode + 1) % eval_every == 0:
            record["eval_success_rate"] = evaluate_q_learning(
                env, agent, episodes=20, seed=12345
            )["success_rate"]

        if callback is not None:
            callback(episode, record)

    return history


# ---------------------------------------------------------------------------
# DQN training (Phase 9)
# ---------------------------------------------------------------------------


def train_dqn(
    env,
    agent: DQNAgent,
    episodes: int,
    seed: int = 0,
    eval_every: int = 0,
    callback: Optional[Callable[[int, Metrics], None]] = None,
    reset_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Metrics]:
    """Trains DQN with epsilon-greedy collection + replay updates (Phase 9)."""
    history: List[Metrics] = []

    for episode in range(episodes):
        obs_dict = env.reset(**(reset_kwargs or {}))
        obs = obs_dict[agent.agent_id]

        total_reward = 0.0
        steps = 0
        collisions = 0
        success = False
        losses: List[float] = []

        for _ in range(env.max_steps):
            if env.get_agent_state(agent.agent_id)["done"]:
                break

            valid = env.get_valid_actions(agent.agent_id)
            action = agent.choose_action(obs, valid)
            next_obs, rewards, dones, _, info = env.step({agent.agent_id: action})

            reward = rewards[agent.agent_id]
            done = dones[agent.agent_id]
            agent.remember(obs, action, reward, next_obs[agent.agent_id], done)
            loss = agent.train_step()

            total_reward += reward
            steps += 1
            if agent.agent_id in info.get("collisions", {}):
                collisions += 1
            if loss is not None:
                losses.append(loss)
            obs = next_obs[agent.agent_id]

            if done:
                success = True
            if done or info.get("truncated", False):
                break

        agent.decay_epsilon()

        record = {
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "success": success,
            "epsilon": agent.epsilon,
            "collisions": collisions,
            "mean_loss": (sum(losses) / len(losses)) if losses else 0.0,
        }
        history.append(record)

        if eval_every > 0 and (episode + 1) % eval_every == 0:
            record["eval_success_rate"] = evaluate_dqn(
                env, agent, episodes=20, seed=12345
            )["success_rate"]

        if callback is not None:
            callback(episode, record)

    return history


# ---------------------------------------------------------------------------
# Evaluation harness (Phases 5 and 11)
# ---------------------------------------------------------------------------


def evaluate_policy(
    env,
    policy: Callable[[List[int], Optional[List[int]], bool], int],
    agent_id: int,
    episodes: int = 100,
    seed: int = 0,
    layout_seed_start: int = 1000,
    reset_kwargs: Optional[Dict[str, Any]] = None,
) -> Metrics:
    """Evaluates any policy over identical seeded random layouts (fairness).

    Every evaluation uses the same `layout_seed_start + i` sequence, so
    Random / Q-learning / DQN face exactly the same scenarios.
    `reset_kwargs` may carry `num_agents` / `num_obstacles` for the random
    layout generator (defaults: 1 agent, 0 obstacles).
    """
    totals: Metrics = {
        "episodes": episodes,
        "successes": 0,
        "reward": 0.0,
        "steps": 0,
        "collisions": 0,
    }

    for i in range(episodes):
        # Caller reset_kwargs fully control the layout mode (randomize True/False).
        # The layout seed is fixed per episode for identical scenario sequences.
        reset_args = {"randomize": False, **(reset_kwargs or {})}
        reset_args.pop("seed", None)
        env.reset(seed=layout_seed_start + i, **reset_args)
        obs = env.get_agent_observation(agent_id)

        total_reward = 0.0
        steps = 0
        collisions = 0
        success = False

        for _ in range(env.max_steps):
            if env.get_agent_state(agent_id)["done"]:
                break
            valid = env.get_valid_actions(agent_id)
            action = policy(obs, valid, greedy=True)
            next_obs, rewards, dones, _, info = env.step({agent_id: action})
            total_reward += rewards[agent_id]
            steps += 1
            if agent_id in info.get("collisions", {}):
                collisions += 1
            obs = next_obs[agent_id]
            if dones[agent_id]:
                success = True
                break
            if info.get("truncated", False):
                break

        totals["successes"] += int(success)
        totals["reward"] += total_reward
        totals["steps"] += steps
        totals["collisions"] += collisions

    n = episodes
    return {
        "algorithm": policy.__self__.__class__.__name__ if hasattr(policy, "__self__") else "policy",
        "episodes": n,
        "success_rate": totals["successes"] / n,
        "avg_reward": totals["reward"] / n,
        "avg_steps": totals["steps"] / n,
        "avg_collisions": totals["collisions"] / n,
    }


def evaluate_q_learning(
    env, agent: QLearningAgent, episodes: int = 100, seed: int = 0,
    reset_kwargs: Optional[Dict[str, Any]] = None,
) -> Metrics:
    """Greedy (epsilon=0) evaluation of a Q-learning agent (Phase 5)."""
    return evaluate_policy(
        env, agent.choose_action, agent_id=0, episodes=episodes, seed=seed,
        reset_kwargs=reset_kwargs,
    )


def evaluate_dqn(
    env, agent: DQNAgent, episodes: int = 100, seed: int = 0,
    reset_kwargs: Optional[Dict[str, Any]] = None,
) -> Metrics:
    """Greedy evaluation of a DQN agent (Phase 11)."""
    return evaluate_policy(
        env, agent.choose_action, agent_id=agent.agent_id, episodes=episodes, seed=seed,
        reset_kwargs=reset_kwargs,
    )


def evaluate_random(
    env, episodes: int = 100, seed: int = 0,
    reset_kwargs: Optional[Dict[str, Any]] = None,
) -> Metrics:
    """Random baseline evaluation (Phase 1 verification + Phase 11 comparison)."""
    policy = RandomPolicy(seed=seed)
    return evaluate_policy(
        env, policy.choose_action, agent_id=0, episodes=episodes, seed=seed,
        reset_kwargs=reset_kwargs,
    )


# ---------------------------------------------------------------------------
# Experiment 2 driver: Random vs Q-learning vs DQN (Phase 11)
# ---------------------------------------------------------------------------


def run_comparison(
    q_agent: QLearningAgent,
    dqn_agent: DQNAgent,
    grid_size: int = 10,
    episodes: int = 100,
    seed: int = 0,
    num_obstacles: int = 0,
) -> Dict[str, Metrics]:
    """Evaluates all three algorithms on identical seeded layouts (Phase 11).

    Fairness: same grid size, same obstacle count, same layout seed sequence
    (1000 + i) and same max_steps for every algorithm.
    """
    from python_backend.environment.grid_world import GridWorld

    env = GridWorld(
        height=grid_size,
        width=grid_size,
        max_steps=grid_size * 5,
        seed=seed,
    )
    reset_kwargs = {"randomize": True, "num_obstacles": num_obstacles}

    return {
        "random": evaluate_random(env, episodes=episodes, seed=seed, reset_kwargs=reset_kwargs),
        "q_learning": evaluate_q_learning(env, q_agent, episodes=episodes, seed=seed, reset_kwargs=reset_kwargs),
        "dqn": evaluate_dqn(env, dqn_agent, episodes=episodes, seed=seed, reset_kwargs=reset_kwargs),
    }
