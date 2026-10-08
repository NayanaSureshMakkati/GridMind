"""RL backend test suite (Developer 2 Phase 14).

Covers: state encoding, Q-table behavior, Q-update math, epsilon-greedy,
network shape, replay buffer, DQN update, model saving/loading, training
convergence, and the evaluation harness determinism.
"""

import pytest
import torch

from python_backend.algorithms.state_encoding import encode_full, encode_relative, get_encoder
from python_backend.algorithms.random_policy import RandomPolicy
from python_backend.algorithms.q_learning import QLearningAgent, QLearningConfig
from python_backend.algorithms.network import QNetwork, OBSERVATION_SIZE, NUM_ACTIONS
from python_backend.algorithms.replay_buffer import ReplayBuffer
from python_backend.algorithms.dqn import DQNAgent, DQNConfig
from python_backend.algorithms.trainer import (
    train_q_learning,
    run_episode,
    evaluate_random,
    evaluate_q_learning,
)
from python_backend.environment.grid_world import GridWorld
from python_backend.environment.constants import Position


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_env() -> GridWorld:
    return GridWorld(
        5, 5, agents_config=[(Position(0, 0), Position(4, 4))], max_steps=60, seed=7
    )


@pytest.fixture
def q_greedy() -> QLearningAgent:
    """Deterministic, non-exploring tabular agent."""
    return QLearningAgent(QLearningConfig(epsilon=0.0, seed=0))


# ---------------------------------------------------------------------------
# Phase 2: state encoding
# ---------------------------------------------------------------------------


class TestStateEncoding:
    def test_encode_full_preserves_all_fields(self) -> None:
        obs = [1, 2, 3, 4] + [0, 1, 0, 1, 1, 0, 1, 0]
        assert encode_full(obs) == (1, 2, 3, 4, 0, 1, 0, 1, 1, 0, 1, 0)

    def test_encode_relative_replaces_positions_with_delta(self) -> None:
        obs = [2, 3, 0, 1] + [0] * 8  # agent (2,3), goal (0,1)
        key = encode_relative(obs)
        assert key[0] == -2  # goal_row - agent_row
        assert key[1] == -2  # goal_col - agent_col
        assert key[2:] == (0, 0, 0, 0, 0, 0, 0, 0)

    def test_relative_states_share_key_across_layouts(self) -> None:
        # Same relative geometry, different absolute positions -> same key
        a = encode_relative([2, 2, 0, 0] + [0] * 8)
        b = encode_relative([5, 5, 3, 3] + [0] * 8)
        assert a == b

    def test_keys_are_hashable(self) -> None:
        assert hash(encode_full([0] * 12)) is not None
        assert hash(encode_relative([0] * 12)) is not None

    def test_get_encoder_rejects_unknown(self) -> None:
        with pytest.raises(ValueError):
            get_encoder("nope")


# ---------------------------------------------------------------------------
# Phase 3: Q-table and Q-update
# ---------------------------------------------------------------------------


class TestQLearning:
    def test_unseen_state_returns_zero_q(self, q_greedy: QLearningAgent) -> None:
        assert q_greedy.get_q_value([0] * 12, 3) == 0.0

    def test_q_update_math(self) -> None:
        agent = QLearningAgent(QLearningConfig(learning_rate=0.1, gamma=0.9, epsilon=0.0))
        obs = [0] * 12
        nxt = [1] * 12
        agent.q_table[encode_relative(nxt)][2] = 10.0

        agent.update(obs, 0, -1.0, nxt, done=False)
        # 0 + 0.1 * (-1 + 0.9 * max_a' Q(s',a') - 0) = 0.1 * (-1 + 9)
        assert agent.get_q_value(obs, 0) == pytest.approx(0.8)

    def test_q_update_terminal_masks_bootstrap(self) -> None:
        agent = QLearningAgent(QLearningConfig(learning_rate=0.1, gamma=0.9, epsilon=0.0))
        obs = [0] * 12
        nxt = [1] * 12
        agent.q_table[encode_relative(nxt)][2] = 10.0

        agent.update(obs, 0, -1.0, nxt, done=True)
        # target = reward only; bootstrap masked
        assert agent.get_q_value(obs, 0) == pytest.approx(-0.1)

    def test_epsilon_greedy_exploits_when_epsilon_zero(self, small_env: GridWorld) -> None:
        small_env.reset()
        obs = small_env.get_agent_observation(0)
        valid = small_env.get_valid_actions(0)

        agent = QLearningAgent(QLearningConfig(epsilon=0.0, seed=1))
        agent.q_table[encode_relative(obs)] = [0.0] * 5
        agent.q_table[encode_relative(obs)][3] = 5.0  # RIGHT best
        for _ in range(20):
            assert agent.choose_action(obs, valid) == 3

    def test_epsilon_greedy_explores_when_epsilon_one(self, small_env: GridWorld) -> None:
        small_env.reset()
        obs = small_env.get_agent_observation(0)
        valid = small_env.get_valid_actions(0)

        agent = QLearningAgent(QLearningConfig(epsilon=1.0, seed=1))
        picks = {agent.choose_action(obs, valid) for _ in range(100)}
        assert picks.issubset(set(valid))

    def test_action_selection_respects_valid_actions(self, small_env: GridWorld) -> None:
        small_env.reset()
        obs = small_env.get_agent_observation(0)
        valid = small_env.get_valid_actions(0)  # corner: DOWN, RIGHT, STAY only
        assert set(valid) == {1, 3, 4}

        agent = QLearningAgent(QLearningConfig(epsilon=1.0, seed=2))
        for _ in range(50):
            assert agent.choose_action(obs, valid) in valid

    def test_decay_epsilon_respects_floor(self) -> None:
        agent = QLearningAgent(QLearningConfig(epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.05))
        for _ in range(20):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.05)

    def test_q_learning_converges_on_fixed_scenario(self, small_env: GridWorld) -> None:
        agent = QLearningAgent(
            QLearningConfig(epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.05, seed=3)
        )
        train_q_learning(small_env, agent, episodes=200, seed=3)
        metrics = evaluate_q_learning(small_env, agent, episodes=1, seed=0)
        assert metrics["success_rate"] == 1.0
        assert metrics["avg_steps"] <= 12  # optimal is 8


# ---------------------------------------------------------------------------
# Phase 6: network
# ---------------------------------------------------------------------------


class TestQNetwork:
    def test_output_shape_is_batch_x_5(self) -> None:
        net = QNetwork(hidden_size=16)
        out = net(torch.zeros(4, OBSERVATION_SIZE))
        assert out.shape == (4, NUM_ACTIONS)

    def test_different_inputs_give_different_outputs(self) -> None:
        torch.manual_seed(0)
        net = QNetwork(hidden_size=16)
        a = net(torch.zeros(1, OBSERVATION_SIZE))
        b = net(torch.ones(1, OBSERVATION_SIZE))
        assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Phase 7: replay buffer
# ---------------------------------------------------------------------------


class TestReplayBuffer:
    def test_push_and_len(self) -> None:
        buf = ReplayBuffer(capacity=3)
        buf.push([0] * 12, 1, -1.0, [1] * 12, False)
        assert len(buf) == 1

    def test_fifo_eviction(self) -> None:
        buf = ReplayBuffer(capacity=2)
        buf.push([0] * 12, 0, 0.0, [0] * 12, False)
        buf.push([1] * 12, 1, 0.0, [1] * 12, False)
        buf.push([2] * 12, 2, 0.0, [2] * 12, False)
        assert len(buf) == 2
        states = {t[0][0] for t in buf.buffer}
        assert states == {1, 2}  # oldest (0) evicted

    def test_sample_size_and_error(self) -> None:
        buf = ReplayBuffer(capacity=10, seed=0)
        for i in range(5):
            buf.push([i] * 12, 0, 0.0, [i] * 12, False)
        batch = buf.sample(3)
        assert len(batch) == 3
        with pytest.raises(ValueError):
            buf.sample(6)


# ---------------------------------------------------------------------------
# Phases 8-9: DQN update
# ---------------------------------------------------------------------------


class TestDQN:
    @pytest.fixture
    def dqn(self) -> DQNAgent:
        return DQNAgent(
            agent_id=0,
            config=DQNConfig(
                hidden_size=16,
                batch_size=8,
                min_buffer_size=8,
                buffer_capacity=100,
                target_update_frequency=5,
                seed=0,
            ),
        )

    def test_train_step_none_before_warmup(self, dqn: DQNAgent) -> None:
        dqn.remember([0] * 12, 0, -1.0, [1] * 12, False)
        assert dqn.train_step() is None

    def test_train_step_returns_loss_after_warmup(self, dqn: DQNAgent) -> None:
        for _ in range(10):
            dqn.remember([0] * 12, 0, -1.0, [1] * 12, False)
        loss = dqn.train_step()
        assert loss is not None and loss > 0.0

    def test_target_sync_changes_target_weights(self, dqn: DQNAgent) -> None:
        dqn.remember([0] * 12, 0, -1.0, [1] * 12, False)
        dqn.remember([1] * 12, 1, -1.0, [2] * 12, False)
        for _ in range(8):
            dqn.remember([2] * 12, 2, 100.0, [3] * 12, True)
            dqn.train_step()
        # force-desync then sync
        with torch.no_grad():
            for p in dqn.online_net.parameters():
                p.add_(1.0)
        dqn.sync_target_network()
        for online_p, target_p in zip(
            dqn.online_net.parameters(), dqn.target_net.parameters()
        ):
            assert torch.equal(online_p, target_p)

    def test_dqn_selects_valid_actions(self, small_env: GridWorld, dqn: DQNAgent) -> None:
        small_env.reset()
        obs = small_env.get_agent_observation(0)
        valid = small_env.get_valid_actions(0)
        for _ in range(30):
            assert dqn.choose_action(obs, valid) in valid

    def test_save_load_roundtrip(self, dqn: DQNAgent, tmp_path) -> None:
        path = str(tmp_path / "dqn.pt")
        dqn.save_model(path, metadata={"grid": "5x5"})

        restored = DQNAgent(agent_id=0, config=DQNConfig(hidden_size=16))
        restored.load_model(path)

        obs = [1, 2, 3, 4] + [0] * 8
        a1 = dqn._q_values(obs)
        a2 = restored._q_values(obs)
        assert a1 == pytest.approx(a2)
        assert restored.epsilon == dqn.epsilon

    def test_rejects_foreign_checkpoint(self, dqn: DQNAgent, tmp_path) -> None:
        path = str(tmp_path / "bad.pt")
        torch.save({"format": "something.else"}, path)
        with pytest.raises(ValueError):
            dqn.load_model(path)


# ---------------------------------------------------------------------------
# Phase 1 + 11: random baseline & evaluation harness
# ---------------------------------------------------------------------------


class TestRandomBaselineAndEvaluation:
    def test_random_policy_stays_in_valid_actions(self, small_env: GridWorld) -> None:
        small_env.reset()
        obs = small_env.get_agent_observation(0)
        valid = small_env.get_valid_actions(0)
        policy = RandomPolicy(seed=0)
        for _ in range(50):
            assert policy.choose_action(obs, valid) in valid

    def test_random_episode_pipeline(self, small_env: GridWorld) -> None:
        """Phase 1 verification: state -> action -> env -> reward -> next state."""
        small_env.reset()
        policy = RandomPolicy(seed=0)
        obs = small_env.get_agent_observation(0)
        steps = 0
        while not small_env.get_agent_state(0)["done"] and steps < small_env.max_steps:
            action = policy.choose_action(obs, small_env.get_valid_actions(0))
            next_obs, rewards, dones, _, info = small_env.step({0: action})
            assert isinstance(rewards[0], float)
            assert len(next_obs[0]) == 12
            assert "truncated" in info
            obs = next_obs[0]
            steps += 1
            if dones[0]:
                break
        assert steps > 0

    def test_run_episode_zero_limit_does_not_crash(self, small_env: GridWorld) -> None:
        """Regression: run_episode with max_steps=0 must return an empty result.

        The step loop never executes, so `info` stays at its pre-initialized
        empty value; the return must not raise UnboundLocalError.
        """
        small_env.reset()
        policy = RandomPolicy(seed=0)
        result = run_episode(small_env, policy.choose_action, agent_id=0, max_steps=0)
        assert result["reward"] == 0.0
        assert result["steps"] == 0
        assert result["success"] is False
        assert result["collisions"] == 0
        assert result["truncated"] is False

    def test_evaluation_metrics_structure(self, small_env: GridWorld) -> None:
        metrics = evaluate_random(small_env, episodes=3, seed=0)
        assert metrics["episodes"] == 3
        assert 0.0 <= metrics["success_rate"] <= 1.0
        for key in ("avg_reward", "avg_steps", "avg_collisions"):
            assert key in metrics

    def test_evaluation_is_deterministic_for_same_seed(self, small_env: GridWorld) -> None:
        m1 = evaluate_random(small_env, episodes=5, seed=42)
        m2 = evaluate_random(small_env, episodes=5, seed=42)
        assert m1["avg_steps"] == m2["avg_steps"]
        assert m1["avg_reward"] == m2["avg_reward"]
        assert m1["success_rate"] == m2["success_rate"]


# ---------------------------------------------------------------------------
# Phases 12-13: independent DQN system
# ---------------------------------------------------------------------------


class TestIndependentDQN:
    def test_multiple_agent_instances(self) -> None:
        from python_backend.multi_agent import IndependentDQNSystem

        system = IndependentDQNSystem(
            3, DQNConfig(hidden_size=16, batch_size=8, min_buffer_size=8, seed=0)
        )
        assert set(system.agents.keys()) == {0, 1, 2}
        # Each agent owns its networks and buffer
        assert system.agents[0].online_net is not system.agents[1].online_net
        assert system.agents[0].replay_buffer is not system.agents[1].replay_buffer

    def test_multi_agent_episode_runs(self) -> None:
        from python_backend.multi_agent import IndependentDQNSystem

        env = GridWorld(
            5,
            5,
            agents_config=[(Position(0, 0), Position(4, 4)), (Position(4, 0), Position(0, 4))],
            max_steps=40,
            seed=5,
        )
        system = IndependentDQNSystem(
            2, DQNConfig(hidden_size=16, batch_size=8, min_buffer_size=8, seed=0)
        )
        result = system.run_episode(env, train=True)
        assert set(result["reward"].keys()) == {0, 1}
        assert result["steps"] > 0

    def test_save_and_load_models(self, tmp_path) -> None:
        import os

        from python_backend.multi_agent import IndependentDQNSystem

        system = IndependentDQNSystem(
            2, DQNConfig(hidden_size=16, batch_size=8, min_buffer_size=8, seed=0)
        )
        directory = str(tmp_path / "models")
        system.save_models(directory, tag="test")

        for aid in (0, 1):
            assert os.path.exists(os.path.join(directory, f"test_agent{aid}.pt"))

        fresh = IndependentDQNSystem(
            2, DQNConfig(hidden_size=16, batch_size=8, min_buffer_size=8, seed=99)
        )
        fresh.load_models(directory, tag="test")
        obs = [0] * 12
        assert fresh.agents[0]._q_values(obs) == pytest.approx(
            system.agents[0]._q_values(obs)
        )
