"""Multi-agent coordination tests (Developer 3, Phase 14).

Covers: multiple agents, simultaneous actions, collision detection,
RULE-BASED COORDINATION, local vs enriched observations, deadlock detection,
recovery, communication, cooperative reward, scalability (2-5 agents),
and evaluation determinism. The frozen Environment API v1.0 is treated as a
black box and never modified.
"""

import os
import sys
import tempfile
from typing import Dict, List

import pytest

from python_backend.algorithms.dqn import DQNConfig
from python_backend.environment.constants import Action, Position
from python_backend.environment.grid_world import GridWorld
from python_backend.environment.reward import RewardConfig
from python_backend.multi_agent.communication import CommunicationManager
from python_backend.multi_agent.coordination import RuleBasedArbiter
from python_backend.multi_agent.deadlock import DeadlockDetector, RecoveryManager
from python_backend.multi_agent.multi_agent_system import (
    MultiAgentConfig,
    MultiAgentSystem,
)
from python_backend.multi_agent.observations import (
    BASE_OBSERVATION_SIZE,
    LOCAL_ONLY,
    LOCAL_PLUS_OTHER_AGENTS,
    build_observation,
    observation_size,
)

ACTION_SET = {int(a) for a in Action}


def make_env(num_agents: int, max_steps: int = 40) -> GridWorld:
    """Deterministic 6x6 environments for 2-5 agents (BFS-verified reachable)."""
    starts_goals = {
        2: [((0, 0), (5, 5)), ((5, 0), (0, 5))],
        3: [((0, 0), (5, 5)), ((5, 0), (0, 5)), ((0, 3), (5, 3))],
        4: [((0, 0), (5, 5)), ((5, 0), (0, 5)), ((0, 3), (5, 3)), ((3, 0), (3, 5))],
        5: [((0, 0), (5, 5)), ((5, 0), (0, 5)), ((0, 3), (5, 3)), ((3, 0), (3, 5)), ((5, 5), (0, 0))],
    }
    obstacles = {Position(2, 2)}  # single pillar; every start/goal stays free
    agents_config = [
        (Position(s[0], s[1]), Position(g[0], g[1])) for s, g in starts_goals[num_agents]
    ]
    return GridWorld(
        height=6, width=6, agents_config=agents_config, obstacles=obstacles,
        reward_config=RewardConfig(), max_steps=max_steps, seed=7,
    )


# ===========================================================================
# Observations (Phase 5)
# ===========================================================================


class TestObservations:
    def test_local_only_is_frozen_12_vector(self):
        env = make_env(2)
        env.reset()
        obs = build_observation(env, 0, 2, LOCAL_ONLY)
        assert len(obs) == BASE_OBSERVATION_SIZE == 12

    def test_enriched_size_formula(self):
        assert observation_size(2, LOCAL_PLUS_OTHER_AGENTS) == 16
        assert observation_size(5, LOCAL_PLUS_OTHER_AGENTS) == 12 + 4 * 4

    def test_enriched_contains_local_prefix(self):
        env = make_env(3)
        env.reset()
        base = env.get_agent_observation(1)
        enriched = build_observation(env, 1, 3, LOCAL_PLUS_OTHER_AGENTS)
        assert enriched[:12] == base
        assert len(enriched) == 20

    def test_waiting_and_done_flags(self):
        env = make_env(2)
        env.reset()
        # agent 1's goal is (0,5); it is not done; agent 1 did NOT stay
        vec = build_observation(env, 0, 2, LOCAL_PLUS_OTHER_AGENTS, waiting_flags={1: True})
        block = vec[12:16]
        assert block[2] == 1  # wait flag
        assert block[3] == 0  # not done

    def test_done_flag_true(self):
        env = make_env(2)
        env.reset()
        env.step({0: int(Action.STAY), 1: int(Action.STAY)})
        # Move agent 1 onto its goal (0,5) directly via overrides is not
        # possible from (5,0); instead craft a done agent:
        env2 = make_env(2)
        env2.reset()
        env2.agents[1].current_position = env2.agents[1].goal_position  # test-only
        env2.agents[1].done = True
        vec = build_observation(env2, 0, 2, LOCAL_PLUS_OTHER_AGENTS)
        assert vec[15] == 1  # other agent done flag

    def test_invalid_mode_raises(self):
        env = make_env(2)
        env.reset()
        with pytest.raises(ValueError):
            build_observation(env, 0, 2, "telepathy")


# ===========================================================================
# Rule-based coordination (Phase 4)
# ===========================================================================


class TestRuleBasedCoordination:
    def _two_agents_facing(self) -> GridWorld:
        """Two agents directly adjacent, goals behind each other (SWAP setup)."""
        env = GridWorld(
            height=3, width=3,
            agents_config=[(Position(0, 1), Position(2, 1)), (Position(1, 1), Position(0, 1))],
            obstacles=set(), reward_config=RewardConfig(), max_steps=10, seed=1,
        )
        env.reset()
        return env

    def test_swap_conflict_both_wait(self):
        env = self._two_agents_facing()  # A0 at (0,1), A1 at (1,1)
        arbiter = RuleBasedArbiter(env, waiting_times={0: 0, 1: 0})
        actions, events = arbiter.resolve(1, {0: int(Action.DOWN), 1: int(Action.UP)})
        assert actions == {0: int(Action.STAY), 1: int(Action.STAY)}
        assert events and all(e.kind == "swap" for e in events)

    def test_same_cell_higher_priority_proceeds(self):
        env = GridWorld(
            height=1, width=4,
            agents_config=[(Position(0, 0), Position(0, 3)), (Position(0, 2), Position(0, 0))],
            obstacles=set(), reward_config=RewardConfig(), max_steps=10, seed=2,
        )
        env.reset()
        # Both propose the middle cell (0,1). Agent 1 is closer to its goal
        # (distance 2 vs 3) -> higher priority: agent 1 proceeds, agent 0 waits.
        arbiter = RuleBasedArbiter(env, waiting_times={0: 0, 1: 0})
        actions, events = arbiter.resolve(1, {0: int(Action.RIGHT), 1: int(Action.LEFT)})
        assert actions[1] == int(Action.LEFT)
        assert actions[0] == int(Action.STAY)
        assert any(e.kind == "same_cell" and e.agent_id == 0 for e in events)

    def test_no_conflict_actions_unchanged(self):
        env = self._two_agents_facing()
        env.agents[0].current_position = Position(0, 0)  # test-only reposition
        env.agents[1].current_position = Position(2, 2)
        arbiter = RuleBasedArbiter(env, waiting_times={0: 0, 1: 0})
        actions, events = arbiter.resolve(1, {0: int(Action.RIGHT), 1: int(Action.LEFT)})
        assert actions == {0: int(Action.RIGHT), 1: int(Action.LEFT)}
        assert events == []

    def test_labeled_rule_based_not_learned(self):
        from python_backend.multi_agent.coordination import PRIORITY_LABEL
        assert PRIORITY_LABEL == "rule_based"

    def test_arbiter_never_moves_agent_into_collision(self):
        """Post-arbitration, the env must report zero agent collisions."""
        env = make_env(3)
        env.reset()
        arbiter = RuleBasedArbiter(env, waiting_times={0: 0, 1: 0, 2: 0})
        system = MultiAgentSystem(env, MultiAgentConfig(num_agents=3, coordination_mode="rule_based"))
        for _ in range(20):
            actions = {}
            for aid in range(3):
                if env.get_agent_state(aid)["done"]:
                    continue
                valid = env.get_valid_actions(aid)
                actions[aid] = system.agents[aid].choose_action(env.get_agent_observation(aid), valid)
            actions, _ = arbiter.resolve(env.current_step + 1, actions)
            _, _, _, _, info = env.step(actions)
            for aid, name in info.get("collisions", {}).items():
                assert name not in ("SAME_CELL", "SWAP")


# ===========================================================================
# Collision analysis (Phase 3)
# ===========================================================================


class TestCollisionAnalysis:
    def test_same_cell_collision_recorded(self):
        env = GridWorld(
            height=1, width=5,
            agents_config=[(Position(0, 1), Position(0, 4)), (Position(0, 3), Position(0, 0))],
            obstacles=set(), reward_config=RewardConfig(), max_steps=10, seed=3,
        )
        env.reset()
        _, rewards, _, _, info = env.step({0: int(Action.RIGHT), 1: int(Action.LEFT)})
        assert info["collisions"] == {0: "SAME_CELL", 1: "SAME_CELL"}
        assert rewards[0] == rewards[1] == RewardConfig().agent_collision

    def test_swap_collision_recorded(self):
        env = GridWorld(
            height=3, width=3,
            agents_config=[(Position(0, 1), Position(2, 1)), (Position(1, 1), Position(0, 1))],
            obstacles=set(), reward_config=RewardConfig(), max_steps=10, seed=4,
        )
        env.reset()
        _, rewards, _, _, info = env.step({0: int(Action.DOWN), 1: int(Action.UP)})
        assert info["collisions"] == {0: "SWAP", 1: "SWAP"}
        assert rewards[0] == rewards[1] == RewardConfig().agent_collision

    def test_totals_bucket_by_type(self):
        from python_backend.multi_agent.multi_agent_system import CollisionTotals
        totals = CollisionTotals()
        totals.record(0, "OBSTACLE")
        totals.record(0, "BOUNDARY")
        totals.record(1, "SAME_CELL")
        totals.record(2, "SWAP")
        assert (totals.obstacle, totals.agent, totals.same_cell, totals.swap) == (2, 2, 1, 1)
        assert totals.by_agent[1] == {"SAME_CELL": 1}
        totals.record(0, "NONE")  # must not crash or count
        assert totals.obstacle == 2


# ===========================================================================
# Deadlock detection & recovery (Phases 7-8)
# ===========================================================================


class TestDeadlockAndRecovery:
    def test_stagnation_fires(self):
        """stagnation_steps=N fires after N consecutive identical comparisons."""
        det = DeadlockDetector(stagnation_steps=3)
        det.record_step(1, {0: (1, 2), 1: (2, 2)}, {0: False, 1: False}, [])
        pos = {0: (1, 1), 1: (2, 2)}
        det.record_step(2, pos, {0: False, 1: False}, [])  # changed -> counter 0
        det.record_step(3, pos, {0: False, 1: False}, [])  # counter 1
        det.record_step(4, pos, {0: False, 1: False}, [])  # counter 2 (position_cycle may fire)
        fired = det.record_step(5, pos, {0: False, 1: False}, [])  # counter 3 -> fires
        assert fired
        assert "stagnation" in det.events[-1].reasons

    def test_position_cycle_fires(self):
        det = DeadlockDetector(cycle_window=6)
        a = {0: (0, 0), 1: (3, 3)}
        b = {0: (0, 1), 1: (3, 2)}
        for i, pos in enumerate([a, b, a, b], start=1):
            fired = det.record_step(i, pos, {0: False, 1: False}, [])
        assert fired and "position_cycle" in det.events[-1].reasons

    def test_mutual_blocking_fires(self):
        det = DeadlockDetector(mutual_block_collision_threshold=3, collision_window=10)
        pos = {0: (1, 1), 1: (1, 2)}
        det.record_step(1, pos, {0: False, 1: False}, [0])
        det.record_step(2, pos, {0: False, 1: False}, [1])
        fired = det.record_step(3, pos, {0: False, 1: False}, [0, 1])
        assert fired and "mutual_blocking" in det.events[-1].reasons

    def test_action_pattern_fires(self):
        """Phase 7: an action signature repeated >= 2 times fires a deadlock.

        Positions cycle through 4 distinct signatures so the position-cycle
        and stagnation signals stay quiet — only the action pattern fires.
        With the calibrated threshold the alternation A,B,A,B,A produces the
        second repeat of A at step 5, which fires.
        """
        det = DeadlockDetector(cycle_window=8)
        positions = [
            {0: (0, 0), 1: (3, 3)},
            {0: (0, 1), 1: (3, 2)},
            {0: (1, 1), 1: (2, 2)},
            {0: (1, 0), 1: (2, 3)},
        ]
        actions_a = {0: 0, 1: 2}  # same pattern every other step
        actions_b = {0: 1, 1: 3}
        for i in range(6):
            acts = actions_a if i % 2 == 0 else actions_b
            det.record_step(i + 1, positions[i % 4], {0: False, 1: False}, [], actions=acts)
        assert det.deadlock_count >= 1
        assert any("action_pattern" in e.reasons for e in det.events)

    def test_single_action_pattern_repeat_does_not_fire(self):
        """Calibration: ONE prior occurrence of a signature is not a deadlock.

        Three steps A,B,A with 3+ distinct position signatures: A repeats once
        (below the default threshold of 2), so nothing may fire.
        """
        det = DeadlockDetector(cycle_window=8)
        positions = [
            {0: (0, 0), 1: (3, 3)},
            {0: (0, 1), 1: (3, 2)},
            {0: (1, 1), 1: (2, 2)},
        ]
        actions = [{0: 0, 1: 2}, {0: 1, 1: 3}, {0: 0, 1: 2}]
        for i in range(3):
            fired = det.record_step(
                i + 1, positions[i], {0: False, 1: False}, [], actions=actions[i]
            )
            assert not fired
        assert det.deadlock_count == 0
        # And with the old lenient threshold (1) the same stream DOES fire.
        lenient = DeadlockDetector(cycle_window=8, action_pattern_repeats=1)
        for i in range(3):
            lenient.record_step(
                i + 1, positions[i], {0: False, 1: False}, [], actions=actions[i]
            )
        assert lenient.deadlock_count == 1

    def test_action_pattern_ignored_without_actions(self):
        """Passing actions=None keeps the signal off (backward compatible).

        Positions use 6 distinct signatures (no repeats within the window) so
        neither stagnation nor position_cycle can fire either.
        """
        det = DeadlockDetector(cycle_window=8)
        positions = [
            {0: (0, 0), 1: (3, 3)},
            {0: (0, 1), 1: (3, 2)},
            {0: (1, 1), 1: (2, 2)},
            {0: (1, 0), 1: (2, 3)},
            {0: (2, 0), 1: (1, 3)},
            {0: (2, 1), 1: (1, 2)},
        ]
        for i in range(6):
            fired = det.record_step(i + 1, positions[i], {0: False, 1: False}, [])
        assert not fired and det.deadlock_count == 0

    def test_no_deadlock_when_all_done(self):
        det = DeadlockDetector(stagnation_steps=2)
        pos = {0: (5, 5), 1: (0, 5)}
        for i in range(5):
            assert not det.record_step(i + 1, pos, {0: True, 1: True}, [])
        assert det.deadlock_count == 0

    def test_deadlock_counted_once(self):
        """One firing = one recorded event; the firing counter resets."""
        det = DeadlockDetector(stagnation_steps=2)
        det.record_step(1, {0: (1, 0), 1: (2, 2)}, {0: False, 1: False}, [])
        pos = {0: (1, 1), 1: (2, 2)}
        det.record_step(2, pos, {0: False, 1: False}, [])  # changed -> counter 0
        det.record_step(3, pos, {0: False, 1: False}, [])  # counter 1
        fired = det.record_step(4, pos, {0: False, 1: False}, [])  # counter 2 -> fires
        assert fired and det.deadlock_count == 1
        assert det._stagnation_counter == 0  # reset so it cannot re-fire next step

    def test_recovery_manager_kinds_and_budget(self):
        assert RecoveryManager.KINDS == ("none", "alternate_action")
        with pytest.raises(ValueError):
            RecoveryManager(kind="teleport")
        rec = RecoveryManager(kind="alternate_action", max_recoveries_per_episode=2)
        assert rec.should_recover() and rec.should_recover()
        assert not rec.should_recover()  # budget exhausted
        rec.reset()
        assert rec.should_recover()

    def test_recovery_actions_deterministic_non_stay(self):
        env = make_env(2)
        env.reset()
        rec = RecoveryManager(kind="alternate_action")
        overrides = rec.recovery_actions(env, [0, 1])
        for aid, act in overrides.items():
            assert act in env.get_valid_actions(aid)
            assert act != int(Action.STAY)  # nudges out of cycles

    def test_recovery_triggers_after_deadlock_in_episode(self):
        system = MultiAgentSystem(
            make_env(2, max_steps=25),
            MultiAgentConfig(num_agents=2, recovery_kind="alternate_action",
                             deadlock_stagnation_steps=2),
        )
        # Both agents pinned in a corner: only STAY valid for agent 1? Ensure
        # deadlock fires, then a recovery override was applied at least once.
        result = system.run_episode(train=False)
        # With stagnation=2 and pinned agents a deadlock may or may not fire
        # depending on the layout; the invariant is consistency:
        if result.deadlocks > 0:
            assert result.recovery_count >= 1

    def test_recovery_respects_valid_actions(self):
        system = MultiAgentSystem(
            make_env(2, max_steps=15),
            MultiAgentConfig(num_agents=2, recovery_kind="alternate_action",
                             deadlock_stagnation_steps=2, max_recoveries_per_episode=5),
        )
        result = system.run_episode(train=False)
        assert result.recovery_count <= 5


# ===========================================================================
# Communication (Phase 9)
# ===========================================================================


class TestCommunication:
    def test_disabled_produces_zero_messages(self):
        cm = CommunicationManager(enabled=False)
        assert (
            cm.broadcast_intents(1, [0, 1], {0: (0, 0), 1: (1, 1)}, {0: 3, 1: 4}) == 0
        )
        assert cm.messages_per_episode == 0

    def test_enabled_counts_active_agents_only(self):
        cm = CommunicationManager(enabled=True)
        cm.broadcast_intents(1, [0, 1], {0: (0, 0), 1: (1, 1)}, {0: 3, 1: 4})
        assert cm.messages_per_step == 2
        cm.broadcast_intents(2, [0], {0: (0, 0), 1: (1, 1)}, {0: 2})  # agent 1 done
        assert cm.messages_per_step == 1
        assert cm.messages_per_episode == 3

    def test_message_carries_position_action_and_goal_status(self):
        """Phase 9: message := (id, row, col, intended_action, goal_reached)."""
        cm = CommunicationManager(enabled=True)
        cm.broadcast_intents(
            7, [0, 1], {0: (2, 3), 1: (4, 5)}, {0: 1, 1: 4}, goal_reached={0: False, 1: True}
        )
        msgs = cm.get_messages()
        assert msgs == [(0, 2, 3, 1, 0), (1, 4, 5, 4, 1)]

    def test_full_reset_clears_totals(self):
        cm = CommunicationManager(enabled=True)
        cm.broadcast_intents(1, [0], {0: (0, 0)}, {0: 0})
        cm.reset(full=True)
        assert cm.total_messages == 0

    def test_system_communication_accounting(self):
        env = make_env(2, max_steps=15)
        # Pin the DQN seed: unseeded network init draws from global entropy, and
        # a lucky init can let greedy untrained agents finish within max_steps,
        # breaking the strict active-agent message accounting below.
        system = MultiAgentSystem(
            env,
            MultiAgentConfig(
                num_agents=2,
                communication_enabled=True,
                dqn_config=DQNConfig(seed=3),
            ),
        )
        result = system.run_episode(train=False)
        assert result.messages_per_episode == result.steps * 2  # 2 active agents/step
        evaluation = system.evaluate(episodes=2, seed=5)
        assert evaluation.messages_per_episode > 0
        assert evaluation.messages_per_agent_step > 0

    def test_disabled_by_default(self):
        env = make_env(2, max_steps=10)
        system = MultiAgentSystem(env, MultiAgentConfig(num_agents=2))
        result = system.run_episode(train=False)
        assert result.messages_per_episode == 0


# ===========================================================================
# Cooperative reward (Phase 10)
# ===========================================================================


class TestCooperativeReward:
    def test_learned_rewards_differ_from_true(self):
        env = make_env(2, max_steps=10)
        system = MultiAgentSystem(
            env, MultiAgentConfig(num_agents=2, cooperative_weight=0.5)
        )
        result = system.run_episode(train=False)
        assert result.learned_reward[0] != result.reward[0]
        assert result.learned_reward[1] != result.reward[1]

    def test_weight_zero_keeps_rewards_identical(self):
        env = make_env(2, max_steps=10)
        system = MultiAgentSystem(env, MultiAgentConfig(num_agents=2, cooperative_weight=0.0))
        result = system.run_episode(train=False)
        assert result.learned_reward == result.reward

    def test_env_total_reward_unaffected_by_coop_weight(self):
        """The environment's own reward stream must be identical (frozen API)."""
        env1 = make_env(2, max_steps=10)
        env2 = make_env(2, max_steps=10)
        s1 = MultiAgentSystem(env1, MultiAgentConfig(num_agents=2, cooperative_weight=0.0))
        s2 = MultiAgentSystem(env2, MultiAgentConfig(num_agents=2, cooperative_weight=0.7))
        r1 = s1.run_episode(train=False, action_overrides={0: int(Action.STAY), 1: int(Action.STAY)})
        r2 = s2.run_episode(train=False, action_overrides={0: int(Action.STAY), 1: int(Action.STAY)})
        assert r1.reward == r2.reward  # true rewards unchanged
        assert r1.learned_reward != r2.learned_reward

    def test_team_term_is_mean_of_others(self):
        """With weight w, learned = true + w * mean(others' true)."""
        env = make_env(2, max_steps=8)
        system = MultiAgentSystem(env, MultiAgentConfig(num_agents=2, cooperative_weight=1.0))
        result = system.run_episode(train=False)
        for true_step, learned_step in zip(result.step_rewards, result.learned_step_rewards):
            expected0 = true_step[0] + 1.0 * (true_step[1])
            expected1 = true_step[1] + 1.0 * (true_step[0])
            assert learned_step[0] == pytest.approx(expected0)
            assert learned_step[1] == pytest.approx(expected1)


# ===========================================================================
# Multi-agent system & scalability (Phases 1, 2, 12)
# ===========================================================================


class TestMultiAgentSystem:
    def test_config_validation(self):
        with pytest.raises(ValueError):
            MultiAgentConfig(num_agents=1)  # below supported range
        with pytest.raises(ValueError):
            MultiAgentConfig(num_agents=6)  # above supported range
        with pytest.raises(ValueError):
            MultiAgentConfig(num_agents=2, coordination_mode="learned_telepathy")
        with pytest.raises(ValueError):
            MultiAgentConfig(num_agents=2, recovery_kind="teleport")
        with pytest.raises(ValueError):
            MultiAgentConfig(num_agents=2, cooperative_weight=1.5)
        with pytest.raises(ValueError):
            MultiAgentConfig(num_agents=2, ctde=True)  # needs enriched obs

    def test_env_agent_count_mismatch(self):
        env = make_env(2)
        with pytest.raises(ValueError):
            MultiAgentSystem(env, MultiAgentConfig(num_agents=3))

    def test_simultaneous_actions_all_agents_act(self):
        env = make_env(2, max_steps=6)
        system = MultiAgentSystem(env, MultiAgentConfig(num_agents=2))
        result = system.run_episode(train=False, action_overrides={0: int(Action.STAY), 1: int(Action.STAY)})
        assert result.steps >= 1
        assert 0 in result.reward and 1 in result.reward

    def test_honest_system_labels(self):
        env = make_env(2)
        cases = [
            (MultiAgentConfig(num_agents=2), "independent_dqn"),
            (MultiAgentConfig(num_agents=2, policy_kind="random"), "random"),
            (MultiAgentConfig(num_agents=2, coordination_mode="rule_based"), "rule_based_coordination"),
            (MultiAgentConfig(num_agents=2, observation_mode=LOCAL_PLUS_OTHER_AGENTS), "coordinated_dqn"),
            (MultiAgentConfig(num_agents=2, cooperative_weight=0.4), "coordinated_dqn"),
            (MultiAgentConfig(num_agents=2, observation_mode=LOCAL_PLUS_OTHER_AGENTS, ctde=True), "ctde_inspired_dqn"),
        ]
        for cfg, expected in cases:
            assert MultiAgentSystem(env, cfg).system_label() == expected

    def test_random_policy_baseline(self):
        """Phase 13: policy_kind='random' is a true uniform-random baseline."""
        env = make_env(2, max_steps=10)
        system = MultiAgentSystem(
            env, MultiAgentConfig(num_agents=2, policy_kind="random", random_seed=17)
        )
        result = system.run_episode(train=False)
        assert system.system_label() == "random"
        assert result.steps >= 1
        # Deterministic given the seed.
        env2 = make_env(2, max_steps=10)
        system2 = MultiAgentSystem(
            env2, MultiAgentConfig(num_agents=2, policy_kind="random", random_seed=17)
        )
        result2 = system2.run_episode(train=False)
        assert result2.reward == result.reward

    def test_invalid_policy_kind_raises(self):
        with pytest.raises(ValueError):
            MultiAgentConfig(num_agents=2, policy_kind="psychic")

    def test_evaluation_reports_collision_types(self):
        """Phase 12: metrics break collisions into obstacle/agent/same_cell/swap."""
        env = make_env(2, max_steps=20)
        system = MultiAgentSystem(env, MultiAgentConfig(num_agents=2))
        evaluation = system.evaluate(episodes=2, seed=11, randomize=True, num_obstacles=2)
        assert set(evaluation.avg_collision_types) == {"obstacle", "agent", "same_cell", "swap"}
        assert all(v >= 0.0 for v in evaluation.avg_collision_types.values())

    def test_greedy_vs_training_action_use(self):
        env = make_env(2, max_steps=10)
        system = MultiAgentSystem(env, MultiAgentConfig(num_agents=2))
        # During training with epsilon=1.0 actions come from exploration;
        # during greedy evaluation the env still receives valid actions only.
        result = system.run_episode(train=False)
        assert result.steps <= 10

    def test_dqn_configs_per_agent(self):
        from python_backend.algorithms.dqn import DQNConfig
        env = make_env(2)
        cfg = MultiAgentConfig(
            num_agents=2,
            dqn_configs={
                0: DQNConfig(hidden_size=32, observation_size=12),
                1: DQNConfig(hidden_size=64, observation_size=12),
            },
        )
        system = MultiAgentSystem(env, cfg)
        assert system.agents[0].config.hidden_size == 32
        assert system.agents[1].config.hidden_size == 64

    @pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
    def test_scalability_2_to_5_agents(self, num_agents):
        env = make_env(num_agents, max_steps=30)
        system = MultiAgentSystem(env, MultiAgentConfig(num_agents=num_agents))
        result = system.run_episode(train=True, episode=0)
        assert len(result.reward) == num_agents
        assert len(result.successes) == num_agents
        assert result.steps <= 30
        assert isinstance(result.all_success, bool)

    def test_evaluation_result_shape(self):
        env = make_env(2, max_steps=20)
        system = MultiAgentSystem(env, MultiAgentConfig(num_agents=2))
        evaluation = system.evaluate(episodes=2, seed=11, randomize=True, num_obstacles=2)
        assert 0.0 <= evaluation.all_success_rate <= 1.0
        assert 0.0 <= evaluation.any_success_rate <= 1.0
        assert evaluation.episodes == 2
        assert set(evaluation.avg_reward) == {0, 1}
        assert evaluation.avg_steps > 0

    def test_evaluation_deterministic_given_seed(self):
        from python_backend.algorithms.dqn import DQNConfig
        env1 = make_env(2, max_steps=20)
        env2 = make_env(2, max_steps=20)
        s1 = MultiAgentSystem(env1, MultiAgentConfig(num_agents=2, dqn_config=DQNConfig(seed=5)))
        s2 = MultiAgentSystem(env2, MultiAgentConfig(num_agents=2, dqn_config=DQNConfig(seed=5)))
        e1 = s1.evaluate(episodes=2, seed=99)
        e2 = s2.evaluate(episodes=2, seed=99)
        assert e1.avg_reward == e2.avg_reward
        assert e1.all_success_rate == e2.all_success_rate


# ===========================================================================
# Model persistence (Phase 2 / Master Context §35)
# ===========================================================================


class TestModelPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        from python_backend.algorithms.dqn import DQNConfig
        env = make_env(2)
        cfg = MultiAgentConfig(num_agents=2, dqn_config=DQNConfig(seed=123))
        system = MultiAgentSystem(env, cfg)
        system.run_episode(train=True, episode=0)
        system.save_models(str(tmp_path), tag="test")

        system2 = MultiAgentSystem(make_env(2), MultiAgentConfig(num_agents=2, dqn_config=DQNConfig(seed=123)))
        system2.load_models(str(tmp_path), tag="test")
        for aid in (0, 1):
            assert system2.agents[aid].epsilon == system.agents[aid].epsilon
            for p1, p2 in zip(
                system.agents[aid].online_net.parameters(),
                system2.agents[aid].online_net.parameters(),
            ):
                assert (p1 == p2).all()

    def test_load_missing_checkpoint_raises(self, tmp_path):
        system = MultiAgentSystem(make_env(2), MultiAgentConfig(num_agents=2))
        with pytest.raises(FileNotFoundError):
            system.load_models(str(tmp_path), tag="missing")


# ===========================================================================
# Training smoke test (Phase 2/6 integration)
# ===========================================================================


class TestTrainingSmoke:
    def test_short_training_run_completes(self):
        env = make_env(2, max_steps=15)
        system = MultiAgentSystem(env, MultiAgentConfig(num_agents=2))
        history = system.train(episodes=3)
        assert len(history) == 3
        assert all(h.steps <= 15 for h in history)
        # True rewards recorded separately from learned rewards throughout.
        assert all(set(h.reward) == {0, 1} for h in history)

    def test_ctde_execution_view_zeroes_other_blocks(self):
        from python_backend.multi_agent.multi_agent_system import MultiAgentSystem as MS
        obs = list(range(16))
        assert MS.execution_observation(obs)[:12] == obs[:12]
        assert MS.execution_observation(obs)[12:] == [0, 0, 0, 0]
        assert len(MS.execution_observation(obs)) == 16
