"""Temporary diagnostic: which budget solves single-agent DQN on the demo scenario?"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_backend.algorithms.dqn import DQNAgent, DQNConfig
from python_backend.algorithms.trainer import train_dqn
from python_backend.evaluation.experiments import build_agent_dqn_config
from python_backend.evaluation.scenarios import load_scenario

scenario = load_scenario("demo_10x10_3agents").with_num_agents(1)


def greedy_episode(env, agent):
    env.reset()
    obs = env.get_agent_observation(0)
    total, steps = 0.0, 0
    while True:
        if env.get_agent_state(0)["done"]:
            return {"reward": total, "success": True, "steps": steps}
        action = agent.choose_action(obs, env.get_valid_actions(0), greedy=True)
        nxt, rew, dones, term, info = env.step({0: action})
        total += rew[0]
        steps += 1
        obs = nxt[0]
        if dones[0]:
            return {"reward": total, "success": True, "steps": steps}
        if info.get("truncated"):
            return {"reward": total, "success": False, "steps": steps}


def run(name, episodes, config_fn):
    env = scenario.build_env()
    agent = DQNAgent(agent_id=0, config=config_fn())
    t0 = time.time()
    train_dqn(env, agent, episodes=episodes, seed=7)
    m = greedy_episode(env, agent)
    print(
        f"{name:34s} eps={episodes:5d} | success={m['success']} reward={m['reward']:7.1f} "
        f"steps={m['steps']:3d} | t={time.time()-t0:.0f}s",
        flush=True,
    )


def scripts_cfg():
    return DQNConfig(gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                     learning_rate=5e-4, batch_size=64, buffer_capacity=50000,
                     min_buffer_size=500, target_update_frequency=250, hidden_size=128,
                     observation_size=12, seed=7)


run("B scripts-cfg lr5e-4 600eps", 600, scripts_cfg)
run("C scripts-cfg lr5e-4 900eps", 900, scripts_cfg)
run("D main-cfg lr2e-4 1200eps", 1200, lambda: build_agent_dqn_config(seed=7, episodes=1200))
