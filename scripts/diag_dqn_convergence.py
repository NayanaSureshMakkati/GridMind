"""Temporary diagnostic: single-agent DQN convergence probe (not a deliverable)."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_backend.algorithms.dqn import DQNAgent
from python_backend.algorithms.trainer import evaluate_dqn, train_dqn
from python_backend.evaluation.experiments import build_agent_dqn_config
from python_backend.evaluation.scenarios import load_scenario

scenario = load_scenario("demo_10x10_3agents").with_num_agents(1)
env = scenario.build_env()

TOTAL = 2000
CHUNK = 100
agent = DQNAgent(agent_id=0, config=build_agent_dqn_config(seed=7, episodes=TOTAL))

t0 = time.time()
for done in range(CHUNK, TOTAL + 1, CHUNK):
    train_dqn(env, agent, episodes=CHUNK, seed=7)
    m = evaluate_dqn(env, agent, episodes=1)
    print(
        f"episode {done:5d} | eps={agent.epsilon:.4f} | fixed success={m['success_rate']:.2f} "
        f"reward={m['avg_reward']:.1f} steps={m['avg_steps']:.0f} | t={time.time()-t0:.0f}s",
        flush=True,
    )
