#!/usr/bin/env bash
# Watch every GridMind algorithm in the Unity frontend, one at a time.
#
# Python owns the simulation and streams authoritative state to Unity; Unity
# only renders it. This script starts one streaming session per algorithm, in
# order, and each session stops itself after --max-seconds so the next one can
# bind the same port.
#
# ONE-TIME SETUP
#   1. Train the algorithms you want to watch (checkpoints already exist for the
#      commands below in models/). If a checkpoint is missing, run e.g.
#         python main.py --mode train --algorithm coordinated --agents 3
#      (single-agent: --algorithm dqn | qlearning, no --agents)
#   2. In Unity: open Assets/Scenes/GridMindSimulation.unity and press Play.
#      The GridMindNetwork object connects automatically and reconnects every
#      2 s, so you can Start/Ctrl-C the Python side as often as you like.
#
# HANDS-FREE (recommended for this script)
#   Select GridMindNetwork -> CommunicationManager -> tick "autoStartOnConnect".
#   Unity then sends "set_speed" + "start" by itself on every reconnect, so the
#   whole sweep runs with no clicking. If you leave it unticked, click "Start"
#   in the panel each time a new algorithm connects (within SECONDS seconds).
#
# USAGE
#   bash scripts/visualize_all_algorithms.sh [SECONDS_PER_ALGORITHM] [ONLY]
#     SECONDS_PER_ALGORITHM  wall clock per algorithm (default 45)
#     ONLY                   comma list of labels to run, e.g. random,coordinated_dqn
#
# EXAMPLES
#   bash scripts/visualize_all_algorithms.sh              # all, 45 s each
#   bash scripts/visualize_all_algorithms.sh 90           # all, 90 s each (slower pace)
#   bash scripts/visualize_all_algorithms.sh 60 ctde_inspired,coordinated_dqn

set -u

cd "$(dirname "$0")/.." || exit 1

PYTHON=${PYTHON:-python}
SECONDS_PER=${1:-45}
ONLY=${2:-}
PORT=${PORT:-8765}
AGENTS=3
SPEED=${SPEED:-250}
EPISODES=${EPISODES:-3}

SCENARIO=demo_10x10_3agents

# label | python algorithm | agent count | extra args
#
# NOTE on agent counts: random / q_learning / dqn are SINGLE-AGENT baselines in
# the live viewer -- the session forces them to one agent (agent 0 of the demo
# layout, start (0,0) -> goal (9,9)). Only independent_dqn, rule_based, ctde and
# coordinated are multi-agent (3 agents). For the apples-to-apples 3-agent
# comparison use: python main.py --mode ablation   (and --mode compare).
ENTRIES=(
  "independent_dqn|independent_dqn|3|--model-tag ${SCENARIO}_independent_dqn_3agents"
  "rule_based|rule_based|3|--model-tag ${SCENARIO}_rule_based_coordination_3agents"
  "coordinated_dqn|coordinated|3|--model-tag ${SCENARIO}_coordinated_dqn_3agents"
  "ctde_inspired|ctde|3|--model-tag ${SCENARIO}_ctde_inspired_dqn_3agents"
  "random|random|1|"
  "q_learning|qlearning|1|--model ${SCENARIO}_q_learning"
  "dqn|dqn|1|--model ${SCENARIO}_dqn"
)

want () {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

echo "GridMind sweep: ${#ENTRIES[@]} algorithms, ${SECONDS_PER}s each, port ${PORT}"
echo "Unity: press Play (and Start, unless autoStartOnConnect is ticked)."
echo

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r label algorithm agents extra <<< "$entry"
  want "$label" || continue

  echo "=============================================================="
  echo ">>> ${label}  (${agents} agent(s), ${SECONDS_PER}s)"
  echo "    python main.py --mode visualize --algorithm ${algorithm} \\"
  echo "        --agents ${agents} --episodes ${EPISODES} --speed-ms ${SPEED} \\"
  echo "        --max-seconds ${SECONDS_PER} --port ${PORT} ${extra}"
  echo "=============================================================="

  # shellcheck disable=SC2086
  "$PYTHON" main.py --mode visualize \
    --algorithm "$algorithm" \
    --agents "$agents" \
    --episodes "$EPISODES" \
    --speed-ms "$SPEED" \
    --max-seconds "$SECONDS_PER" \
    --port "$PORT" \
    $extra

  echo ">>> ${label} done. Unity reconnects automatically for the next one."
  echo
done

echo "Sweep complete."
