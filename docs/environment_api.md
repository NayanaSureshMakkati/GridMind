# GridMind — Environment API v1.0 Specification

> **Status: FROZEN.** This document is the contract for Developers 2, 3, and 4.
> Any change requires team agreement per the API freeze rule (see Master Context §61).

## 1. Coordinate System

- Logical positions are discrete tuples `(row, column)`.
- Origin `(0, 0)` is the **top-left** cell.
- Rows increment **downwards**: valid range `[0, height)`.
- Columns increment **rightwards**: valid range `[0, width)`.
- Positions are represented by `Position(NamedTuple)` from `python_backend.environment.constants`.

### 1.1 Python → Unity Coordinate Conversion (Authoritative Convention)

One conversion rule is used **everywhere** in the Unity frontend, implemented in
`unity_frontend/Assets/Scripts/CoordinateConverter.cs`:

```
Unity X =  col * CELL_SIZE
Unity Y =  AGENT_HEIGHT          (fixed vertical offset for agents)
Unity Z = -row * CELL_SIZE       (negated so row 0 renders at the BACK of the scene)
```

Example (with `CELL_SIZE = 1.0`, `AGENT_HEIGHT = 0.5`):

| Python (row, col) | Unity (x, y, z)  |
|-------------------|------------------|
| (0, 0)            | (0.0, 0.5, 0.0)  |
| (2, 3)            | (3.0, 0.5, -2.0) |
| (8, 8)            | (8.0, 0.5, -8.0) |

**Note:** The Master Context recommends `Z = +row * CELL_SIZE`. This implementation
negates Z so that, with Unity's default camera looking down -Z, row 0 appears at the
top of the screen (map-like orientation). This negation is the single documented
deviation; do not mix conventions. To revert, change one line in `CoordinateConverter.cs`.

## 2. Action Space

Fixed integer action per agent, defined in `Action(IntEnum)`:

| ID | Action | Δrow | Δcol |
|----|--------|------|------|
| 0  | UP     | -1   | 0    |
| 1  | DOWN   | +1   | 0    |
| 2  | LEFT   | 0    | -1   |
| 3  | RIGHT  | 0    | +1   |
| 4  | STAY   | 0    | 0    |

No diagonal actions exist. The mapping above must never change between modules.

## 3. Observation Representation

`env.get_agent_observation(agent_id)` returns a 12-element `List[int]`:

| Index | Field          | Meaning                                              |
|-------|----------------|------------------------------------------------------|
| 0     | agent_row      | Agent's current row                                  |
| 1     | agent_col      | Agent's current column                               |
| 2     | goal_row       | Agent's goal row                                     |
| 3     | goal_col       | Agent's goal column                                  |
| 4     | obstacle_up    | 1 if boundary or obstacle in the cell above, else 0  |
| 5     | obstacle_down  | 1 if boundary or obstacle in the cell below, else 0  |
| 6     | obstacle_left  | 1 if boundary or obstacle to the left, else 0        |
| 7     | obstacle_right | 1 if boundary or obstacle to the right, else 0       |
| 8     | agent_up       | 1 if another agent occupies the cell above, else 0   |
| 9     | agent_down     | 1 if another agent occupies the cell below, else 0   |
| 10    | agent_left     | 1 if another agent occupies the cell to the left, else 0 |
| 11    | agent_right    | 1 if another agent occupies the cell to the right, else 0 |

Semantics: obstacle flags treat **out-of-bounds cells as blocked** (1).
The vector is compact, interpretable, and has a fixed size regardless of grid dimensions.

## 4. Reward Schedule

Configured via `RewardConfig` (frozen dataclass — never hardcode these values):

| Event                                          | Default value |
|------------------------------------------------|---------------|
| Goal reached                                   | `+100.0`      |
| Normal step (movement)                         | `-1.0`        |
| Obstacle collision                             | `-10.0`       |
| Agent collision (same-cell/swap)               | `-20.0`       |
| Invalid movement (boundary/obstacle attempt)   | `-10.0`       |
| STAY action                                    | `-2.0`        |

Notes:
- Penalties replace the step reward for that tick (they are not additive).
- Rewards are computed **after** conflict resolution, on the agent's final position.
- Terminal agents (`done=True`) receive `0.0` thereafter.

## 5. Environment API Reference

Construction:

```python
from python_backend.environment.grid_world import GridWorld
from python_backend.environment.constants import Position

env = GridWorld(
    height=10, width=10,
    agents_config=[(Position(0, 0), Position(9, 9))],  # (start, goal) pairs
    obstacles={Position(4, 4)},
    reward_config=None,     # None -> default RewardConfig()
    max_steps=100,
    seed=42,
)
```

### `reset(seed=None, randomize=False, num_agents=1, num_obstacles=0) -> Dict[int, List[int]]`
- Resets step counter, agents, and collision reports.
- If `randomize=True`, generates a fresh seeded layout (obstacles first, then
  start/goal pairs, BFS-validated for reachability — see §8).
- Returns: `{agent_id: observation_vector}`.

### `step(actions) -> Tuple[observations, rewards, dones, terminated, info]`
- `actions`: `Dict[int, int]` mapping agent_id to action id. **Missing agents default to STAY.**
- `observations`: `Dict[int, List[int]]` — next observation per agent.
- `rewards`: `Dict[int, float]`.
- `dones`: `Dict[int, bool]` — per-agent goal attainment.
- `terminated`: `bool` — True iff **all** agents are done (see §7).
- `info`: `Dict[str, Any]` with keys:
  - `"truncated"`: bool — True iff `current_step >= max_steps`
  - `"step"`: int — current step count
  - `"collisions"`: `Dict[int, str]` — agent_id to collision type name, only agents that collided this step

### `get_agent_state(agent_id) -> Dict[str, Any]`
Keys: `agent_id`, `current_position` (Position), `goal_position` (Position),
`steps`, `total_reward`, `done`.

### `get_all_states() -> Dict[int, Dict[str, Any]]`
Same structure as `get_agent_state`, for every agent.

### `get_valid_actions(agent_id) -> List[int]`
Actions that keep the agent inside the grid and out of obstacles.
Terminal agents receive `[4]` (STAY) only.

### `get_agent_position(agent_id) -> Position`
### `get_goal_position(agent_id) -> Position`
### `is_obstacle(position) -> bool`
### `is_valid_position(position) -> bool`
Inside-grid check (does **not** consider obstacles).

### `is_goal_reached(agent_id) -> bool`
### `is_collision(agent_id) -> bool`
True iff the agent collided during the **most recent** `step()` call.

### `is_reachable(start, goal, obstacles, height, width) -> bool`
Static method. Bounded 4-connected BFS. Returns False if the goal is outside the
grid or on an obstacle. **Validation only** — never used as a navigation policy.

### `render() -> str`
ASCII depiction: `A0`/`A1`... agents, `G0`/`G1`... goals, `#` obstacles, `.` free cells.

### `close() -> None`
Clears all agents and obstacles.

## 6. Simultaneous Step Pipeline

`step()` never moves agents sequentially. The order of operations is:

1. Collect actions (missing -> STAY; terminal agents hold position).
2. Compute **proposed positions** for all agents simultaneously.
3. `ConflictArbiter.resolve_simultaneous_moves()`:
   a. Static breaches (boundary, obstacle) -> bounce.
   b. Swap conflicts (A to B's cell while B to A's cell) -> both bounce.
   c. Same-cell conflicts (multiple agents to one cell) -> all bounce.
   d. **Cascade:** any mover whose destination is now held by a bounced/stationary
      agent also bounces (iterated until stable).
4. Conflicting agents remain at their original positions.
5. Rewards computed on final positions; goals checked; episode state updated.
6. New observations returned.

## 7. Termination vs Truncation

- `terminated = True` iff **all** agents reached their goals (success).
- `truncated = True` iff `current_step >= max_steps` (time limit; not a success/failure verdict).
- Both may be True on the same step (goal reached exactly at the step limit).
- These flags are returned separately and must never be merged by consumers
  (matches the Gymnasium convention).

## 8. Random Layouts & Reachability Guarantee

- `reset(randomize=True, ...)` draws obstacles first, then start/goal pairs from
  the remaining free cells, all from a seeded `random.Random`.
- Each pair is BFS-validated; unreachable draws are re-shuffled (max 100 attempts,
  then `RuntimeError`).
- Placement validation rejects starts/goals that overlap obstacles or fall outside
  the grid (`ValueError`).

## 9. Collision Taxonomy

| Type      | Trigger                            | Resolution   | Penalty               |
|-----------|------------------------------------|--------------|-----------------------|
| BOUNDARY  | Proposed cell outside grid         | Bounce back  | `obstacle_collision`  |
| OBSTACLE  | Proposed cell in obstacle          | Bounce back  | `obstacle_collision`  |
| SAME_CELL | Two or more agents propose same cell | All bounce | `agent_collision`     |
| SWAP      | Two agents cross edges             | Both bounce  | `agent_collision`     |

Collisions are recorded in `info["collisions"]` and queryable via
`is_collision(agent_id)` until the next step.
