# Environment Reference

ArenaRL currently ships with four built-in environments, carefully designed to introduce progressive complexity in state spaces and physics.

## 1. GridWorld-v1

The foundational reinforcement learning environment. The agent must navigate an $N \times N$ grid to reach a goal while avoiding obstacles. 

- **Observation Space:** `Discrete(N * N)` — Represented as a single integer (row * grid_size + col).
- **Action Space:** `Discrete(4)` — (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).
- **Rewards:** 
  - `+1.0` Goal reached
  - `-1.0` Hit obstacle
  - `-0.1` Time penalty per step

**Configuration:**
- `grid_size` (int): Size of the grid.
- `random_obstacles` (bool): If True, procedurally generates random solvable obstacles on every reset.

## 2. Maze-v1

A complex procedural maze generator. The agent must navigate corridors to find the exit. The state space is similar to GridWorld, but the topology forces the agent to learn long-term planning and backtracking.

- **Observation Space:** `Discrete(N * N)`
- **Action Space:** `Discrete(4)`
- **Rewards:** Same as GridWorld.

**Configuration:**
- `maze_size` (int): Must be an odd number (automatically incremented if even).

## 3. Snake-v1

The classic Snake game. Introduces a complex, dynamic observation space (the grid itself) since the state cannot be represented by a single position integer (the snake grows and takes up multiple cells).

- **Observation Space:** `Box` — A 2D grid of integers representing Empty(0), Head(1), Body(2), and Apple(3).
- **Action Space:** `Discrete(4)`
- **Rewards:**
  - `+1.0` Eating an apple
  - `-1.0` Hitting a wall or self (death)
  - `0.0` Step

## 4. CarTrack-v1

Introduces continuous physics (momentum, velocity, turning angles). The agent drives a car around an oval track and is rewarded for completing laps.

- **Observation Space:** `Box(4,)` — Continuous vector `[x, y, velocity, angle]`.
- **Action Space:** `Discrete(5)` — (0: COAST, 1: ACCEL, 2: BRAKE, 3: LEFT, 4: RIGHT).
- **Rewards:**
  - `+10.0` Completing a full lap
  - `-1.0` Crashing off the track
  - `+distance` Proportional reward for driving forward
  - `-0.01` Time penalty
