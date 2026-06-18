"""
Maze-v1 — Navigate through a procedurally generated maze.

The agent starts at the top-left of a maze and must find its way
to the exit at the bottom-right. Mazes are generated using a
randomized depth-first search algorithm, guaranteeing a solvable path.

State:  (row, col) encoded as a single integer — agent position
Actions: 0=up, 1=down, 2=left, 3=right
Rewards:
    +10.0  — reaching the exit
    -0.01  — each step (encourages finding shorter paths)

Configuration:
    maze_size: int — size of the maze (default 7, must be odd)
    max_steps: int — maximum steps before truncation (default 500)
    random_maze: bool — if True, generate a new maze on each reset()
"""

from __future__ import annotations

import numpy as np

from arenarl.core.base_env import BaseEnv
from arenarl.core.spaces import Discrete

# Action constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Movement deltas: (row_change, col_change)
MOVES = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


class MazeEnv(BaseEnv):
    """A procedurally generated maze navigation environment.

    Mazes are generated using randomized depth-first search (DFS),
    which guarantees exactly one path between any two points.
    The agent must find its way from the start to the exit.

    The maze grid uses a wall/path system:
        - Odd-indexed cells are potential paths
        - Even-indexed cells are walls or carved passages between paths
        - This ensures walls always separate paths cleanly

    RL concepts taught: sparse rewards, exploration vs exploitation,
    long-horizon planning.
    """

    def __init__(
        self,
        maze_size: int = 7,
        max_steps: int = 500,
        random_maze: bool = False,
    ):
        super().__init__()

        # Maze size must be odd to ensure proper wall/path grid
        if maze_size < 5:
            raise ValueError(f"maze_size must be at least 5, got {maze_size}")
        if maze_size % 2 == 0:
            import warnings
            warnings.warn(
                f"maze_size {maze_size} is even, incrementing to {maze_size + 1} "
                "to ensure proper wall/path grid."
            )
            maze_size += 1  # Make it odd

        self.maze_size = maze_size
        self.max_steps = max_steps
        self.random_maze = random_maze

        # Start at top-left path cell, exit at bottom-right path cell
        self.start_pos = (1, 1)
        self.exit_pos = (self.maze_size - 2, self.maze_size - 2)

        # Action space: 4 directions
        self.action_space = Discrete(4)

        # Observation space: each cell in the maze
        self.observation_space = Discrete(self.maze_size * self.maze_size)

        # Internal state
        self._agent_pos: tuple[int, int] = self.start_pos
        self._step_count: int = 0
        self._last_seed: int | None = None

        # Generate initial maze (will be overwritten on reset if random_maze=True)
        self._grid: np.ndarray | None = None
        self._maze_generated: bool = False

    def _generate_maze(self) -> None:
        """Generate a maze using randomized depth-first search (DFS).

        Algorithm:
        1. Start with a grid full of walls (1s)
        2. Pick a starting cell, mark it as path (0)
        3. Randomly choose an unvisited neighbor (2 cells away)
        4. Remove the wall between current cell and chosen neighbor
        5. Move to the neighbor and repeat from step 3
        6. If no unvisited neighbors, backtrack
        7. Continue until all reachable cells are visited

        The "2 cells away" step ensures walls always separate paths.
        """
        size = self.maze_size
        grid = np.ones((size, size), dtype=np.int8)  # 1 = wall

        # Starting cell for generation (top-left path cell)
        start_r, start_c = 1, 1
        grid[start_r, start_c] = 0  # 0 = path

        # Stack-based DFS (avoids Python recursion limit for large mazes)
        stack = [(start_r, start_c)]
        # Directions to check: move 2 cells at a time
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            current_r, current_c = stack[-1]

            # Find unvisited neighbors (2 cells away, still a wall)
            neighbors = []
            for dr, dc in directions:
                nr, nc = current_r + dr, current_c + dc
                if 1 <= nr < size - 1 and 1 <= nc < size - 1 and grid[nr, nc] == 1:
                    neighbors.append((nr, nc, dr, dc))

            if neighbors:
                # Pick a random unvisited neighbor
                idx = self.np_random.integers(len(neighbors))
                nr, nc, dr, dc = neighbors[idx]

                # Carve the wall between current cell and neighbor
                wall_r = current_r + dr // 2
                wall_c = current_c + dc // 2
                grid[wall_r, wall_c] = 0  # Remove wall
                grid[nr, nc] = 0           # Mark neighbor as path

                stack.append((nr, nc))
            else:
                # Backtrack — no unvisited neighbors
                stack.pop()

        # Ensure start is path, verify exit is reachable
        grid[self.start_pos[0], self.start_pos[1]] = 0
        if grid[self.exit_pos[0], self.exit_pos[1]] != 0:
            raise RuntimeError(
                "Generated maze has no path to exit. "
                "This implies a broken generation algorithm."
            )

        self._grid = grid
        self._maze_generated = True

    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        """Reset the maze environment.

        If random_maze is True, generates a new maze layout each time.
        Otherwise, regenerates only on the first call (or with a new seed).

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            (observation, info) where observation is the encoded position.
        """
        super().reset(seed=seed)

        seed_changed = seed is not None and seed != getattr(self, "_last_seed", None)
        if seed is not None:
            self._last_seed = seed

        # Generate maze: always on first call, or every reset if random mode, or if seed changed
        if self.random_maze or not self._maze_generated or seed_changed:
            self._generate_maze()

        self._agent_pos = self.start_pos
        self._step_count = 0

        return self._encode_pos(self._agent_pos), self._get_info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Take one step in the maze.

        The agent can only move into path cells (value 0).
        Moving into a wall keeps the agent in place.

        Args:
            action: An integer 0-3 representing up/down/left/right.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. Must be in {{0, 1, 2, 3}}."
            )

        super().step(action)
        self._step_count += 1

        # Calculate new position
        dr, dc = MOVES[action]
        new_row = self._agent_pos[0] + dr
        new_col = self._agent_pos[1] + dc

        # Check if the new position is a valid path (not a wall, not out of bounds)
        hit_wall = False
        if (
            0 <= new_row < self.maze_size
            and 0 <= new_col < self.maze_size
            and self._grid[new_row, new_col] == 0
        ):
            self._agent_pos = (new_row, new_col)
        else:
            hit_wall = True

        # Check if agent reached the exit
        terminated = self._agent_pos == self.exit_pos
        truncated = self._step_count >= self.max_steps and not terminated

        # Reward: big bonus for exit, penalty for wall, small penalty per step
        if terminated:
            reward = 10.0
        elif hit_wall:
            reward = -1.0
        else:
            reward = -0.01

        self._track_step(reward, terminated, truncated)

        return (
            self._encode_pos(self._agent_pos),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _encode_pos(self, pos: tuple[int, int]) -> int:
        """Encode a (row, col) position as a single integer."""
        return pos[0] * self.maze_size + pos[1]

    def _decode_pos(self, encoded: int) -> tuple[int, int]:
        """Decode a single integer back to (row, col)."""
        return (encoded // self.maze_size, encoded % self.maze_size)

    def _get_info(self) -> dict:
        """Return extra information about the current state."""
        return {
            "agent_pos": self._agent_pos,
            "exit_pos": self.exit_pos,
            "steps": self._step_count,
            "maze_size": self.maze_size,
        }

    def render(self, mode: str = "ascii") -> str | None:
        """Render the maze as ASCII art.

        Wall cells are drawn as blocks, path cells as dots.
        The agent is marked with 'A' and the exit with 'E'.

        Returns:
            A string representation of the maze, or None.
        """
        if mode != "ascii":
            return None

        lines = []
        for row in range(self.maze_size):
            line = ""
            for col in range(self.maze_size):
                pos = (row, col)
                if pos == self._agent_pos and pos == self.exit_pos:
                    line += " ★"
                elif pos == self._agent_pos:
                    line += " A"
                elif pos == self.exit_pos:
                    line += " E"
                elif self._grid[row, col] == 1:
                    line += "██"
                else:
                    line += "  "
            lines.append(line)

        output = "\n".join(lines)
        return output
