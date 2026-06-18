"""
GridWorld-v1 — Navigate an N×N grid from start to goal, avoiding obstacles.

The agent starts at a position on a grid and must reach the goal.
Obstacles block movement. The agent receives a reward for reaching
the goal and a penalty for each step taken.

State:  (row, col) — agent position on the grid
Actions: 0=up, 1=down, 2=left, 3=right
Rewards:
    +1.0  — reaching the goal
    -1.0  — hitting an obstacle
    -0.1  — each step (encourages finding short paths)

Configuration:
    grid_size: int — size of the grid (default 5)
    obstacles: list — list of (row, col) obstacle positions (default auto-generated)
    start_pos: tuple — starting position (default top-left)
    goal_pos: tuple — goal position (default bottom-right)
    max_steps: int — maximum steps before truncation (default 100)
"""

from __future__ import annotations

from arenarl.core.base_env import BaseEnv
from arenarl.core.spaces import Discrete

# Action constants for readability
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Action name mapping for rendering
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

# Movement deltas: (row_change, col_change) for each action
MOVES = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


class GridWorldEnv(BaseEnv):
    """A simple grid world navigation environment.

    The agent moves on an N×N grid, trying to reach the goal while
    avoiding obstacles. This is the most fundamental RL environment,
    ideal for learning Q-learning and basic RL concepts.
    """

    def __init__(
        self,
        grid_size: int = 5,
        obstacles: list[tuple[int, int]] | None = None,
        start_pos: tuple[int, int] | None = None,
        goal_pos: tuple[int, int] | None = None,
        max_steps: int = 100,
        random_obstacles: bool = False,
        num_obstacles: int | None = None,
    ):
        super().__init__()

        self.grid_size = grid_size
        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")

        self.max_steps = max_steps
        self.random_obstacles = random_obstacles

        # How many random obstacles to place (default: ~15% of grid cells)
        self.num_obstacles = num_obstacles or max(1, (grid_size * grid_size) // 7)

        # Default positions
        self.start_pos = (0, 0) if start_pos is None else start_pos
        self.goal_pos = (grid_size - 1, grid_size - 1) if goal_pos is None else goal_pos

        if self.start_pos == self.goal_pos:
            raise ValueError("start_pos and goal_pos must be different")
        for pos_name, pos in [("start_pos", self.start_pos), ("goal_pos", self.goal_pos)]:
            if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
                raise ValueError(
                    f"{pos_name} {pos} is out of bounds for grid size {self.grid_size}"
                )

        # Set obstacles — random mode generates new ones on each reset()
        if obstacles is not None:
            self.obstacles = set(tuple(o) for o in obstacles)
            self.random_obstacles = False  # explicit obstacles override random mode
        elif not self.random_obstacles:
            self.obstacles = self._generate_default_obstacles()
        else:
            self.obstacles: set[tuple[int, int]] = set()  # filled on reset()

        # Validate that start and goal are not on obstacles
        if self.start_pos in self.obstacles:
            raise ValueError(f"Start position {self.start_pos} is on an obstacle")
        if self.goal_pos in self.obstacles:
            raise ValueError(f"Goal position {self.goal_pos} is on an obstacle")

        # Define spaces
        # Action space: 4 discrete actions (up, down, left, right)
        self.action_space = Discrete(4)

        # Observation space: 2 discrete values (row, col)
        # We represent the state as a single integer: row * grid_size + col
        self.observation_space = Discrete(grid_size * grid_size)

        # Internal state
        self._agent_pos: tuple[int, int] = self.start_pos
        self._step_count: int = 0

    def _is_reachable(self, obstacles: set[tuple[int, int]]) -> bool:
        """Check if the goal is reachable from the start using BFS."""
        from collections import deque
        q, seen = deque([self.start_pos]), {self.start_pos}
        while q:
            r, c = q.popleft()
            if (r, c) == self.goal_pos:
                return True
            for dr, dc in MOVES.values():
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.grid_size
                    and 0 <= nc < self.grid_size
                    and (nr, nc) not in obstacles
                    and (nr, nc) not in seen
                ):
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return False

    def _generate_default_obstacles(self) -> set[tuple[int, int]]:
        """Generate a reasonable set of default obstacles for the grid."""
        obstacles = set()
        size = self.grid_size

        if size >= 4:
            # Place some obstacles in the middle area
            mid = size // 2
            obstacles.add((mid, mid))
            obstacles.add((mid - 1, mid))
            obstacles.add((mid, mid - 1))

        if size >= 6:
            obstacles.add((mid + 1, mid + 1))
            obstacles.add((1, mid))

        # Make sure start and goal are clear
        obstacles.discard(self.start_pos)
        obstacles.discard(self.goal_pos)

        return obstacles

    def _generate_random_obstacles(self) -> set[tuple[int, int]]:
        """Generate random obstacle positions using the seeded RNG."""
        all_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) != self.start_pos and (r, c) != self.goal_pos
        ]

        for _ in range(100):  # Retry cap to avoid infinite loop
            obstacles: set[tuple[int, int]] = set()
            # Shuffle and pick the first num_obstacles cells
            indices = self.np_random.permutation(len(all_cells))
            for i in range(min(self.num_obstacles, len(all_cells))):
                obstacles.add(all_cells[indices[i]])

            if self._is_reachable(obstacles):
                return obstacles

        return set()  # Fallback to no obstacles if impossible to generate valid layout

    def reset(self, seed: int | None = None) -> tuple[int, dict]:
        """Reset the grid world to its initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            (observation, info) where observation is the encoded position.
        """
        super().reset(seed=seed)

        # Generate new random obstacles if in random mode
        if self.random_obstacles:
            self.obstacles = self._generate_random_obstacles()

        self._agent_pos = self.start_pos
        self._step_count = 0

        return self._encode_pos(self._agent_pos), self._get_info()

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """Take one step in the grid world.

        Args:
            action: An integer 0-3 representing up/down/left/right.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        super().step(action)

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. Must be in {{0, 1, 2, 3}}."
            )

        self._step_count += 1

        # Calculate new position
        dr, dc = MOVES[action]
        new_row = self._agent_pos[0] + dr
        new_col = self._agent_pos[1] + dc

        # Check boundaries
        if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
            new_pos = (new_row, new_col)

            if new_pos in self.obstacles:
                # Hit an obstacle — stay in place, get penalty
                reward = -1.0
                terminated = False
            elif new_pos == self.goal_pos:
                # Reached the goal
                self._agent_pos = new_pos
                reward = 1.0
                terminated = True
            else:
                # Valid move
                self._agent_pos = new_pos
                reward = -0.1
                terminated = False
        else:
            # Hit a wall — stay in place, step penalty
            reward = -0.1
            terminated = False

        # Check if we've exceeded max steps
        truncated = self._step_count >= self.max_steps and not terminated

        # Track metrics in the base class
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
        return pos[0] * self.grid_size + pos[1]

    def _decode_pos(self, encoded: int) -> tuple[int, int]:
        """Decode a single integer back to (row, col)."""
        return (encoded // self.grid_size, encoded % self.grid_size)

    def _get_info(self) -> dict:
        """Return extra information about the current state."""
        return {
            "agent_pos": self._agent_pos,
            "goal_pos": self.goal_pos,
            "steps": self._step_count,
        }

    def render(self, mode: str = "ascii") -> str | None:
        """Render the grid world as ASCII art.

        Returns:
            A string representation of the grid, or None.
        """
        if mode != "ascii":
            return None

        lines = []
        border = "┌" + "───" * self.grid_size + "┐"
        lines.append(border)

        for row in range(self.grid_size):
            line = "│"
            for col in range(self.grid_size):
                pos = (row, col)
                if pos == self._agent_pos and pos == self.goal_pos:
                    cell = " ★ "  # agent on goal
                elif pos == self._agent_pos:
                    cell = " A "
                elif pos == self.goal_pos:
                    cell = " G "
                elif pos in self.obstacles:
                    cell = " ■ "
                else:
                    cell = " · "
                line += cell
            line += "│"
            lines.append(line)

        border = "└" + "───" * self.grid_size + "┘"
        lines.append(border)

        output = "\n".join(lines)
        print(output)
        return output
