"""
Snake-v1 — Classic Snake game for reinforcement learning.

The agent controls a snake that grows when it eats apples.
The episode ends if the snake hits a wall or its own body.

State:  An N×N grid representing the board:
        0 = Empty
        1 = Snake Head
        2 = Snake Body
        3 = Apple

Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
         Note: The snake cannot reverse directly into its own body.

Rewards:
    +1.0   — eating an apple
    -1.0   — dying (hitting wall or self)
     0.0   — regular step
"""

from __future__ import annotations

import numpy as np

from arenarl.core.base_env import BaseEnv
from arenarl.core.spaces import Box, Discrete

# Grid values
EMPTY = 0
HEAD = 1
BODY = 2
APPLE = 3

# Actions
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

# Opposite directions (cannot reverse into yourself)
OPPOSITES = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}

class SnakeEnv(BaseEnv):
    """Classic Snake environment.

    The observation is a 2D grid of the game state.
    This teaches RL agents spatial awareness, planning, and dealing
    with a changing state space (growing body).
    """

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 1000,
    ):
        super().__init__()

        if grid_size < 4:
            raise ValueError(f"grid_size must be at least 4, got {grid_size}")

        self.grid_size = grid_size
        self.max_steps = max_steps

        # Action space: 4 directions
        self.action_space = Discrete(4)

        # Observation space: 2D grid of integers (0 to 3)
        self.observation_space = Box(
            low=0,
            high=3,
            shape=(grid_size, grid_size),
            dtype=np.int8
        )

        # Internal state
        # The snake body is a list of (row, col) tuples.
        # Index 0 is the head, the rest is the body trailing behind.
        self._snake: list[tuple[int, int]] = []
        self._apple_pos: tuple[int, int] = (0, 0)
        self._current_direction: int = RIGHT
        self._step_count: int = 0

    def _place_apple(self) -> None:
        """Place an apple in a random empty cell."""
        empty_cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) not in self._snake:
                    empty_cells.append((r, c))

        if not empty_cells:
            # Snake fills the entire board! You win!
            self._apple_pos = (-1, -1)
            return

        idx = self.np_random.integers(len(empty_cells))
        self._apple_pos = empty_cells[idx]

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the Snake environment.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        self._step_count = 0

        # Start in the middle facing right, length 3
        mid_r = self.grid_size // 2
        mid_c = self.grid_size // 2

        self._snake = [
            (mid_r, mid_c),         # Head
            (mid_r, mid_c - 1),     # Body 1
            (mid_r, mid_c - 2),     # Body 2
        ]
        self._current_direction = RIGHT

        self._place_apple()

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take one step in the game.

        Args:
            action: An integer 0-3 representing up/down/left/right.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {{0, 1, 2, 3}}.")

        super().step(action)
        self._step_count += 1

        # Prevent 180-degree immediate reversal
        if OPPOSITES[action] == self._current_direction:
            # Ignore the reversal, keep moving in current direction
            action = self._current_direction

        self._current_direction = action

        # Calculate new head position
        head_r, head_c = self._snake[0]
        dr, dc = MOVES[action]
        new_head = (head_r + dr, head_c + dc)

        # 1. Check Wall Collision
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            reward = -1.0
            terminated = True
            self._track_step(reward, terminated, False)
            return self._get_obs(), reward, terminated, False, self._get_info()

        # 2. Check Self Collision
        # Note: The tail will move forward this step, so hitting the current tail is safe
        # unless we just ate an apple and the tail is staying put.
        # But to be perfectly precise: if the new head is in the body
        # (excluding the very tip of the tail)
        if new_head in self._snake[:-1]:
            reward = -1.0
            terminated = True
            self._track_step(reward, terminated, False)
            return self._get_obs(), reward, terminated, False, self._get_info()

        # Move the snake
        self._snake.insert(0, new_head)  # Add new head

        # 3. Check Apple
        reward = 0.0
        terminated = False

        if new_head == self._apple_pos:
            reward = 1.0
            self._place_apple()

            # Check for perfect game win condition
            if self._apple_pos == (-1, -1):
                reward += 10.0  # Big bonus for perfect game
                terminated = True
        else:
            # Didn't eat, remove the tail tip so snake length stays the same
            self._snake.pop()

        truncated = self._step_count >= self.max_steps and not terminated

        self._track_step(reward, terminated, truncated)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Create the 2D grid observation."""
        obs = np.full((self.grid_size, self.grid_size), EMPTY, dtype=np.int8)

        # Place apple
        if self._apple_pos != (-1, -1):
            obs[self._apple_pos[0], self._apple_pos[1]] = APPLE

        # Place snake body
        for r, c in self._snake[1:]:
            obs[r, c] = BODY

        # Place snake head (overwrites body if they overlap during death step)
        if self._snake:
            head_r, head_c = self._snake[0]
            if 0 <= head_r < self.grid_size and 0 <= head_c < self.grid_size:
                obs[head_r, head_c] = HEAD

        return obs

    def _get_info(self) -> dict:
        """Return extra information."""
        return {
            "snake_length": len(self._snake),
            "steps": self._step_count,
        }

    def render(self, mode: str = "ascii") -> str | None:
        """Render the Snake game as ASCII art."""
        if mode != "ascii":
            return None

        obs = self._get_obs()
        lines = []

        border = "┌" + "──" * self.grid_size + "┐"
        lines.append(border)

        for r in range(self.grid_size):
            line = "│"
            for c in range(self.grid_size):
                val = obs[r, c]
                if val == EMPTY:
                    line += " ·"
                elif val == APPLE:
                    line += " 🍎"
                elif val == HEAD:
                    line += " ▣"
                elif val == BODY:
                    line += " ▢"
            line += "│"
            lines.append(line)

        border = "└" + "──" * self.grid_size + "┘"
        lines.append(border)

        return "\n".join(lines)
