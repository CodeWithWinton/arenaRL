"""
BaseEnv — Abstract base class for all ArenaRL environments.

Every environment in ArenaRL inherits from this class. It provides:
    - Abstract interface that all environments must follow
    - Automatic seeding for reproducibility
    - Built-in episode tracking (rewards, steps, episode count)

Subclasses must implement:
    - reset(seed=None) -> (observation, info)
    - step(action) -> (observation, reward, terminated, truncated, info)

Subclasses may optionally implement:
    - render(mode="ascii") -> str | None
    - close() -> None
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from arenarl.core.spaces import Space
from arenarl.utils.seeding import create_rng


class BaseEnv(ABC):
    """Abstract base class for all ArenaRL environments.

    Attributes:
        action_space: Defines valid actions the agent can take.
        observation_space: Defines the structure of observations.
        np_random: Seeded random number generator for reproducibility.
    """

    # Subclasses must set these in __init__
    action_space: Space
    observation_space: Space

    def __init__(self):
        # Random number generator — seeded on reset()
        self.np_random: np.random.Generator = np.random.default_rng()

        # Episode tracking
        self._current_episode_reward: float = 0.0
        self._current_episode_steps: int = 0
        self._episode_count: int = 0

        # History of completed episodes
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

        # Track whether reset() has been called before step()
        self._has_reset: bool = False

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple:
        """Reset the environment to an initial state.

        This must be called before the first step(). Subclasses must call
        super().reset(seed=seed) at the start of their reset() method.

        Args:
            seed: Optional seed for the random number generator.
                  Pass the same seed to reproduce identical episodes.

        Returns:
            A tuple of (observation, info) where:
                - observation: The initial state of the environment
                - info: A dict with optional extra information
        """
        # Create a seeded random generator
        self.np_random, _ = create_rng(seed)

        # Propagate the RNG to spaces so space.sample() is also reproducible
        self.action_space.np_random = self.np_random
        self.observation_space.np_random = self.np_random

        # Reset episode counters for the new episode
        self._current_episode_reward = 0.0
        self._current_episode_steps = 0
        self._has_reset = True

    @abstractmethod
    def step(self, action) -> tuple:
        """Take one step in the environment.

        Subclasses implement game logic here. After computing the result,
        subclasses must call super()._track_step(reward, terminated, truncated)
        to record metrics.

        Args:
            action: The action to take. Must be valid in self.action_space.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info) where:
                - observation: The new state after the action
                - reward: A float reward signal
                - terminated: True if the episode ended naturally (goal/failure)
                - truncated: True if the episode was cut short (max steps)
                - info: A dict with optional extra information
        """
        if not self._has_reset:
            raise RuntimeError(
                "Environment must be reset before calling step(). Call env.reset() first."
            )

    def _track_step(self, reward: float, terminated: bool, truncated: bool) -> None:
        """Record metrics for the current step. Called by subclasses after step logic."""
        self._current_episode_reward += reward
        self._current_episode_steps += 1

        # If episode is over, save it to history
        if terminated or truncated:
            self._episode_rewards.append(self._current_episode_reward)
            self._episode_lengths.append(self._current_episode_steps)
            self._episode_count += 1
            self._has_reset = False

    def render(self, mode: str = "ascii") -> str | None:
        """Render the environment. Override in subclasses for visualization."""
        pass

    def close(self) -> None:
        """Clean up resources. Override if the environment uses external resources."""
        pass

    def get_metrics(self) -> dict:
        """Return a summary of training metrics.

        Returns:
            A dict containing episode rewards, lengths, and summary statistics.
        """
        metrics = {
            "episode_count": self._episode_count,
            "episode_rewards": list(self._episode_rewards),
            "episode_lengths": list(self._episode_lengths),
        }

        if self._episode_rewards:
            rewards = np.array(self._episode_rewards)
            metrics["mean_reward"] = float(np.mean(rewards))
            metrics["std_reward"] = float(np.std(rewards))
            metrics["min_reward"] = float(np.min(rewards))
            metrics["max_reward"] = float(np.max(rewards))
            metrics["mean_length"] = float(np.mean(self._episode_lengths))

        return metrics

    def export_data(self, path: str) -> None:
        """Export episode data to a file.

        Supports CSV and JSON formats based on file extension.

        Args:
            path: File path ending in .csv or .json.
        """
        import json

        metrics = self.get_metrics()

        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(metrics, f, indent=2)

        elif path.endswith(".csv"):
            with open(path, "w") as f:
                f.write("episode,reward,length\n")
                for i, (r, length) in enumerate(
                    zip(metrics["episode_rewards"], metrics["episode_lengths"])
                ):
                    f.write(f"{i + 1},{r},{length}\n")

        else:
            raise ValueError(f"Unsupported file format: {path}. Use .csv or .json.")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} | action_space={self.action_space}>"
