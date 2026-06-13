"""
ArenaRL — An open-source educational reinforcement learning simulation framework.

Usage:
    import arenarl

    env = arenarl.make("GridWorld-v1")
    state, info = env.reset()
    state, reward, terminated, truncated, info = env.step(action)
"""

__version__ = "0.1.0"

from arenarl.core.base_env import BaseEnv  # noqa: F401
from arenarl.core.registry import list_envs, make, register  # noqa: F401

# Auto-register all built-in environments
import arenarl.envs  # noqa: F401, E402
