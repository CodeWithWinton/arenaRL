"""
ArenaRL — An open-source educational reinforcement learning simulation framework.

Usage:
    import arenarl

    env = arenarl.make("GridWorld-v1")
    state, info = env.reset()
    state, reward, terminated, truncated, info = env.step(action)
"""

__version__ = "0.1.0"

# Core API will be exposed here once implemented.
# from arenarl.core.registry import make, register, list_envs
# from arenarl.core.base_env import BaseEnv
# import arenarl.envs
