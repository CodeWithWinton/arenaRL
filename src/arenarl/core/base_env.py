"""
BaseEnv — Abstract base class for all ArenaRL environments.

All environments must inherit from this class and implement:
    - reset(seed=None) -> (observation, info)
    - step(action) -> (observation, reward, terminated, truncated, info)

Optional:
    - render(mode="ascii") -> str | None
    - close() -> None
"""

# TODO: Implement BaseEnv class (Phase 1)
