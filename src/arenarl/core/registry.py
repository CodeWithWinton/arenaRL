"""
Registry — Environment registration and instantiation system.

The registry is a simple dictionary mapping string IDs to environment classes.
This lets users create environments by name: arenarl.make("GridWorld-v1")
without needing to import the class directly.

Functions:
    make(env_id, **kwargs)  — Create an environment instance by its ID
    register(env_id, cls)   — Register a new environment class
    list_envs()             — List all registered environment IDs
"""

from __future__ import annotations

from arenarl.core.base_env import BaseEnv


# Internal registry: maps env IDs to their classes
_REGISTRY: dict[str, type[BaseEnv]] = {}


def register(env_id: str, env_class: type[BaseEnv]) -> None:
    """Register an environment class with a string ID.

    Args:
        env_id: A unique identifier like "GridWorld-v1".
        env_class: The environment class (must inherit from BaseEnv).

    Raises:
        ValueError: If the ID is already registered or class is invalid.
    """
    if env_id in _REGISTRY:
        raise ValueError(
            f"Environment '{env_id}' is already registered. "
            f"Use a different ID or unregister it first."
        )

    if not (isinstance(env_class, type) and issubclass(env_class, BaseEnv)):
        raise ValueError(
            f"env_class must be a subclass of BaseEnv, got {env_class}"
        )

    _REGISTRY[env_id] = env_class


def make(env_id: str, **kwargs) -> BaseEnv:
    """Create an environment instance by its registered ID.

    Args:
        env_id: The registered environment ID (e.g. "GridWorld-v1").
        **kwargs: Arguments passed to the environment constructor.

    Returns:
        An instance of the requested environment.

    Raises:
        KeyError: If the environment ID is not registered.
    """
    if env_id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Environment '{env_id}' not found. Available: {available}"
        )

    return _REGISTRY[env_id](**kwargs)


def list_envs() -> list[str]:
    """List all registered environment IDs.

    Returns:
        A sorted list of registered environment ID strings.
    """
    return sorted(_REGISTRY.keys())
