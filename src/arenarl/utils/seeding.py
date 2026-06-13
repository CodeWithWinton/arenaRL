"""
Seeding — Random seed management for reproducibility.

Provides utilities for creating seeded numpy random generators,
ensuring that environment behavior can be reproduced exactly.
"""

from __future__ import annotations

import numpy as np


def create_rng(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Create a seeded numpy random generator.

    Args:
        seed: Optional integer seed. If None, a random seed is generated.

    Returns:
        A tuple of (generator, seed_used).
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**63 - 1))
    seed = int(seed)
    rng = np.random.default_rng(seed)
    return rng, seed
