"""
Spaces — Define action and observation spaces for environments.

Spaces describe the valid structure of actions and observations.
Each space supports sampling, containment checks, and shape inspection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Space:
    """Base class for all spaces."""

    def __init__(self, shape: tuple[int, ...] | None = None, dtype: np.dtype | type = np.float32):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._np_random: np.random.Generator | None = None

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    def sample(self) -> int | NDArray:
        """Return a random valid value from this space."""
        raise NotImplementedError

    def contains(self, x) -> bool:
        """Check if x is a valid member of this space."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__


class Discrete(Space):
    """A discrete space of n values: {0, 1, ..., n-1}.

    Useful for environments with a finite set of actions.

    Example:
        >>> space = Discrete(4)  # 4 actions: up, down, left, right
        >>> space.sample()
        2
        >>> space.contains(3)
        True
        >>> space.contains(5)
        False
    """

    def __init__(self, n: int):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.n = n
        super().__init__(shape=(), dtype=np.int64)

    def sample(self) -> int:
        """Return a random integer in {0, 1, ..., n-1}."""
        return int(self.np_random.integers(self.n))

    def contains(self, x) -> bool:
        """Check if x is a valid discrete value."""
        if isinstance(x, (np.generic, np.ndarray)):
            if np.ndim(x) != 0:
                return False
            x = int(x)
        return isinstance(x, int) and not isinstance(x, bool) and 0 <= x < self.n

    def __eq__(self, other) -> bool:
        return isinstance(other, Discrete) and self.n == other.n

    def __hash__(self) -> int:
        return hash(("Discrete", self.n))

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class Box(Space):
    """A continuous n-dimensional space with element-wise bounds.

    Each element of the space is bounded by [low, high].

    Example:
        >>> space = Box(low=-1.0, high=1.0, shape=(3,))
        >>> space.sample()  # random array of shape (3,) in [-1, 1]
        array([0.23, -0.85, 0.11])
        >>> space.contains(np.array([0.5, 0.5, 0.5]))
        True
    """

    def __init__(
        self,
        low: float | NDArray,
        high: float | NDArray,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | type = np.float32,
    ):
        if shape is None:
            if isinstance(low, np.ndarray):
                shape = low.shape
            elif isinstance(high, np.ndarray):
                shape = high.shape
            else:
                shape = ()

        self.low = np.full(shape, low, dtype=dtype) if not isinstance(low, np.ndarray) \
            else low.astype(dtype, copy=True)
        self.high = np.full(shape, high, dtype=dtype) if not isinstance(high, np.ndarray) \
            else high.astype(dtype, copy=True)

        if self.low.shape != shape or self.high.shape != shape:
            raise ValueError(
                f"low shape {self.low.shape} and high shape {self.high.shape} "
                f"must match shape {shape}"
            )

        if np.any(self.low > self.high):
            raise ValueError("All low values must be <= corresponding high values")

        super().__init__(shape=shape, dtype=dtype)

    def sample(self) -> NDArray:
        """Return a random array uniformly sampled within bounds."""
        return self.np_random.uniform(
            low=self.low, high=self.high, size=self.shape
        ).astype(self.dtype)

    def contains(self, x) -> bool:
        """Check if x falls within the bounded space."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=self.dtype)
        return (
            x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def __repr__(self) -> str:
        return f"Box(low={self.low.min()}, high={self.high.max()}, shape={self.shape})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Box)
            and self.shape == other.shape
            and self.dtype == other.dtype
            and np.array_equal(self.low, other.low)
            and np.array_equal(self.high, other.high)
        )

    def __hash__(self) -> int:
        return hash(("Box", self.shape, self.dtype))


class MultiDiscrete(Space):
    """Multiple independent discrete spaces combined into one.

    Each element i can take values in {0, 1, ..., nvec[i]-1}.

    Example:
        >>> space = MultiDiscrete([3, 2])
        >>> space.sample()  # e.g. array([2, 1])
        >>> space.contains(np.array([1, 0]))
        True
    """

    def __init__(self, nvec: list[int] | NDArray):
        self.nvec = np.asarray(nvec, dtype=np.int64)
        if np.any(self.nvec <= 0):
            raise ValueError(f"All values in nvec must be positive, got {nvec}")
        super().__init__(shape=(len(self.nvec),), dtype=np.int64)

    def sample(self) -> NDArray:
        """Return a random array with each element sampled from its range."""
        return np.array(
            [self.np_random.integers(n) for n in self.nvec], dtype=np.int64
        )

    def contains(self, x) -> bool:
        """Check if x is valid across all discrete dimensions."""
        arr = np.asarray(x)
        if arr.shape != self.shape or arr.dtype.kind not in ("i", "u"):
            return False
        return bool(np.all(arr >= 0) and np.all(arr < self.nvec))

    def __repr__(self) -> str:
        return f"MultiDiscrete({self.nvec.tolist()})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, MultiDiscrete)
            and np.array_equal(self.nvec, other.nvec)
        )

    def __hash__(self) -> int:
        return hash(("MultiDiscrete", tuple(self.nvec.tolist())))
