# Custom Environments

ArenaRL makes it easy to create your own environments. Just inherit from `BaseEnv` and implement a few methods.

## Basic Template

```python
from arenarl import BaseEnv, register
from arenarl.core.spaces import Discrete, Box
import numpy as np


class MyEnv(BaseEnv):
    """A custom environment."""

    def __init__(self):
        super().__init__()
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=10, shape=(2,))

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0])
        return self.state.copy(), {}

    def step(self, action):
        # Apply action logic here
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return self.state.copy(), reward, terminated, truncated, info

    def render(self, mode="ascii"):
        print(f"State: {self.state}")


# Register your environment
register("MyEnv-v1", MyEnv)
```

## Guidelines

- Always call `super().__init__()` and `super().reset(seed=seed)`
- Return copies of state arrays to prevent external mutation
- Define `action_space` and `observation_space` in `__init__`
- Use `self.np_random` (provided by BaseEnv) for all random operations

<!-- TODO: Expand with more examples after implementation -->
