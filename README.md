<p align="center">
  <h1 align="center">ArenaRL</h1>
  <p align="center">
    An open-source reinforcement learning simulation framework for education and research.
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://github.com/CodeWithWinton/arenarl/issues"><img src="https://img.shields.io/github/issues/CodeWithWinton/arenarl" alt="Issues"></a>
  </p>
</p>

---

ArenaRL provides lightweight, well-documented simulation environments for training and evaluating reinforcement learning agents. It ships with a Gymnasium-compatible API, built-in metrics tracking, and terminal-based rendering — with no heavy dependencies.

## Installation

```bash
pip install arenarl
```

For development:

```bash
git clone https://github.com/CodeWithWinton/arenarl.git
cd arenarl
pip install -e ".[dev]"
```

## Quick Start

```python
import arenarl

env = arenarl.make("GridWorld-v1")
state, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        state, info = env.reset()

metrics = env.get_metrics()
print(metrics.summary())
```

## Environments

| Environment | ID | Action Space | Description |
|-------------|-----|-------------|-------------|
| Grid World | `GridWorld-v1` | Discrete | Navigate an N×N grid to reach a goal while avoiding obstacles |
| Maze | `Maze-v1` | Discrete | Solve procedurally generated mazes |
| Snake | `Snake-v1` | Discrete | Classic snake game with growing state complexity |
| Car Track | `CarTrack-v1` | Continuous | 2D car driving with velocity and steering physics |

## Custom Environments

ArenaRL supports user-defined environments. Inherit from `BaseEnv`, implement `reset()` and `step()`, and register:

```python
from arenarl import BaseEnv, register
from arenarl.core.spaces import Discrete

class MyEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(2)
        self.observation_space = Discrete(5)

    def reset(self, seed=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        return 0, 0.0, False, False, {}

register("MyEnv-v1", MyEnv)
```

See the [custom environments guide](docs/custom_environments.md) for details.

## Data Export

```python
env.export_data("results.csv")
env.export_data("results.json")
```

## Documentation

- [Getting Started](docs/getting_started.md)
- [Environment Reference](docs/environments.md)
- [Custom Environments](docs/custom_environments.md)

## Testing

```bash
pytest tests/ -v
```

## Roadmap

- Core framework (BaseEnv, Spaces, Registry)
- GridWorld, Maze, Snake, and CarTrack environments
- Metrics collection and data export
- Terminal-based ASCII rendering
- AI-powered custom environment generation

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE).
