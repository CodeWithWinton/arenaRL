# Getting Started with ArenaRL

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

## Your First Environment

```python
import arenarl

# See all available environments
print(arenarl.list_envs())

# Create an environment
env = arenarl.make("GridWorld-v1")

# Reset to get initial state
state, info = env.reset()

# Run a simple loop
for step in range(100):
    action = env.action_space.sample()  # random action
    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        state, info = env.reset()

# Check your results
metrics = env.get_metrics()
print(metrics.summary())
```

## Next Steps

- See [Environments](environments.md) for details on each built-in environment
- See [Custom Environments](custom_environments.md) to build your own
