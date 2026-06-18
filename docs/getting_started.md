# Getting Started with ArenaRL

ArenaRL is a lightweight, easy-to-use reinforcement learning simulation framework designed for education and research. It provides classic environments without the heavy dependency footprint of larger frameworks.

## Installation

Install via pip:

```bash
pip install arenarl
```

## Basic Usage

The core API is designed to be familiar if you have used Gymnasium.

### 1. Creating an Environment

Use `arenarl.make()` to instantiate an environment by its registered ID.

```python
import arenarl

# Create a standard 5x5 GridWorld
env = arenarl.make("GridWorld-v1")

# You can also pass custom configuration parameters
env = arenarl.make("GridWorld-v1", grid_size=10, random_obstacles=True)
```

### 2. The RL Loop

Interact with the environment using `reset()` and `step()`.

```python
# Reset returns the initial observation and an info dictionary
obs, info = env.reset(seed=42)

done = False
while not done:
    # Sample a random action from the environment's action space
    action = env.action_space.sample()
    
    # Take a step
    obs, reward, terminated, truncated, info = env.step(action)
    
    # An episode ends if it is terminated (goal reached/crashed) or truncated (timeout)
    done = terminated or truncated

# You can render the final state to the terminal
env.render()
```

### 3. Collecting Metrics

ArenaRL automatically tracks your agent's performance (rewards and episode lengths) internally. 

```python
metrics = env.get_metrics()
print(f"Total episodes played: {metrics['episode_count']}")
print(f"Mean reward: {metrics['mean_reward']}")

# Export data to CSV or JSON for external analysis
env.export_data("results.csv")
```

### 4. Plotting

ArenaRL includes built-in plotting utilities to visualize learning progress.

```python
from arenarl.utils.plotting import plot_learning_curve

# Plots the learning curve and saves it
plot_learning_curve(metrics, save_path="learning_curve.png")
```

Next, check out the [Environment Reference](environments.md) to see what simulations are available!
