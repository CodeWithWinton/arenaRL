import pytest
from arenarl import make

def test_snake_initialization():
    env = make("Snake-v1", grid_size=6)
    obs, info = env.reset(seed=42)
    
    assert info["snake_length"] == 3
    assert env.observation_space.contains(obs)
    
def test_snake_movement_and_apple():
    env = make("Snake-v1", grid_size=6)
    env.reset(seed=42)
    
    # Manually place apple right in front of snake (head is at 3,3 facing right)
    # So right is 3,4.
    env._apple_pos = (3, 4)
    
    obs, reward, term, trunc, info = env.step(3)  # RIGHT
    
    assert reward == 1.0  # Ate apple
    assert info["snake_length"] == 4
    assert not term
    
def test_snake_collision():
    env = make("Snake-v1", grid_size=4)
    env.reset()
    env._snake = [(1, 1), (1, 2), (1, 3)]  # Facing left, length 3
    env._current_direction = 2  # LEFT
    
    # Move UP to (0, 1), then UP to (-1, 1) which is a wall
    env.step(0)  # UP
    obs, reward, term, trunc, info = env.step(0)  # UP into wall
    
    assert reward == -1.0
    assert term
