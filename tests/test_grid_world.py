import pytest
from arenarl import make

def test_gridworld_basic():
    env = make("GridWorld-v1", grid_size=4)
    obs, info = env.reset(seed=42)
    assert obs == 0  # Starts at (0, 0)
    
    # Try invalid action
    with pytest.raises(ValueError):
        env.step(99)
        
    # Move right
    obs, reward, terminated, truncated, info = env.step(3)  # RIGHT
    assert obs == 1  # (0, 1) -> 0*4 + 1
    assert reward == -0.1
    assert not terminated
    
def test_gridworld_goal():
    env = make("GridWorld-v1", grid_size=3, obstacles=[])
    env.reset(seed=42)
    
    # Grid is 3x3. Start is 0,0. Goal is 2,2.
    env.step(3)  # R
    env.step(3)  # R
    env.step(1)  # D
    obs, reward, terminated, truncated, info = env.step(1)  # D -> reaches (2, 2)
    
    assert terminated
    assert reward == 1.0
    
def test_gridworld_obstacles():
    env = make("GridWorld-v1", grid_size=3, obstacles=[(0, 1)])
    env.reset()
    
    # Try to move right into obstacle
    obs, reward, terminated, truncated, info = env.step(3)
    assert obs == 0  # Stays in place
    assert reward == -1.0
    
def test_gridworld_validation():
    with pytest.raises(ValueError, match="start_pos and goal_pos must be different"):
        make("GridWorld-v1", grid_size=3, start_pos=(0,0), goal_pos=(0,0))
