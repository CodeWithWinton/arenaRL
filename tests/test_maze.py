import pytest
from arenarl import make

def test_maze_even_size_warning():
    # Should warn and increment size to 7
    with pytest.warns(UserWarning, match="is even, incrementing"):
        env = make("Maze-v1", maze_size=6)
    assert env.maze_size == 7

def test_maze_fixed_seed():
    env = make("Maze-v1", maze_size=5)
    env.reset(seed=42)
    grid1 = env._grid.copy()
    
    env.reset(seed=42)
    grid2 = env._grid.copy()
    
    assert (grid1 == grid2).all()

def test_maze_step_logic():
    env = make("Maze-v1", maze_size=5)
    env.reset(seed=42)
    
    # Starting at (1,1). Try moving UP (0) into a wall.
    obs, reward, term, trunc, info = env.step(0)
    assert reward == -1.0
    assert not term
    assert info["agent_pos"] == (1, 1)
