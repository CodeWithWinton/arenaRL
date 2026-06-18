import pytest
from arenarl import make, list_envs, register
from arenarl.core.base_env import BaseEnv
from arenarl.core.spaces import Discrete

class DummyEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(2)
        self.observation_space = Discrete(2)

    def reset(self, seed=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        super().step(action)
        return 0, 0.0, True, False, {}

def test_registry():
    # Register custom env
    register("Dummy-v0", DummyEnv)
    
    # Check listing
    envs = list_envs()
    assert "Dummy-v0" in envs
    assert "GridWorld-v1" in envs
    assert "Maze-v1" in envs
    assert "Snake-v1" in envs
    assert "CarTrack-v1" in envs
    
    # Check make
    env = make("Dummy-v0")
    assert isinstance(env, DummyEnv)
    
    # Check failure
    with pytest.raises(KeyError):
        make("NonExistent-v0")
        
    with pytest.raises(ValueError):
        register("Dummy-v0", DummyEnv)  # Already registered
