import pytest
from arenarl.core.base_env import BaseEnv
from arenarl.core.spaces import Discrete

class MockEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(2)
        self.observation_space = Discrete(2)

    def reset(self, seed=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        super().step(action)
        self._track_step(1.0, True, False)
        return 0, 1.0, True, False, {}

def test_base_env_must_reset():
    env = MockEnv()
    with pytest.raises(RuntimeError, match="must be reset before calling step"):
        env.step(0)
        
def test_metrics_tracking():
    env = MockEnv()
    env.reset()
    env.step(0)
    
    metrics = env.get_metrics()
    assert metrics["episode_count"] == 1
    assert metrics["mean_reward"] == 1.0
    assert metrics["mean_length"] == 1.0
