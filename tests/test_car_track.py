import pytest
from arenarl import make

def test_cartrack_initialization():
    env = make("CarTrack-v1")
    obs, info = env.reset()
    
    assert env.observation_space.contains(obs)
    assert info["velocity"] == 0.0
    
def test_cartrack_acceleration():
    env = make("CarTrack-v1")
    env.reset()
    
    # Action 1 is ACCEL
    obs, reward, term, trunc, info = env.step(1)
    assert info["velocity"] > 0.0
    
def test_cartrack_crash():
    env = make("CarTrack-v1", track_radius_outer=20.0, track_radius_inner=10.0)
    env.reset()
    
    # Start radius is 15.0
    # Let's teleport the car out of bounds to simulate a crash
    env._x = 30.0 
    obs, reward, term, trunc, info = env.step(0)  # COAST
    
    assert term
    assert reward == -1.0
