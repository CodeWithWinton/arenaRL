import pytest
import numpy as np
from arenarl.core.spaces import Discrete, Box, MultiDiscrete

def test_discrete():
    space = Discrete(4)
    assert space.n == 4
    
    # Test containment
    assert space.contains(0)
    assert space.contains(3)
    assert not space.contains(4)
    assert not space.contains(-1)
    
    # Test sampling
    samples = [space.sample() for _ in range(100)]
    assert all(0 <= s < 4 for s in samples)
    
    # Equality
    assert Discrete(4) == Discrete(4)
    assert Discrete(4) != Discrete(5)

def test_box():
    space = Box(low=-1.0, high=1.0, shape=(2,))
    
    # Containment
    assert space.contains(np.array([0.5, -0.5]))
    assert not space.contains(np.array([1.5, 0.0]))
    assert not space.contains(np.array([0.0]))
    
    # Sampling
    sample = space.sample()
    assert sample.shape == (2,)
    assert np.all(sample >= -1.0) and np.all(sample <= 1.0)
    
    # Equality
    space2 = Box(low=-1.0, high=1.0, shape=(2,))
    assert space == space2
    space3 = Box(low=-1.0, high=2.0, shape=(2,))
    assert space != space3

def test_multidiscrete():
    space = MultiDiscrete([2, 3])
    
    # Containment
    assert space.contains([0, 2])
    assert space.contains(np.array([1, 1]))
    assert not space.contains([2, 2])
    assert not space.contains([0, 3])
    
    # Sampling
    sample = space.sample()
    assert sample.shape == (2,)
    assert 0 <= sample[0] < 2
    assert 0 <= sample[1] < 3
    
    # Equality
    assert space == MultiDiscrete([2, 3])
    assert space != MultiDiscrete([3, 2])
