"""
Built-in environments for ArenaRL.

Environments are auto-registered when this module is imported.
"""

from arenarl.core.registry import register
from arenarl.envs.grid_world import GridWorldEnv
from arenarl.envs.maze import MazeEnv
from arenarl.envs.snake import SnakeEnv
from arenarl.envs.car_track import CarTrackEnv

# Register all built-in environments
register("GridWorld-v1", GridWorldEnv)
register("Maze-v1", MazeEnv)
register("Snake-v1", SnakeEnv)
register("CarTrack-v1", CarTrackEnv)
