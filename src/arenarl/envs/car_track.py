"""
CarTrack-v1 — A continuous physics environment for car racing.

The agent controls a car driving around a track. The car has momentum,
velocity, and a steering angle. The goal is to complete laps as fast
as possible without crashing into the track boundaries.

State (Continuous Box):
    [0] x position
    [1] y position
    [2] velocity
    [3] heading angle (radians)

Actions (Discrete):
    0 = Coast (do nothing)
    1 = Accelerate
    2 = Brake
    3 = Steer Left
    4 = Steer Right

Rewards:
    +10.0  — completing a lap
    -1.0   — crashing (driving off track)
    +0.1   — moving forward (progress reward)
    -0.01  — time penalty per step
"""

from __future__ import annotations

import math
import numpy as np

from arenarl.core.base_env import BaseEnv
from arenarl.core.spaces import Box, Discrete

# Actions
COAST = 0
ACCEL = 1
BRAKE = 2
LEFT = 3
RIGHT = 4

# Physics Constants
MAX_VELOCITY = 5.0
ACCELERATION_FORCE = 0.5
BRAKING_FORCE = 1.0
FRICTION = 0.05
TURN_RATE = math.radians(15)  # 15 degrees per step

class CarTrackEnv(BaseEnv):
    """Car racing environment with continuous physics.

    The car drives on an oval track. The state space is continuous
    (x, y, velocity, angle), but the action space is discrete to make
    it accessible for basic algorithms like DQN.
    """

    def __init__(
        self,
        track_radius_outer: float = 20.0,
        track_radius_inner: float = 10.0,
        max_steps: int = 1000,
    ):
        super().__init__()

        self.track_radius_outer = track_radius_outer
        self.track_radius_inner = track_radius_inner
        self.max_steps = max_steps

        # Action space: 5 discrete commands
        self.action_space = Discrete(5)

        # Observation space: Continuous Box
        # [x, y, velocity, angle]
        high = np.array([
            track_radius_outer * 2,    # Max X
            track_radius_outer * 2,    # Max Y
            MAX_VELOCITY,              # Max Velocity
            math.pi * 2                # Max Angle
        ], dtype=np.float32)
        
        low = np.array([
            -track_radius_outer * 2,   # Min X
            -track_radius_outer * 2,   # Min Y
            -MAX_VELOCITY,             # Min Velocity (reverse)
            -math.pi * 2               # Min Angle
        ], dtype=np.float32)

        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        # State variables
        self._x: float = 0.0
        self._y: float = 0.0
        self._velocity: float = 0.0
        self._angle: float = 0.0
        self._step_count: int = 0
        
        # Track progress (angles around the origin)
        self._last_angle_from_center: float = 0.0
        self._total_angle_traveled: float = 0.0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the car to the starting line."""
        super().reset(seed=seed)

        # Start on the right side of the track, facing UP (Y+ direction)
        start_radius = (self.track_radius_inner + self.track_radius_outer) / 2.0
        self._x = start_radius
        self._y = 0.0
        self._velocity = 0.0
        self._angle = math.pi / 2.0  # Facing UP (90 degrees)
        self._step_count = 0
        
        self._last_angle_from_center = math.atan2(self._y, self._x)
        self._total_angle_traveled = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Update car physics by one time step."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}.")
            
        super().step(action)
        self._step_count += 1

        # 1. Apply Actions (Steering)
        if action == LEFT:
            self._angle += TURN_RATE
        elif action == RIGHT:
            self._angle -= TURN_RATE
            
        # Normalize car angle between -2PI and 2PI
        self._angle = self._angle % (2 * math.pi)

        # 2. Apply Actions (Acceleration/Braking)
        if action == ACCEL:
            self._velocity += ACCELERATION_FORCE
        elif action == BRAKE:
            self._velocity -= BRAKING_FORCE

        # 3. Apply Friction (slows down naturally)
        if self._velocity > 0:
            self._velocity = max(0.0, self._velocity - FRICTION)
        elif self._velocity < 0:
            self._velocity = min(0.0, self._velocity + FRICTION)

        # Clamp max velocity
        self._velocity = max(-MAX_VELOCITY, min(MAX_VELOCITY, self._velocity))

        # 4. Update Position using Trigonometry
        self._x += math.cos(self._angle) * self._velocity
        self._y += math.sin(self._angle) * self._velocity

        # 5. Calculate Distance from Center (to check if on track)
        distance_from_center = math.hypot(self._x, self._y)

        terminated = False
        reward = -0.01  # Small time penalty
        
        # Check if crashed (off the track)
        if distance_from_center > self.track_radius_outer or distance_from_center < self.track_radius_inner:
            reward = -1.0
            terminated = True
            
        # Calculate Progress (did we drive forward around the circle?)
        current_angle = math.atan2(self._y, self._x)
        
        # Calculate angle difference (handling the wrap-around at PI / -PI)
        angle_diff = current_angle - self._last_angle_from_center
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        self._last_angle_from_center = current_angle
        
        # Only reward moving forward (counter-clockwise)
        if angle_diff > 0 and not terminated:
            # Reward proportional to distance traveled around the curve
            reward += angle_diff * 1.0  
            self._total_angle_traveled += angle_diff
            
            # Did we complete a full lap?
            if self._total_angle_traveled >= 2 * math.pi:
                reward += 10.0  # Lap complete bonus!
                terminated = True  # We'll end the episode on lap completion

        truncated = self._step_count >= self.max_steps and not terminated
        
        self._track_step(reward, terminated, truncated)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Return the continuous state vector."""
        return np.array([self._x, self._y, self._velocity, self._angle], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "x": self._x,
            "y": self._y,
            "velocity": self._velocity,
            "heading": self._angle,
            "distance_from_center": math.hypot(self._x, self._y),
            "lap_progress_pct": (self._total_angle_traveled / (2 * math.pi)) * 100,
        }

    def render(self, mode: str = "ascii") -> str | None:
        """Render a minimal top-down view of the track and car."""
        if mode != "ascii":
            return None
            
        grid_size = 21  # 21x21 characters
        center = grid_size // 2
        
        # Scale to map coordinates to grid
        scale = (self.track_radius_outer * 1.2) / center
        
        lines = []
        for r in range(grid_size):
            line = ""
            for c in range(grid_size):
                # Convert grid (r,c) to world (x,y)
                world_x = (c - center) * scale
                world_y = (center - r) * scale  # r=0 is top, y is positive up
                
                dist = math.hypot(world_x, world_y)
                
                # Check if this cell is where the car is
                car_dist = math.hypot(world_x - self._x, world_y - self._y)
                
                if car_dist < scale * 1.5:  # Tolerance for drawing car
                    # Draw an arrow pointing in the car's direction
                    # Normalize angle to 0 - 2pi
                    ang = self._angle % (2 * math.pi)
                    if math.pi/4 <= ang < 3*math.pi/4:
                        line += "↑ "
                    elif 3*math.pi/4 <= ang < 5*math.pi/4:
                        line += "← "
                    elif 5*math.pi/4 <= ang < 7*math.pi/4:
                        line += "↓ "
                    else:
                        line += "→ "
                # Draw Track Inner/Outer boundaries
                elif abs(dist - self.track_radius_inner) < scale / 2:
                    line += "· "
                elif abs(dist - self.track_radius_outer) < scale / 2:
                    line += "· "
                else:
                    line += "  "
                    
            lines.append(line)
            
        output = "\n".join(lines)
        return output
