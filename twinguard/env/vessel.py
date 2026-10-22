"""
3-DOF Vessel Kinematics and Dynamics Model with Nomoto Steering.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import copy
import numpy as np


@dataclass
class VesselState:
    """
    State representation of a surface vessel in 2D Euclidean space.
    Course convention: 0 rad = North (+y), pi/2 rad = East/Starboard (+x).
    """
    id: str = "vessel"
    x: float = 0.0
    y: float = 0.0
    speed: float = 0.0          # Surge speed (m/s)
    course: float = 0.0         # Heading angle in radians [0, 2*pi)
    rudder_angle: float = 0.0   # Current rudder angle in radians
    yaw_rate: float = 0.0       # Current turning rate (rad/s)
    length: float = 50.0        # Vessel length overall (m)
    beam: float = 10.0          # Vessel beam/width (m)
    max_speed: float = 12.0     # Maximum forward speed (m/s)
    max_rudder_rad: float = field(default_factory=lambda: float(np.deg2rad(35.0)))
    max_yaw_rate: float = field(default_factory=lambda: float(np.deg2rad(15.0)))
    k_nomoto: float = 0.15      # Nomoto gain constant
    t_nomoto: float = 3.5       # Nomoto time constant (s)

    @property
    def position(self) -> np.ndarray:
        """2D position array [x, y] in meters."""
        return np.array([self.x, self.y], dtype=np.float64)

    @property
    def velocity_vector(self) -> np.ndarray:
        """2D velocity vector [vx, vy] in m/s."""
        vx = self.speed * np.sin(self.course)
        vy = self.speed * np.cos(self.course)
        return np.array([vx, vy], dtype=np.float64)

    def copy(self) -> VesselState:
        """Return an independent deep copy of this vessel state."""
        return copy.deepcopy(self)

    def step(
        self,
        commanded_speed: float,
        commanded_rudder: float,
        dt: float = 1.0,
        drift_x: float = 0.0,
        drift_y: float = 0.0
    ) -> None:
        """
        Advances vessel state by dt seconds using 1st-order Nomoto steering dynamics.
        """
        # 1. Rudder rate limit
        cmd_rudder = float(np.clip(commanded_rudder, -self.max_rudder_rad, self.max_rudder_rad))
        max_rudder_rate = float(np.deg2rad(10.0)) * dt
        dr = float(np.clip(cmd_rudder - self.rudder_angle, -max_rudder_rate, max_rudder_rate))
        self.rudder_angle += dr

        # 2. Nomoto steering dynamics: T * dr/dt + r = K * delta
        r_dot = (self.k_nomoto * self.rudder_angle - self.yaw_rate) / max(self.t_nomoto, 0.1)
        self.yaw_rate = float(np.clip(
            self.yaw_rate + r_dot * dt,
            -self.max_yaw_rate,
            self.max_yaw_rate
        ))

        # 3. Update course
        self.course = float((self.course + self.yaw_rate * dt) % (2 * np.pi))

        # 4. Engine throttle and acceleration limit
        cmd_speed = float(np.clip(commanded_speed, 0.0, self.max_speed))
        max_accel = 1.5 * dt
        ds = float(np.clip(cmd_speed - self.speed, -max_accel, max_accel))
        self.speed = float(np.clip(self.speed + ds, 0.0, self.max_speed))

        # 5. Position update with environmental drift
        vx = self.speed * np.sin(self.course) + drift_x
        vy = self.speed * np.cos(self.course) + drift_y
        self.x += vx * dt
        self.y += vy * dt