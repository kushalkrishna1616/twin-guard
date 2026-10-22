"""
Rule-Based Navigation Agents (Artificial Potential Field & Waypoint Tracker).
"""

from __future__ import annotations
import math
import numpy as np
from twinguard.agents.base import BaseAgent
from twinguard.env.vessel import VesselState


class APFMaritimeAgent(BaseAgent):
    """
    Artificial Potential Field (APF) and Corridor Navigation Agent.
    If colreg_aware=True: executes clean, realistic COLREG-compliant corridor passing.
    If colreg_aware=False: heads naively toward the waypoint goal, ignoring traffic rules.
    """
    def __init__(
        self,
        target_speed: float = 8.0,
        k_att: float = 1.0,
        k_rep: float = 150000.0,
        repulsion_radius: float = 600.0,
        colreg_aware: bool = True,
        name: str = "APFAgent"
    ):
        super().__init__(name=name)
        self.target_speed = target_speed
        self.k_att = k_att
        self.k_rep = k_rep
        self.repulsion_radius = repulsion_radius
        self.colreg_aware = colreg_aware

    def predict(self, observation: np.ndarray, info: dict | None = None) -> np.ndarray:
        if info is None or "own_pos" not in info:
            return np.array([self.target_speed, 0.0], dtype=np.float32)

        own_pos = np.array(info["own_pos"])
        own_course_rad = np.deg2rad(info["own_course_deg"])
        goal_pos = np.array(info.get("goal_pos", [0.0, 1200.0]))
        goal_x, goal_y = float(goal_pos[0]), float(goal_pos[1])

        # If naive (colreg_aware=False): track goal directly, ignoring other traffic
        if not self.colreg_aware:
            d_goal = goal_pos - own_pos
            desired_heading = math.atan2(d_goal[0], d_goal[1])
            h_err = (desired_heading - own_course_rad) % (2.0 * math.pi)
            if h_err > math.pi: h_err -= 2.0 * math.pi
            cmd_rudder = float(np.clip(1.2 * h_err, -math.radians(25.0), math.radians(25.0)))
            return np.array([self.target_speed, cmd_rudder], dtype=np.float32)

        # Smart / COLREG-aware navigation
        target_speed = self.target_speed
        desired_x = goal_x

        target_positions = info.get("target_positions", {})
        active_threat = None
        min_threat_d = float("inf")

        for tgt_id, pos in target_positions.items():
            tx, ty = float(pos[0]), float(pos[1])
            d = math.hypot(own_pos[0] - tx, own_pos[1] - ty)
            dy = ty - own_pos[1]

            # Threat closing in ahead
            if -40.0 < dy < 700.0 and d < 700.0:
                if d < min_threat_d:
                    min_threat_d = d
                    active_threat = (tx, ty)

        if active_threat is not None:
            tx, ty = active_threat
            dx = tx - own_pos[0]
            dy = ty - own_pos[1]
            dist = math.hypot(dx, dy)

            # Meeting or head-on: pass port-to-port by shifting lane to Starboard (+115m)
            if abs(dx) < 80.0 and dy > 0:
                desired_x = max(tx + 115.0, 110.0)
            elif dx > 40.0 and dist < 450.0:
                # Starboard crossing: ease speed to pass astern
                target_speed = 5.0
            elif dx < -40.0 and dist < 400.0:
                # Port crossing: ease speed slightly
                target_speed = 4.5

        # Check static obstacles (islands)
        for ob in info.get("static_obstacles", []):
            if isinstance(ob, (list, tuple)) and len(ob) >= 3:
                ox, oy, orad = ob[0], ob[1], ob[2]
                if math.hypot(own_pos[0] - ox, own_pos[1] - oy) < (orad + 60.0):
                    if own_pos[0] < ox:
                        desired_x = min(desired_x, ox - orad - 50.0)
                    else:
                        desired_x = max(desired_x, ox + orad + 50.0)

        lookahead = 170.0
        cross_err = desired_x - own_pos[0]
        desired_heading = math.atan2(cross_err, lookahead)

        dist_goal = math.hypot(goal_x - own_pos[0], goal_y - own_pos[1])
        if dist_goal < 250.0:
            desired_heading = math.atan2(goal_x - own_pos[0], goal_y - own_pos[1])

        heading_error = (desired_heading - own_course_rad) % (2.0 * math.pi)
        if heading_error > math.pi: heading_error -= 2.0 * math.pi

        cmd_rudder = float(np.clip(1.3 * heading_error, -math.radians(25.0), math.radians(25.0)))
        return np.array([target_speed, cmd_rudder], dtype=np.float32)


class WaypointFollowerAgent(BaseAgent):
    """Simple waypoint tracking agent with proportional steering."""
    def __init__(self, target_speed: float = 8.0, name: str = "WaypointAgent"):
        super().__init__(name=name)
        self.target_speed = target_speed

    def predict(self, observation: np.ndarray, info: dict | None = None) -> np.ndarray:
        if info is None or "own_pos" not in info:
            return np.array([self.target_speed, 0.0], dtype=np.float32)

        own_pos = np.array(info["own_pos"])
        own_course_rad = np.deg2rad(info["own_course_deg"])
        goal_pos = np.array(info.get("goal_pos", [0.0, own_pos[1] + 1000.0]))

        d_goal = goal_pos - own_pos
        desired_heading = (np.arctan2(d_goal[0], d_goal[1])) % (2 * np.pi)

        heading_error = (desired_heading - own_course_rad) % (2 * np.pi)
        if heading_error > np.pi:
            heading_error -= 2 * np.pi

        cmd_rudder = float(np.clip(1.0 * heading_error, -np.deg2rad(30.0), np.deg2rad(30.0)))
        return np.array([self.target_speed, cmd_rudder], dtype=np.float32)
