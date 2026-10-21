"""
Rule-Based Navigation Agents (Artificial Potential Field & Waypoint Tracker).
"""

from __future__ import annotations
import numpy as np
from twinguard.agents.base import BaseAgent
from twinguard.env.vessel import VesselState


class APFMaritimeAgent(BaseAgent):
    """
    Artificial Potential Field (APF) navigation agent.
    Attractive potential pulls towards the waypoint goal.
    Repulsive potential pushes away from obstacles/target ships.

    If colreg_aware=False, the agent uses symmetric repulsion which often
    causes dangerous illegal PORT turns in head-on/crossing encounters (classic black-box failure).
    """
    def __init__(
        self,
        target_speed: float = 8.0,
        k_att: float = 1.0,
        k_rep: float = 150000.0,
        repulsion_radius: float = 600.0,
        colreg_aware: bool = False,
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
            # Fallback to straight ahead if no info provided
            return np.array([self.target_speed, 0.0], dtype=np.float32)

        own_pos = np.array(info["own_pos"])
        own_course_rad = np.deg2rad(info["own_course_deg"])

        # Attractive force towards goal (assume goal along +y / extracted from info)
        goal_pos = np.array(info.get("goal_pos", [0.0, own_pos[1] + 1000.0]))
        d_goal = goal_pos - own_pos
        dist_goal = float(np.linalg.norm(d_goal))
        f_att = (self.k_att * d_goal / max(dist_goal, 1.0))

        # Repulsive force from target vessels
        f_rep = np.zeros(2)
        target_positions = info.get("target_positions", {})
        for tgt_id, pos in target_positions.items():
            tgt_pos = np.array(pos)
            diff = own_pos - tgt_pos # points away from target
            dist = float(np.linalg.norm(diff))

            if dist < self.repulsion_radius and dist > 1.0:
                # Magnitude inversely proportional to distance
                rep_mag = self.k_rep * (1.0 / dist - 1.0 / self.repulsion_radius) / (dist ** 2)
                rep_dir = diff / dist

                # If COLREG aware, bias repulsive force to the right (starboard)
                if self.colreg_aware:
                    # Rotate repulsion vector slightly starboard
                    rot_matrix = np.array([[0.866, 0.5], [-0.5, 0.866]])
                    rep_dir = np.dot(rot_matrix, rep_dir)

                f_rep += rep_mag * rep_dir

        # Total steering force vector in world frame
        f_total = f_att + f_rep
        desired_heading = (np.arctan2(f_total[0], f_total[1])) % (2 * np.pi)

        # Heading error relative to own course
        heading_error = (desired_heading - own_course_rad) % (2 * np.pi)
        if heading_error > np.pi:
            heading_error -= 2 * np.pi

        # Proportional rudder control
        cmd_rudder = float(np.clip(1.2 * heading_error, -np.deg2rad(35.0), np.deg2rad(35.0)))

        # Reduce speed in tight turns or heavy repulsion
        cmd_speed = self.target_speed
        if abs(cmd_rudder) > np.deg2rad(15.0) or np.linalg.norm(f_rep) > 1.0:
            cmd_speed = max(4.0, self.target_speed * 0.75)

        return np.array([cmd_speed, cmd_rudder], dtype=np.float32)


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
