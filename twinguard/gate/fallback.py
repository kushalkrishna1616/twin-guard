"""
Deterministic Safe Fallback Controllers for Emergency Gate Interventions.
"""

from __future__ import annotations
import math
import numpy as np
from twinguard.env.vessel import VesselState


class EmergencyFallbackController:
    """
    Emergency fail-safe controller activated when the safety gate vetoes an agent's action.
    Executes smooth, proportional COLREG-compliant evasive maneuvers without spinning in circles.
    """
    def __init__(self, emergency_speed: float = 6.5, evasive_rudder_deg: float = 22.0):
        self.emergency_speed = emergency_speed
        self.evasive_rudder_rad = math.radians(evasive_rudder_deg)

    def compute_fallback_action(
        self,
        own_ship: VesselState,
        target_ships: list[VesselState]
    ) -> tuple[np.ndarray, str]:
        """
        Determines the safest immediate action:
        - Steers to an evasive Starboard corridor (+110m cushion)
        - Uses proportional heading-hold control so the vessel DOES NOT spin in circles
        - Resumes forward fairway course once traffic passes astern
        """
        if not target_ships:
            h_err = (0.0 - own_ship.course) % (2.0 * math.pi)
            if h_err > math.pi: h_err -= 2.0 * math.pi
            cmd_r = float(np.clip(1.4 * h_err, -math.radians(20.0), math.radians(20.0)))
            return np.array([8.0, cmd_r], dtype=np.float32), "Fallback: Clear fairway"

        # Find closest threatening vessel
        distances = [
            (float(np.linalg.norm(own_ship.position - t.position)), t)
            for t in target_ships
        ]
        distances.sort(key=lambda x: x[0])
        min_dist, closest_target = distances[0]

        from twinguard.risk.colreg import compute_relative_bearing
        from twinguard.risk.cpa import compute_dcpa_tcpa

        dcpa, tcpa = compute_dcpa_tcpa(
            own_ship.position, own_ship.velocity_vector,
            closest_target.position, closest_target.velocity_vector
        )
        rel_b = compute_relative_bearing(own_ship, closest_target)
        rel_b_deg = float(np.rad2deg(rel_b))

        # Check if traffic has already passed (diverging or astern)
        is_past = (tcpa <= 0.0 and min_dist > 80.0) or (105.0 <= rel_b_deg <= 255.0)
        if is_past:
            # Safe! Align back to forward fairway heading (North / 0 rad)
            h_err = (0.0 - own_ship.course) % (2.0 * math.pi)
            if h_err > math.pi: h_err -= 2.0 * math.pi
            cmd_r = float(np.clip(1.4 * h_err, -math.radians(22.0), math.radians(22.0)))
            return np.array([8.0, cmd_r], dtype=np.float32), f"Fallback: Target {closest_target.id} astern. Resuming fairway."

        # Active forward threat:
        target_speed = self.emergency_speed
        desired_x = 0.0

        # Target movement vector
        t_course = closest_target.course
        t_vx = closest_target.speed * math.sin(t_course)
        dx = closest_target.x - own_ship.x
        dy = closest_target.y - own_ship.y

        # Crossing from Starboard (Give-Way)
        if dx > 40.0 and t_vx < -1.0:
            target_speed = 4.5 # ease speed to pass astern
            desired_x = 0.0
            reason = f"Emergency Veto: Easing speed (4.5m/s) to pass astern of crossing vessel {closest_target.id}."
        # Crossing from Port (Stand-On emergency)
        elif dx < -40.0 and t_vx > 1.0:
            target_speed = 4.0
            desired_x = 0.0
            reason = f"Emergency Veto: Braking to 4.0m/s to allow rogue port crossing ship to pass ahead."
        # Head-on or meeting
        else:
            desired_x = max(closest_target.x + 115.0, 110.0)
            reason = f"Emergency Veto: Shifting to starboard lane ({desired_x:.0f}m) to clear {closest_target.id}."

        lookahead = 160.0
        cross_err = desired_x - own_ship.x
        desired_heading = math.atan2(cross_err, lookahead)

        h_err = (desired_heading - own_ship.course) % (2.0 * math.pi)
        if h_err > math.pi: h_err -= 2.0 * math.pi

        cmd_rudder = float(np.clip(1.5 * h_err, -math.radians(25.0), math.radians(25.0)))
        return np.array([target_speed, cmd_rudder], dtype=np.float32), reason
