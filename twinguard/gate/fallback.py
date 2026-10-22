"""
Deterministic Safe Fallback Controllers for Emergency Gate Interventions.
"""

from __future__ import annotations
import math
import numpy as np
from twinguard.env.vessel import VesselState
from twinguard.risk.colreg import compute_relative_bearing
from twinguard.risk.cpa import compute_dcpa_tcpa


class EmergencyFallbackController:
    """
    Emergency fail-safe controller activated when the safety gate vetoes an agent's action.
    Executes smooth, proportional COLREG-compliant evasive maneuvers without spinning 360 degrees.
    """
    def __init__(self, emergency_speed: float = 6.0, evasive_rudder_deg: float = 25.0):
        self.emergency_speed = emergency_speed
        self.evasive_rudder_rad = math.radians(evasive_rudder_deg)

    def compute_fallback_action(
        self,
        own_ship: VesselState,
        target_ships: list[VesselState]
    ) -> tuple[np.ndarray, str]:
        """
        Determines the safest immediate action:
        - Steers to an evasive Starboard heading (+35 deg from threat line of bearing)
        - Uses proportional heading-hold control so the vessel DOES NOT spin in circles
        - Resumes forward fairway course immediately once traffic passes astern
        """
        if not target_ships:
            return np.array([self.emergency_speed, 0.0], dtype=np.float32), "Fallback: Low-speed idle glide"

        # Find closest threatening vessel
        distances = [
            (float(np.linalg.norm(own_ship.position - t.position)), t)
            for t in target_ships
        ]
        distances.sort(key=lambda x: x[0])
        min_dist, closest_target = distances[0]

        dcpa, tcpa = compute_dcpa_tcpa(
            own_ship.position, own_ship.velocity_vector,
            closest_target.position, closest_target.velocity_vector
        )
        rel_b = compute_relative_bearing(own_ship, closest_target)
        rel_b_deg = float(np.rad2deg(rel_b))

        # Check if traffic has already passed (diverging or astern in aft quadrant)
        if tcpa <= 0.0 or (105.0 <= rel_b_deg <= 255.0):
            # Safe! Align back to forward fairway heading (North / 0 rad)
            h_err = (0.0 - own_ship.course) % (2.0 * math.pi)
            if h_err > math.pi:
                h_err -= 2.0 * math.pi
            cmd_rudder = float(np.clip(1.5 * h_err, -math.radians(25.0), math.radians(25.0)))
            return np.array([7.0, cmd_rudder], dtype=np.float32), f"Fallback: Target {closest_target.id} safely astern. Resuming fairway."

        # Active forward threat: Calculate desired starboard passing heading
        dx = closest_target.x - own_ship.x
        dy = closest_target.y - own_ship.y
        lob = math.atan2(dx, dy) # bearing to target

        # Steer 35 degrees to the right of threat line of bearing (COLREG Starboard pass)
        target_heading = (lob + math.radians(35.0)) % (2.0 * math.pi)

        # Proportional heading-hold control (stops turning once desired course is reached)
        h_err = (target_heading - own_ship.course) % (2.0 * math.pi)
        if h_err > math.pi:
            h_err -= 2.0 * math.pi

        kp = 1.4
        cmd_rudder = float(np.clip(kp * h_err, -math.radians(30.0), math.radians(30.0)))
        cmd_speed = max(5.0, self.emergency_speed)

        reason = (
            f"Emergency Veto: Evasive Starboard heading alteration ({math.degrees(target_heading):.1f}° COG) "
            f"clearing {closest_target.id} at {min_dist:.0f}m."
        )
        return np.array([cmd_speed, cmd_rudder], dtype=np.float32), reason
