"""
Deterministic Safe Fallback Controllers for Emergency Gate Interventions.
"""

from __future__ import annotations
import numpy as np
from twinguard.env.vessel import VesselState
from twinguard.risk.colreg import compute_relative_bearing


class EmergencyFallbackController:
    """
    Emergency fail-safe controller activated when the safety gate vetoes an agent's action.
    Executes emergency braking and COLREG-compliant evasive maneuvers.
    """
    def __init__(self, emergency_speed: float = 1.5, evasive_rudder_deg: float = 25.0):
        self.emergency_speed = emergency_speed
        self.evasive_rudder_rad = np.deg2rad(evasive_rudder_deg)

    def compute_fallback_action(
        self,
        own_ship: VesselState,
        target_ships: list[VesselState]
    ) -> tuple[np.ndarray, str]:
        """
        Determines the safest immediate action:
        - Drastically drop speed
        - If nearest target is dead ahead or on port/starboard, turn away according to COLREGs
        """
        if not target_ships:
            # Standstill / slow glide
            return np.array([self.emergency_speed, 0.0], dtype=np.float32), "Fallback: Low-speed idle glide"

        # Find closest threatening vessel
        distances = [
            (float(np.linalg.norm(own_ship.position - t.position)), t)
            for t in target_ships
        ]
        distances.sort(key=lambda x: x[0])
        min_dist, closest_target = distances[0]

        rel_bearing = compute_relative_bearing(own_ship, closest_target)
        rel_b_deg = float(np.rad2deg(rel_bearing))

        # Decision logic:
        # If target is in front sector (330 to 30 deg), COLREG standard evasive turn is Starboard (turn right)
        if rel_b_deg < 35.0 or rel_b_deg > 325.0:
            cmd_rudder = self.evasive_rudder_rad # Turn starboard
            reason = f"Emergency Veto: Hazard {closest_target.id} at {min_dist:.1f}m dead ahead. Decelerating to {self.emergency_speed}m/s and hard evasive Starboard turn."
        elif 35.0 <= rel_b_deg <= 180.0:
            # Target is on starboard side: give-way turn starboard further away or slow down sharply
            cmd_rudder = self.evasive_rudder_rad
            reason = f"Emergency Veto: Target {closest_target.id} on Starboard ({rel_b_deg:.1f} deg). Executing emergency braking and opening distance."
        else:
            # Target is on port side: steer to starboard to clear path
            cmd_rudder = self.evasive_rudder_rad * 0.7
            reason = f"Emergency Veto: Target {closest_target.id} on Port ({rel_b_deg:.1f} deg). Braking to avoid crossing."

        cmd_speed = max(0.5, self.emergency_speed)
        action = np.array([cmd_speed, cmd_rudder], dtype=np.float32)
        return action, reason
