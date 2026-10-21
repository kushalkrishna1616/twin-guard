"""
Closest Point of Approach (DCPA and TCPA) Calculations for Maritime Navigation.
"""

from __future__ import annotations
import numpy as np


def compute_dcpa_tcpa(
    own_pos: np.ndarray,      # [x, y]
    own_vel: np.ndarray,      # [vx, vy]
    target_pos: np.ndarray,   # [x, y]
    target_vel: np.ndarray    # [vx, vy]
) -> tuple[float, float]:
    """
    Computes Distance at Closest Point of Approach (DCPA in meters)
    and Time to Closest Point of Approach (TCPA in seconds).

    Positive TCPA means ships are closing in towards CPA.
    Negative TCPA means ships are diverging (already passed CPA).
    """
    rel_pos = target_pos - own_pos
    rel_vel = target_vel - own_vel

    rel_speed_sq = float(np.dot(rel_vel, rel_vel))

    # If relative speed is negligible, DCPA is current distance, TCPA is inf
    if rel_speed_sq < 1e-6:
        dcpa = float(np.linalg.norm(rel_pos))
        tcpa = float("inf")
        return dcpa, tcpa

    # TCPA formula: - (p_rel . v_rel) / |v_rel|^2
    tcpa = -float(np.dot(rel_pos, rel_vel)) / rel_speed_sq

    if tcpa <= 0:
        # Ships are already diverging; closest distance was or is now
        dcpa = float(np.linalg.norm(rel_pos))
    else:
        pos_at_cpa = rel_pos + rel_vel * tcpa
        dcpa = float(np.linalg.norm(pos_at_cpa))

    return dcpa, tcpa
