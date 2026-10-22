"""
Closest Point of Approach (DCPA and TCPA) Calculations for Maritime Navigation.
"""

from __future__ import annotations
import math
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
    Optimized for high-throughput scalar calculation.
    """
    rx = float(target_pos[0] - own_pos[0])
    ry = float(target_pos[1] - own_pos[1])
    vx = float(target_vel[0] - own_vel[0])
    vy = float(target_vel[1] - own_vel[1])

    rel_speed_sq = vx * vx + vy * vy

    if rel_speed_sq < 1e-6:
        dcpa = math.hypot(rx, ry)
        tcpa = float("inf")
        return dcpa, tcpa

    dot_pv = rx * vx + ry * vy
    tcpa = -dot_pv / rel_speed_sq

    # CPA position relative to own ship
    cpa_x = rx + vx * tcpa
    cpa_y = ry + vy * tcpa
    dcpa = math.hypot(cpa_x, cpa_y)

    return dcpa, tcpa
