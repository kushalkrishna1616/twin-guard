"""
Safety Action Filter: Searches for minimal corrections to unsafe actions.
"""

from __future__ import annotations
import math
import numpy as np
from twinguard.env.vessel import VesselState
from twinguard.counterfactual.rollout import CounterfactualRolloutEngine, CounterfactualSummary
from twinguard.calibration.conformal import ConformalRiskCalibrator


def _clamp(val: float, low: float, high: float) -> float:
    if val < low:
        return low
    if val > high:
        return high
    return val


class SafetyActionFilter:
    """
    Evaluates candidate action variations in the local neighborhood
    to identify a minimally-modified action satisfying the risk bound.
    """
    def __init__(
        self,
        rollout_engine: CounterfactualRolloutEngine,
        calibrator: ConformalRiskCalibrator,
        safety_threshold: float = 0.35
    ):
        self.rollout_engine = rollout_engine
        self.calibrator = calibrator
        self.safety_threshold = safety_threshold

    def search_safe_action(
        self,
        own_ship: VesselState,
        target_ships: list[VesselState],
        proposed_action: np.ndarray,
        base_drift: tuple[float, float] = (0.0, 0.0)
    ) -> tuple[np.ndarray | None, CounterfactualSummary | None, str]:
        """
        Explores candidate adjustments ordered by minimal perturbation cost.
        Returns immediately upon finding the lowest-cost safe action.
        """
        orig_speed = float(proposed_action[0])
        orig_rudder = float(proposed_action[1])

        # Prioritize COLREG starboard turns and slight speed adjustments
        candidate_offsets = [
            (0.0, 0.1745),    # +10 deg
            (0.0, 0.2618),    # +15 deg
            (-1.5, 0.1745),   # -1.5 m/s, +10 deg
            (-1.5, 0.2618),   # -1.5 m/s, +15 deg
            (0.0, 0.0873),    # +5 deg
            (0.0, 0.3491),    # +20 deg
            (-1.5, 0.0873),   # -1.5 m/s, +5 deg
            (-3.0, 0.1745),   # -3.0 m/s, +10 deg
            (-1.5, 0.3491),   # -1.5 m/s, +20 deg
            (-3.0, 0.2618),   # -3.0 m/s, +15 deg
            (0.0, -0.0873),   # -5 deg
            (0.0, -0.1745),   # -10 deg
        ]

        for ds, dr in candidate_offsets:
            cand_speed = _clamp(orig_speed + ds, 2.0, 12.0)
            cand_rudder = _clamp(orig_rudder + dr, -0.6108, 0.6108)
            cand_act = np.array([cand_speed, cand_rudder], dtype=np.float32)

            summary = self.rollout_engine.evaluate_action(
                own_ship, target_ships, cand_act, base_drift
            )

            _, _, upper_risk = self.calibrator.get_calibrated_risk_bound(summary.p95_risk)

            if upper_risk < self.safety_threshold and summary.collision_probability == 0.0:
                deg_change = math.degrees(dr)
                reason = (
                    f"Action Modified: Adjusted rudder by {deg_change:+.1f} deg and speed by {ds:+.1f} m/s "
                    f"(Risk reduced from raw to {upper_risk:.2f} <= {self.safety_threshold})."
                )
                return cand_act, summary, reason

        return None, None, "No safe action found within search neighborhood."
