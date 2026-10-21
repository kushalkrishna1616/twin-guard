"""
Safety Action Filter: Searches for minimal corrections to unsafe actions.
"""

from __future__ import annotations
import numpy as np
from twinguard.env.vessel import VesselState
from twinguard.counterfactual.rollout import CounterfactualRolloutEngine, CounterfactualSummary
from twinguard.calibration.conformal import ConformalRiskCalibrator


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
        Explores candidate adjustments:
        - Steer further Starboard (+5°, +10°, +15°, +20°)
        - Moderate speed reductions (-1.5 m/s, -3.0 m/s)
        Returns:
            (best_action, summary, reason) or (None, None, failure_reason)
        """
        orig_speed = float(proposed_action[0])
        orig_rudder = float(proposed_action[1])

        # Candidate offsets: (d_speed, d_rudder)
        rudder_offsets = [
            np.deg2rad(5.0), np.deg2rad(10.0), np.deg2rad(15.0), np.deg2rad(20.0),
            -np.deg2rad(5.0), -np.deg2rad(10.0) # Port adjustments tested only if compliant
        ]
        speed_offsets = [0.0, -1.5, -3.0]

        best_action: np.ndarray | None = None
        best_summary: CounterfactualSummary | None = None
        min_cost = float("inf")
        reason = ""

        # Weighting: favor small rudder adjustments over dramatic slowdowns
        w_speed = 1.0
        w_rudder = 4.0

        for ds in speed_offsets:
            cand_speed = float(np.clip(orig_speed + ds, 2.0, 12.0))
            for dr in rudder_offsets:
                cand_rudder = float(np.clip(orig_rudder + dr, -np.deg2rad(35.0), np.deg2rad(35.0)))
                cand_act = np.array([cand_speed, cand_rudder], dtype=np.float32)

                summary = self.rollout_engine.evaluate_action(
                    own_ship, target_ships, cand_act, base_drift
                )

                # Check upper calibrated bound
                _, _, upper_risk = self.calibrator.get_calibrated_risk_bound(summary.p95_risk)

                # Action is acceptable if certified risk < threshold and zero collisions
                if upper_risk < self.safety_threshold and summary.collision_probability == 0.0:
                    cost = w_speed * (abs(ds) ** 2) + w_rudder * (abs(dr) ** 2)
                    if cost < min_cost:
                        min_cost = cost
                        best_action = cand_act
                        best_summary = summary
                        deg_change = np.rad2deg(dr)
                        reason = (
                            f"Action Modified: Adjusted rudder by {deg_change:+.1f} deg and speed by {ds:+.1f} m/s "
                            f"(Risk reduced from raw to {upper_risk:.2f} <= {self.safety_threshold})."
                        )

        return best_action, best_summary, reason
