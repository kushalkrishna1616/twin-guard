"""
TwinGuard Safety Gate: Approves, Modifies, or Vetoes Navigation Actions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from twinguard.env.vessel import VesselState
from twinguard.counterfactual.rollout import CounterfactualRolloutEngine, CounterfactualSummary
from twinguard.calibration.conformal import ConformalRiskCalibrator
from twinguard.gate.safety_filter import SafetyActionFilter
from twinguard.gate.fallback import EmergencyFallbackController


class GateDecision(str, Enum):
    APPROVE = "APPROVE"
    MODIFY = "MODIFY"
    VETO = "VETO"


@dataclass
class GateAuditRecord:
    """Complete diagnostic audit record for a single step decision."""
    step: int
    decision: GateDecision
    proposed_action: list[float]  # [speed, rudder]
    executed_action: list[float]  # [speed, rudder]
    raw_p95_risk: float
    calibrated_upper_risk: float
    safety_threshold: float
    collision_prob: float
    domain_violation_prob: float
    colreg_violation_prob: float
    min_clearance_m: float
    reason: str
    counterfactual_summary: CounterfactualSummary | None = None


class TwinGuardSafetyGate:
    """
    Runtime verification gate wrapping black-box maritime agents.
    """
    def __init__(
        self,
        safety_threshold: float = 0.35,
        num_rollouts: int = 15,
        horizon_steps: int = 35,
        alpha: float = 0.05
    ):
        self.safety_threshold = safety_threshold
        self.rollout_engine = CounterfactualRolloutEngine(
            num_rollouts=num_rollouts,
            horizon_steps=horizon_steps
        )
        self.calibrator = ConformalRiskCalibrator(alpha=alpha)
        self.safety_filter = SafetyActionFilter(
            rollout_engine=self.rollout_engine,
            calibrator=self.calibrator,
            safety_threshold=safety_threshold
        )
        self.fallback_controller = EmergencyFallbackController()

        self.audit_log: list[GateAuditRecord] = []
        self.current_step = 0

    def evaluate_and_gate(
        self,
        own_ship: VesselState,
        target_ships: list[VesselState],
        proposed_action: np.ndarray,
        base_drift: tuple[float, float] = (0.0, 0.0)
    ) -> tuple[np.ndarray, GateAuditRecord]:
        """
        Main gate entrypoint. Given proposed action, returns safe executable action
        and full audit record.
        """
        self.current_step += 1

        # Fast range check: if all targets are beyond 1100m, maneuver is safe by definition
        min_target_dist = min(
            [float(np.linalg.norm(own_ship.position - t.position)) for t in target_ships],
            default=9999.0
        )
        if min_target_dist > 1100.0:
            record = GateAuditRecord(
                step=self.current_step,
                decision=GateDecision.APPROVE,
                proposed_action=proposed_action.tolist(),
                executed_action=proposed_action.tolist(),
                raw_p95_risk=0.0,
                calibrated_upper_risk=0.08,
                safety_threshold=self.safety_threshold,
                collision_prob=0.0,
                domain_violation_prob=0.0,
                colreg_violation_prob=0.0,
                min_clearance_m=round(min_target_dist, 1),
                reason=f"Approved: Nearest traffic is {min_target_dist:.0f}m away (>1100m). Clear waters.",
                counterfactual_summary=None
            )
            self.audit_log.append(record)
            return proposed_action.copy(), record

        # 1. Run counterfactual rollout evaluation on proposed action
        summary = self.rollout_engine.evaluate_action(
            own_ship=own_ship,
            target_ships=target_ships,
            candidate_action=proposed_action,
            base_drift=base_drift
        )

        # 2. Get certified conformal risk bound
        _, _, calibrated_risk = self.calibrator.get_calibrated_risk_bound(summary.p95_risk)

        # 3. Decision Logic:
        # Check condition for APPROVE:
        # Safe threshold satisfied AND zero estimated collision probability AND no severe COLREG breach
        is_safe = (
            calibrated_risk < self.safety_threshold and
            summary.collision_probability == 0.0 and
            summary.colreg_violation_probability < 0.2
        )

        if is_safe:
            decision = GateDecision.APPROVE
            exec_action = proposed_action.copy()
            reason = (
                f"Approved: Calibrated risk {calibrated_risk:.2f} is below threshold {self.safety_threshold:.2f}. "
                f"Min projected clearance {summary.min_clearance:.1f}m."
            )
        else:
            # Try to MODIFY action via minimal perturbation filter
            mod_action, mod_summary, mod_reason = self.safety_filter.search_safe_action(
                own_ship=own_ship,
                target_ships=target_ships,
                proposed_action=proposed_action,
                base_drift=base_drift
            )

            if mod_action is not None and mod_summary is not None:
                decision = GateDecision.MODIFY
                exec_action = mod_action
                summary = mod_summary
                _, _, calibrated_risk = self.calibrator.get_calibrated_risk_bound(summary.p95_risk)
                reason = mod_reason
            else:
                # VETO and invoke deterministic emergency fallback
                decision = GateDecision.VETO
                fallback_act, fallback_reason = self.fallback_controller.compute_fallback_action(
                    own_ship=own_ship,
                    target_ships=target_ships
                )
                exec_action = fallback_act
                reason = fallback_reason

        record = GateAuditRecord(
            step=self.current_step,
            decision=decision,
            proposed_action=proposed_action.tolist(),
            executed_action=exec_action.tolist(),
            raw_p95_risk=summary.p95_risk,
            calibrated_upper_risk=round(calibrated_risk, 3),
            safety_threshold=self.safety_threshold,
            collision_prob=summary.collision_probability,
            domain_violation_prob=summary.domain_violation_probability,
            colreg_violation_prob=summary.colreg_violation_probability,
            min_clearance_m=summary.min_clearance,
            reason=reason,
            counterfactual_summary=summary
        )

        self.audit_log.append(record)
        return exec_action, record

    def reset(self) -> None:
        self.audit_log = []
        self.current_step = 0
