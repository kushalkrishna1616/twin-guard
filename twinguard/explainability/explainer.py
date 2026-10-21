"""
Explainability and Safety Audit Layer for TwinGuard.
"""

from __future__ import annotations
from dataclasses import dataclass
from twinguard.gate.gate import GateAuditRecord, GateDecision


@dataclass
class ExplanationReport:
    """Detailed human-readable explanation of a safety gate decision."""
    step: int
    decision: GateDecision
    headline: str
    key_factors: list[str]
    risk_breakdown: dict[str, float]
    recommendation: str


class DecisionExplainer:
    """
    Translates raw simulation logs and audit records into clear operator explanations.
    """
    @staticmethod
    def explain(record: GateAuditRecord) -> ExplanationReport:
        summary = record.counterfactual_summary
        breakdown: dict[str, float] = {
            "Certified Upper Risk": record.calibrated_upper_risk,
            "Raw P95 Risk": record.raw_p95_risk,
            "Collision Probability": record.collision_prob,
            "Domain Violation Prob": record.domain_violation_prob,
            "COLREG Violation Prob": record.colreg_violation_prob,
        }

        factors = []
        if record.collision_prob > 0.0:
            factors.append(f"Severe collision hazard predicted in {record.collision_prob*100:.1f}% of what-if rollouts.")
        if record.domain_violation_prob > 0.3:
            factors.append(f"High risk of entering vessel safety domain ({record.domain_violation_prob*100:.1f}% prob).")
        if record.colreg_violation_prob > 0.0:
            factors.append(f"Proposed rudder angle breaches maritime COLREG navigational guidelines.")
        if record.min_clearance_m < 200.0:
            factors.append(f"Closest projected clearance is critical: {record.min_clearance_m:.1f} meters.")

        if not factors:
            factors.append("All maritime safety margins, CPA thresholds, and COLREG rules satisfied.")

        # Headline
        if record.decision == GateDecision.APPROVE:
            headline = f"Step {record.step}: Action APPROVED (Safe trajectory confirmed across all counterfactuals)"
            rec = "Proceed with agent's autonomous trajectory."
        elif record.decision == GateDecision.MODIFY:
            headline = f"Step {record.step}: Action MODIFIED (Minimal course/speed trim applied)"
            rec = "Autonomous agent course trimmed to ensure COLREG compliance and safe domain clearance."
        else:
            headline = f"Step {record.step}: Action VETOED (Emergency safety fallback active)"
            rec = "Agent action suppressed; emergency evasive maneuver and engine reduction engaged."

        return ExplanationReport(
            step=record.step,
            decision=record.decision,
            headline=headline,
            key_factors=factors,
            risk_breakdown=breakdown,
            recommendation=rec
        )
