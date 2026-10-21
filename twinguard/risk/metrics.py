"""
Risk Scoring, Collision Probability and Aggregation Layer.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from twinguard.env.vessel import VesselState
from twinguard.risk.cpa import compute_dcpa_tcpa
from twinguard.risk.ship_domain import ShipDomain
from twinguard.risk.colreg import classify_colreg_encounter, check_colreg_compliance, EncounterType


@dataclass
class EncounterRiskReport:
    """Detailed risk report for a single target vessel encounter."""
    target_id: str
    distance: float
    dcpa: float
    tcpa: float
    domain_ratio: float
    domain_violated: bool
    encounter_type: EncounterType
    colreg_compliant: bool
    colreg_message: str
    risk_score: float  # [0.0, 1.0]


class MaritimeRiskEvaluator:
    """
    Evaluates instant and projected risk between own ship and target vessels.
    """
    def __init__(
        self,
        domain_model: ShipDomain | None = None,
        dcpa_threshold: float = 250.0, # m (safe CPA buffer)
        tcpa_horizon: float = 120.0     # s (time horizon considered hazardous)
    ):
        self.domain = domain_model or ShipDomain()
        self.dcpa_threshold = dcpa_threshold
        self.tcpa_horizon = tcpa_horizon

    def evaluate_target(
        self,
        own_ship: VesselState,
        target_ship: VesselState,
        commanded_rudder: float = 0.0
    ) -> EncounterRiskReport:
        """
        Evaluate full maritime risk against one target vessel.
        """
        dist = float(np.linalg.norm(own_ship.position - target_ship.position))
        dcpa, tcpa = compute_dcpa_tcpa(
            own_ship.position, own_ship.velocity_vector,
            target_ship.position, target_ship.velocity_vector
        )

        domain_violated, domain_ratio = self.domain.evaluate(own_ship, target_ship)
        enc_type, _ = classify_colreg_encounter(own_ship, target_ship)
        colreg_ok, colreg_msg = check_colreg_compliance(
            enc_type, commanded_rudder, dist, dcpa, self.dcpa_threshold
        )

        # Risk Score Calculation (0 = Safe, 1 = Imminent danger)
        # Component 1: DCPA penalty (exponential rise as DCPA falls below threshold)
        if dcpa < self.dcpa_threshold:
            r_dcpa = np.exp(-2.0 * (dcpa / max(self.dcpa_threshold, 1.0)))
        else:
            r_dcpa = 0.0

        # Component 2: TCPA urgency factor (higher if TCPA is positive and small)
        if 0.0 < tcpa <= self.tcpa_horizon:
            r_tcpa = np.exp(-1.5 * (tcpa / self.tcpa_horizon))
        elif tcpa <= 0.0 and dist > 300.0:
            r_tcpa = 0.0 # Diverging safely
        else:
            r_tcpa = 0.1

        # Component 3: Ship domain violation penalty
        r_domain = 0.8 if domain_violated else max(0.0, 1.0 - domain_ratio)

        # Component 4: COLREG rule penalty
        r_colreg = 0.4 if not colreg_ok else 0.0

        # Combine components
        # If ships are far apart (>2000m), risk is negligible
        distance_dampening = float(np.clip(1.0 - (dist / 2500.0), 0.0, 1.0))
        raw_risk = (0.35 * r_dcpa * r_tcpa + 0.35 * r_domain + 0.30 * r_colreg) * distance_dampening

        # Hard minimum if already inside domain or collision
        if dist < (own_ship.length + target_ship.length):
            raw_risk = 1.0
        elif domain_violated:
            raw_risk = max(raw_risk, 0.75)

        risk_score = float(np.clip(raw_risk, 0.0, 1.0))

        return EncounterRiskReport(
            target_id=target_ship.id,
            distance=round(dist, 1),
            dcpa=round(dcpa, 1),
            tcpa=round(tcpa, 1),
            domain_ratio=round(domain_ratio, 2),
            domain_violated=domain_violated,
            encounter_type=enc_type,
            colreg_compliant=colreg_ok,
            colreg_message=colreg_msg,
            risk_score=round(risk_score, 3)
        )

    def aggregate_multi_ship_risk(
        self,
        reports: list[EncounterRiskReport]
    ) -> tuple[float, EncounterRiskReport | None]:
        """
        Aggregate risks across all surrounding vessels.
        Returns (max_risk, most_critical_report).
        """
        if not reports:
            return 0.0, None

        # Sort by risk descending
        reports_sorted = sorted(reports, key=lambda r: r.risk_score, reverse=True)
        max_report = reports_sorted[0]
        return max_report.risk_score, max_report
