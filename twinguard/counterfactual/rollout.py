"""
Counterfactual Rollout Engine: Fast-forwards K parallel what-if futures
to quantify risk before action execution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from twinguard.env.vessel import VesselState
from twinguard.counterfactual.perturb import CounterfactualPerturbator, PerturbationParams
from twinguard.risk.metrics import MaritimeRiskEvaluator, EncounterRiskReport


@dataclass
class RolloutResult:
    """Outcomes from a single counterfactual rollout trajectory."""
    rollout_id: int
    own_trajectory: list[tuple[float, float]]  # List of [x, y]
    target_trajectories: dict[str, list[tuple[float, float]]]
    min_distance: float
    collision_occurred: bool
    domain_violation_occurred: bool
    colreg_violation_occurred: bool
    max_risk: float
    critical_report: EncounterRiskReport | None


@dataclass
class CounterfactualSummary:
    """Aggregated assessment across all K rollouts for a candidate action."""
    action: np.ndarray
    num_rollouts: int
    collision_probability: float
    domain_violation_probability: float
    colreg_violation_probability: float
    mean_risk: float
    max_risk: float
    p95_risk: float
    min_clearance: float
    mean_min_clearance: float
    rollouts: list[RolloutResult] = field(default_factory=list)


class CounterfactualRolloutEngine:
    """
    Simulates K prospective rollout trajectories from the current digital-twin state.
    """
    def __init__(
        self,
        num_rollouts: int = 15,
        horizon_steps: int = 35,
        dt: float = 1.0,
        perturbator: CounterfactualPerturbator | None = None,
        risk_evaluator: MaritimeRiskEvaluator | None = None
    ):
        self.num_rollouts = num_rollouts
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.perturbator = perturbator or CounterfactualPerturbator()
        self.risk_evaluator = risk_evaluator or MaritimeRiskEvaluator()

    def evaluate_action(
        self,
        own_ship: VesselState,
        target_ships: list[VesselState],
        candidate_action: np.ndarray,
        base_drift: tuple[float, float] = (0.0, 0.0),
        seed: int | None = None
    ) -> CounterfactualSummary:
        """
        Runs K what-if rollouts applying candidate_action = [cmd_speed, cmd_rudder].
        """
        rng = np.random.default_rng(seed)
        cmd_speed = float(candidate_action[0])
        cmd_rudder = float(candidate_action[1])

        rollout_results: list[RolloutResult] = []

        for k in range(self.num_rollouts):
            # Nominal rollout for k=0; perturbed for k >= 1
            if k == 0:
                rollout_own = own_ship.copy()
                rollout_targets = [t.copy() for t in target_ships]
                drift_x, drift_y = base_drift
            else:
                rollout_own = own_ship.copy()
                rollout_targets = [
                    self.perturbator.generate_perturbed_target(t, rng)
                    for t in target_ships
                ]
                drift_x, drift_y = self.perturbator.sample_drift(base_drift[0], base_drift[1], rng)

            own_traj: list[tuple[float, float]] = [(rollout_own.x, rollout_own.y)]
            tgt_trajs: dict[str, list[tuple[float, float]]] = {
                t.id: [(t.x, t.y)] for t in rollout_targets
            }

            min_dist = float("inf")
            collision = False
            domain_violation = False
            colreg_violation = False
            max_risk_in_rollout = 0.0
            worst_report: EncounterRiskReport | None = None

            for _ in range(self.horizon_steps):
                # Step own ship with proposed candidate action
                rollout_own.step(
                    commanded_speed=cmd_speed,
                    commanded_rudder=cmd_rudder,
                    dt=self.dt,
                    drift_x=drift_x,
                    drift_y=drift_y
                )
                own_traj.append((rollout_own.x, rollout_own.y))

                # Step target vessels
                step_reports: list[EncounterRiskReport] = []
                for tgt in rollout_targets:
                    tgt.step(
                        commanded_speed=tgt.speed,
                        commanded_rudder=0.0,
                        dt=self.dt,
                        drift_x=drift_x,
                        drift_y=drift_y
                    )
                    tgt_trajs[tgt.id].append((tgt.x, tgt.y))

                    # Evaluate metrics
                    d = float(np.linalg.norm(rollout_own.position - tgt.position))
                    if d < min_dist:
                        min_dist = d
                    if d < (rollout_own.length / 2 + tgt.length / 2):
                        collision = True

                    rep = self.risk_evaluator.evaluate_target(rollout_own, tgt, cmd_rudder)
                    step_reports.append(rep)
                    if rep.domain_violated:
                        domain_violation = True
                    if not rep.colreg_compliant:
                        colreg_violation = True

                agg_r, crit_r = self.risk_evaluator.aggregate_multi_ship_risk(step_reports)
                if agg_r > max_risk_in_rollout:
                    max_risk_in_rollout = agg_r
                    worst_report = crit_r

            rollout_results.append(
                RolloutResult(
                    rollout_id=k,
                    own_trajectory=own_traj,
                    target_trajectories=tgt_trajs,
                    min_distance=min_dist,
                    collision_occurred=collision,
                    domain_violation_occurred=domain_violation,
                    colreg_violation_occurred=colreg_violation,
                    max_risk=max_risk_in_rollout,
                    critical_report=worst_report
                )
            )

        # Aggregate summary statistics across all K rollouts
        all_max_risks = [r.max_risk for r in rollout_results]
        all_min_dists = [r.min_distance for r in rollout_results]

        collision_prob = sum(1 for r in rollout_results if r.collision_occurred) / self.num_rollouts
        domain_prob = sum(1 for r in rollout_results if r.domain_violation_occurred) / self.num_rollouts
        colreg_prob = sum(1 for r in rollout_results if r.colreg_violation_occurred) / self.num_rollouts

        mean_risk = float(np.mean(all_max_risks))
        max_risk = float(np.max(all_max_risks))
        p95_risk = float(np.percentile(all_max_risks, 95))
        min_clearance = float(np.min(all_min_dists))
        mean_clearance = float(np.mean(all_min_dists))

        return CounterfactualSummary(
            action=candidate_action,
            num_rollouts=self.num_rollouts,
            collision_probability=round(collision_prob, 3),
            domain_violation_probability=round(domain_prob, 3),
            colreg_violation_probability=round(colreg_prob, 3),
            mean_risk=round(mean_risk, 3),
            max_risk=round(max_risk, 3),
            p95_risk=round(p95_risk, 3),
            min_clearance=round(min_clearance, 1),
            mean_min_clearance=round(mean_clearance, 1),
            rollouts=rollout_results
        )
