"""
Unit and Integration Tests for TwinGuard.
"""

import numpy as np
import pytest

from twinguard.env.vessel import VesselState
from twinguard.risk.cpa import compute_dcpa_tcpa
from twinguard.risk.ship_domain import ShipDomain
from twinguard.risk.colreg import (
    EncounterType,
    classify_colreg_encounter,
    check_colreg_compliance,
)
from twinguard.risk.metrics import MaritimeRiskEvaluator
from twinguard.counterfactual.rollout import CounterfactualRolloutEngine
from twinguard.calibration.conformal import ConformalRiskCalibrator
from twinguard.gate.gate import TwinGuardSafetyGate, GateDecision
from twinguard.agents.rule_based import APFMaritimeAgent
from twinguard.env.maritime_env import MaritimeEnv
from twinguard.scenarios.encounters import create_head_on_scenario, create_crossing_scenario


def test_vessel_stepping():
    vessel = VesselState(id="v1", x=0.0, y=0.0, speed=10.0, course=0.0)
    # Stepping forward heading North
    vessel.step(commanded_speed=10.0, commanded_rudder=0.0, dt=1.0)
    assert vessel.y > 0.0
    assert abs(vessel.x) < 0.1


def test_dcpa_tcpa():
    # Head-on: own ship at (0, 0) moving North at 10m/s; target at (0, 1000) moving South at 10m/s
    own_pos = np.array([0.0, 0.0])
    own_vel = np.array([0.0, 10.0])
    tgt_pos = np.array([0.0, 1000.0])
    tgt_vel = np.array([0.0, -10.0])

    dcpa, tcpa = compute_dcpa_tcpa(own_pos, own_vel, tgt_pos, tgt_vel)
    assert dcpa == pytest.approx(0.0, abs=1e-3)
    assert tcpa == pytest.approx(50.0, abs=1e-3)


def test_ship_domain_violation():
    domain = ShipDomain()
    own_ship = VesselState(id="own", x=0.0, y=0.0, course=0.0, length=50.0)
    # Far away target
    safe_target = VesselState(id="tgt_safe", x=1000.0, y=1000.0)
    violated, ratio = domain.evaluate(own_ship, safe_target)
    assert not violated
    assert ratio > 1.0

    # Very close target inside fore domain (e.g. 50m ahead)
    close_target = VesselState(id="tgt_close", x=0.0, y=50.0)
    violated_close, ratio_close = domain.evaluate(own_ship, close_target)
    assert violated_close
    assert ratio_close < 1.0


def test_colreg_classification():
    # Head on
    own = VesselState(x=0.0, y=0.0, course=0.0, speed=8.0)
    tgt_headon = VesselState(x=0.0, y=1000.0, course=np.pi, speed=8.0)
    enc, _ = classify_colreg_encounter(own, tgt_headon)
    assert enc == EncounterType.HEAD_ON

    # Crossing give-way (target on starboard heading West)
    tgt_crossing = VesselState(x=800.0, y=400.0, course=3*np.pi/2, speed=8.0)
    enc_cross, _ = classify_colreg_encounter(own, tgt_crossing)
    assert enc_cross == EncounterType.CROSSING_GIVE_WAY

    # Compliance check: turning PORT in head-on is a violation
    ok, msg = check_colreg_compliance(
        encounter_type=EncounterType.HEAD_ON,
        commanded_rudder=-np.deg2rad(10.0), # Turn port
        current_distance=600.0,
        dcpa=10.0
    )
    assert not ok
    assert "Rule 14 Violation" in msg


def test_conformal_calibrator():
    calibrator = ConformalRiskCalibrator(alpha=0.05)
    preds = [0.2, 0.4, 0.6, 0.1, 0.5]
    targets = [0.3, 0.5, 0.6, 0.2, 0.5]
    q = calibrator.fit_calibration_data(preds, targets)
    assert q >= 0.0
    low, pt, up = calibrator.get_calibrated_risk_bound(0.3)
    assert up >= pt


def test_safety_gate_flow():
    gate = TwinGuardSafetyGate(num_rollouts=5, horizon_steps=15)
    own = VesselState(x=0.0, y=0.0, course=0.0, speed=8.0)

    # Distant target -> Should APPROVE
    tgt_safe = VesselState(x=2000.0, y=2000.0, course=np.pi, speed=5.0)
    action_safe = np.array([8.0, 0.0])
    exec_act, record = gate.evaluate_and_gate(own, [tgt_safe], action_safe)
    assert record.decision == GateDecision.APPROVE
    assert np.allclose(exec_act, action_safe)


def test_environment_run():
    scenario = create_head_on_scenario(distance=800.0)
    env = MaritimeEnv(scenario_config=scenario)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape

    action = np.array([8.0, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    assert "min_distance" in next_info
    assert "aggregate_risk" in next_info
