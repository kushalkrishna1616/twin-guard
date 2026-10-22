"""
Gymnasium-compatible Maritime Navigation Simulation Environment.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from twinguard.env.vessel import VesselState
from twinguard.scenarios.encounters import ScenarioConfig
from twinguard.risk.metrics import MaritimeRiskEvaluator
from twinguard.risk.cpa import compute_dcpa_tcpa


@dataclass
class Disturbances:
    """Environmental drift and ocean current parameters."""
    current_speed: float = 0.0
    current_direction: float = 0.0 # radians, 0 = North
    wind_speed: float = 0.0
    sensor_noise_std: float = 0.0

    @property
    def drift(self) -> tuple[float, float]:
        dx = self.current_speed * np.sin(self.current_direction)
        dy = self.current_speed * np.cos(self.current_direction)
        return float(dx), float(dy)


class MaritimeEnv(gym.Env):
    """
    Maritime simulation environment for autonomous vessel safety benchmarking.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario_config: ScenarioConfig,
        enable_disturbances: bool = False,
        risk_evaluator: MaritimeRiskEvaluator | None = None
    ):
        super().__init__()
        self.scenario_config = scenario_config
        self.enable_disturbances = enable_disturbances
        self.disturbances = Disturbances()
        self.risk_evaluator = risk_evaluator or MaritimeRiskEvaluator()

        self.dt = scenario_config.dt
        self.max_steps = scenario_config.max_steps

        # Action: [commanded_speed (0-12 m/s), commanded_rudder (-35 deg to +35 deg in rad)]
        max_rudder = float(np.deg2rad(35.0))
        self.action_space = spaces.Box(
            low=np.array([0.0, -max_rudder], dtype=np.float32),
            high=np.array([12.0, max_rudder], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: 4 own-ship features + (3 targets * 6 features) = 22 dims
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32
        )

        self.own_ship: VesselState = self.scenario_config.own_ship.copy()
        self.target_ships: list[VesselState] = [t.copy() for t in self.scenario_config.target_ships]
        self.goal_position: np.ndarray = np.array(self.scenario_config.goal_position, dtype=np.float64)
        self.static_obstacles = getattr(self.scenario_config, 'static_obstacles', [])

        self.current_step = 0
        self.min_distance_overall = float("inf")
        self.collision_occurred = False
        self.domain_violation_occurred = False
        self.colreg_violation_occurred = False
        self.trajectory_history: list[dict] = []

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.own_ship = self.scenario_config.own_ship.copy()
        self.target_ships = [t.copy() for t in self.scenario_config.target_ships]
        self.goal_position = np.array(self.scenario_config.goal_position, dtype=np.float64)

        self.current_step = 0
        self.min_distance_overall = float("inf")
        self.collision_occurred = False
        self.domain_violation_occurred = False
        self.colreg_violation_occurred = False
        self.trajectory_history = []

        self._record_history()
        obs = self._get_obs()
        info = self._get_info(cmd_rudder=0.0)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.current_step += 1
        cmd_speed = float(action[0])
        cmd_rudder = float(action[1])

        drift_x, drift_y = self.disturbances.drift if self.enable_disturbances else (0.0, 0.0)

        prev_dist_goal = float(np.linalg.norm(self.own_ship.position - self.goal_position))

        # 1. Step own ship
        self.own_ship.step(
            commanded_speed=cmd_speed,
            commanded_rudder=cmd_rudder,
            dt=self.dt,
            drift_x=drift_x,
            drift_y=drift_y
        )

        # 2. Step target vessels
        for tgt in self.target_ships:
            tgt.step(
                commanded_speed=tgt.speed,
                commanded_rudder=0.0,
                dt=self.dt,
                drift_x=drift_x,
                drift_y=drift_y
            )

        # 3. Evaluate Safety Metrics
        step_min_dist = float("inf")
        step_collision = False
        step_domain_viol = False
        step_colreg_viol = False
        reports = []

        for tgt in self.target_ships:
            d = float(np.linalg.norm(self.own_ship.position - tgt.position))
            if d < step_min_dist:
                step_min_dist = d
            if d < (self.own_ship.length / 2.0 + tgt.length / 2.0):
                step_collision = True

            rep = self.risk_evaluator.evaluate_target(self.own_ship, tgt, cmd_rudder)
            reports.append(rep)
            if rep.domain_violated:
                step_domain_viol = True
            if not rep.colreg_compliant:
                step_colreg_viol = True


        # Check static obstacles / reef grounding
        for (ox, oy, orad) in self.static_obstacles:
            d_obs = float(np.hypot(self.own_ship.x - ox, self.own_ship.y - oy))
            if d_obs < step_min_dist:
                step_min_dist = d_obs
            if d_obs < (self.own_ship.length / 2.0 + orad):
                step_collision = True
            elif d_obs < (orad + 100.0):
                step_domain_viol = True

        if step_min_dist < self.min_distance_overall:
            self.min_distance_overall = step_min_dist

        if step_collision:
            self.collision_occurred = True
        if step_domain_viol:
            self.domain_violation_occurred = True
        if step_colreg_viol:
            self.colreg_violation_occurred = True

        agg_risk, worst_report = self.risk_evaluator.aggregate_multi_ship_risk(reports)

        curr_dist_goal = float(np.linalg.norm(self.own_ship.position - self.goal_position))
        goal_reached = curr_dist_goal < 60.0

        # Termination & Truncation
        terminated = self.collision_occurred or goal_reached
        truncated = self.current_step >= self.max_steps

        # Reward
        reward = (prev_dist_goal - curr_dist_goal) * 0.1 - 0.1
        if self.collision_occurred:
            reward -= 100.0
        elif step_domain_viol:
            reward -= 10.0
        if step_colreg_viol:
            reward -= 5.0
        if goal_reached:
            reward += 50.0

        self._record_history()
        obs = self._get_obs()
        info = self._get_info(cmd_rudder=cmd_rudder, agg_risk=agg_risk, worst_report=worst_report, goal_reached=goal_reached)

        return obs, float(reward), terminated, truncated, info

    def _record_history(self) -> None:
        self.trajectory_history.append({
            "step": self.current_step,
            "own_x": float(self.own_ship.x),
            "own_y": float(self.own_ship.y),
            "own_speed": float(self.own_ship.speed),
            "own_course": float(self.own_ship.course),
            "targets": [
                {
                    "id": t.id,
                    "x": float(t.x),
                    "y": float(t.y),
                    "speed": float(t.speed),
                    "course": float(t.course)
                }
                for t in self.target_ships
            ],
            "static_obstacles": list(self.static_obstacles)
        })

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(22, dtype=np.float32)
        d_goal = self.goal_position - self.own_ship.position
        dist_goal = float(np.linalg.norm(d_goal))
        desired_heading = float(np.arctan2(d_goal[0], d_goal[1]) % (2 * np.pi))
        heading_err = (desired_heading - self.own_ship.course) % (2 * np.pi)
        if heading_err > np.pi:
            heading_err -= 2 * np.pi

        obs[0] = np.clip(self.own_ship.speed / 12.0, 0.0, 1.5)
        obs[1] = np.clip(self.own_ship.yaw_rate / max(self.own_ship.max_yaw_rate, 1e-3), -1.0, 1.0)
        obs[2] = np.clip(dist_goal / 2000.0, 0.0, 2.0)
        obs[3] = np.clip(heading_err / np.pi, -1.0, 1.0)

        for i, tgt in enumerate(self.target_ships[:3]):
            idx = 4 + i * 6
            dx = tgt.x - self.own_ship.x
            dy = tgt.y - self.own_ship.y
            rel_v = tgt.speed - self.own_ship.speed
            rel_course = (tgt.course - self.own_ship.course) % (2 * np.pi)
            if rel_course > np.pi:
                rel_course -= 2 * np.pi
            dcpa, tcpa = compute_dcpa_tcpa(
                self.own_ship.position, self.own_ship.velocity_vector,
                tgt.position, tgt.velocity_vector
            )
            obs[idx] = np.clip(dx / 2000.0, -2.0, 2.0)
            obs[idx + 1] = np.clip(dy / 2000.0, -2.0, 2.0)
            obs[idx + 2] = np.clip(rel_v / 15.0, -1.5, 1.5)
            obs[idx + 3] = np.clip(rel_course / np.pi, -1.0, 1.0)
            obs[idx + 4] = np.clip(dcpa / 1000.0, 0.0, 2.0)
            obs[idx + 5] = np.clip(tcpa / 120.0, -1.0, 2.0)

        return obs

    def _get_info(
        self,
        cmd_rudder: float = 0.0,
        agg_risk: float = 0.0,
        worst_report=None,
        goal_reached: bool = False
    ) -> dict:
        min_dist = float("inf")
        for tgt in self.target_ships:
            d = float(np.linalg.norm(self.own_ship.position - tgt.position))
            if d < min_dist:
                min_dist = d

        return {
            "min_distance": float(self.min_distance_overall if self.min_distance_overall != float("inf") else min_dist),
            "current_min_distance": float(min_dist),
            "aggregate_risk": float(agg_risk),
            "worst_report": worst_report,
            "collision": self.collision_occurred,
            "domain_violation": self.domain_violation_occurred,
            "colreg_violation": self.colreg_violation_occurred,
            "goal_reached": goal_reached,
            "own_pos": [float(self.own_ship.x), float(self.own_ship.y)],
            "own_course_deg": float(np.rad2deg(self.own_ship.course)),
            "goal_pos": [float(self.goal_position[0]), float(self.goal_position[1])],
            "target_positions": {t.id: [float(t.x), float(t.y)] for t in self.target_ships}
        }
