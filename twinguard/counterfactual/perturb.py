"""
Perturbation Models for Counterfactual 'What-If' Scenario Generation.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from twinguard.env.vessel import VesselState


@dataclass
class PerturbationParams:
    """Parameters controlling magnitude of counterfactual variations."""
    target_course_noise_deg: float = 12.0  # Possible course changes by other vessels (deg)
    target_speed_noise_pct: float = 0.15   # Possible speed variation (+/- 15%)
    sensor_pos_noise_std: float = 8.0      # Radar/GPS position uncertainty (m)
    sensor_bearing_noise_deg: float = 2.0  # Bearing uncertainty (deg)
    current_speed_noise: float = 0.3       # Current variation (m/s)
    current_dir_noise_rad: float = np.pi/6 # Current angle fluctuation (rad)


class CounterfactualPerturbator:
    """
    Applies plausible alternative variations to vessels and the environment.
    """
    def __init__(self, params: PerturbationParams | None = None):
        self.params = params or PerturbationParams()

    def generate_perturbed_target(
        self,
        base_target: VesselState,
        rng: np.random.Generator
    ) -> VesselState:
        """
        Creates a perturbed target vessel with randomized course and speed shifts.
        """
        tgt = base_target.copy()

        # Course perturbation
        d_course = np.deg2rad(rng.normal(0, self.params.target_course_noise_deg))
        tgt.course = (tgt.course + d_course) % (2 * np.pi)

        # Speed perturbation
        speed_factor = 1.0 + rng.uniform(-self.params.target_speed_noise_pct, self.params.target_speed_noise_pct)
        tgt.speed = float(np.clip(tgt.speed * speed_factor, 1.0, tgt.max_speed))

        # Position sensing jitter
        tgt.x += float(rng.normal(0, self.params.sensor_pos_noise_std))
        tgt.y += float(rng.normal(0, self.params.sensor_pos_noise_std))

        return tgt

    def sample_drift(
        self,
        base_drift_x: float,
        base_drift_y: float,
        rng: np.random.Generator
    ) -> tuple[float, float]:
        """
        Sample perturbed environmental drift.
        """
        dx = base_drift_x + float(rng.normal(0, self.params.current_speed_noise))
        dy = base_drift_y + float(rng.normal(0, self.params.current_speed_noise))
        return dx, dy
