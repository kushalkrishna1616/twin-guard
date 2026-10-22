"""
Ship Domain Modelling and Violation Detection (Fujii and Goodwin Models).
"""

from __future__ import annotations
import math
import numpy as np
from twinguard.env.vessel import VesselState


class ShipDomain:
    """
    Fujii-style elliptical safety domain around own-ship.
    Fore semi-axis a_fore = k_fore * L
    Aft semi-axis a_aft = k_aft * L
    Starboard semi-axis b_starboard = k_stbd * L
    Port semi-axis b_port = k_port * L
    """
    def __init__(
        self,
        k_fore: float = 6.0,
        k_aft: float = 2.5,
        k_stbd: float = 3.5,
        k_port: float = 2.5
    ):
        self.k_fore = k_fore
        self.k_aft = k_aft
        self.k_stbd = k_stbd
        self.k_port = k_port

    def evaluate(self, own_ship: VesselState, target_ship: VesselState) -> tuple[bool, float]:
        """
        Evaluate if target_ship penetrates own_ship's safety domain.
        Returns:
            (is_violated, domain_ratio)
            domain_ratio < 1.0 means inside the domain (violation).
            domain_ratio = 1.0 means on the perimeter.
            domain_ratio > 1.0 means safe outside.
        """
        # Relative vector in world frame
        dx = target_ship.x - own_ship.x
        dy = target_ship.y - own_ship.y

        # Rotate into own-ship body frame
        cos_psi = math.cos(own_ship.course)
        sin_psi = math.sin(own_ship.course)

        # Body coordinates: x_b = starboard (+), y_b = fore/bow (+)
        x_body = dx * cos_psi - dy * sin_psi
        y_body = dx * sin_psi + dy * cos_psi

        L = own_ship.length

        # Asymmetric axes depending on quadrant
        a = (self.k_fore * L) if y_body >= 0 else (self.k_aft * L)
        b = (self.k_stbd * L) if x_body >= 0 else (self.k_port * L)

        # Ellipse normalized equation: (x/b)^2 + (y/a)^2
        val = (x_body / max(b, 1.0)) ** 2 + (y_body / max(a, 1.0)) ** 2
        domain_ratio = math.sqrt(val)

        is_violated = domain_ratio < 1.0
        return is_violated, domain_ratio

    def get_domain_ellipse_points(self, own_ship: VesselState, num_points: int = 60) -> np.ndarray:
        """
        Generate (num_points, 2) array of [x, y] coordinates in world frame for plotting.
        """
        thetas = np.linspace(0, 2 * np.pi, num_points)
        L = own_ship.length

        xs_body = []
        ys_body = []
        for theta in thetas:
            dx_unit = np.sin(theta)
            dy_unit = np.cos(theta)

            a = (self.k_fore * L) if dy_unit >= 0 else (self.k_aft * L)
            b = (self.k_stbd * L) if dx_unit >= 0 else (self.k_port * L)

            xs_body.append(b * dx_unit)
            ys_body.append(a * dy_unit)

        xs_b = np.array(xs_body)
        ys_b = np.array(ys_body)

        # Rotate back to world frame
        cos_psi = np.cos(own_ship.course)
        sin_psi = np.sin(own_ship.course)

        x_world = own_ship.x + (xs_b * cos_psi + ys_b * sin_psi)
        y_world = own_ship.y + (-xs_b * sin_psi + ys_b * cos_psi)

        return np.column_stack([x_world, y_world])
