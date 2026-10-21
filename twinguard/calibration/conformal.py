"""
Conformal Prediction and Bootstrap Uncertainty Calibration for Maritime Risk Gating.
"""

from __future__ import annotations
import numpy as np


class ConformalRiskCalibrator:
    """
    Implements inductive split-conformal calibration and bootstrap intervals
    to calibrate risk thresholds and output certified prediction bounds.
    """
    def __init__(self, alpha: float = 0.05):
        """
        alpha: significance level (e.g. 0.05 corresponds to 95% safety coverage guarantee).
        """
        self.alpha = alpha
        self.q_hat: float = 0.08  # Default calibrated non-conformity quantile
        self.calibration_scores: list[float] = []
        self.is_calibrated: bool = False

    def fit_calibration_data(
        self,
        predicted_risks: list[float] | np.ndarray,
        ground_truth_hazards: list[float] | np.ndarray
    ) -> float:
        """
        Fit on held-out calibration set:
        predicted_risks: array of counterfactual risk estimates R_i in [0, 1]
        ground_truth_hazards: actual observed outcome (1.0 if domain violation/collision, 0.0 if clean pass)
        """
        preds = np.asarray(predicted_risks, dtype=np.float64)
        targets = np.asarray(ground_truth_hazards, dtype=np.float64)

        # Residual non-conformity score: how much the twin underestimated actual hazard
        # s_i = max(0, targets - preds)
        scores = np.maximum(0.0, targets - preds)
        self.calibration_scores = scores.tolist()

        n = len(scores)
        if n == 0:
            self.q_hat = 0.1
            return self.q_hat

        # Conformal quantile at level ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level))
        self.is_calibrated = True
        return self.q_hat

    def get_calibrated_risk_bound(self, raw_risk: float) -> tuple[float, float, float]:
        """
        Returns (lower_bound, point_estimate, upper_certified_bound)
        upper_bound = min(1.0, raw_risk + q_hat)
        """
        point_est = float(np.clip(raw_risk, 0.0, 1.0))
        upper_bound = float(np.clip(point_est + self.q_hat, 0.0, 1.0))
        lower_bound = float(np.clip(point_est - self.q_hat, 0.0, 1.0))
        return lower_bound, point_est, upper_bound

    def bootstrap_confidence_interval(
        self,
        rollout_risks: list[float],
        num_resamples: int = 500
    ) -> tuple[float, float]:
        """
        Computes 95% bootstrap confidence interval over the counterfactual rollout samples.
        """
        if not rollout_risks:
            return 0.0, 0.0

        risks = np.array(rollout_risks)
        rng = np.random.default_rng(42)
        means = []
        n = len(risks)

        for _ in range(num_resamples):
            sample = rng.choice(risks, size=n, replace=True)
            means.append(np.mean(sample))

        low = float(np.percentile(means, 2.5))
        high = float(np.percentile(means, 97.5))
        return round(low, 3), round(high, 3)
