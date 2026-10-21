"""
Reinforcement Learning Navigation Agent wrapper (PPO).
"""

from __future__ import annotations
import os
import numpy as np
from twinguard.agents.base import BaseAgent

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class RLAgent(BaseAgent):
    """
    Learned Deep RL navigation agent (PPO).
    Can load a trained policy or train dynamically.
    """
    def __init__(
        self,
        model_path: str | None = None,
        name: str = "PPO_RL_Agent"
    ):
        super().__init__(name=name)
        self.model_path = model_path
        self.model = None

        if model_path and os.path.exists(model_path) and SB3_AVAILABLE:
            try:
                self.model = PPO.load(model_path)
            except Exception as e:
                print(f"Warning: Failed to load PPO model: {e}")

    def predict(self, observation: np.ndarray, info: dict | None = None) -> np.ndarray:
        if self.model is not None:
            action, _ = self.model.predict(observation, deterministic=True)
            return np.array(action, dtype=np.float32)

        # Fallback neural/heuristic policy if no weights loaded yet:
        # Extract features from observation:
        # obs[0] = own_speed/12, obs[1] = yaw_rate/max_r, obs[2] = dist_goal, obs[3] = angle_goal/pi
        angle_to_goal = observation[3] * np.pi
        target_speed = 8.0

        # Reactive collision avoidance heuristic mimicking an imperfect RL policy
        # which turns left or right depending on nearest target relative position
        rudder = float(np.clip(1.2 * angle_to_goal, -np.deg2rad(30.0), np.deg2rad(30.0)))

        # Target 1 info starts at obs[4]: [dx, dy, rel_v, rel_course, dcpa, tcpa]
        if len(observation) >= 10:
            dx = observation[4] * 2000.0
            dy = observation[5] * 2000.0
            dcpa = observation[8] * 1000.0
            tcpa = observation[9] * 120.0

            dist = np.hypot(dx, dy)
            if dist < 650.0 and (0.0 < tcpa < 90.0):
                # Imperfect RL agent might turn slightly port (negative rudder)
                # or fail to give way, triggering the Safety Gate!
                rudder += float(np.sign(dx) * np.deg2rad(15.0))

        return np.array([target_speed, rudder], dtype=np.float32)
