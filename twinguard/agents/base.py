"""
Base Navigation Agent Interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract interface for all black-box navigation agents."""

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    @abstractmethod
    def predict(self, observation: np.ndarray, info: dict | None = None) -> np.ndarray:
        """
        Produce [commanded_speed, commanded_rudder] action.
        """
        pass
