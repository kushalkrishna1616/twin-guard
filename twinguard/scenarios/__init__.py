"""
TwinGuard Scenarios Module.
"""

from twinguard.scenarios.encounters import (
    ScenarioConfig,
    create_head_on_scenario,
    create_crossing_scenario,
    create_overtaking_scenario,
    create_multi_vessel_scenario,
    SCENARIO_BUILDERS,
)
from twinguard.scenarios.ais_loader import generate_synthetic_ais_dataset

__all__ = [
    "ScenarioConfig",
    "create_head_on_scenario",
    "create_crossing_scenario",
    "create_overtaking_scenario",
    "create_multi_vessel_scenario",
    "SCENARIO_BUILDERS",
    "generate_synthetic_ais_dataset",
]
