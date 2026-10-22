"""
Maritime Encounter Scenario Generator for TwinGuard.
Implements COLREG Rule 14 (Head-on), Rule 15 (Crossing), Rule 13 (Overtaking),
and multi-vessel scenarios.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from twinguard.env.vessel import VesselState


@dataclass
class ScenarioConfig:
    """Configuration for an encounter scenario."""
    name: str
    description: str
    own_ship: VesselState
    target_ships: list[VesselState]
    goal_position: np.ndarray  # [x, y] in meters
    max_steps: int = 150
    dt: float = 1.0
    static_obstacles: list[tuple[float, float, float]] = field(default_factory=list)


def create_head_on_scenario(distance: float = 1200.0, speed: float = 8.0) -> ScenarioConfig:
    """
    COLREG Rule 14: Head-on situation.
    Two vessels meeting on reciprocal courses with risk of collision.
    Both vessels must alter course to starboard (turn right).
    """
    own_ship = VesselState(
        id="own_ship",
        x=0.0,
        y=0.0,
        speed=speed,
        course=0.0, # Heading North
        length=50.0,
        beam=10.0
    )

    # Target ship approaching directly from North heading South
    target_ship = VesselState(
        id="target_head_on",
        x=0.0,
        y=distance,
        speed=speed,
        course=np.pi, # Heading South (180 deg)
        length=60.0,
        beam=12.0
    )

    return ScenarioConfig(
        name="COLREG Rule 14: Head-on Encounter",
        description="Two vessels meeting head-on on reciprocal courses. Both must turn starboard.",
        own_ship=own_ship,
        target_ships=[target_ship],
        goal_position=np.array([0.0, distance + 300.0]),
        max_steps=220,
        dt=1.0
    )


def create_crossing_scenario(
    distance: float = 1000.0,
    own_is_give_way: bool = True,
    speed: float = 8.0
) -> ScenarioConfig:
    """
    COLREG Rule 15: Crossing situation.
    If target is on own ship's starboard side, own ship is GIVE-WAY.
    If target is on port side, own ship is STAND-ON.
    """
    own_ship = VesselState(
        id="own_ship",
        x=0.0,
        y=0.0,
        speed=speed,
        course=0.0, # Heading North
        length=50.0,
        beam=10.0
    )

    # Collision point ahead around y = 600m
    t_collision = 600.0 / speed

    if own_is_give_way:
        # Target crossing from Starboard (+x, East) towards West (-x)
        target_x = distance * 0.8
        target_y = 600.0
        target_speed = target_x / t_collision
        target_ship = VesselState(
            id="target_crossing_starboard",
            x=target_x,
            y=target_y,
            speed=target_speed,
            course=3 * np.pi / 2, # Heading West (270 deg)
            length=55.0,
            beam=11.0
        )
        name = "COLREG Rule 15: Crossing (Own Ship Give-Way)"
        desc = "Target approaches from starboard. Own ship must keep out of the way (alter course starboard)."
    else:
        # Target crossing from Port (-x, West) towards East (+x)
        target_x = -distance * 0.8
        target_y = 600.0
        target_speed = abs(target_x) / t_collision
        target_ship = VesselState(
            id="target_crossing_port",
            x=target_x,
            y=target_y,
            speed=target_speed,
            course=np.pi / 2, # Heading East (90 deg)
            length=55.0,
            beam=11.0
        )
        name = "COLREG Rule 15: Crossing (Own Ship Stand-On)"
        desc = "Target approaches from port. Target is give-way; own ship maintains course & speed unless target fails."

    return ScenarioConfig(
        name=name,
        description=desc,
        own_ship=own_ship,
        target_ships=[target_ship],
        goal_position=np.array([0.0, 1200.0]),
        max_steps=180,
        dt=1.0
    )


def create_overtaking_scenario(own_is_overtaking: bool = True, speed: float = 9.0) -> ScenarioConfig:
    """
    COLREG Rule 13: Overtaking situation.
    The overtaking vessel must keep clear of the overtaken vessel.
    """
    if own_is_overtaking:
        own_ship = VesselState(
            id="own_ship",
            x=0.0,
            y=0.0,
            speed=speed,
            course=0.0,
            length=50.0,
            beam=10.0
        )
        # Slower target ahead
        target_ship = VesselState(
            id="target_slower_ahead",
            x=5.0,
            y=350.0,
            speed=4.5,
            course=0.0,
            length=70.0,
            beam=14.0
        )
        name = "COLREG Rule 13: Overtaking (Own Ship Overtaking)"
        desc = "Own ship is faster and overtaking slower vessel ahead. Own ship must keep clear."
    else:
        own_ship = VesselState(
            id="own_ship",
            x=0.0,
            y=350.0,
            speed=4.5,
            course=0.0,
            length=50.0,
            beam=10.0
        )
        # Faster target overtaking own ship from behind
        target_ship = VesselState(
            id="target_faster_behind",
            x=5.0,
            y=0.0,
            speed=speed,
            course=0.0,
            length=70.0,
            beam=14.0
        )
        name = "COLREG Rule 13: Overtaking (Target Overtaking)"
        desc = "Faster target is overtaking own ship from astern. Target vessel must keep clear."

    return ScenarioConfig(
        name=name,
        description=desc,
        own_ship=own_ship,
        target_ships=[target_ship],
        goal_position=np.array([0.0, 1400.0]),
        max_steps=200,
        dt=1.0
    )


def create_multi_vessel_scenario() -> ScenarioConfig:
    """
    Complex multi-vessel encounter with 3 target vessels:
    1 head-on vessel, 1 starboard crossing vessel, and 1 overtaking vessel.
    """
    own_ship = VesselState(
        id="own_ship",
        x=0.0,
        y=0.0,
        speed=7.5,
        course=0.0,
        length=50.0,
        beam=10.0
    )

    t1 = VesselState(
        id="target_1_headon",
        x=20.0,
        y=1100.0,
        speed=7.0,
        course=np.pi,
        length=60.0,
        beam=12.0
    )

    t2 = VesselState(
        id="target_2_crossing",
        x=750.0,
        y=550.0,
        speed=6.0,
        course=3 * np.pi / 2, # West
        length=50.0,
        beam=10.0
    )

    t3 = VesselState(
        id="target_3_diagonal",
        x=-600.0,
        y=700.0,
        speed=5.5,
        course=np.pi / 4, # North-East
        length=45.0,
        beam=9.0
    )

    return ScenarioConfig(
        name="Multi-Vessel Cluttered Encounter",
        description="Complex 3-ship encounter requiring multi-hazard collision avoidance and COLREG prioritisation.",
        own_ship=own_ship,
        target_ships=[t1, t2, t3],
        goal_position=np.array([0.0, 1500.0]),
        max_steps=240,
        dt=1.0
    )



def create_island_channel_scenario() -> ScenarioConfig:
    """
    Restricted Channel with Island / Coral Reef Barrier and Crossing Vessel.
    Navigating narrow strait while avoiding grounding and crossing ship.
    """
    own_ship = VesselState(
        id="own_ship",
        x=0.0,
        y=0.0,
        speed=7.5,
        course=0.0,
        length=50.0,
        beam=10.0
    )

    crossing_target = VesselState(
        id="target_channel_cross",
        x=800.0,
        y=650.0,
        speed=6.5,
        course=3 * np.pi / 2, # West
        length=55.0,
        beam=11.0
    )

    # Island at (150m, 500m) with 110m radius reef
    obstacles = [(180.0, 520.0, 110.0), (-250.0, 850.0, 80.0)]

    return ScenarioConfig(
        name="Narrow Strait with Rocky Island & Reef",
        description="Restricted navigation channel flanked by a rocky island barrier and an oncoming crossing vessel.",
        own_ship=own_ship,
        target_ships=[crossing_target],
        goal_position=np.array([0.0, 1400.0]),
        max_steps=200,
        dt=1.0,
        static_obstacles=obstacles
    )

SCENARIO_BUILDERS = {
    "head_on": create_head_on_scenario,
    "crossing_give_way": lambda: create_crossing_scenario(own_is_give_way=True),
    "crossing_stand_on": lambda: create_crossing_scenario(own_is_give_way=False),
    "overtaking": create_overtaking_scenario,
    "multi_vessel": create_multi_vessel_scenario,
    "island_channel": create_island_channel_scenario,
}
