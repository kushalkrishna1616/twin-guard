"""
COLREG Rules Classification and Compliance Verification (Rules 13, 14, 15, 16, 17).
"""

from __future__ import annotations
from enum import Enum
import numpy as np
from twinguard.env.vessel import VesselState


class EncounterType(str, Enum):
    NONE = "none"
    HEAD_ON = "head_on"           # Rule 14
    CROSSING_GIVE_WAY = "crossing_give_way" # Rule 15 (Target on starboard)
    CROSSING_STAND_ON = "crossing_stand_on" # Rule 15 (Target on port)
    OVERTAKING_GIVE_WAY = "overtaking_give_way" # Rule 13 (Own ship is overtaking)
    OVERTAKEN_STAND_ON = "overtaken_stand_on"   # Rule 13 (Own ship is being overtaken)


def compute_relative_bearing(source: VesselState, target: VesselState) -> float:
    """
    Computes relative bearing of target from source vessel (in radians [0, 2*pi)).
    0 rad = dead ahead, pi/2 rad = beam starboard, pi rad = dead astern, 3*pi/2 = beam port.
    """
    dx = target.x - source.x
    dy = target.y - source.y
    true_bearing = (np.arctan2(dx, dy)) % (2 * np.pi)
    rel_bearing = (true_bearing - source.course) % (2 * np.pi)
    return float(rel_bearing)


def classify_colreg_encounter(own_ship: VesselState, target_ship: VesselState) -> tuple[EncounterType, str]:
    """
    Determines encounter type between own_ship and target_ship according to COLREG rules.
    """
    rel_bearing_own_to_tgt = compute_relative_bearing(own_ship, target_ship)
    rel_bearing_tgt_to_own = compute_relative_bearing(target_ship, own_ship)

    # Relative course angle in degrees
    course_diff_deg = float(np.rad2deg(abs(own_ship.course - target_ship.course)) % 360)
    if course_diff_deg > 180:
        course_diff_deg = 360 - course_diff_deg

    rel_b_deg = float(np.rad2deg(rel_bearing_own_to_tgt))
    tgt_b_deg = float(np.rad2deg(rel_bearing_tgt_to_own))

    # Rule 13: Overtaking (coming up > 22.5 deg abaft the beam, i.e. 112.5 to 247.5 deg)
    if 112.5 <= tgt_b_deg <= 247.5 and own_ship.speed > target_ship.speed:
        return EncounterType.OVERTAKING_GIVE_WAY, "Rule 13: Own ship is overtaking target. Must keep well clear."

    if 112.5 <= rel_b_deg <= 247.5 and target_ship.speed > own_ship.speed:
        return EncounterType.OVERTAKEN_STAND_ON, "Rule 13: Target ship is overtaking own ship. Own ship stands on."

    # Rule 14: Head-on (reciprocal courses within ~15 deg, meeting nearly dead ahead)
    if (165.0 <= course_diff_deg <= 180.0 or 0.0 <= course_diff_deg <= 15.0) and \
       (rel_b_deg <= 15.0 or rel_b_deg >= 345.0) and \
       (tgt_b_deg <= 15.0 or tgt_b_deg >= 345.0):
        return EncounterType.HEAD_ON, "Rule 14: Head-on encounter. Both vessels must alter course to starboard."

    # Rule 15: Crossing
    if 15.0 < rel_b_deg <= 112.5:
        # Target is on own-ship's starboard side -> Own ship is GIVE-WAY
        return EncounterType.CROSSING_GIVE_WAY, "Rule 15: Crossing encounter. Target on starboard; own ship is Give-way."

    if 247.5 <= rel_b_deg < 345.0:
        # Target is on own-ship's port side -> Own ship is STAND-ON
        return EncounterType.CROSSING_STAND_ON, "Rule 15: Crossing encounter. Target on port; own ship is Stand-on."

    return EncounterType.NONE, "Normal clearance or non-conflicting courses."


def check_colreg_compliance(
    encounter_type: EncounterType,
    commanded_rudder: float,
    current_distance: float,
    dcpa: float,
    safe_distance: float = 300.0
) -> tuple[bool, str]:
    """
    Verify if commanded rudder angle obeys COLREG rules for the active encounter.
    commanded_rudder > 0 is Starboard (turn right).
    commanded_rudder < 0 is Port (turn left).
    """
    if current_distance > 1500.0:
        # Outside critical encounter interaction zone
        return True, "Clear range (>1500m); maneuvering permitted."

    # In Head-on situation (Rule 14), turning to PORT (left) is strictly prohibited
    if encounter_type == EncounterType.HEAD_ON:
        if commanded_rudder < -np.deg2rad(1.0):
            return False, "COLREG Rule 14 Violation: Cannot turn to PORT in a Head-on encounter! Must alter course to Starboard."
        return True, "Compliant with Rule 14 (maintaining or turning starboard)."

    # In Crossing Give-Way (Rule 15), turning port across target's bow is forbidden
    if encounter_type == EncounterType.CROSSING_GIVE_WAY:
        if commanded_rudder < -np.deg2rad(1.0) and dcpa < safe_distance:
            return False, "COLREG Rule 15 Violation: Give-way vessel turning to PORT towards crossing target."
        return True, "Compliant with Rule 15 (giving way to starboard or holding clear)."

    # In Crossing Stand-On (Rule 17), vessel should maintain course unless collision is imminent
    if encounter_type == EncounterType.CROSSING_STAND_ON:
        if abs(commanded_rudder) > np.deg2rad(5.0) and current_distance > 500.0:
            return True, "Rule 17 advisory: Stand-on vessel altering course early."
        return True, "Compliant with Rule 17."

    return True, "Compliant."
