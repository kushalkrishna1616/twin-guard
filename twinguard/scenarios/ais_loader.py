"""
AIS Data Loader and Synthetic AIS Trajectory Generator.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from twinguard.env.vessel import VesselState


def generate_synthetic_ais_dataset(num_encounters: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic AIS trajectories representing varied maritime encounters.
    Returns DataFrame with columns: ['encounter_id', 'step', 'time_s', 'mmsi', 'ship_type', 'x', 'y', 'sog', 'cog'].
    """
    rng = np.random.default_rng(seed)
    records = []

    for enc_id in range(num_encounters):
        enc_type = rng.choice(["head_on", "crossing", "overtaking"])
        duration = int(rng.integers(80, 150))
        dt = 1.0

        # Own ship
        own_sog = float(rng.uniform(6.0, 9.0))
        own_cog = 0.0 # North
        own_x, own_y = 0.0, 0.0

        # Target vessel based on encounter type
        if enc_type == "head_on":
            tgt_x = float(rng.normal(0.0, 30.0))
            tgt_y = float(rng.uniform(900.0, 1400.0))
            tgt_sog = float(rng.uniform(6.0, 9.0))
            tgt_cog = np.pi + float(rng.normal(0.0, 0.05))
        elif enc_type == "crossing":
            side = rng.choice([-1.0, 1.0]) # -1 = port, 1 = starboard
            tgt_x = side * float(rng.uniform(600.0, 900.0))
            tgt_y = float(rng.uniform(400.0, 700.0))
            tgt_sog = float(rng.uniform(5.5, 8.5))
            tgt_cog = (3 * np.pi / 2 if side > 0 else np.pi / 2) + float(rng.normal(0.0, 0.05))
        else: # overtaking
            tgt_x = float(rng.normal(0.0, 20.0))
            tgt_y = float(rng.uniform(250.0, 450.0))
            tgt_sog = float(rng.uniform(3.5, 5.0))
            tgt_cog = float(rng.normal(0.0, 0.05))

        tgt_curr_x, tgt_curr_y = tgt_x, tgt_y
        for step in range(duration):
            t = step * dt
            # Own ship progression
            records.append({
                "encounter_id": enc_id,
                "encounter_type": enc_type,
                "step": step,
                "time_s": t,
                "mmsi": 9990001,
                "role": "own_ship",
                "x": round(own_x, 2),
                "y": round(own_y, 2),
                "sog": round(own_sog, 2),
                "cog_deg": round(float(np.rad2deg(own_cog)) % 360, 2)
            })
            own_y += own_sog * dt

            # Target ship progression
            records.append({
                "encounter_id": enc_id,
                "encounter_type": enc_type,
                "step": step,
                "time_s": t,
                "mmsi": 9990002 + enc_id,
                "role": "target_ship",
                "x": round(tgt_curr_x, 2),
                "y": round(tgt_curr_y, 2),
                "sog": round(tgt_sog, 2),
                "cog_deg": round(float(np.rad2deg(tgt_cog)) % 360, 2)
            })
            tgt_curr_x += tgt_sog * np.sin(tgt_cog) * dt
            tgt_curr_y += tgt_sog * np.cos(tgt_cog) * dt

    df = pd.DataFrame(records)
    return df
