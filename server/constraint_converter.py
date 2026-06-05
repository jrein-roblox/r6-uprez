"""Convert plugin world-space constraints to Kimodo constraint format.

The plugin stores constraints as world-space positions (studs, Y-up, -Z forward).
Kimodo expects Y-up, meters, +Z forward with specific effector naming.

Reuses coordinate math from python/roblox_to_kimodo.py.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

STUD_TO_METER = 0.30

EFFECTOR_TO_KIMODO = {
    "left_hand": "LeftHand",
    "right_hand": "RightHand",
    "left_foot": "LeftFoot",
    "right_foot": "RightFoot",
    "hips": "Hips",
    "root": "Root2D",
}

EFFECTOR_TYPES = {
    "left_hand": "EndEffector",
    "right_hand": "EndEffector",
    "left_foot": "EndEffector",
    "right_foot": "EndEffector",
    "hips": "Hips",
    "root": "Root2D",
}


def roblox_pos_to_kimodo(pos: list[float]) -> list[float]:
    """Convert Roblox position (studs, Y-up, -Z forward) to Kimodo (meters, Y-up, +Z forward)."""
    x, y, z = pos
    return [
        -x * STUD_TO_METER,
        y * STUD_TO_METER,
        -z * STUD_TO_METER,
    ]


def roblox_quat_to_kimodo(quat: list[float]) -> list[float]:
    """Convert Roblox quaternion [qx,qy,qz,qw] to Kimodo [qw,qx,qy,qz] with axis flip."""
    qx, qy, qz, qw = quat
    return [qw, -qx, qy, -qz]


def convert_constraints(
    constraints: list[dict[str, Any]],
    duration: float,
    floor_y: float = 0.0,
) -> dict[str, Any]:
    """Convert plugin constraints to Kimodo constraints.json format.

    Args:
        constraints: List of {effector, time, position, rotation} from plugin.
        duration: Total animation duration in seconds.
        floor_y: Y offset for floor normalization (studs). Usually lowest foot Y.

    Returns:
        Kimodo-format constraints dict ready for kimodo_gen.
    """
    fps = 30
    n_frames = int(math.ceil(duration * fps)) + 1

    kimodo_constraints: dict[str, Any] = {
        "fps": fps,
        "n_frames": n_frames,
        "duration_s": duration,
        "effectors": {},
    }

    grouped: dict[str, list[dict]] = {}
    for c in constraints:
        eff = c["effector"]
        grouped.setdefault(eff, []).append(c)

    for effector_name, keyframes in grouped.items():
        kimodo_name = EFFECTOR_TO_KIMODO.get(effector_name)
        if not kimodo_name:
            continue

        eff_type = EFFECTOR_TYPES[effector_name]
        frames: list[int] = []
        positions: list[list[float]] = []
        rotations: list[list[float]] = []

        for kf in sorted(keyframes, key=lambda k: k["time"]):
            frame_idx = int(round(kf["time"] * fps))
            frame_idx = max(0, min(frame_idx, n_frames - 1))
            frames.append(frame_idx)

            pos = kf["position"]
            pos_adjusted = [pos[0], pos[1] - floor_y, pos[2]]
            positions.append(roblox_pos_to_kimodo(pos_adjusted))

            if "rotation" in kf and kf["rotation"]:
                rotations.append(roblox_quat_to_kimodo(kf["rotation"]))

        entry: dict[str, Any] = {
            "type": eff_type,
            "frames": frames,
            "positions": positions,
        }
        if rotations:
            entry["rotations"] = rotations

        kimodo_constraints["effectors"][kimodo_name] = entry

    return kimodo_constraints
