# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Rig-agnostic helpers extracted from motion-matching's `uefn_to_soma.py`.

Only the pieces that don't depend on UEFN bind tables. Used by
`roblox_to_kimodo.py` for sparse-keyframe selection.
"""
from __future__ import annotations

import numpy as np


# SOMA30 neutral joint positions, frozen from
# `kimodo.skeleton.SOMASkeleton30().neutral_joints`. Same data Kimodo's
# constraint loader feeds to `SOMASkeleton30.fk()` so an identity
# axis-angle rotation chain places each leaf at exactly these world
# offsets when the root is at the origin. Used by roblox_to_kimodo to
# back-solve a "fake" root_positions value that puts a single effector
# at a desired world position under T-pose FK.
SOMA30_BONE_ORDER = [
    "Hips", "Spine1", "Spine2", "Chest",
    "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]
SOMA30_INDEX = {n: i for i, n in enumerate(SOMA30_BONE_ORDER)}
SOMA30_NEUTRAL_JOINTS = np.array([
    [+0.0000000000, +0.0000000000, +0.0000000000],  #  0 Hips
    [-0.0001372700, +0.0500376256, -0.0005372667],  #  1 Spine1
    [-0.0001372719, +0.1212906395, -0.0008355152],  #  2 Spine2
    [-0.0001372776, +0.1967912701, -0.0089952252],  #  3 Chest
    [-0.0019540428, +0.4599042226, -0.0145287081],  #  4 Neck1
    [-0.0019540713, +0.5369981890, +0.0084971465],  #  5 Neck2
    [-0.0019541173, +0.5982873485, +0.0280342327],  #  6 Head
    [-0.0019277485, +0.6030432710, +0.0589836388],  #  7 Jaw
    [+0.0301096906, +0.6520893997, +0.1039030634],  #  8 LeftEye
    [-0.0341785190, +0.6519060385, +0.1036165686],  #  9 RightEye
    [+0.0160792399, +0.4291629107, +0.0421389072],  # 10 LeftShoulder
    [+0.1652776971, +0.4291629327, -0.0128843504],  # 11 LeftArm
    [+0.4526707750, +0.4291629352, -0.0129102291],  # 12 LeftForeArm
    [+0.7236105870, +0.4291629281, -0.0128841394],  # 13 LeftHand
    [+0.8462968538, +0.3969611708, +0.0354465482],  # 14 LeftHandThumbEnd
    [+0.9137301824, +0.4260341442, -0.0132237098],  # 15 LeftHandMiddleEnd
    [-0.0139384600, +0.4285943556, +0.0431463534],  # 16 RightShoulder
    [-0.1643104221, +0.4285944730, -0.0123096903],  # 17 RightArm
    [-0.4516768153, +0.4285944917, -0.0123356612],  # 18 RightForeArm
    [-0.7230130129, +0.4285944906, -0.0123095343],  # 19 RightHand
    [-0.8456554961, +0.3964799458, +0.0357308561],  # 20 RightHandThumbEnd
    [-0.9130189577, +0.4255283351, -0.0126252686],  # 21 RightHandMiddleEnd
    [+0.1004321400, -0.0843452671, +0.0259565473],  # 22 LeftLeg
    [+0.1004321300, -0.5165628043, +0.0179274193],  # 23 LeftShin
    [+0.1004321400, -0.9381137634, -0.0168878105],  # 24 LeftFoot
    [+0.1004321400, -0.9887084839, +0.1154274833],  # 25 LeftToeBase
    [-0.1004727800, -0.0829525995, +0.0262031695],  # 26 RightLeg
    [-0.1004727700, -0.5165746585, +0.0181476112],  # 27 RightShin
    [-0.1004727500, -0.9377486018, -0.0166363673],  # 28 RightFoot
    [-0.1004727534, -0.9885446950, +0.1162055883],  # 29 RightToeBase
], dtype=np.float64)


def _gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.copy()
    radius = int(np.ceil(3.0 * sigma))
    width = 2 * radius + 1
    kernel = np.exp(-0.5 * ((np.arange(width) - radius) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_velocity_extremes(
    joint_world_pos: np.ndarray,  # (F, 3) in meters
    fps: int = 30,
    smooth_sigma: float = 1.5,
    min_separation_frames: int = 8,
    max_picks: int | None = None,
) -> list[int]:
    """Local minima AND maxima of XZ-speed, NMS within each type independently.

    Valleys = planted/stationary moments; peaks = mid-swing fastest moments.
    In a walk cycle valleys and peaks alternate every quarter-cycle (~5 frames
    at 30 fps for a typical 0.7s gait), so a single shared NMS at 8 frames
    would force us to keep only one type per cycle. We NMS minima and maxima
    independently (slowest-first and fastest-first respectively), then merge.
    A valley and peak can land 1-2 frames apart — that's fine, they encode
    different motion moments.
    """
    F = joint_world_pos.shape[0]
    if F < 3:
        return list(range(F))
    vel = np.zeros((F, 2))
    vel[1:-1] = 0.5 * (joint_world_pos[2:, [0, 2]] - joint_world_pos[:-2, [0, 2]]) * fps
    vel[0] = vel[1]
    vel[-1] = vel[-2]
    speed = np.linalg.norm(vel, axis=-1)
    speed_smooth = _gaussian_smooth_1d(speed, smooth_sigma)

    minima: list[int] = []
    maxima: list[int] = []
    for i in range(1, F - 1):
        s, sp, sn = speed_smooth[i], speed_smooth[i - 1], speed_smooth[i + 1]
        if s <= sp and s <= sn:
            minima.append(i)
        elif s >= sp and s >= sn:
            maxima.append(i)
    # Edges: classify by which neighbor relation they satisfy (both is OK).
    if speed_smooth[0] <= speed_smooth[1]:
        minima.insert(0, 0)
    elif speed_smooth[0] >= speed_smooth[1]:
        maxima.insert(0, 0)
    if speed_smooth[-1] <= speed_smooth[-2]:
        minima.append(F - 1)
    elif speed_smooth[-1] >= speed_smooth[-2]:
        maxima.append(F - 1)

    def _nms(candidates: list[int], rank_key) -> list[int]:
        if not candidates:
            return []
        candidates_sorted = sorted(candidates, key=rank_key)
        picked: list[int] = []
        for f in candidates_sorted:
            if all(abs(f - p) >= min_separation_frames for p in picked):
                picked.append(f)
        return picked

    picked_min = _nms(minima, rank_key=lambda i: speed_smooth[i])           # slowest first
    picked_max = _nms(maxima, rank_key=lambda i: -speed_smooth[i])          # fastest first

    combined = sorted(set(picked_min) | set(picked_max))
    if max_picks is not None and len(combined) > max_picks:
        # Keep the most-extreme half of each type, biased toward whichever is
        # over budget. Simple cap: just truncate.
        combined = combined[:max_picks]
    return combined


def detect_velocity_valleys(
    joint_world_pos: np.ndarray,  # (F, 3) in meters
    fps: int = 30,
    smooth_sigma: float = 1.5,
    min_separation_frames: int = 8,
    max_picks: int | None = None,
) -> list[int]:
    """Local minima of XZ-speed, NMS at `min_separation_frames` (slowest first).

    For feet during a walk/run the valleys are footplants. For hands they
    are swing-reversal points. Either way these are semantically meaningful
    keyframes that tell Kimodo "the limb was here at this moment".
    """
    F = joint_world_pos.shape[0]
    if F < 3:
        return list(range(F))
    vel = np.zeros((F, 2))
    vel[1:-1] = 0.5 * (joint_world_pos[2:, [0, 2]] - joint_world_pos[:-2, [0, 2]]) * fps
    vel[0] = vel[1]
    vel[-1] = vel[-2]
    speed = np.linalg.norm(vel, axis=-1)
    speed_smooth = _gaussian_smooth_1d(speed, smooth_sigma)

    minima: list[int] = []
    for i in range(1, F - 1):
        if speed_smooth[i] <= speed_smooth[i - 1] and speed_smooth[i] <= speed_smooth[i + 1]:
            minima.append(i)
    if speed_smooth[0] <= speed_smooth[1]:
        minima.insert(0, 0)
    if speed_smooth[-1] <= speed_smooth[-2]:
        minima.append(F - 1)

    if not minima:
        return [int(np.argmin(speed_smooth))]

    minima.sort(key=lambda i: speed_smooth[i])
    picked: list[int] = []
    for f in minima:
        if all(abs(f - p) >= min_separation_frames for p in picked):
            picked.append(f)
            if max_picks is not None and len(picked) >= max_picks:
                break
    return sorted(picked)


def dedupe_frames_across_effectors(
    frames_per_effector: dict[str, list[int]],
    n_frames: int,
) -> dict[str, list[int]]:
    """Nudge any frame collisions to the nearest free frame within ±n_frames.

    Workaround for a Kimodo editor quirk where multiple EE constraints
    sharing a frame index cause earlier ones to be overwritten on load.
    """
    used: set[int] = set()
    out: dict[str, list[int]] = {}
    for eff, frames in frames_per_effector.items():
        resolved: list[int] = []
        for f in frames:
            if f not in used:
                resolved.append(f)
                used.add(f)
                continue
            for d in range(1, n_frames):
                for s in (1, -1):
                    cand = f + s * d
                    if 0 <= cand < n_frames and cand not in used:
                        resolved.append(cand)
                        used.add(cand)
                        break
                else:
                    continue
                break
        out[eff] = sorted(resolved)
    return out
