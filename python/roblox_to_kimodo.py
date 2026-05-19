# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Stage 2: Roblox pose.json → Kimodo constraints.json (+ meta.json).

Reads the per-frame world CFrames produced by `extract_pose.lua`, converts
them into Kimodo's coordinate space, picks sparse keyframes per limb (by
velocity-valley heuristic, or by explicit user-supplied times), and emits
the Kimodo constraint JSON consumed by `run_kimodo.py`.

Coordinate / unit contract Kimodo expects (see motion-matching's
`extract_constraints.py:16-25`):
    Y-up, meters, +Z forward.

Roblox is also Y-up but characters face -Z and units are studs. We:
    1. Apply 180-degree Y-conjugation: pos.x → -pos.x, pos.z → -pos.z;
       quat (w,x,y,z) → (w, -x, y, -z).
    2. Scale studs → meters: STUD_TO_M = 1/3 (the inverse of the
       CM_TO_STUD = 0.03 constant baked into export_r15.py).
    3. Canonicalize root XZ to (0, 0) at frame 0.
    4. Resample to 30 fps if the input fps differs.

Effector constraints are POSITION-ONLY in v1 — `local_joints_rot` is sent as
zeros (identity T-pose). Kimodo's EE loss is dominated by FK position so
this is sufficient for "anchor the new motion to the source's hands & feet".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal as signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from effector_helpers import (  # noqa: E402
    detect_velocity_extremes,
    dedupe_frames_across_effectors,
    SOMA30_INDEX,
    SOMA30_NEUTRAL_JOINTS,
)
from soma_rig import SOMA_BIND_WORLD  # noqa: E402
from vendor import quat  # noqa: E402


# Per-SOMA-bone bind correction for the Roblox→SOMA retarget.
#
# Background: with Kimodo's FK applied to all-identity local rotations, SOMA
# bones land at their T-pose — legs hanging −Y and ARMS EXTENDED ±X. Roblox
# rigs (both R6 and R15) sit at axis-aligned rest with arms hanging −Y. For
# legs the rests match, so bind = identity (D[k] = source_world[k]) gives
# the right answer.
#
# For arms the rests differ, so we need a non-identity bind. We pick the
# rotation `bind_left = Rz(+90°)` such that at source rest (source_world =
# identity) the formula `D = source * inv(bind)` makes `local = inv(bind) =
# Rz(−90°)`, which when applied via FK to SOMA's LeftArm rest direction
# `+X` yields `−Y` — i.e., SOMA's arm hangs straight down, matching Roblox.
# Right arm is mirrored: `bind_right = Rz(−90°)`.
RZ_PLUS_90 = np.array([0.7071067811865476, 0.0, 0.0, 0.7071067811865476])    # Rz(+90°), wxyz
RZ_MINUS_90 = np.array([0.7071067811865476, 0.0, 0.0, -0.7071067811865476])  # Rz(-90°), wxyz
SOMA_BIND_CORRECTION: dict[str, np.ndarray] = {
    "LeftShoulder":  RZ_PLUS_90,
    "LeftArm":       RZ_PLUS_90,
    "LeftForeArm":   RZ_PLUS_90,
    "LeftHand":      RZ_PLUS_90,
    "RightShoulder": RZ_MINUS_90,
    "RightArm":      RZ_MINUS_90,
    "RightForeArm":  RZ_MINUS_90,
    "RightHand":     RZ_MINUS_90,
}
_BIND_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


# Per-rig chain mapping: SOMA bone name → source bone key emitted by the
# Lua side under `frames[t].chain`. `None` means "no source bone exists in
# this rig at this chain position" (e.g. R6 has no shin or clavicle); the
# retargeter fills those SOMA joints with identity local rotation, which
# leaves them collinear with their parent — matching how a single rigid
# Roblox limb part propagates orientation through its implied subjoints.
R6_CHAINS: dict[str, list[tuple[str, str | None]]] = {
    "left-foot":  [("Hips", "torso"),  ("LeftLeg",   "left_leg"),  ("LeftShin",    None), ("LeftFoot",  None)],
    "right-foot": [("Hips", "torso"),  ("RightLeg",  "right_leg"), ("RightShin",   None), ("RightFoot", None)],
    "left-hand":  [("Hips", "torso"),  ("LeftArm",   "left_arm"),  ("LeftForeArm", None), ("LeftHand",  None)],
    "right-hand": [("Hips", "torso"),  ("RightArm",  "right_arm"), ("RightForeArm",None), ("RightHand", None)],
}
R15_CHAINS: dict[str, list[tuple[str, str | None]]] = {
    "left-foot":  [("Hips", "lower_torso"), ("LeftLeg",  "left_upper_leg"),  ("LeftShin",   "left_lower_leg"),  ("LeftFoot",  "left_foot")],
    "right-foot": [("Hips", "lower_torso"), ("RightLeg", "right_upper_leg"), ("RightShin",  "right_lower_leg"), ("RightFoot", "right_foot")],
    "left-hand":  [("Hips", "lower_torso"), ("LeftArm",  "left_upper_arm"),  ("LeftForeArm","left_lower_arm"),  ("LeftHand",  "left_hand")],
    "right-hand": [("Hips", "lower_torso"), ("RightArm", "right_upper_arm"), ("RightForeArm","right_lower_arm"),("RightHand", "right_hand")],
}

KIMODO_FPS = 30
# Default stud→meter scale. The naïve 1/3 (mirrors motion-matching's
# CM_TO_STUD=0.03) makes the converted Roblox character ~1.67 m tall,
# noticeably bigger than Kimodo's ~1.54 m SOMA character. 0.30 lands a
# 5-stud-tall R6/R15 character at ~1.5 m which fits the Kimodo viewer
# without further tuning. Override via --scale if your source is sized
# unusually.
DEFAULT_STUD_TO_M = 0.30
SOMA30_N_JOINTS = 30

EFFECTORS = ("left_hand", "right_hand", "left_foot", "right_foot")
EFFECTOR_TYPE = {
    "left_hand":  "left-hand",
    "right_hand": "right-hand",
    "left_foot":  "left-foot",
    "right_foot": "right-foot",
}
EFFECTOR_JOINT_NAME = {
    "left_hand":  "LeftHand",
    "right_hand": "RightHand",
    "left_foot":  "LeftFoot",
    "right_foot": "RightFoot",
}


def _quat_to_axis_angle(q: np.ndarray, *, canonical: bool = True) -> np.ndarray:
    """(F, 4) wxyz quat → (F, 3) axis-angle (radians).

    canonical=True (default): forces w >= 0 first → angle ∈ [0, π], the
    shortest-arc representation. Used when callers don't care about
    temporal continuity.

    canonical=False: preserves the input sign so a temporally-unrolled
    quat path produces a continuous axis-angle path. The resulting
    magnitude can exceed π — that's fine for Kimodo since
    `axis_angle_to_matrix` (Rodrigues) is correct for any angle. Without
    this option, a unrolled quat that crosses w=0 would still flip the
    axis at the canonical conversion, undoing the unroll.
    """
    q = np.asarray(q, dtype=np.float64)
    if canonical:
        sign = np.where(q[..., 0:1] < 0.0, -1.0, 1.0)
        q = q * sign
    w = np.clip(q[..., 0], -1.0, 1.0)
    xyz = q[..., 1:4]
    s2 = np.sqrt(np.maximum(1.0 - w * w, 0.0))  # = |sin(θ/2)|
    near_zero = s2 < 1e-8
    theta = 2.0 * np.arccos(w)
    safe_s2 = np.where(near_zero, 1.0, s2)
    aa = (xyz / safe_s2[..., None]) * theta[..., None]
    aa = np.where(near_zero[..., None], 2.0 * xyz, aa)
    return aa


def _quat_y180_conjugate(q: np.ndarray) -> np.ndarray:
    """(F, 4) wxyz → wxyz, rotated by 180° about Y.

    Conjugating a quaternion q by R_y(180) = (0, 0, 1, 0) flips the X and Z
    components of the imaginary part: q' = (w, -x, y, -z). This corresponds
    to the same 180° change-of-basis we apply to position vectors.
    """
    out = q.copy()
    out[..., 1] = -q[..., 1]   # x
    out[..., 3] = -q[..., 3]   # z
    return out


def _quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(F, 4) wxyz quat applied to a single (3,) vector → (F, 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v
    # v' = q * v * conj(q). Direct expansion:
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return np.stack([rx, ry, rz], axis=-1)


def _smooth_savgol(x: np.ndarray, window: int, order: int = 3) -> np.ndarray:
    if x.shape[0] < window:
        return x
    return signal.savgol_filter(x, window, order, axis=0, mode="interp")


def _load_pose(pose_path: Path) -> dict:
    with Path(pose_path).open() as f:
        pose = json.load(f)

    n = pose["n_frames"]
    fps = pose["fps"]

    def _stack_top(channel: str) -> tuple[np.ndarray, np.ndarray]:
        pos = np.zeros((n, 3), dtype=np.float64)
        rot = np.zeros((n, 4), dtype=np.float64)  # wxyz
        for i, frame in enumerate(pose["frames"]):
            ch = frame[channel]
            pos[i] = ch["pos"]
            qx, qy, qz, qw = ch["rot"]
            rot[i] = (qw, qx, qy, qz)
        return pos, rot

    channels = {}
    for ch in ("hrp",) + EFFECTORS:
        channels[ch] = _stack_top(ch)

    # Per-frame chain bones (rig-specific). Stack into the same (F, 4) /
    # (F, 3) shape so the converter applies a uniform Y-180 conjugation
    # to everything.
    chain_keys: set[str] = set()
    for frame in pose["frames"]:
        for k in frame.get("chain", {}).keys():
            chain_keys.add(k)
    chain_channels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k in chain_keys:
        pos = np.zeros((n, 3), dtype=np.float64)
        rot = np.zeros((n, 4), dtype=np.float64)
        for i, frame in enumerate(pose["frames"]):
            ch = frame["chain"].get(k)
            if ch is None:
                rot[i] = (1.0, 0.0, 0.0, 0.0)
                continue
            pos[i] = ch["pos"]
            qx, qy, qz, qw = ch["rot"]
            rot[i] = (qw, qx, qy, qz)
        chain_channels[k] = (pos, rot)

    looped = bool(pose.get("looped", False))
    if looped and n >= 2:
        # Mirror frame 0 into the final frame for every channel so the
        # constraints emitted at frame F-1 match frame 0 exactly. The
        # downstream chain retarget reads these arrays directly so the
        # overwrite is sufficient — no special-case needed in the
        # constraint loop beyond pinning the endpoint frame indices.
        for ch_name, (pos, rot) in channels.items():
            pos[-1] = pos[0]
            rot[-1] = rot[0]
        for ch_name, (pos, rot) in chain_channels.items():
            pos[-1] = pos[0]
            rot[-1] = rot[0]

    return {
        "fps": fps,
        "n_frames": n,
        "duration_s": pose["duration_s"],
        "rig_type": pose["rig_type"],
        "asset_id": pose.get("asset_id", 0),
        "looped": looped,
        "channels": channels,
        "chain_channels": chain_channels,
    }


def _retarget_chain_quats(
    chain: list[tuple[str, str | None]],
    chain_world_rots: dict[str, np.ndarray],  # source key → (4,) wxyz, ALREADY in Kimodo frame
) -> dict[int, np.ndarray]:
    """Return SOMA30_INDEX → local quaternion (4,) wxyz for the chain joints at one frame.

    Math (mirrors motion-matching/python/uefn_to_soma.retarget_chain, but
    returns quats instead of axis-angles so the caller can unroll across
    time before the lossy canonical-form conversion):

        D[k]            = source_world[k] * inv(SOURCE_BIND_WORLD[k])
        local_q[soma_j] = inv(D[parent_chain_idx]) * D[child_chain_idx]

    SOURCE_BIND_WORLD is the source rig's rest-pose world rotation. For
    R6/R15 we spawn at identity HRP so source bind = identity, except for
    arms which need the Rz(±90°) bind correction (SOMA T-pose extends
    arms ±X but Roblox arms hang -Y at rest). A `None` source key
    inherits D from the parent so the missing SOMA joint comes out at
    identity local — the upstream rigid limb propagates orientation
    through the implied subjoints.
    """
    Ds: list[np.ndarray] = []
    for i, (soma_name, src_key) in enumerate(chain):
        if src_key is None:
            Ds.append(Ds[i - 1].copy() if i > 0 else np.array([1.0, 0.0, 0.0, 0.0]))
        else:
            bind = SOMA_BIND_CORRECTION.get(soma_name, _BIND_IDENTITY)
            w = chain_world_rots[src_key]
            Ds.append(quat.mul(w, quat.inv(bind)))

    out: dict[int, np.ndarray] = {}
    for i, (soma_name, _) in enumerate(chain):
        if i == 0:
            local_q = Ds[i]
        else:
            local_q = quat.mul(quat.inv(Ds[i - 1]), Ds[i])
        out[SOMA30_INDEX[soma_name]] = local_q
    return out


def _unroll_quats_inplace(quats: np.ndarray) -> None:
    """For each row i > 0, negate quats[i] if it has a negative 4D dot
    product with quats[i-1]. Maintains temporal continuity of the
    quaternion path so subsequent axis-angle conversion doesn't see a
    spurious sign-flip — which would otherwise produce flickering rotation
    axes near 180° rotations and visible spin in dense constraints.
    """
    for i in range(1, len(quats)):
        if float(np.dot(quats[i - 1], quats[i])) < 0.0:
            quats[i] = -quats[i]


def _to_kimodo(
    pos_studs: np.ndarray,
    rot_wxyz: np.ndarray,
    *,
    stud_to_m: float,
    y_offset_studs: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Roblox studs + 180°-Y flip → Kimodo meters.

    `y_offset_studs` is subtracted from Y in the source frame BEFORE the
    stud→meter scale, so the same offset (computed once from the lowest
    foot Y in the clip) applies coherently to every channel.
    """
    pos = pos_studs.copy()
    pos[..., 1] -= y_offset_studs
    pos_m = pos * stud_to_m
    pos_k = pos_m.copy()
    pos_k[..., 0] = -pos_k[..., 0]
    pos_k[..., 2] = -pos_k[..., 2]
    rot_k = _quat_y180_conjugate(rot_wxyz)
    return pos_k, rot_k


def _compute_y_offset_studs(channels: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    """Pick a Y shift that lands the lowest-ever foot at Kimodo Y=0.

    Returns the shift in source (stud) units so callers can apply it
    inside `_to_kimodo` BEFORE the stud→meter scale.
    """
    foot_y = np.concatenate([
        channels["left_foot"][0][:, 1],
        channels["right_foot"][0][:, 1],
    ])
    return float(np.min(foot_y))


def _resample_frames(
    pos: np.ndarray, rot: np.ndarray, src_fps: int, dst_fps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interp positions, slerp-by-nearest for quats. v1 only handles
    the case src_fps == dst_fps — anything else fails loudly so we don't
    silently corrupt timing. We extract at 30 fps in extract_pose so this
    rarely triggers."""
    if src_fps == dst_fps:
        return pos, rot
    raise NotImplementedError(
        f"Resampling {src_fps}→{dst_fps} fps not implemented. "
        f"Re-run extract_pose with --fps {dst_fps}."
    )


def _explicit_frames(times_s: list[float] | None, fps: int, n_frames: int) -> list[int] | None:
    if times_s is None:
        return None
    out = []
    for t in times_s:
        f = int(round(float(t) * fps))
        if 0 <= f < n_frames:
            out.append(f)
    return sorted(set(out))


def extract_constraints(
    pose_path: Path,
    *,
    out_dir: Path,
    name: str | None = None,
    no_effectors: list[str] | None = None,
    foot_min_separation: int = 8,
    hand_min_separation: int = 8,
    explicit_times: list[float] | None = None,
    explicit_per_limb: dict[str, list[float]] | None = None,
    constraints_filename: str = "constraints.json",
    stud_to_m: float = DEFAULT_STUD_TO_M,
    y_offset_mode: str = "auto",
    dedupe: bool = False,
    loop_window: int = 2,
) -> dict:
    pose_path = Path(pose_path)
    pose = _load_pose(pose_path)
    if name is None:
        name = pose_path.parent.name
    clip_dir = Path(out_dir) / name
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Pick the Y shift in source coords first (lowest foot Y across the
    # whole clip), then apply it consistently to every channel during
    # conversion.
    if y_offset_mode == "auto":
        y_offset_studs = _compute_y_offset_studs(pose["channels"])
    elif y_offset_mode == "none":
        y_offset_studs = 0.0
    else:
        raise ValueError(f"y_offset_mode must be 'auto' or 'none' (got {y_offset_mode!r})")

    converted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ch, (pos, rot) in pose["channels"].items():
        pk, rk = _to_kimodo(pos, rot, stud_to_m=stud_to_m, y_offset_studs=y_offset_studs)
        pk, rk = _resample_frames(pk, rk, pose["fps"], KIMODO_FPS)
        converted[ch] = (pk, rk)

    # Same conversion for the chain bones — Y-180 conjugation + scale +
    # Y-offset. Positions aren't used directly (FK derives them from
    # neutral_joints + retargeted rotations), but we run the converter
    # for symmetry and so debug dumps work.
    chain_converted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ch, (pos, rot) in pose.get("chain_channels", {}).items():
        pk, rk = _to_kimodo(pos, rot, stud_to_m=stud_to_m, y_offset_studs=y_offset_studs)
        pk, rk = _resample_frames(pk, rk, pose["fps"], KIMODO_FPS)
        chain_converted[ch] = (pk, rk)

    # Optional swing-twist clean-up on arm bones. Both R6 and R15 arm
    # segments rest along world -Y in their identity-HRP spawn pose, so
    # the twist axis (= rest bone direction) is -Y for every arm bone.
    # We strip any rotation around that axis so Kimodo's diffusion can't
    # amplify the singular axis ambiguity at overhead poses into a
    # visible spin.
    F = converted["hrp"][0].shape[0]
    if F < 2:
        raise RuntimeError(f"{pose_path}: too few frames after conversion ({F})")

    # Canonicalize XZ so the HRP starts at (0, 0). Apply the same offset
    # to every channel so effector positions stay coherent. Y is left in
    # absolute world coords — feet should sit just above 0 m for a
    # character of typical height.
    hrp_pos, _ = converted["hrp"]
    origin_xz = hrp_pos[0, [0, 2]].copy()

    def _shift_xz(p: np.ndarray) -> np.ndarray:
        out = p.copy()
        out[:, 0] -= origin_xz[0]
        out[:, 2] -= origin_xz[1]
        return out

    eff_pos_s: dict[str, np.ndarray] = {}
    for eff in EFFECTORS:
        eff_pos_s[eff] = _shift_xz(converted[eff][0])

    # Per-limb keyframe selection (heuristic by default).
    no_effectors = no_effectors or []
    explicit_per_limb = explicit_per_limb or {}
    fps = KIMODO_FPS

    per_eff: dict[str, list[int]] = {}
    eff_keyframes_meta: dict[str, list[int]] = {}

    for eff in EFFECTORS:
        family = "feet" if eff.endswith("_foot") else "hands"
        if family in no_effectors:
            continue

        limb_explicit = _explicit_frames(
            explicit_per_limb.get(eff), fps, F
        )
        shared_explicit = _explicit_frames(explicit_times, fps, F)

        if limb_explicit:
            picks = limb_explicit
        elif shared_explicit:
            picks = shared_explicit
        else:
            sep = foot_min_separation if family == "feet" else hand_min_separation
            picks = detect_velocity_extremes(
                eff_pos_s[eff], fps=fps, min_separation_frames=sep,
            )
        per_eff[eff] = picks

    # If the source clip is marked Loop, force a small WINDOW of frames at
    # each end into every effector's keyframe set, not just the endpoints.
    # Pinning {0, 1, …, W-1} ∪ {F-W, …, F-1} forces Kimodo to follow the
    # actual source pose right around the boundary, so the GENERATED motion
    # has matching VELOCITY (not just position) at the loop seam. With
    # W=1 we'd only get position match — common cause of the visible pop
    # when the diffusion produces different velocities entering vs leaving
    # the boundary. The frame F-1 channel data was already overwritten with
    # frame 0 in `_load_pose` so the endpoint poses literally match.
    if pose.get("looped", False) and F >= 2:
        w = max(1, int(loop_window))
        # Edge cases: clip too short to fit two non-overlapping windows.
        head = set(range(min(w, F)))
        tail = set(range(max(0, F - w), F))
        forced = head | tail
        for eff, frames in per_eff.items():
            per_eff[eff] = sorted(set(frames) | forced)

    # Optional editor-bug workaround: dedupe shifts colliding frames to a
    # nearest free slot to avoid the editor's "hand constraint nuked on
    # shared frame" quirk. Default OFF — kimodo_gen CLI doesn't have the
    # editor's quirk.
    if per_eff and dedupe:
        per_eff = dedupe_frames_across_effectors(per_eff, n_frames=F)
    # Refresh metadata to reflect what each effector ACTUALLY ends up with
    # (post round-robin / dedupe), not the pre-assignment candidate list.
    eff_keyframes_meta = {eff: list(frames) for eff, frames in per_eff.items()}

    # Build the per-effector "cheat root" constraints.
    #
    # Kimodo's JSON loader (kimodo/constraints.py:486-512, EndEffectorConstraintSet
    # .from_dict) only accepts `local_joints_rot` + `root_positions`, then runs
    # SOMA30 FK to compute the constrained leaf-joint world position. There is
    # no JSON field for raw world position. With identity (T-pose) local
    # rotations the FK reduces to:
    #
    #     leaf_world = root_position + SOMA30_NEUTRAL_JOINTS[leaf]
    #
    # So for each keyframe we set:
    #
    #     root_position[t] = desired_leaf_world_pos - SOMA30_NEUTRAL[leaf]
    #
    # which makes Kimodo's FK land the effector exactly at the source R6/R15
    # foot or hand position. Each effector ships in its own constraint object
    # so the per-constraint root_positions are independent — they don't fight
    # each other (the actual generated root motion is driven by the implicit
    # natural prior, not by these synthetic root_positions which are only used
    # by THIS constraint's FK).
    # Pick the rig-specific chain table.
    rig = pose["rig_type"]
    if rig == "R6":
        rig_chains = R6_CHAINS
    elif rig == "R15":
        rig_chains = R15_CHAINS
    else:
        raise ValueError(f"unknown rig_type {rig!r}")

    # SOMA Hips tracks the source TORSO bone, not the HRP. R6 anchors HRP
    # but moves all body motion through the Torso Motor6D (e.g. dance bobs
    # ride on the Torso while HRP stays fixed); R15 has the same pattern
    # via LowerTorso. Using the torso captures both locomotion and
    # in-place body motion. Y-180 conjugation, stud→m scale, and
    # y_offset_studs were already baked in by `_to_kimodo` upstream.
    torso_key = "torso" if rig == "R6" else "lower_torso"
    if torso_key not in chain_converted:
        raise RuntimeError(
            f"pose.json missing chain bone {torso_key!r} for rig {rig!r}; "
            f"re-extract with the latest extract_pose.lua"
        )
    root_pos_kimodo, _ = chain_converted[torso_key]

    constraints: list[dict] = []
    for eff, frames in per_eff.items():
        if not frames:
            continue
        if eff_to_constraint_type := EFFECTOR_TYPE.get(eff):
            ctype = eff_to_constraint_type
        else:
            continue
        chain_def = rig_chains[ctype]
        joint_name = EFFECTOR_JOINT_NAME[eff]

        T = len(frames)
        # First pass: per-frame local quats per joint. Stored so we can
        # unroll across time before the canonical axis-angle conversion.
        per_joint_quats: dict[int, np.ndarray] = {}
        root_positions = np.zeros((T, 3), dtype=np.float64)
        smooth_root_2d = np.zeros((T, 2), dtype=np.float64)
        for t_idx, f in enumerate(frames):
            chain_rots: dict[str, np.ndarray] = {}
            for soma_name, src_key in chain_def:
                if src_key is None:
                    continue
                if src_key not in chain_converted:
                    raise RuntimeError(
                        f"pose.json missing chain bone {src_key!r} for "
                        f"rig {rig!r} (constraint {ctype})"
                    )
                chain_rots[src_key] = chain_converted[src_key][1][f]

            quats_per_joint = _retarget_chain_quats(chain_def, chain_rots)
            for soma_idx, q in quats_per_joint.items():
                if soma_idx not in per_joint_quats:
                    per_joint_quats[soma_idx] = np.zeros((T, 4), dtype=np.float64)
                per_joint_quats[soma_idx][t_idx] = q

            root_positions[t_idx] = root_pos_kimodo[f]
            smooth_root_2d[t_idx, 0] = root_pos_kimodo[f, 0]
            smooth_root_2d[t_idx, 1] = root_pos_kimodo[f, 2]

        # Second pass: unroll each joint's quat path across time, then
        # collapse to canonical axis-angle. The unroll keeps consecutive
        # frames on the same 4D hemisphere; without it, the per-frame
        # `_quat_to_axis_angle`'s w>=0 normalization can flip the axis
        # direction whenever the rotation's `w` crosses zero (≈180°),
        # which Kimodo follows literally in dense mode and renders as
        # bone spin.
        local_joints_rot = np.zeros((T, SOMA30_N_JOINTS, 3), dtype=np.float64)
        for soma_idx, quats in per_joint_quats.items():
            _unroll_quats_inplace(quats)
            # canonical=False so the conversion respects the unroll; the
            # axis-angle path can exceed |π| but Kimodo's Rodrigues
            # decoding handles arbitrary angles correctly.
            local_joints_rot[:, soma_idx, :] = _quat_to_axis_angle(quats, canonical=False)

        constraints.append({
            "type": ctype,
            "frame_indices": [int(f) for f in frames],
            "local_joints_rot": local_joints_rot.tolist(),
            "root_positions": root_positions.tolist(),
            "smooth_root_2d": smooth_root_2d.tolist(),
            "joint_names": [joint_name],
        })

    duration_s = F / fps
    meta = {
        "fps": fps,
        "n_frames": int(F),
        "duration_s": duration_s,
        "source_pose": str(pose_path),
        "name": name,
        "rig_type": pose["rig_type"],
        "asset_id": pose["asset_id"],
        "stud_to_m": stud_to_m,
        "y_offset_studs": y_offset_studs,
        "y_offset_mode": y_offset_mode,
        "looped": pose.get("looped", False),
        "effectors": [e for e in EFFECTORS if e in per_eff],
        "effector_keyframes": eff_keyframes_meta,
        "prompt": None,
    }

    constraints_path = clip_dir / constraints_filename
    meta_path = clip_dir / "meta.json"
    with constraints_path.open("w") as f:
        json.dump(constraints, f, indent=2)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "clip_dir": str(clip_dir),
        "constraints_path": str(constraints_path),
        "meta_path": str(meta_path),
        "n_frames": int(F),
        "duration_s": duration_s,
        "fps": fps,
        "effector_keyframes": eff_keyframes_meta,
    }


def _parse_csv_floats(s: str | None) -> list[float] | None:
    if not s:
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pose", type=Path,
                   help="Path to pose.json from extract_pose.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (constraints written to <out>/<name>/).")
    p.add_argument("--name", type=str, default=None,
                   help="Output dir name (default: pose.json's parent dir name).")
    p.add_argument("--no-effectors", action="append", default=[],
                   choices=["feet", "hands"],
                   help="Drop a class of effector constraints. Repeatable.")
    p.add_argument("--foot-min-separation", type=int, default=8,
                   help="NMS window in frames for foot keyframe picks. Larger "
                        "= sparser = Kimodo more creative.")
    p.add_argument("--hand-min-separation", type=int, default=8,
                   help="NMS window in frames for hand keyframe picks.")
    p.add_argument("--explicit-times", type=str, default=None,
                   help="Comma-separated explicit times (seconds), applied to "
                        "ALL effectors. Overrides velocity-valley heuristic.")
    p.add_argument("--left-foot-times", type=str, default=None)
    p.add_argument("--right-foot-times", type=str, default=None)
    p.add_argument("--left-hand-times", type=str, default=None)
    p.add_argument("--right-hand-times", type=str, default=None)
    p.add_argument("--scale", type=float, default=DEFAULT_STUD_TO_M,
                   help=f"Stud→meter scale (default {DEFAULT_STUD_TO_M}). "
                        "Lower = smaller character in Kimodo space.")
    p.add_argument("--y-offset-mode", choices=["auto", "none"], default="auto",
                   help="auto = subtract min foot Y so the lowest foot lands "
                        "at Kimodo Y=0 (default). none = pass Y through.")
    p.add_argument("--loop-window", type=int, default=2,
                   help="When the source clip is marked Loop, pin this many "
                        "frames at each end to source pose so velocity "
                        "matches at the loop seam. 1 = endpoints only "
                        "(positions match, velocity may not). Default 2.")
    p.add_argument("--dedupe", dest="dedupe", action="store_true",
                   help="Shift across-effector frame collisions to nearest "
                        "free frames. Workaround for a Kimodo editor quirk "
                        "where multi-keyed-on-one-frame nukes the hand "
                        "constraint. kimodo_gen CLI doesn't need it.")
    p.set_defaults(dedupe=False)
    args = p.parse_args(argv)

    explicit_per_limb = {
        "left_foot":  _parse_csv_floats(args.left_foot_times),
        "right_foot": _parse_csv_floats(args.right_foot_times),
        "left_hand":  _parse_csv_floats(args.left_hand_times),
        "right_hand": _parse_csv_floats(args.right_hand_times),
    }
    explicit_per_limb = {k: v for k, v in explicit_per_limb.items() if v}

    result = extract_constraints(
        args.pose,
        out_dir=args.out,
        name=args.name,
        no_effectors=args.no_effectors,
        foot_min_separation=args.foot_min_separation,
        hand_min_separation=args.hand_min_separation,
        explicit_times=_parse_csv_floats(args.explicit_times),
        explicit_per_limb=explicit_per_limb,
        stud_to_m=args.scale,
        y_offset_mode=args.y_offset_mode,
        dedupe=args.dedupe,
        loop_window=args.loop_window,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
