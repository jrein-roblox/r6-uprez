# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "ufbx"]
# ///
"""
Retarget a Geno-skeleton BVH clip onto a Roblox R15 rig.

Output: JSON with per-frame quaternions for each R15 Motor6D (as Transform
rotation deltas from rest) and per-frame HumanoidRootPart CFrame components
(position + yaw rotation as world motion).

Quaternions in output are (x, y, z, w) to match Roblox's CFrame rotation
component order. Internally we use (w, x, y, z) to stay consistent with
vendor/quat.py from GenoViewPython-MotionMatching.

Run:
    python3 export_r15.py \
        --bind data/Geno_bind.bvh \
        --anim data/bvh/walk1_subject5.bvh \
        --start 160 --count 300 \
        --out data/r15_walk.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vendor import bvh, quat  # noqa: E402

# Rig tables (R15_JOINTS / R15_CHAINS / HARDCODED_BIND_WORLD) live as
# module globals below and default to the Geno (lafan1) layout. Call
# `set_rig("motusman")` before `retarget()` to swap in the MocapOnline
# MotusMan tables — same retarget math, different source bone names
# and bind reference.
_ACTIVE_RIG = "geno"


def set_rig(name: str) -> None:
    """Swap R15_JOINTS / R15_CHAINS / HARDCODED_BIND_WORLD for the named
    source rig. "geno" (default, lafan1 BVH), "motusman" (MocapOnline
    Mobility Starter 27B FBX), or "uefn" (Unreal-Engine Mobility Starter
    GASP rig FBX). Mutates module globals in place so downstream code
    sees the new tables without parameter threading."""
    global R15_JOINTS, R15_CHAINS, HARDCODED_BIND_WORLD, _ACTIVE_RIG
    name = name.lower()
    if name == _ACTIVE_RIG:
        return
    if name == "geno":
        R15_JOINTS = _GENO_R15_JOINTS
        R15_CHAINS = _GENO_R15_CHAINS
        HARDCODED_BIND_WORLD = _GENO_BIND_WORLD
    elif name == "motusman":
        from motusman_rig import (
            MOTUSMAN_R15_JOINTS,
            MOTUSMAN_R15_CHAINS,
            MOTUSMAN_BIND_WORLD,
        )
        R15_JOINTS = MOTUSMAN_R15_JOINTS
        R15_CHAINS = MOTUSMAN_R15_CHAINS
        HARDCODED_BIND_WORLD = MOTUSMAN_BIND_WORLD
    elif name == "uefn":
        from uefn_rig import (
            UEFN_R15_JOINTS,
            UEFN_R15_CHAINS,
            UEFN_BIND_WORLD,
        )
        R15_JOINTS = UEFN_R15_JOINTS
        R15_CHAINS = UEFN_R15_CHAINS
        HARDCODED_BIND_WORLD = UEFN_BIND_WORLD
    elif name == "soma":
        from soma_rig import (
            SOMA_R15_JOINTS,
            SOMA_R15_CHAINS,
            SOMA_BIND_WORLD,
        )
        R15_JOINTS = SOMA_R15_JOINTS
        R15_CHAINS = SOMA_R15_CHAINS
        HARDCODED_BIND_WORLD = SOMA_BIND_WORLD
    elif name == "soma_r15plus":
        from soma_r15plus_rig import (
            SOMA_R15PLUS_JOINTS,
            SOMA_R15PLUS_CHAINS,
            SOMA_R15PLUS_BIND_WORLD,
        )
        R15_JOINTS = SOMA_R15PLUS_JOINTS
        R15_CHAINS = SOMA_R15PLUS_CHAINS
        HARDCODED_BIND_WORLD = SOMA_R15PLUS_BIND_WORLD
    else:
        raise ValueError(
            f"Unknown rig {name!r} (expected 'geno', 'motusman', 'uefn', 'soma', or 'soma_r15plus')"
        )
    _ACTIVE_RIG = name


def _load_anim_any(path: Path) -> dict:
    """Dispatch by file extension: .bvh -> BVH loader, .fbx -> FBX loader.
    Both return the same dict schema."""
    ext = path.suffix.lower()
    if ext == ".bvh":
        return load_bvh_world_rotations(path)
    if ext == ".fbx":
        from fbx_loader import load_fbx_world_rotations
        return load_fbx_world_rotations(path)
    raise ValueError(f"Unsupported animation format: {path.suffix} ({path})")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# BVH is Y-up cm, 60fps. Roblox R15 HRP sits ~2 studs off the ground.
# 1 cm -> 0.03 studs puts Geno's ~85 cm hips at ~2.55 studs, close enough.
CM_TO_STUD = 0.03

# R15 HumanoidRootPart rest height in studs (Y). Geno hips Y is discarded.
HRP_REST_Y = 2.0

# R15 default faces -Z; Geno faces +Z. Flip by negating X,Z on position and
# adding a 180 degree yaw to HRP rotation.
FLIP_WORLD = True

# Split character yaw into a smoothed "heading" and a high-frequency
# residual. HRP keeps the smoothed heading; the residual falls through to
# LowerTorso's Transform automatically (because Transform = inv(HRP_world)
# * hips_world). Without this split the hips' per-step yaw sway is baked
# entirely into HRP's Y rotation, which MM overrides at runtime — so
# LowerTorso ends up with ~0 Y rotation and the animation looks stiff
# (no hip-sway with each step).
#
# SIGMA is in SOURCE frames (typically 60 Hz). Scaled by `stride` below
# to keep the effective smoothing time window constant when the source
# is downsampled. Matches the reference's smoothed simDir (savgol
# window=61 at 60 Hz, ≈ 1 s) in spirit but using Gaussian smoothing
# since we're numpy-only.
#   Source: /Users/jrein/git/orangeduck/Motion-Matching/resources/
#           generate_database.py:116
YAW_SMOOTH_SIGMA_SOURCE_FRAMES = 10

# Position smoothing for HRP path (separate from yaw). The residual
# `pelvis - smoothed_HRP` becomes LowerTorso's Position curve. With a
# wide window (sigma ≥ ~half a step), the smoothed path lags the
# actual pelvis through turns — at run speed (~12 stud/s) that lag
# can show up as the LowerTorso drifting many studs off HRP, way
# beyond a natural bob/sway. Keeping sigma small (~0.1 s window)
# lets the smoothed path track the trajectory while still removing
# the highest-frequency per-step jitter.
POSITION_SMOOTH_SIGMA_SOURCE_FRAMES = 3

# Rotation channel orientation of BVH ("zyx" is what the Geno BVH uses).
# bvh.load auto-detects this; we read bvhData['order'].

# Map R15 Motor6D target -> (source BVH joint driving target WORLD rotation,
# R15 parent joint that the Motor6D is attached to).
#
# Root motion is handled separately. The Transform of each Motor6D is derived
# as inv(world rotation of the R15 parent's mapped source joint) * world
# rotation of this joint's mapped source joint. For LowerTorso the R15 parent
# is HRP, whose rotation is yaw-only, and the residual (non-yaw) component of
# Hips goes on the Root joint.
_GENO_R15_JOINTS = [
    # R15 part, source BVH joint (last in chain), R15 parent part, R15 Motor6D name
    ("LowerTorso",     "Hips",         "HumanoidRootPart", "Root"),
    ("UpperTorso",     "Spine3",       "LowerTorso",       "Waist"),
    ("Head",           "Head",         "UpperTorso",       "Neck"),
    ("LeftUpperArm",   "LeftArm",      "UpperTorso",       "LeftShoulder"),
    ("LeftLowerArm",   "LeftForeArm",  "LeftUpperArm",     "LeftElbow"),
    ("LeftHand",       "LeftHand",     "LeftLowerArm",     "LeftWrist"),
    ("RightUpperArm",  "RightArm",     "UpperTorso",       "RightShoulder"),
    ("RightLowerArm",  "RightForeArm", "RightUpperArm",    "RightElbow"),
    ("RightHand",      "RightHand",    "RightLowerArm",    "RightWrist"),
    ("LeftUpperLeg",   "LeftUpLeg",    "LowerTorso",       "LeftHip"),
    ("LeftLowerLeg",   "LeftLeg",      "LeftUpperLeg",     "LeftKnee"),
    ("LeftFoot",       "LeftFoot",     "LeftLowerLeg",     "LeftAnkle"),
    ("RightUpperLeg",  "RightUpLeg",   "LowerTorso",       "RightHip"),
    ("RightLowerLeg",  "RightLeg",     "RightUpperLeg",    "RightKnee"),
    ("RightFoot",      "RightFoot",    "RightLowerLeg",    "RightAnkle"),
]
R15_JOINTS = _GENO_R15_JOINTS  # active rig — reassigned by set_rig()

# Chain of Geno joints that collectively produce the R15 Motor6D's rotation.
# R15 skeleton is sparser than Geno's (no clavicle, no intermediate spine,
# no neck1). The Motor6D Transform for R15's waist must absorb the entire
# Spine→Spine1→Spine2→Spine3 chain of Geno locals, so that the child part's
# world rotation tracks Spine3.
#
# Each entry maps R15 part -> ordered list of Geno joints. The list is the
# chain of local rotations between R15's parent joint's mapped source and
# the R15 joint's mapped source.
_GENO_R15_CHAINS: dict[str, list[str]] = {
    "LowerTorso":    ["Hips"],  # special: Hips is root; yaw stripped for HRP
    "UpperTorso":    ["Spine", "Spine1", "Spine2", "Spine3"],
    "Head":          ["Neck", "Neck1", "Head"],
    "LeftUpperArm":  ["LeftShoulder", "LeftArm"],
    "LeftLowerArm":  ["LeftForeArm"],
    "LeftHand":      ["LeftHand"],
    "RightUpperArm": ["RightShoulder", "RightArm"],
    "RightLowerArm": ["RightForeArm"],
    "RightHand":     ["RightHand"],
    "LeftUpperLeg":  ["LeftUpLeg"],
    "LeftLowerLeg":  ["LeftLeg"],
    "LeftFoot":      ["LeftFoot"],
    "RightUpperLeg": ["RightUpLeg"],
    "RightLowerLeg": ["RightLeg"],
    "RightFoot":     ["RightFoot"],
}
R15_CHAINS = _GENO_R15_CHAINS

# Hardcoded bind world rotations for all motion-matching retargeting.
# Extracted once from walk1_subject5.bvh frame 160 — a walking-start pose
# that sits visually close to an I-pose (arms ~15° off vertical, legs
# straight). Using this as the fixed bind reference for every clip means:
#   - R15's rest pose (all Motor6Ds = I) corresponds to this same Geno pose.
#   - Every retargeted clip starts from the same baseline, so clips can
#     blend and share the same motion-matching feature database.
#   - Frame 0 of a clip only has a Motor6D delta from this reference, which
#     is small for clips with similar starting poses (other walking/running
#     clips) and larger for dissimilar clips (pushAndStumble, etc.).
_GENO_BIND_WORLD: dict[str, np.ndarray] = {
    "Hips":          np.array([+0.99954680, +0.01226743, -0.02678681, -0.00617769]),
    "Spine":         np.array([+0.99949398, +0.01609549, -0.02691185, -0.00533573]),
    "Spine1":        np.array([+0.99934221, +0.02375081, -0.02716073, -0.00365160]),
    "Spine2":        np.array([+0.99912896, +0.03140468, -0.02740796, -0.00196725]),
    "Spine3":        np.array([+0.99885424, +0.03905663, -0.02765349, -0.00028277]),
    "Neck":          np.array([+0.99773547, +0.06228133, -0.02454113, -0.00653423]),
    "Neck1":         np.array([+0.99681221, +0.07579048, -0.02127603, -0.01298279]),
    "Head":          np.array([+0.99875387, +0.04526197, -0.01981972, -0.00701695]),
    "LeftShoulder":  np.array([+0.57779187, +0.01796081, +0.04888005, -0.81452115]),
    "LeftArm":       np.array([+0.07921692, +0.19227043, +0.13201813, -0.96918934]),
    "LeftForeArm":   np.array([+0.11242334, +0.16504812, -0.07324295, -0.97711595]),
    "LeftHand":      np.array([+0.10031650, +0.15747703, -0.10814721, -0.97644343]),
    "RightShoulder": np.array([+0.56313948, -0.00644799, -0.06328113, +0.82391010]),
    "RightArm":      np.array([+0.11350304, +0.09134029, -0.16547855, +0.97539267]),
    "RightForeArm":  np.array([+0.14857319, +0.10636389, +0.16316057, +0.96953152]),
    "RightHand":     np.array([+0.19555716, +0.07726574, +0.17400821, +0.96203355]),
    "LeftUpLeg":     np.array([+0.99931173, -0.01478183, -0.02454114, -0.02356465]),
    "LeftLeg":       np.array([+0.98872199, +0.10127209, -0.10879836, -0.01832250]),
    "LeftFoot":      np.array([+0.97915095, +0.18721605, -0.02172328, +0.07577376]),
    "RightUpLeg":    np.array([+0.99757145, -0.00078573, -0.06050167, -0.03449835]),
    "RightLeg":      np.array([+0.99689965, +0.07644443, -0.00672510, -0.01738112]),
    "RightFoot":     np.array([+0.97698278, +0.21095927, -0.01284570, -0.02891071]),
}
HARDCODED_BIND_WORLD = _GENO_BIND_WORLD

# ----------------------------------------------------------------------------
# Quaternion helpers (wrap vendor/quat.py which uses (w, x, y, z))
# ----------------------------------------------------------------------------

def quat_identity(n: int | None = None) -> np.ndarray:
    if n is None:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))


def quat_yaw_from_forward_xz(fwd: np.ndarray) -> np.ndarray:
    """Compute yaw quaternion (rotation around Y) from a forward vector.

    fwd: (..., 3). Only X/Z matter; Y is projected to zero and the vector is
    re-normalized before extracting the angle.
    """
    proj = fwd.copy()
    proj[..., 1] = 0.0
    n = np.linalg.norm(proj, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-8)
    proj = proj / n
    # atan2(X, Z) gives the signed angle around +Y from +Z to `proj`.
    angle = np.arctan2(proj[..., 0], proj[..., 2])
    half = angle / 2.0
    q = np.zeros(list(fwd.shape[:-1]) + [4])
    q[..., 0] = np.cos(half)
    q[..., 2] = np.sin(half)
    return q


def gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass filter on a 1-D series. Kernel truncated at 3σ,
    input padded with edge values so the smoothed series is the same length
    as x. numpy-only (no scipy dependency)."""
    if sigma <= 0:
        return x.copy()
    radius = int(np.ceil(3.0 * sigma))
    width = 2 * radius + 1
    kernel = np.exp(-0.5 * ((np.arange(width) - radius) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    out = np.empty_like(q)
    out[..., 0] = q[..., 1]
    out[..., 1] = q[..., 2]
    out[..., 2] = q[..., 3]
    out[..., 3] = q[..., 0]
    return out


# ----------------------------------------------------------------------------
# BVH load + FK
# ----------------------------------------------------------------------------

def load_bvh_world_rotations(path: Path) -> dict:
    """Load a BVH and compute world-space rotations per frame per joint.

    Returns dict with:
        names: list[str]
        parents: np.ndarray[int]  (parents[i] is parent index of joint i, -1 for root)
        order: str
        frame_time: float         (may be None if static pose file)
        local_rot: (F, J, 4)      (w, x, y, z)
        world_rot: (F, J, 4)      (w, x, y, z)
        local_pos: (F, J, 3)      (cm, BVH space)
        world_pos: (F, J, 3)      (cm, BVH space)
        offsets:   (J, 3)         (cm; static bone offsets from bind pose)
    """
    data = bvh.load(str(path))
    rot_euler = np.radians(data["rotations"])  # (F, J, 3)
    pos = data["positions"].astype(np.float64)  # (F, J, 3), cm
    parents = data["parents"]
    order = data["order"]

    local_rot = quat.from_euler(rot_euler, order=order)  # (F, J, 4)
    # FK: world rotations and positions. Need to pass quats and positions.
    world_rot, world_pos = quat.fk(local_rot, pos, parents)

    return {
        "names": data["names"],
        "parents": parents,
        "order": order,
        "frame_time": data.get("frame_time"),
        "local_rot": local_rot,
        "world_rot": world_rot,
        "local_pos": pos,
        "world_pos": world_pos,
        "offsets": data["offsets"],
    }


# ----------------------------------------------------------------------------
# Retarget
# ----------------------------------------------------------------------------

def retarget(bind_path: Path, anim_path: Path, start: int, count: int | None,
             bind_from_anim_start: bool = False,
             synthetic_ipose_bind: bool = False,
             hardcoded_bind: bool = False,
             stride: int = 1) -> dict:
    """Retarget a BVH animation onto R15 Motor6D transforms + HRP CFrame.

    Math:
      For each source joint j, the *delta from bind* in source world space is
        D[t,j] = R_world[t,j] * inv(B_world[j])
      We map this delta onto R15 (assuming R15 parts at rest are world-identity):
        R15_world[t,j] = D[t, src(j)]
      And convert to the Motor6D's local rotation delta relative to parent:
        Transform[t,j] = inv(R15_world[t, parent(j)]) * R15_world[t, j]
      For the bind pose, D[j] = identity for all j, so Transform is identity.
    """
    bind = _load_anim_any(bind_path)
    anim = _load_anim_any(anim_path)

    # --bind-from-anim-start treats the anim's `start` frame as the bind
    # reference. Use this when R15's rest pose differs visibly from the
    # --bind BVH's rest pose (e.g., Geno's bind is a loose A-pose with arms
    # 45 deg outward, but the R15 rig in Studio is closer to an I-pose with
    # arms nearly straight down). Using the first anim frame as bind makes
    # Motor6D transforms = identity at animation start, so the rig starts in
    # its natural I-pose and only the walk-cycle deltas are applied on top.
    if bind_from_anim_start:
        bind = dict(anim)
        # Slice to the start frame for BOTH world_rot and local_rot, since
        # the new local-delta retargeting path reads local_rot[0] as bind.
        bind["world_rot"] = anim["world_rot"][start : start + 1]
        bind["local_rot"] = anim["local_rot"][start : start + 1]

    if hardcoded_bind:
        # Use the hardcoded walk1-frame-160 bind world rotations for every
        # joint we retarget. Consistent across all clips, independent of the
        # --bind BVH file (which is still loaded for skeleton/hierarchy info
        # but its rotations are overridden for joints in HARDCODED_BIND_WORLD).
        old_world_rot = np.array(bind["world_rot"][0], copy=True)
        new_world_rot = old_world_rot.copy()
        names = list(anim["names"])
        overridden = set()
        for jname, wq in HARDCODED_BIND_WORLD.items():
            if jname in names:
                new_world_rot[names.index(jname)] = wq
                overridden.add(names.index(jname))
        # Propagate bind corrections to descendants of overridden joints.
        # When a joint's bind world rotation is overridden but its children
        # are not, the children still carry the old (e.g. T-pose) world
        # rotation. This breaks the parent-child delta math because
        # inv(D_parent) * D_child assumes both were computed from a
        # consistent FK chain. Fix: for each non-overridden joint whose
        # BVH-parent (or ancestor) was overridden, apply the same rigid
        # correction: new_bind[child] = correction * old_bind[child],
        # where correction = new_bind[parent] * inv(old_bind[parent]).
        parents_arr = anim["parents"]
        for j in range(len(names)):
            if j in overridden:
                continue
            p = parents_arr[j]
            if p < 0:
                continue
            # Walk up to find the nearest ancestor that was overridden or
            # already corrected (any ancestor whose new != old).
            correction_src = p
            while correction_src >= 0:
                if not np.allclose(new_world_rot[correction_src], old_world_rot[correction_src], atol=1e-7):
                    break
                correction_src = parents_arr[correction_src]
            if correction_src < 0:
                continue
            # correction = new_parent * inv(old_parent)
            correction = quat.mul(
                new_world_rot[correction_src:correction_src+1],
                quat.inv(old_world_rot[correction_src:correction_src+1]),
            )
            new_world_rot[j] = quat.mul(correction, old_world_rot[j:j+1])[0]
        bind = dict(bind)
        bind["world_rot"] = new_world_rot[None, :, :]

    names = anim["names"]
    if not isinstance(names, list):
        names = list(names)

    if not bind_from_anim_start and list(bind["names"]) != names:
        raise RuntimeError(
            f"Bind and animation skeletons differ: {bind_path.name} has {len(bind['names'])} joints, "
            f"{anim_path.name} has {len(names)} joints. Only matching skeletons are supported."
        )

    name_to_idx = {n: i for i, n in enumerate(names)}
    required = {src for (_, src, _, _) in R15_JOINTS}
    # The first rig row maps R15.LowerTorso → the source rig's hips
    # bone (e.g. "Hips" for Geno/MotusMan, "pelvis" for UEFN). Its
    # world position drives HRP root motion below.
    hips_src_name = R15_JOINTS[0][1]
    required.add(hips_src_name)
    missing = [n for n in required if n not in name_to_idx]
    if missing:
        raise RuntimeError(f"BVH missing expected joints: {missing}")

    # Select frame slice.
    F_total = anim["world_rot"].shape[0]
    if count is None or start + count > F_total:
        count = F_total - start
    if start < 0 or start >= F_total or count <= 0:
        raise RuntimeError(f"Invalid slice start={start} count={count} (total={F_total})")

    # Slice + stride. stride=2 → 30Hz from 60Hz source, stride=3 → 20Hz, etc.
    world_rot = anim["world_rot"][start : start + count : stride]
    world_pos = anim["world_pos"][start : start + count : stride]
    out_frame_count = world_rot.shape[0]

    # Bind pose world rotations.
    bind_world_rot = bind["world_rot"][0]                    # (J, 4)

    # --- World-delta from bind, per source joint per frame -----------------
    # D[t, j] = world[t, j] * inv(bind[j])
    D = quat.mul(world_rot, quat.inv(bind_world_rot)[None, :, :])  # (F, J, 4)

    # --- Root motion: HRP position + yaw from Hips -------------------------
    hips_idx = name_to_idx[hips_src_name]
    hips_world_pos = world_pos[:, hips_idx]                  # (F, 3), cm
    # Use the DELTA hips rotation (relative to bind) to compute character
    # facing — yaw = 0 at bind by construction.
    hips_delta = D[:, hips_idx]                              # (F, 4)
    hips_fwd = quat.mul_vec(hips_delta, np.array([0.0, 0.0, 1.0]))
    yaw_q = quat_yaw_from_forward_xz(hips_fwd)               # (F, 4)

    # Split yaw into smoothed "heading" (stays on HRP) and high-frequency
    # residual (falls through to LowerTorso via the inv(parent)*child
    # computation below, because parent_world here is the smoothed
    # yaw_q and child_world for LowerTorso is still the full hips_delta).
    # See YAW_SMOOTH_SIGMA_SOURCE_FRAMES comment.
    #
    # Unwrap yaw before smoothing so wraps through ±π (e.g., character
    # turning through south) don't smear across the wrap boundary.
    # Recompose back to a (w, x, y, z) quat with only Y rotation.
    yaw_angle = 2.0 * np.arctan2(yaw_q[..., 2], yaw_q[..., 0])
    yaw_angle = np.unwrap(yaw_angle)
    sigma = max(1.0, YAW_SMOOTH_SIGMA_SOURCE_FRAMES / max(1, stride))
    yaw_angle = gaussian_smooth_1d(yaw_angle, sigma)
    yaw_q = np.zeros_like(yaw_q)
    yaw_q[..., 0] = np.cos(yaw_angle * 0.5)   # w
    yaw_q[..., 2] = np.sin(yaw_angle * 0.5)   # y

    # HRP rotation: the R15 rig's default CFrame faces -Z (LookVector = -Z),
    # Geno's default faces +Z. With FLIP_WORLD we also flip HRP position
    # (negate X, Z) so R15 walks in its natural -Z direction. The yaw we
    # extracted from Geno's hips forward is a rotation around +Y that takes
    # +Z to the forward direction; applying the same quaternion to R15
    # takes -Z to the "flipped" forward — which is exactly what we want. So
    # no extra 180° here, just use yaw_q directly.
    if FLIP_WORLD:
        hrp_rot_wxyz = yaw_q
        sign = -1.0
    else:
        hrp_rot_wxyz = yaw_q
        sign = 1.0

    # Smooth the ground-plane HRP path so HRP carries only the
    # "intended trajectory" (the smooth arc the body follows) and the
    # high-frequency residual — the per-step lateral sway — falls
    # through to LowerTorso's Position curve below. Without this
    # split, sway is invisible because HRP rotation tracks it 1:1.
    # Same sigma the yaw branch uses, so heading and position are
    # smoothed on the same time-scale.
    hips_x_studs = sign * hips_world_pos[:, 0] * CM_TO_STUD
    hips_z_studs = sign * hips_world_pos[:, 2] * CM_TO_STUD
    hips_y_studs = hips_world_pos[:, 1] * CM_TO_STUD  # for bob below
    pos_sigma = max(1.0, POSITION_SMOOTH_SIGMA_SOURCE_FRAMES / max(1, stride))
    # Linear-detrend the forward translation before smoothing. Without
    # this the gaussian smoother's edge-padded boundary biases against
    # the linear trend (it pads with frame[0] / frame[-1] values, so
    # the smoothed signal "lags" the actual motion at the start and
    # "leads" at the end). Looping clips wrap from frame N back to
    # frame 0 — and an O(0.7 stud) residual on each side popped the
    # LowerTorso for a frame or two right at the seam. Detrending
    # makes the smoother see only the small per-step wobble (which
    # has well-behaved edges), so the residual is consistent
    # throughout the clip.
    if out_frame_count >= 2:
        x_trend = np.linspace(hips_x_studs[0], hips_x_studs[-1], out_frame_count)
        z_trend = np.linspace(hips_z_studs[0], hips_z_studs[-1], out_frame_count)
    else:
        x_trend = np.zeros_like(hips_x_studs)
        z_trend = np.zeros_like(hips_z_studs)
    hips_x_smooth = gaussian_smooth_1d(hips_x_studs - x_trend, pos_sigma) + x_trend
    hips_z_smooth = gaussian_smooth_1d(hips_z_studs - z_trend, pos_sigma) + z_trend
    hrp_pos = np.stack(
        [
            hips_x_smooth,
            np.full(out_frame_count, HRP_REST_Y),
            hips_z_smooth,
        ],
        axis=-1,
    )

    # --- Per-joint Motor6D Transform rotations -----------------------------
    # World-delta retargeting: R15 body part at time t has world rotation
    # equal to Geno joint's world delta from bind. Motor6D Transform =
    # inv(R15 parent world) * R15 joint world.
    if FLIP_WORLD:
        base = np.array([0.0, 0.0, 1.0, 0.0])  # 180 deg about Y (w,x,y,z)
    else:
        base = np.array([1.0, 0.0, 0.0, 0.0])
    base_inv = quat.inv(base)

    r15_world: dict[str, np.ndarray] = {"HumanoidRootPart": hrp_rot_wxyz}
    for r15_part, src_joint, _r15_parent, _m6d in R15_JOINTS:
        j = name_to_idx[src_joint]
        r15_world[r15_part] = D[:, j]

    transforms: dict[str, np.ndarray] = {}
    for r15_part, _src_joint, r15_parent, _m6d in R15_JOINTS:
        child_world = r15_world[r15_part]
        parent_world = r15_world[r15_parent]
        t = quat.inv_mul(parent_world, child_world)
        if FLIP_WORLD:
            # Re-express rotation in R15's flipped body frame (180° Y).
            # Conjugating by `base` negates X and Z components of quaternion.
            t = quat.mul(quat.mul(base[None, :], t), base_inv[None, :])
        transforms[r15_part] = quat.unroll(t)

    # LowerTorso Position curve = pelvis position residual, in
    # HRP-local frame. The Y component is anchored to the BIND
    # pelvis height (not HRP_REST_Y) so that at the bind pose the
    # offset is zero and the R15 rig's natural hip-to-foot length
    # places feet on the ground. Without this anchor, a source rig
    # whose pelvis is higher than R15's rest hip (e.g. UEFN at 2.7
    # stud vs R15 at ~2.0) lifts the whole rig — feet hover.
    # XZ residual is `raw - smoothed_HRP` regardless of bind, since
    # smoothing already centers it around zero.
    bind_pelvis_y_studs = float(bind["world_pos"][0, hips_idx, 1] * CM_TO_STUD)
    pelvis_pos_r15 = np.stack(
        [sign * hips_world_pos[:, 0] * CM_TO_STUD,
         hips_y_studs,
         sign * hips_world_pos[:, 2] * CM_TO_STUD],
        axis=-1,
    )
    world_delta = pelvis_pos_r15 - hrp_pos                    # (F, 3)
    # Re-anchor Y to bind: `pelvis_y - HRP_REST_Y` → `pelvis_y - bind_y`.
    world_delta[:, 1] = hips_y_studs - bind_pelvis_y_studs
    lt_pos_local = quat.inv_mul_vec(hrp_rot_wxyz, world_delta)  # (F, 3)

    hrp_rot_wxyz = quat.unroll(hrp_rot_wxyz)

    # --- Pack output --------------------------------------------------------
    frame_time = float(anim.get("frame_time") or 1.0 / 60.0)
    src_hz = round(1.0 / frame_time) if frame_time > 0 else 60
    out_hz = src_hz // stride
    result = {
        "schemaVersion": 1,
        "source": {
            "bind": str(bind_path),
            "anim": str(anim_path),
            "startFrame": int(start),
            "sourceFrameCount": int(count),
            "stride": int(stride),
        },
        "frameRate": int(out_hz),
        "frameCount": int(out_frame_count),
        "cmToStud": CM_TO_STUD,
        "flipWorld": bool(FLIP_WORLD),
        "jointOrder": [r[0] for r in R15_JOINTS],
        "parts": {},
        "root": {
            "posX": hrp_pos[:, 0].tolist(),
            "posY": hrp_pos[:, 1].tolist(),
            "posZ": hrp_pos[:, 2].tolist(),
            # Roblox quaternion order: (x, y, z, w)
            **_rot_dict(hrp_rot_wxyz),
        },
    }
    for r15_part, _src, _par, m6d in R15_JOINTS:
        q_wxyz = transforms[r15_part]
        entry = {
            "motor6d": m6d,
            **_rot_dict(q_wxyz),
        }
        if r15_part == "LowerTorso":
            # Bob + sway carried as the Root Motor6D's Transform.Position.
            entry["posX"] = lt_pos_local[:, 0].tolist()
            entry["posY"] = lt_pos_local[:, 1].tolist()
            entry["posZ"] = lt_pos_local[:, 2].tolist()
        result["parts"][r15_part] = entry
    return result


def _rot_dict(q_wxyz: np.ndarray) -> dict:
    """Convert (..., 4) w,x,y,z quats to Roblox-style x,y,z,w column lists."""
    q = quat_wxyz_to_xyzw(q_wxyz)
    return {
        "rotX": q[..., 0].tolist(),
        "rotY": q[..., 1].tolist(),
        "rotZ": q[..., 2].tolist(),
        "rotW": q[..., 3].tolist(),
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def emit_luau(result: dict, clip_name: str) -> str:
    """Emit a Luau literal declaring the clip data + a small builder that
    constructs the CurveAnimation under ReplicatedStorage.Animations[name]
    and returns the registered clip id.

    Designed to be pasted into `mcp__Roblox_Studio__execute_luau`. For long
    clips the literal gets big; consider chunking (upload data in pieces into
    a shared global, then call the builder) once clips exceed ~500 KB.
    """
    parts = result["parts"]
    root = result["root"]
    fr = result["frameCount"]
    hz = result["frameRate"]

    lines: list[str] = []
    w = lines.append

    w("local ReplicatedStorage = game:GetService('ReplicatedStorage')")
    w("local ACP = game:GetService('AnimationClipProvider')")
    w("local Cubic = Enum.KeyInterpolationMode.Cubic")
    w(f"local FRAME_HZ = {hz}")
    w(f"local FRAMES = {fr}")
    w("")
    w(f"local animsFolder = ReplicatedStorage:FindFirstChild('Animations')")
    w("if not animsFolder then animsFolder = Instance.new('Folder', ReplicatedStorage); animsFolder.Name = 'Animations' end")
    w(f"local prior = animsFolder:FindFirstChild('{clip_name}')")
    w("if prior then prior:Destroy() end")
    w("")
    w("local ca = Instance.new('CurveAnimation')")
    w(f"ca.Name = '{clip_name}'")
    w("ca.Parent = animsFolder")
    w("")
    w("local function t(i) return (i - 1) / FRAME_HZ end")
    w("")
    w("local function addRot(partName, qx, qy, qz, qw)")
    w("    local f = Instance.new('Folder', ca); f.Name = partName")
    w("    local rc = Instance.new('RotationCurve', f); rc.Name = 'Rotation'")
    w("    for i = 1, FRAMES do")
    w("        rc:InsertKey(RotationCurveKey.new(t(i), CFrame.new(0,0,0, qx[i], qy[i], qz[i], qw[i]), Cubic))")
    w("    end")
    w("    return f")
    w("end")
    w("")
    w("local function addHrp(px, py, pz, qx, qy, qz, qw)")
    w("    local f = Instance.new('Folder', ca); f.Name = 'HumanoidRootPart'")
    w("    local rc = Instance.new('RotationCurve', f); rc.Name = 'Rotation'")
    w("    local v3 = Instance.new('Vector3Curve', f); v3.Name = 'Position'")
    w("    local cx, cy, cz = v3:X(), v3:Y(), v3:Z()")
    w("    for i = 1, FRAMES do")
    w("        local ti = t(i)")
    w("        rc:InsertKey(RotationCurveKey.new(ti, CFrame.new(0,0,0, qx[i], qy[i], qz[i], qw[i]), Cubic))")
    w("        cx:InsertKey(FloatCurveKey.new(ti, px[i], Cubic))")
    w("        cy:InsertKey(FloatCurveKey.new(ti, py[i], Cubic))")
    w("        cz:InsertKey(FloatCurveKey.new(ti, pz[i], Cubic))")
    w("    end")
    w("end")
    w("")

    def _floats(xs: list[float]) -> str:
        return "{" + ",".join(f"{v:.6f}" for v in xs) + "}"

    # Emit per-part arrays + addRot call.
    for pname, pdata in parts.items():
        w(f"-- {pname}")
        w(f"local _{pname}_qx = {_floats(pdata['rotX'])}")
        w(f"local _{pname}_qy = {_floats(pdata['rotY'])}")
        w(f"local _{pname}_qz = {_floats(pdata['rotZ'])}")
        w(f"local _{pname}_qw = {_floats(pdata['rotW'])}")
        w(f"addRot('{pname}', _{pname}_qx, _{pname}_qy, _{pname}_qz, _{pname}_qw)")
        w("")

    # HRP
    w("-- HumanoidRootPart (root motion)")
    w(f"local _hrp_px = {_floats(root['posX'])}")
    w(f"local _hrp_py = {_floats(root['posY'])}")
    w(f"local _hrp_pz = {_floats(root['posZ'])}")
    w(f"local _hrp_qx = {_floats(root['rotX'])}")
    w(f"local _hrp_qy = {_floats(root['rotY'])}")
    w(f"local _hrp_qz = {_floats(root['rotZ'])}")
    w(f"local _hrp_qw = {_floats(root['rotW'])}")
    w("addHrp(_hrp_px, _hrp_py, _hrp_pz, _hrp_qx, _hrp_qy, _hrp_qz, _hrp_qw)")
    w("")
    w("local clipId = ACP:RegisterAnimationClip(ca)")
    w(f"return string.format('Built CurveAnimation {clip_name} with %d frames -> clipId=%s', FRAMES, tostring(clipId))")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind", required=True, type=Path, help="Path to bind-pose BVH")
    p.add_argument("--anim", required=True, type=Path, help="Path to animation BVH")
    p.add_argument("--start", type=int, default=0, help="First frame (0-indexed)")
    p.add_argument("--count", type=int, default=None, help="Number of frames (default: to end)")
    p.add_argument("--out", required=True, type=Path, help="Output JSON path")
    p.add_argument("--emit-luau", type=Path, default=None, help="Also write a Luau builder script")
    p.add_argument("--clip-name", type=str, default=None, help="Clip name for Luau builder (default: derived from --anim)")
    p.add_argument(
        "--bind-from-anim-start",
        action="store_true",
        help="Use the anim's --start frame as the bind reference instead of the --bind file. "
        "Useful when the target R15 rig's rest pose doesn't match the --bind file's pose "
        "(e.g., R15 is I-pose but --bind is loose A-pose). Motor6D transform = identity at "
        "animation start, and only the clip's motion variation is applied to the rig. "
        "Per-clip bind means clips can't be cleanly blended — use --synthetic-ipose-bind "
        "for a fixed shared reference across clips.",
    )
    p.add_argument(
        "--synthetic-ipose-bind",
        action="store_true",
        help="[Dead end — doesn't work] Override bind world rotations with a hypothetical "
        "Geno I-pose (arms/legs straight down, torso upright). Produces ~180° deltas "
        "because Geno's BVH data never shows bones in that hypothetical orientation.",
    )
    p.add_argument(
        "--hardcoded-bind",
        action="store_true",
        help="Override bind world rotations with the HARDCODED_BIND_WORLD dict "
        "(extracted from walk1_subject5.bvh frame 160, visually close to I-pose). "
        "Consistent across all clips — use this for motion-matching databases.",
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help="Sample every Nth source frame (stride=2 → 30 Hz from 60 Hz BVH source). "
        "Reduces output size and Studio CurveAnimation key count proportionally.",
    )
    p.add_argument(
        "--rig", choices=("geno", "motusman", "uefn", "soma", "soma_r15plus"), default="geno",
        help="Source skeleton layout. 'geno' (default) is the lafan1 BVH rig; "
        "'motusman' is the MocapOnline MotusMan_v55 FBX rig; 'uefn' is the "
        "Unreal-Engine Mobility Starter GASP rig (Z-up FBX, auto-converted "
        "to Y-up by the loader); 'soma' is Kimodo's somaskel77 BVH rig "
        "(synthetic Root + Hips, T-pose bind, retargets to standard R15); "
        "'soma_r15plus' is the same somaskel77 source but retargets to the "
        "R15-plus rig (spine, clavicles, fingers, toes). Selects which "
        "R15_JOINTS / R15_CHAINS / bind tables are used — does NOT affect "
        "file-format dispatch (that's by --anim's extension).",
    )
    p.add_argument(
        "--hrp-scale", type=float, default=1.0,
        help="Multiply HRP horizontal trajectory (X, Z) and the LowerTorso "
        "horizontal residual by this factor. Compensates for rig-vs-source "
        "leg-length mismatch when the source skeleton has shorter (or longer) "
        "legs than R15. R15's default leg length is ~1.4× SOMA's, so its feet "
        "sweep further per stride than the source HRP travels — using <1.0 "
        "slows HRP to match the visible step distance; >1.0 stretches the "
        "trajectory to match R15's leg sweep. Y is left untouched (vertical "
        "bob shouldn't scale). Try 0.72 (= SOMA/R15 leg ratio) first.",
    )
    args = p.parse_args(argv)

    set_rig(args.rig)

    result = retarget(args.bind, args.anim, args.start, args.count,
                      bind_from_anim_start=args.bind_from_anim_start,
                      synthetic_ipose_bind=args.synthetic_ipose_bind,
                      hardcoded_bind=args.hardcoded_bind,
                      stride=args.stride)

    if args.hrp_scale != 1.0:
        s = float(args.hrp_scale)
        root = result["root"]
        root["posX"] = [v * s for v in root["posX"]]
        root["posZ"] = [v * s for v in root["posZ"]]
        # Scale LowerTorso XZ residual identically so pelvis_world = HRP +
        # LowerTorso stays consistent (the residual was raw_pelvis - smooth_HRP
        # in pre-scale units; both factors get the same scale).
        if "LowerTorso" in result["parts"]:
            lt = result["parts"]["LowerTorso"]
            if "posX" in lt:
                lt["posX"] = [v * s for v in lt["posX"]]
                lt["posZ"] = [v * s for v in lt["posZ"]]
        result["hrpScale"] = s

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, separators=(",", ":"))

    # Quick summary for the operator.
    root = result["root"]
    dx = root["posX"][-1] - root["posX"][0]
    dz = root["posZ"][-1] - root["posZ"][0]
    dist = math.hypot(dx, dz)
    duration = result["frameCount"] / result["frameRate"]
    speed = dist / duration if duration > 0 else 0.0
    print(
        f"Wrote {args.out} — {result['frameCount']} frames @ {result['frameRate']} Hz "
        f"({duration:.2f}s). Root traveled {dist:.2f} studs ({speed:.2f} studs/s)."
    )

    if args.emit_luau is not None:
        clip_name = args.clip_name or args.anim.stem
        luau_src = emit_luau(result, clip_name)
        args.emit_luau.parent.mkdir(parents=True, exist_ok=True)
        args.emit_luau.write_text(luau_src)
        print(f"Wrote {args.emit_luau} — {len(luau_src) // 1024} KiB Luau source, clip '{clip_name}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
