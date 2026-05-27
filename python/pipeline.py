# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""End-to-end r6-uprez pipeline: Roblox asset id → Roblox CurveAnimation rbxm.

Stages:
    1. extract_pose.py  → work/<name>/pose.json
    2. roblox_to_kimodo.py → work/<name>/constraints.json + meta.json
    3. run_kimodo.py    → work/<name>/generated.bvh
    4. export_r15.retarget + hrp_scale → work/<name>/r15.json
    5. build_rbxm.py    → work/<name>/r15.rbxm

Usage:
    uv run --with numpy --with scipy python/pipeline.py \
        --asset-id 507771019 --out work/walk

Rig type (R6 vs R15) is auto-detected from the clip's bone names.
Root motion is folded into LowerTorso by default; pass --root-motion to
emit HRP curves instead (useful for clips that should drive locomotion).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import extract_pose  # noqa: E402
import roblox_to_kimodo  # noqa: E402
import run_kimodo  # noqa: E402
import build_rbxm  # noqa: E402

BIND_BVH = REPO_ROOT / "data" / "soma_tpose.bvh"
RIG = "soma"
HRP_SCALE = 0.72


import math as _math


def _quat_slerp_from_identity(q: tuple, t: float) -> tuple:
    """SLERP from identity (0,0,0,1) to unit quaternion q with parameter t.

    Returns the rotation `q ** t` (axis preserved, angle scaled by t).
    Handles short-arc selection via the standard w-sign flip.
    """
    qx, qy, qz, qw = q
    if qw < 0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    qw_clamped = max(-1.0, min(1.0, qw))
    angle = 2.0 * _math.acos(qw_clamped)
    if abs(angle) < 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    sin_half = _math.sin(angle * 0.5)
    if sin_half < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    ax, ay, az = qx / sin_half, qy / sin_half, qz / sin_half
    new_half = angle * 0.5 * t
    s = _math.sin(new_half)
    c = _math.cos(new_half)
    return (ax * s, ay * s, az * s, c)


def _quat_mul_xyzw(a: tuple, b: tuple) -> tuple:
    """Hamilton product, both inputs (x, y, z, w). build_rbxm.lua reads
    (rotX, rotY, rotZ, rotW) which is xyzw."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_rotate_xyzw(q: tuple, v: tuple) -> tuple:
    """Rotate vector v by quaternion q (xyzw)."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _inertial_blend_loop_seam(result: dict, blend_frames: int) -> None:
    """Mask the loop seam by faking an inertial blend over the first
    `blend_frames` frames. Treats the clip's last frame as the "previous"
    pose feeding into frame 0 and decays the resulting offset to zero by
    frame `blend_frames`.

    Per curve:
        offset            = last - first
        blended[0]        = first + offset * 1            == last
        blended[i in 0..N) = first[i] + offset * decay(i)
        blended[N..F-1]   = unchanged

    `decay(i)` is `1 - smoothstep(i / blend_frames)` so position and
    velocity are continuous at both ends of the blend window. Rotations
    use a SLERP-from-identity by t=decay(i) of the offset rotation, then
    composed with the original frame's rotation.

    With blended[0] == last, Studio's loop wraps `last → blended[0]` with
    zero pop. The next blend_frames frames smoothly arrive at the
    original animation's natural progression.
    """
    n = result.get("frameCount", 0)
    if blend_frames <= 0 or n < blend_frames + 2:
        return

    def _decay(i: int) -> float:
        if i >= blend_frames:
            return 0.0
        # `t = i / (blend_frames - 1)` so decay(blend_frames-1) = 0 exactly,
        # i.e. the last blended frame lands on the original animation —
        # avoids a tiny residual offset at the blend tail.
        if blend_frames == 1:
            return 1.0 if i == 0 else 0.0
        t = i / float(blend_frames - 1)
        return 1.0 - (3.0 * t * t - 2.0 * t * t * t)

    def _blend_pos(arr_dict: dict) -> None:
        for k in ("posX", "posY", "posZ"):
            arr = arr_dict.get(k)
            if not arr:
                continue
            offset = arr[-1] - arr[0]
            if abs(offset) < 1e-12:
                continue
            for i in range(blend_frames):
                arr[i] = arr[i] + offset * _decay(i)

    def _blend_rot(arr_dict: dict) -> None:
        if not all(k in arr_dict for k in ("rotX", "rotY", "rotZ", "rotW")):
            return
        last_q = (arr_dict["rotX"][-1], arr_dict["rotY"][-1], arr_dict["rotZ"][-1], arr_dict["rotW"][-1])
        first_q = (arr_dict["rotX"][0],  arr_dict["rotY"][0],  arr_dict["rotZ"][0],  arr_dict["rotW"][0])
        # Hemisphere-align last_q to first_q before computing the offset.
        # `q` and `-q` represent the same rotation, but their `q1 * inv(q2)`
        # product differs in sign — and our SLERP would then take the long
        # arc through the 4D sphere even though the underlying rotation
        # delta is small. Aligning by dot product ensures the resulting
        # offset_q represents the short-arc difference between the two
        # rotations.
        dot = (last_q[0] * first_q[0] + last_q[1] * first_q[1]
               + last_q[2] * first_q[2] + last_q[3] * first_q[3])
        if dot < 0.0:
            last_q = (-last_q[0], -last_q[1], -last_q[2], -last_q[3])
        first_inv = (-first_q[0], -first_q[1], -first_q[2], first_q[3])
        offset_q = _quat_mul_xyzw(last_q, first_inv)
        for i in range(blend_frames):
            d = _decay(i)
            blend_q = _quat_slerp_from_identity(offset_q, d)
            cur = (arr_dict["rotX"][i], arr_dict["rotY"][i], arr_dict["rotZ"][i], arr_dict["rotW"][i])
            nx, ny, nz, nw = _quat_mul_xyzw(blend_q, cur)
            arr_dict["rotX"][i] = nx
            arr_dict["rotY"][i] = ny
            arr_dict["rotZ"][i] = nz
            arr_dict["rotW"][i] = nw

    def _unroll(arr_dict: dict) -> None:
        """Walk consecutive rotation keyframes and negate any quat that
        has a negative dot product with its predecessor. Roblox's
        RotationCurve interpolator takes the long-arc path between two
        keys when dot(q[i], q[i+1]) < 0, which manifests as a "spinning"
        bone. Running this once over the full curve guarantees every
        adjacent pair is short-arc.
        """
        if not all(k in arr_dict for k in ("rotX", "rotY", "rotZ", "rotW")):
            return
        rx = arr_dict["rotX"]; ry = arr_dict["rotY"]
        rz = arr_dict["rotZ"]; rw = arr_dict["rotW"]
        for i in range(1, len(rw)):
            dot = rx[i - 1] * rx[i] + ry[i - 1] * ry[i] + rz[i - 1] * rz[i] + rw[i - 1] * rw[i]
            if dot < 0.0:
                rx[i] = -rx[i]; ry[i] = -ry[i]; rz[i] = -rz[i]; rw[i] = -rw[i]

    if "root" in result:
        _blend_pos(result["root"])
        _blend_rot(result["root"])
        _unroll(result["root"])
    for part in result.get("parts", {}).values():
        _blend_pos(part)
        _blend_rot(part)
        _unroll(part)


def _trim_middle_cycle(result: dict, source_n_frames: int, loop_passes: int) -> None:
    """For multi-pass looped extractions, slice the retargeted result to
    just the middle cycle. Cycles share their boundary frame in the looped
    sample (frame 0 of cycle k+1 == frame F-1 of cycle k, both at source
    TimePosition 0), so the middle cycle occupies indices
    [source_F-1, 2*(source_F-1)] inclusive — `source_F` frames preserving
    the loop-point frame at the end.

    Mutates `result` in place.
    """
    if loop_passes <= 1 or source_n_frames < 2:
        return
    s = source_n_frames - 1
    if loop_passes < 3:
        # 2-pass: take cycle 1's range. Less ideal than 3-pass (no symmetric
        # context on both sides) but better than nothing.
        start = 0
    else:
        # Middle cycle index = the (loop_passes // 2)-th cycle.
        mid = loop_passes // 2
        start = mid * s
    end = start + s + 1  # inclusive endpoint preserved for clean rbxm loop
    n_total = result.get("frameCount", 0)
    if end > n_total:
        return

    def _slice(arr_dict: dict) -> None:
        for k, v in list(arr_dict.items()):
            if isinstance(v, list) and len(v) == n_total:
                arr_dict[k] = v[start:end]

    if "root" in result:
        _slice(result["root"])
    for part_name, part in result.get("parts", {}).items():
        _slice(part)
    result["frameCount"] = end - start


def _fold_root_into_lower_torso(result: dict) -> None:
    """Fold HRP motion into LowerTorso as a delta from frame 0, then DROP
    the HRP curves so Studio plays the animation at the character's spawn
    pose rather than teleporting it to origin.

    Math (per frame, with t=0 as the reference):
        delta_HRP[t]    = inv(T_HRP[0]) * T_HRP[t]
        new_LT_local[t] = delta_HRP[t] * old_LT_local[t]

    At t=0, delta_HRP = identity ⇒ new_LT_local = old_LT_local (unchanged
    from rest), which avoids the "double-applies HRP world position to
    LowerTorso Motor6D Transform" hover bug. For t>0, the LT Motor6D
    Transform absorbs whatever HRP drift the source had relative to the
    starting pose.

    Removing `result["root"]` (instead of zeroing it) signals build_rbxm.lua
    to skip emitting the HumanoidRootPart curves entirely, so Studio uses
    the spawn HRP pose (feet on ground) without override.
    """
    parts = result.get("parts", {})
    if "LowerTorso" not in parts:
        return
    n = result["frameCount"]
    root = result["root"]
    lt = parts["LowerTorso"]

    # LT often has no position curve in the input (Motor6Ds at rest = no
    # translation delta). Once we fold the HRP delta in there's animated
    # translation, so create the curves if missing.
    for axis in ("posX", "posY", "posZ"):
        if axis not in lt:
            lt[axis] = [0.0] * n

    ref_pos = (root["posX"][0], root["posY"][0], root["posZ"][0])
    ref_rot = (root["rotX"][0], root["rotY"][0], root["rotZ"][0], root["rotW"][0])
    # Inverse of a unit quaternion = its conjugate.
    ref_rot_inv = (-ref_rot[0], -ref_rot[1], -ref_rot[2], ref_rot[3])

    new_pos_x = [0.0] * n
    new_pos_y = [0.0] * n
    new_pos_z = [0.0] * n
    new_rot_x = [0.0] * n
    new_rot_y = [0.0] * n
    new_rot_z = [0.0] * n
    new_rot_w = [0.0] * n
    for i in range(n):
        rp = (root["posX"][i], root["posY"][i], root["posZ"][i])
        rq = (root["rotX"][i], root["rotY"][i], root["rotZ"][i], root["rotW"][i])
        lp = (lt["posX"][i], lt["posY"][i], lt["posZ"][i])
        lq = (lt["rotX"][i], lt["rotY"][i], lt["rotZ"][i], lt["rotW"][i])

        # delta_HRP[t] = inv(T_HRP[0]) * T_HRP[t], decomposed as:
        #   delta_pos_world = T_HRP[t].pos - T_HRP[0].pos
        #   delta_pos_local = inv(R_HRP[0]) * delta_pos_world
        #   delta_rot       = inv(R_HRP[0]) * R_HRP[t]
        dp_world = (rp[0] - ref_pos[0], rp[1] - ref_pos[1], rp[2] - ref_pos[2])
        dp_local = _quat_rotate_xyzw(ref_rot_inv, dp_world)
        d_rot = _quat_mul_xyzw(ref_rot_inv, rq)

        # new_LT_local = delta_HRP * old_LT_local
        rotated_lp = _quat_rotate_xyzw(d_rot, lp)
        new_pos_x[i] = dp_local[0] + rotated_lp[0]
        new_pos_y[i] = dp_local[1] + rotated_lp[1]
        new_pos_z[i] = dp_local[2] + rotated_lp[2]
        nq = _quat_mul_xyzw(d_rot, lq)
        new_rot_x[i] = nq[0]
        new_rot_y[i] = nq[1]
        new_rot_z[i] = nq[2]
        new_rot_w[i] = nq[3]

    lt["posX"] = new_pos_x
    lt["posY"] = new_pos_y
    lt["posZ"] = new_pos_z
    lt["rotX"] = new_rot_x
    lt["rotY"] = new_rot_y
    lt["rotZ"] = new_rot_z
    lt["rotW"] = new_rot_w

    # Drop HRP curves entirely. build_rbxm.lua checks `data.root` and skips
    # the HumanoidRootPart folder when it's nil.
    result.pop("root", None)


def _retarget_bvh_to_r15_json(
    bvh_path: Path,
    json_path: Path,
    *,
    root_motion: bool,
    source_n_frames: int = 0,
    loop_passes: int = 1,
    looped: bool = False,
    inertial_blend_frames: int = 0,
    hrp_scale: float | None = None,
) -> dict:
    """Run the SOMA→R15 retarget. Mirrors batch_retarget._process_one.

    `hrp_scale` overrides the module-level HRP_SCALE for a single run
    (None ⇒ use HRP_SCALE). Pass it when retargeting onto a non-stock
    rig (e.g., Rthro) so the hip XZ stride matches the target rig's
    leg length and feet stop sliding.
    """
    import export_r15

    export_r15.set_rig(RIG)
    result = export_r15.retarget(
        BIND_BVH, bvh_path,
        start=0, count=None,
        bind_from_anim_start=False,
        synthetic_ipose_bind=False,
        hardcoded_bind=True,
        stride=1,
    )
    effective_hrp_scale = HRP_SCALE if hrp_scale is None else float(hrp_scale)
    if effective_hrp_scale != 1.0:
        s = float(effective_hrp_scale)
        root = result["root"]
        root["posX"] = [v * s for v in root["posX"]]
        root["posZ"] = [v * s for v in root["posZ"]]
        if "LowerTorso" in result["parts"]:
            lt = result["parts"]["LowerTorso"]
            if "posX" in lt:
                lt["posX"] = [v * s for v in lt["posX"]]
                lt["posZ"] = [v * s for v in lt["posZ"]]
        result["hrpScale"] = s
    # Trim BEFORE folding so the fold's frame-0 reference is the start of
    # the middle cycle (where the LowerTorso should rest at identity), not
    # the start of cycle 1.
    if loop_passes > 1 and source_n_frames >= 2:
        _trim_middle_cycle(result, source_n_frames, loop_passes)
    if not root_motion:
        _fold_root_into_lower_torso(result)
    # Inertial blend last so it operates on the final per-curve arrays
    # that build_rbxm.lua actually emits — works for both root-motion and
    # folded-LT modes.
    if looped and inertial_blend_frames > 0:
        _inertial_blend_loop_seam(result, blend_frames=inertial_blend_frames)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(result, f, separators=(",", ":"))
    return {
        "frameCount": result.get("frameCount"),
        "frameRate": result.get("frameRate"),
        "root_motion": root_motion,
    }


def _build_rbxm(out_dir: Path, name: str, roblox_cli: str | None) -> Path:
    """Invoke build_rbxm.py against the single-clip layout we built.

    build_rbxm walks `<repo_root>/<in_rel>` for r15.json files. We pass
    --repo-root=out_dir.parent and --in=out_dir.name so the walk root is
    exactly our out_dir. The Lua's `categoryAndClip` parser handles the
    1-level layout `<name>/r15.json` by synthesizing a category from the
    clip name's first underscore token. Per-clip rbxm is what we want;
    per-category and corpus emission are disabled.
    """
    cmd_args = [
        "--in", out_dir.name,
        "--repo-root", str(out_dir.parent),
        "--per-clip",
        "--no-per-category",
        "--no-corpus",
    ]
    if roblox_cli:
        cmd_args.extend(["--roblox-cli", roblox_cli])
    print(f"[pipeline] build_rbxm {' '.join(cmd_args)}", flush=True)
    rc = build_rbxm.main(cmd_args)
    if rc != 0:
        raise RuntimeError(f"build_rbxm exited with {rc}")
    rbxm = out_dir / name / "r15.rbxm"
    if not rbxm.is_file():
        raise RuntimeError(f"build_rbxm did not produce {rbxm}")
    return rbxm


def _parse_csv_floats(s: str | None) -> list[float] | None:
    if not s:
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-id", type=int, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory. Final rbxm at <out>/<name>/r15.rbxm.")
    p.add_argument("--name", type=str, default=None,
                   help="Clip name (default: 'asset_<id>').")
    p.add_argument("--prompt", type=str, default="",
                   help="Kimodo text prompt. Empty = unconditioned, let "
                        "constraints drive the result.")
    p.add_argument("--cfg-weight", type=float, default=2.0,
                   help="Kimodo constraint adherence (separated CFG). "
                        "Higher = stricter, less creative.")
    p.add_argument("--cfg-text-weight", type=float, default=2.0)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--explicit-times", type=str, default=None,
                   help="Comma-separated explicit keyframe times (s) for ALL effectors.")
    p.add_argument("--left-foot-times", type=str, default=None)
    p.add_argument("--right-foot-times", type=str, default=None)
    p.add_argument("--left-hand-times", type=str, default=None)
    p.add_argument("--right-hand-times", type=str, default=None)
    p.add_argument("--foot-min-separation", type=int, default=8)
    p.add_argument("--hand-min-separation", type=int, default=8)
    p.add_argument("--scale", type=float, default=None,
                   help="Stud→meter scale for stage 2 (default: roblox_to_kimodo's "
                        "0.30). Lower = smaller character in Kimodo space.")
    p.add_argument("--y-offset-mode", choices=["auto", "none"], default="auto")
    p.add_argument("--loop-window", type=int, default=2,
                   help="For looped clips, pin N frames at each end to the "
                        "source pose so velocity matches at the loop seam. "
                        "Default 2.")
    p.add_argument("--no-effectors", action="append", default=[],
                   choices=["feet", "hands"])
    p.add_argument("--min-duration", type=float, default=0.0,
                   help="Minimum output duration (s); loops the source clip "
                        "if shorter. Default 0 = no looping.")
    p.add_argument("--loop-passes", type=int, default=3,
                   help="Sample N copies of looped clips so Kimodo gets "
                        "periodic context, then trim to the middle cycle "
                        "on export. Only applies when AnimationClip.Loop. "
                        "Default 3.")
    p.add_argument("--inertial-blend", type=int, default=8,
                   help="For looped clips, fake an inertial blend over the "
                        "first N frames using the clip's last frame as the "
                        "'previous' pose. Soaks up residual seam mismatch. "
                        "Default 8. Pass 0 to disable.")
    p.add_argument("--root-motion", dest="root_motion", action="store_true",
                   help="Keep HumanoidRootPart curves in the rbxm (default: "
                        "fold root motion into LowerTorso so HRP stays at "
                        "rest). Disable for assets that shouldn't drive "
                        "character locomotion (waves, dances, gestures).")
    p.set_defaults(root_motion=False)
    p.add_argument("--roblox-cli", type=str, default=None)
    p.add_argument("--skip", action="append", default=[],
                   choices=["pose", "constraints", "kimodo", "retarget", "rbxm"],
                   help="Skip a stage (re-using existing output). Repeatable.")
    args = p.parse_args(argv)

    name = args.name or f"asset_{args.asset_id}"
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = out_dir / name
    clip_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: extract pose ----
    pose_path = clip_dir / "pose.json"
    if "pose" in args.skip and pose_path.is_file():
        print(f"[pipeline] skip stage 1, using {pose_path}")
    else:
        extract_pose.extract_pose(
            asset_id=args.asset_id,
            out_dir=clip_dir,
            sample_fps=30,
            min_duration_s=args.min_duration,
            loop_passes=args.loop_passes,
            roblox_cli=args.roblox_cli,
        )

    # ---- Stage 2: constraints ----
    constraints_path = clip_dir / "constraints.json"
    if "constraints" in args.skip and constraints_path.is_file():
        print(f"[pipeline] skip stage 2, using {constraints_path}")
    else:
        explicit_per_limb = {
            "left_foot":  _parse_csv_floats(args.left_foot_times),
            "right_foot": _parse_csv_floats(args.right_foot_times),
            "left_hand":  _parse_csv_floats(args.left_hand_times),
            "right_hand": _parse_csv_floats(args.right_hand_times),
        }
        explicit_per_limb = {k: v for k, v in explicit_per_limb.items() if v}
        kw = dict(
            out_dir=out_dir,
            name=name,
            no_effectors=args.no_effectors,
            foot_min_separation=args.foot_min_separation,
            hand_min_separation=args.hand_min_separation,
            explicit_times=_parse_csv_floats(args.explicit_times),
            explicit_per_limb=explicit_per_limb,
            y_offset_mode=args.y_offset_mode,
            loop_window=args.loop_window,
        )
        if args.scale is not None:
            kw["stud_to_m"] = args.scale
        roblox_to_kimodo.extract_constraints(pose_path, **kw)

    # ---- Stage 3: Kimodo ----
    bvh_path = clip_dir / "generated.bvh"
    if "kimodo" in args.skip and bvh_path.is_file():
        print(f"[pipeline] skip stage 3, using {bvh_path}")
    else:
        extra = [
            "--cfg_type", "separated",
            "--cfg_weight", str(args.cfg_text_weight), str(args.cfg_weight),
        ]
        run_kimodo.run_kimodo(
            clip_dir,
            prompt=args.prompt,
            seed=args.seed,
            diffusion_steps=args.diffusion_steps,
            extra_args=extra,
        )

    # ---- Stage 4: BVH → R15 JSON ----
    r15_json = clip_dir / "r15.json"
    if "retarget" in args.skip and r15_json.is_file():
        print(f"[pipeline] skip stage 4, using {r15_json}")
    else:
        # Read pose meta to know source cycle length for the trim step.
        pose_meta = json.loads(pose_path.read_text()) if pose_path.is_file() else {}
        source_n_frames = int(pose_meta.get("source_n_frames", 0))
        actual_passes = int(pose_meta.get("loop_passes", 1))
        looped_flag = bool(pose_meta.get("looped", False))
        info = _retarget_bvh_to_r15_json(
            bvh_path, r15_json,
            root_motion=args.root_motion,
            source_n_frames=source_n_frames,
            loop_passes=actual_passes,
            looped=looped_flag,
            inertial_blend_frames=args.inertial_blend,
        )
        print(f"[pipeline] retarget OK: {info}")

    # ---- Stage 5: rbxm ----
    if "rbxm" in args.skip and (clip_dir / "r15.rbxm").is_file():
        rbxm_path = clip_dir / "r15.rbxm"
        print(f"[pipeline] skip stage 5, using {rbxm_path}")
    else:
        rbxm_path = _build_rbxm(out_dir, name, args.roblox_cli)

    pose_meta = json.loads((clip_dir / "pose.json").read_text())
    print(json.dumps({
        "name": name,
        "asset_id": args.asset_id,
        "rig": pose_meta.get("rig_type"),
        "rbxm": str(rbxm_path),
        "duration_s": json.loads((clip_dir / "meta.json").read_text())["duration_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
