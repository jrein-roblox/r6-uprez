# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Prompt-only pipeline: text prompt → Roblox CurveAnimation rbxm.

Reuses the back half of `pipeline.py` (BVH→R15 retarget, HRP-scale,
optional root-motion fold, optional loop-seam inertial blend, rbxm build).
Skips pose extraction and constraint synthesis since there is no source
asset.

Stages:
    A. run kimodo_gen "<prompt>" --duration <secs>  → work/<name>/generated.bvh
    B. export_r15.retarget + hrp_scale              → work/<name>/r15.json
    C. build_rbxm.py                                → work/<name>/r15.rbxm

Usage:
    uv run --with numpy --with scipy python/prompt_pipeline.py \
        --prompt "a person waves hello" --out work --name wave --duration 3.0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import build_rbxm  # noqa: E402
import export_r15  # noqa: E402
import pipeline as parent_pipeline  # noqa: E402
import run_kimodo  # noqa: E402

DEFAULT_MODEL = run_kimodo.DEFAULT_MODEL


# Default target rig dimensions: the Rthro "Rig" in workspace (the user's
# canonical playback rig). Numbers measured directly from the Studio
# instance (workspace.Rig) via MCP inspect. Stock R15 numbers are kept in
# the dict below for reference / for users who want to override.
#
#   HumanoidRootPart.Position.Y  = 4.1197 stud (HRP rest world Y)
#   LeftAnkleRigAttachment.WorldPosition.Y = 0.4505 stud
#   ⇒ HRP-to-ankle = 4.1197 - 0.4505 = 3.6692 stud
#
# Foot block bottom Y = 0.0069 stud (≈ floor); ankle is 0.4436 stud above
# foot bottom. So `target_rest_ankle_y = HRP_REST_Y - HRP_TO_ANKLE` ≈ 0.45,
# which is what we want as the rest-pose anchor for grounding.
_RTHRO_HRP_REST_Y = 4.1197
_RTHRO_HRP_TO_ANKLE = 3.6693

# Stock R15 (no avatar scaling) for reference: HRP_REST_Y = 2.0, HRP-to-
# ankle = 1.6. Pass these via --target-hrp-rest-y / --target-hrp-to-ankle
# if retargeting onto stock R15 instead of the Rthro Rig.
_STOCK_R15_HRP_REST_Y = 2.0
_STOCK_R15_HRP_TO_ANKLE = 1.6

_DEFAULT_TARGET_HRP_REST_Y = _RTHRO_HRP_REST_Y
_DEFAULT_TARGET_HRP_TO_ANKLE = _RTHRO_HRP_TO_ANKLE


def _soma_bind_hip_to_ankle_studs() -> float:
    """Hip-to-ankle Y of the soma bind BVH, in studs.

    Loaded at runtime so the geometric HRP_SCALE auto-derivation tracks
    whatever bind file `pipeline.BIND_BVH` points at. Currently
    `data/soma_tpose.bvh` ⇒ 88.11 cm = 2.643 stud.
    """
    export_r15.set_rig(parent_pipeline.RIG)
    bind = export_r15._load_anim_any(parent_pipeline.BIND_BVH)
    names = list(bind["names"])
    bp = bind["world_pos"]
    hi = names.index("Hips")
    li = names.index("LeftFoot")
    ri = names.index("RightFoot")
    bind_hip_y = float(bp[0, hi, 1])
    bind_ankle_y = min(float(bp[0, li, 1]), float(bp[0, ri, 1]))
    return (bind_hip_y - bind_ankle_y) * export_r15.CM_TO_STUD


def _ground_y(
    result: dict,
    bvh_path: Path,
    mode: str,
    *,
    target_hrp_rest_y: float = _DEFAULT_TARGET_HRP_REST_Y,
    target_hrp_to_ankle: float = _DEFAULT_TARGET_HRP_TO_ANKLE,
    extra_bias: float = 0.0,
) -> float:
    """Shift LowerTorso.posY so the rig's feet sit on the floor.

    Why a post-pass instead of fixing it in retarget: the export_r15
    anchor (`LT.posY = (bvh_hip_y - bind_pelvis_y) * cmToStud`) keeps the
    R15 hip tracking the BVH hip 1:1, but ignores the actual leg-chain
    Y projection. Two failure modes:

      1. Wave-style clips: kimodo's frame-0 hip is slightly above bind
         and the legs are near rest, so feet float by 1-2 inches.
      2. Crouch-style clips: kimodo emits bent knees with a fixed-Y hip
         (no pelvis drop), so the R15 leg chain compresses but the rig
         doesn't lower — feet hover several inches above ground.

    Strategy: use the BVH's already-FK'd ankle world positions to predict
    the R15 ankle world Y per frame, then shift LT.posY by a single
    constant so the anchor frame's foot lands on the ground.

    Proportional model: at any frame the R15 hip-to-ankle Y projection
    scales linearly with the BVH's, by the rest-pose ratio. Specifically:

        soma_chain_cm[i] = bvh_hip_y[i] - bvh_ankle_y[i]
        R15_chain[i]     = target_hrp_to_ankle * (soma_chain_cm[i]
                                                   / soma_bind_chain_cm)
        R15_ankle_y[i]   = target_hrp_rest_y + LT.posY[i] - R15_chain[i]

    The soma_bind_chain term comes from the bind BVH at runtime, so the
    only per-rig knobs are `target_hrp_rest_y` and `target_hrp_to_ankle`
    (defaulted to stock R15 — see args).

    Mode 'first' anchors at frame 0 (best when the prompt's first frame
    is a standing/planted pose); 'min' anchors at the frame with the
    lowest predicted ankle Y (no ground penetration); 'off' disables.
    `extra_bias` is added in studs (positive raises the rig) for manual
    fine-tuning if the proportional model over- or under-shoots.

    `target_hrp_rest_y` / `target_hrp_to_ankle` describe the rig the
    rbxm is played on. Defaults match the Rthro "Rig" in the user's
    workspace (4.1197 / 3.6693, measured from the live Studio instance).
    Pass `--target-hrp-rest-y 2.0 --target-hrp-to-ankle 1.6` for stock
    R15. The leg-chain scale used per frame is proportional to
    `target_hrp_to_ankle`, so the wrong target chain length both
    over/under-corrects standing pose AND amplifies the error in
    crouches (where soma_chain shrinks and the proportional model
    multiplies any error in the rest ratio).
    """
    import numpy as np  # local: avoid hard dep if pipeline runs without

    if mode == "off" and extra_bias == 0.0:
        return 0.0

    target = result.get("root") if "root" in result else result.get("parts", {}).get("LowerTorso")
    if not target or "posY" not in target or not target["posY"]:
        return 0.0

    pos_y = target["posY"]
    n = len(pos_y)

    if mode == "off":
        offset = 0.0
    else:
        # Load BVH world positions for hip + ankle joints (already FK'd
        # by export_r15's BVH parser — same data the retarget consumed).
        export_r15.set_rig(parent_pipeline.RIG)
        anim = export_r15._load_anim_any(bvh_path)
        bind = export_r15._load_anim_any(parent_pipeline.BIND_BVH)
        names = list(anim["names"])
        # Soma joint names: hip = 'Hips', ankle joints = 'LeftFoot',
        # 'RightFoot' (BVH "Foot" is the ankle joint, with Toe descendants
        # beneath it). Falls back gracefully on other rigs that name
        # ankles differently.
        hip_name   = "Hips"
        l_ankle    = "LeftFoot"
        r_ankle    = "RightFoot"
        for nm in (hip_name, l_ankle, r_ankle):
            if nm not in names:
                print(f"[prompt_pipeline] _ground_y: BVH missing {nm}, "
                      f"skipping grounding")
                return 0.0

        wp = anim["world_pos"]   # (F, J, 3) cm, BVH space
        bp = bind["world_pos"]   # (1, J, 3) cm, bind frame
        hi = names.index(hip_name)
        li = names.index(l_ankle)
        ri = names.index(r_ankle)

        bind_chain_cm = float(bp[0, hi, 1] - min(bp[0, li, 1], bp[0, ri, 1]))
        if bind_chain_cm <= 1e-3:
            print(f"[prompt_pipeline] _ground_y: bind hip-to-ankle "
                  f"({bind_chain_cm:.2f} cm) is too small, skipping")
            return 0.0

        # Per-frame predicted R15 ankle world Y, taking the lower of the
        # two ankles (whichever is closer to the ground controls the
        # visible float).
        F = min(wp.shape[0], n)
        soma_hip = wp[:F, hi, 1]
        soma_lank = wp[:F, li, 1]
        soma_rank = wp[:F, ri, 1]
        soma_lower_ank = np.minimum(soma_lank, soma_rank)
        soma_chain_cm = soma_hip - soma_lower_ank   # (F,)
        bind_chain_studs = bind_chain_cm * export_r15.CM_TO_STUD
        soma_chain_studs = soma_chain_cm * export_r15.CM_TO_STUD
        # Target-rig leg chain (HRP→ankle) scaled proportionally to the
        # BVH's per-frame chain. For Rthro Rig (default 3.6693) this is
        # 3.67 * (soma/bind); for stock R15 (1.6) override it's smaller.
        # A crouch frame with soma_chain at 50% of bind drops the ankle
        # by 0.5 * target_hrp_to_ankle below rest, so the right value
        # here is critical for crouch grounding.
        r15_chain = soma_chain_studs * (target_hrp_to_ankle / bind_chain_studs)

        lt_y = np.asarray(pos_y[:F], dtype=float)
        target_rest_ankle_y = target_hrp_rest_y - target_hrp_to_ankle
        predicted_ankle_y = target_hrp_rest_y + lt_y - r15_chain
        floats = predicted_ankle_y - target_rest_ankle_y   # >0 = floats; <0 = penetrates

        if mode == "first":
            offset = float(floats[0])
        elif mode == "min":
            offset = float(floats.min())
        else:
            raise ValueError(f"unknown ground-y-mode: {mode}")

    offset -= extra_bias  # +bias raises character → less subtracted
    if abs(offset) < 1e-9:
        return 0.0
    target["posY"] = [v - offset for v in pos_y]
    return offset


def _run_kimodo_promptonly(
    clip_dir: Path,
    *,
    prompt: str,
    duration_s: float,
    model: str,
    seed: int | None,
    diffusion_steps: int,
    cfg_type: str,
    cfg_weight: list[float],
    extra_args: list[str] | None = None,
) -> Path:
    """Variant of run_kimodo.run_kimodo that does not require pre-built
    meta.json + constraints.json. Writes generated.bvh into clip_dir and
    returns its path."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    bin_path = run_kimodo.resolve_kimodo_gen()
    out_stem = clip_dir / "generated"

    cmd = [
        bin_path,
        prompt,
        "--model", model,
        "--duration", f"{duration_s}",
        "--output", str(out_stem),
        "--bvh",
        "--diffusion_steps", str(diffusion_steps),
        "--cfg_type", cfg_type,
    ]
    if cfg_weight:
        cmd += ["--cfg_weight", *(str(w) for w in cfg_weight)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if extra_args:
        cmd += list(extra_args)

    print(f"[prompt_pipeline] kimodo_gen (prompt={prompt!r}, duration={duration_s}s)")
    sys.stdout.flush()
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    subprocess.run(cmd, check=True, env=env)

    bvh_path = Path(f"{out_stem}.bvh")
    if not bvh_path.is_file():
        raise RuntimeError(f"Expected {bvh_path} after kimodo_gen but it is missing")
    return bvh_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", type=str, required=True,
                   help="Text prompt for Kimodo. Multiple prompts can be "
                        "joined with '.' per kimodo_gen's syntax.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory. Final rbxm at <out>/<name>/r15.rbxm.")
    p.add_argument("--name", type=str, default=None,
                   help="Clip name (default: derived from --prompt).")
    p.add_argument("--duration", type=float, default=3.0,
                   help="Generated motion duration in seconds. Default 3.0.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cfg-type", choices=["nocfg", "regular", "separated"],
                   default="regular",
                   help="Kimodo CFG mode. 'regular' (default) uses a single "
                        "cfg_weight on the text prompt; 'separated' is for "
                        "joint text+constraint guidance (use only if you "
                        "really know why).")
    p.add_argument("--cfg-weight", type=float, default=5.0,
                   help="CFG weight on the prompt for cfg_type=regular. "
                        "Higher = stricter prompt adherence. Default 5.0.")
    p.add_argument("--cfg-text-weight", type=float, default=2.0,
                   help="Text weight for cfg_type=separated. Ignored for "
                        "regular/nocfg.")
    p.add_argument("--cfg-constraint-weight", type=float, default=2.0,
                   help="Constraint weight for cfg_type=separated. Ignored "
                        "for regular/nocfg. (No constraints are passed in "
                        "this pipeline; included only for completeness.)")
    p.add_argument("--root-motion", dest="root_motion", action="store_true",
                   help="Keep HumanoidRootPart curves in the rbxm (default: "
                        "fold root motion into LowerTorso so HRP stays at "
                        "rest, like the parent pipeline).")
    p.set_defaults(root_motion=False)
    p.add_argument("--inertial-blend", type=int, default=0,
                   help="Fake inertial blend over the first N frames using "
                        "the clip's last frame as the 'previous' pose. Use "
                        "for prompts that should loop. Default 0 (off) "
                        "since prompt motions aren't periodic by default.")
    p.add_argument("--looped", action="store_true",
                   help="Mark the output as a looping clip. Required for "
                        "--inertial-blend to take effect (mirrors the "
                        "parent pipeline's behavior).")
    p.add_argument("--ground-y-mode", choices=["first", "min", "off"],
                   default="first",
                   help="Shift the root-translation Y curve so the rest "
                        "pose sits on the floor. 'first' (default) zeroes "
                        "frame 0; 'min' zeroes the lowest frame (no "
                        "ground penetration); 'off' disables. Without "
                        "constraints anchoring the feet, Kimodo's BVH "
                        "places the hip at an arbitrary Y and the "
                        "character would otherwise float by ~0.1-0.2 studs.")
    p.add_argument("--target-hrp-rest-y", type=float,
                   default=_DEFAULT_TARGET_HRP_REST_Y,
                   help="Target rig's HumanoidRootPart world Y at rest. "
                        f"Default {_DEFAULT_TARGET_HRP_REST_Y} matches "
                        "the Rthro Rig in workspace (measured directly). "
                        "For stock R15 pass 2.0.")
    p.add_argument("--target-hrp-to-ankle", type=float,
                   default=_DEFAULT_TARGET_HRP_TO_ANKLE,
                   help="Target rig's HRP-to-ankle Y distance at rest. "
                        f"Default {_DEFAULT_TARGET_HRP_TO_ANKLE} matches "
                        "the Rthro Rig (HRP=4.1197 minus ankle "
                        "attachment Y=0.4505). For stock R15 pass 1.6. "
                        "Drives the proportional leg-chain model in "
                        "grounding AND the geometric HRP_SCALE auto-"
                        "derivation, so getting this number right is "
                        "the single most important knob.")
    p.add_argument("--ground-y-bias", type=float, default=0.0,
                   help="Manual offset added in studs after the "
                        "proportional grounding model (positive raises "
                        "the rig). Use to nudge if the model under- or "
                        "over-shoots on a specific clip.")
    p.add_argument("--hrp-scale", type=float, default=None,
                   help="Override the BVH-hip-XZ → target-rig-hip-XZ "
                        "scale. Default: pure geometric leg-length ratio "
                        "= target_hrp_to_ankle / soma_bind_hip_to_ankle "
                        "(loaded from pipeline.BIND_BVH). For Rthro "
                        "(3.6693 / 2.643) ≈ 1.388. The no-slide "
                        "constraint is geometric: when a foot is "
                        "planted, hip XZ velocity = leg_length × "
                        "angular_velocity; same Motor6D rotations on "
                        "both rigs ⇒ scale = leg ratio.")
    p.add_argument("--roblox-cli", type=str, default=None)
    p.add_argument("--skip", action="append", default=[],
                   choices=["kimodo", "retarget", "rbxm"],
                   help="Skip a stage (re-using existing output). Repeatable.")
    args = p.parse_args(argv)

    name = args.name or _slug_from_prompt(args.prompt)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = out_dir / name
    clip_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage A: Kimodo ----
    bvh_path = clip_dir / "generated.bvh"
    if "kimodo" in args.skip and bvh_path.is_file():
        print(f"[prompt_pipeline] skip stage A, using {bvh_path}")
    else:
        if args.cfg_type == "regular":
            cfg_weight = [args.cfg_weight]
        elif args.cfg_type == "separated":
            cfg_weight = [args.cfg_text_weight, args.cfg_constraint_weight]
        else:  # nocfg
            cfg_weight = []
        bvh_path = _run_kimodo_promptonly(
            clip_dir,
            prompt=args.prompt,
            duration_s=args.duration,
            model=args.model,
            seed=args.seed,
            diffusion_steps=args.diffusion_steps,
            cfg_type=args.cfg_type,
            cfg_weight=cfg_weight,
        )
        # Stash a meta.json for parity with the asset-id pipeline (helps
        # downstream tooling / debugging).
        (clip_dir / "meta.json").write_text(json.dumps({
            "source": "prompt",
            "prompt": args.prompt,
            "duration_s": float(args.duration),
            "kimodo_model": args.model,
            "kimodo_seed": args.seed,
            "kimodo_diffusion_steps": args.diffusion_steps,
            "kimodo_cfg_type": args.cfg_type,
            "looped": bool(args.looped),
        }, indent=2))

    # ---- Stage B: BVH → R15 JSON ----
    r15_json = clip_dir / "r15.json"
    if "retarget" in args.skip and r15_json.is_file():
        print(f"[prompt_pipeline] skip stage B, using {r15_json}")
    else:
        # Auto-derive hrp_scale from target rig if user didn't override.
        # No-slide condition (planted-foot kinematics):
        #     hip_velocity_world = leg_length × leg_angular_velocity
        # Motor6D rotations are identical on any target rig, so the
        # scale that preserves "feet stay planted" is purely geometric:
        #     hrp_scale = target_HRP_to_ankle / soma_bind_HRP_to_ankle
        # For Rthro Rig (default 3.6693) and current bind (2.643 stud)
        # ⇒ 1.388. For stock R15 override (1.6) ⇒ 0.605. We override
        # the historical 0.72 baseline (which was empirically tuned and
        # over-translated stock R15 by ~19%) in favor of geometry.
        if args.hrp_scale is not None:
            effective_hrp_scale = float(args.hrp_scale)
        else:
            soma_bind_chain_studs = _soma_bind_hip_to_ankle_studs()
            effective_hrp_scale = (
                args.target_hrp_to_ankle / soma_bind_chain_studs
            )
        info = parent_pipeline._retarget_bvh_to_r15_json(
            bvh_path, r15_json,
            root_motion=args.root_motion,
            # Prompt motion has no source cycle, so no trim. Looping is
            # opt-in via --looped; only meaningful in combination with
            # --inertial-blend.
            source_n_frames=0,
            loop_passes=1,
            looped=bool(args.looped),
            inertial_blend_frames=args.inertial_blend,
            hrp_scale=effective_hrp_scale,
        )
        print(f"[prompt_pipeline] retarget OK (hrp_scale={effective_hrp_scale:.3f}): {info}")
        # Ground the rest pose. Done as a post-pass on the dumped JSON
        # to avoid threading a new arg through the parent pipeline's
        # retarget helper.
        result = json.loads(r15_json.read_text())
        offset = _ground_y(
            result, bvh_path, args.ground_y_mode,
            target_hrp_rest_y=args.target_hrp_rest_y,
            target_hrp_to_ankle=args.target_hrp_to_ankle,
            extra_bias=args.ground_y_bias,
        )
        if offset != 0.0:
            r15_json.write_text(json.dumps(result, separators=(",", ":")))
            print(f"[prompt_pipeline] grounded Y by {offset:+.4f} studs "
                  f"(mode={args.ground_y_mode}, "
                  f"target HRP={args.target_hrp_rest_y:.2f}/"
                  f"chain={args.target_hrp_to_ankle:.2f})")

    # ---- Stage C: rbxm ----
    rbxm_path = clip_dir / "r15.rbxm"
    if "rbxm" in args.skip and rbxm_path.is_file():
        print(f"[prompt_pipeline] skip stage C, using {rbxm_path}")
    else:
        rbxm_path = parent_pipeline._build_rbxm(out_dir, name, args.roblox_cli)

    print(json.dumps({
        "name": name,
        "prompt": args.prompt,
        "duration_s": args.duration,
        "rbxm": str(rbxm_path),
    }, indent=2))
    return 0


def _slug_from_prompt(prompt: str, max_len: int = 40) -> str:
    """Cheap kebab-case slug for default --name."""
    cleaned = "".join(c.lower() if c.isalnum() else "-" for c in prompt)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")[:max_len].strip("-")
    return cleaned or "prompt"


if __name__ == "__main__":
    sys.exit(main())
