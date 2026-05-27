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
import pipeline as parent_pipeline  # noqa: E402
import run_kimodo  # noqa: E402

DEFAULT_MODEL = run_kimodo.DEFAULT_MODEL


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
        )
        print(f"[prompt_pipeline] retarget OK: {info}")

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
