#!/usr/bin/env python3
"""Stage 1 driver: invoke `lua/extract_pose.lua` via roblox-cli to download
a Roblox animation asset and emit a per-frame world-CFrame pose.json.

The Lua side does the actual work (asset download, character spawn,
Animator scrubbing, sampling). This Python wrapper only:
  - resolves the roblox-cli binary (env / PATH / known fallback paths)
  - writes <work>/_extract_config.json with the params the Lua reads
  - subprocesses roblox-cli with --fs.readwrite <work>
  - prints a summary of the resulting pose.json

Stdlib-only so it can run without uv / a virtualenv.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
LUA_SCRIPT = REPO_ROOT / "lua" / "extract_pose.lua"

# Same fallback list build_rbxm.py uses; the studio build is preferred
# because it has the full instance-type catalog (CurveAnimation, etc.).
DEFAULT_CLI_FALLBACKS = [
    "/Users/jrein/git/roblox/game-engine/build/ninja/studio/xcode-16.2/arm64/noopt/Client/CLI/app/roblox-cli",
    "/Users/jrein/git/roblox/game-engine/build/ninja/client/xcode-16.2/arm64/release/_deps/darwin-arm64.robloxdev-cli-src/robloxdev-cli",
    "/Users/jrein/git/roblox/game-engine/build/ninja/client/xcode-16.2/arm64/optimized/_deps/darwin-arm64.robloxdev-cli-src/robloxdev-cli",
    "/Users/jrein/git/roblox/game-engine/build/ninja/client/xcode-16.2/arm64/noopt/_deps/darwin-arm64.robloxdev-cli-src/robloxdev-cli",
]


def resolve_cli(explicit: str | None) -> str:
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"--roblox-cli not a file: {explicit}")
        return explicit
    env = os.environ.get("ROBLOX_CLI")
    if env and Path(env).is_file():
        return env
    for name in ("robloxdev-cli", "roblox-cli"):
        found = shutil.which(name)
        if found:
            return found
    for cand in DEFAULT_CLI_FALLBACKS:
        if Path(cand).is_file():
            return cand
    raise FileNotFoundError(
        "Could not locate robloxdev-cli / roblox-cli. Set $ROBLOX_CLI or pass "
        "--roblox-cli <path>."
    )


def extract_pose(
    *,
    asset_id: int,
    out_dir: Path,
    sample_fps: int = 30,
    min_duration_s: float = 0.0,
    loop_passes: int = 1,
    roblox_cli: str | None = None,
    work_root: Path | None = None,
) -> dict:
    """Run the Lua extractor and return the parsed pose.json. Rig type
    (R6 vs R15) is auto-detected by the Lua side from the clip's bone names
    and reported back in pose.json."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # The Lua script needs --fs.readwrite to cover BOTH the config file and
    # the output. Use a single root that's the parent of both — by default
    # the repo's work/ dir.
    if work_root is None:
        work_root = REPO_ROOT / "work"
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    pose_path = out_dir / "pose.json"
    config_path = work_root / "_extract_config.json"
    config = {
        "asset_id": int(asset_id),
        "out_path": str(pose_path),
        "sample_fps": int(sample_fps),
        "min_duration_s": float(min_duration_s),
        "loop_passes": int(loop_passes),
    }
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    cli = resolve_cli(roblox_cli)
    # The fs.readwrite path needs to be a common ancestor of config_path
    # and pose_path. Compute it so callers can put `out_dir` outside `work/`.
    try:
        common = Path(os.path.commonpath([str(work_root), str(out_dir)]))
    except ValueError:
        # Different drives on Windows — fall back to user home.
        common = Path.home()

    cmd = [
        cli, "run",
        "--run", str(LUA_SCRIPT.resolve()),
        "--fs.readwrite", str(common),
        "--load.asRobloxScript",
    ]
    print(f"[extract_pose] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    if not pose_path.is_file():
        raise RuntimeError(f"Lua did not produce {pose_path}")
    with pose_path.open() as f:
        pose = json.load(f)

    print(
        f"[extract_pose] {pose['rig_type']} clip "
        f"{pose.get('clip_class','?')} duration={pose['duration_s']:.2f}s "
        f"frames={pose['n_frames']} fps={pose['fps']}"
    )
    return pose


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-id", type=int, required=True,
                   help="Roblox asset id (Animation, KeyframeSequence, or "
                        "CurveAnimation). Wrapper Animations are followed.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (pose.json written here).")
    p.add_argument("--fps", type=int, default=30,
                   help="Sample rate for pose extraction (default: 30).")
    p.add_argument("--min-duration", type=float, default=0.0,
                   help="Minimum output duration in seconds (default: 0 = "
                        "no looping). If set and the source clip is shorter, "
                        "it loops by wrapping TimePosition.")
    p.add_argument("--loop-passes", type=int, default=1,
                   help="Sample N copies of the cycle for clips marked "
                        "AnimationClip.Loop=true. Pipeline trims to the "
                        "middle cycle on export, giving Kimodo periodic "
                        "context on both sides of the kept cycle so the "
                        "loop seam is much smoother. Default 1 (off).")
    p.add_argument("--roblox-cli", type=str, default=None,
                   help="Override roblox-cli binary path.")
    p.add_argument("--work-root", type=Path, default=None,
                   help="Directory for the Lua-side config file. Default: "
                        "<repo>/work.")
    args = p.parse_args(argv)

    pose = extract_pose(
        asset_id=args.asset_id,
        out_dir=args.out,
        sample_fps=args.fps,
        min_duration_s=args.min_duration,
        loop_passes=args.loop_passes,
        roblox_cli=args.roblox_cli,
        work_root=args.work_root,
    )
    print(json.dumps({
        "pose_path": str(Path(args.out) / "pose.json"),
        "n_frames": pose["n_frames"],
        "duration_s": pose["duration_s"],
        "rig_type": pose["rig_type"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
