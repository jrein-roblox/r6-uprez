#!/usr/bin/env python3
"""Stage C deliverable builder. Invokes `robloxdev-cli run` on
`lua/build_rbxm.lua` to produce CurveAnimation `.rbxm` files for every
clip in `data/Kimodo_Constraints/` — no Studio in the loop.

Pre-requisites:
  - `r15.json` already produced for each clip (run `python/batch_retarget.py`).
  - `robloxdev-cli` available somewhere; located via:
      1. $ROBLOX_CLI env var
      2. `which robloxdev-cli` / `which roblox-cli` on PATH
      3. The user's game-engine build output (default fallback).

Output layout (defaults: all three ON):
  data/Kimodo_Constraints/<Cat>/<Clip>/r15.rbxm    (per-clip)
  data/Kimodo_Animations/<Category>.rbxm           (per-category)
  data/Kimodo_Animations/all.rbxm                  (whole corpus)

Stdlib only. Just builds the CLI invocation and shells out.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# Default fallback CLI binaries to probe (macOS-only; the user's machine
# was confirmed to have one of these). The "studio" build is preferred
# because it has the same instance type catalog as Studio (covers
# CurveAnimation, RotationCurve, etc.). The "client" robloxdev-cli is the
# downloaded SDK binary used by anim-simularity.
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roblox-cli", type=str, default=None,
                   help="Path to robloxdev-cli. Default: $ROBLOX_CLI, then "
                        "PATH lookup, then known fallback locations.")
    p.add_argument("--lua-script", type=Path,
                   default=REPO_ROOT / "lua" / "build_rbxm.lua",
                   help="Lua entry point (default: lua/build_rbxm.lua).")
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                   help="Repo root passed to the Lua script (used for "
                        "constraints + output paths). Default: this repo.")
    p.add_argument("--per-clip", dest="per_clip",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Emit one rbxm per clip alongside r15.json (default ON).")
    p.add_argument("--per-category", dest="per_category",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Emit one rbxm per category under data/Kimodo_Animations/ "
                        "(default ON).")
    p.add_argument("--corpus", dest="corpus",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Emit a single all.rbxm with every clip (default ON).")
    p.add_argument("--in", dest="in_rel", type=str, default="data/Kimodo_Constraints",
                   help="Constraints input dir, relative to --repo-root "
                        "(default: data/Kimodo_Constraints). For procedural "
                        "clips: data/Kimodo_Procedural.")
    p.add_argument("--out-name", type=str, default="Kimodo_Animations",
                   help="Per-category + corpus rbxm output dir name under "
                        "data/ (default: Kimodo_Animations). Use a different "
                        "name to keep procedural outputs separate.")
    p.add_argument("--pattern", type=str, default=None,
                   help="Glob filter on clip relative path (e.g. 'Crouch/*' "
                        "or '*Loop_F*').")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N clips after filtering.")
    p.add_argument("--rig-data", type=Path, default=None,
                   help="Override the embedded rig data rbxm path. Default: "
                        "data/RigData.rbxm. For R15-plus retargets pass "
                        "data/r15plus-rigdata-hrd.rbxm.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved CLI invocation and exit.")
    args = p.parse_args(argv)

    cli = resolve_cli(args.roblox_cli)

    cmd: list[str] = [
        cli, "run",
        "--run", str(args.lua_script.resolve()),
        "--fs.readwrite", str(args.repo_root.resolve()),
        "--fs.readwrite", str(REPO_ROOT.resolve()),
        "--load.asRobloxScript",
    ]

    # --lua.globals takes "name=value" pairs and may be repeated.
    def add_global(name: str, value):
        cmd.extend(["--lua.globals", f"{name}={value}"])

    add_global("BUILD_RBXM_REPO_ROOT", str(args.repo_root.resolve()))
    rig_data_path = args.rig_data if args.rig_data else (REPO_ROOT / "data" / "RigData.rbxm")
    add_global("BUILD_RBXM_RIG_DATA_PATH", str(rig_data_path.resolve()))
    add_global("BUILD_RBXM_PER_CLIP", "1" if args.per_clip else "0")
    add_global("BUILD_RBXM_PER_CATEGORY", "1" if args.per_category else "0")
    add_global("BUILD_RBXM_CORPUS", "1" if args.corpus else "0")
    add_global("BUILD_RBXM_IN_DIR", args.in_rel)
    add_global("BUILD_RBXM_OUT_NAME", args.out_name)
    if args.pattern:
        add_global("BUILD_RBXM_PATTERN", args.pattern)
    if args.limit is not None:
        add_global("BUILD_RBXM_LIMIT", str(int(args.limit)))

    print("[build_rbxm] using cli:", cli)
    print("[build_rbxm] command:")
    print("  " + " \\\n  ".join(_shell_quote(c) for c in cmd))
    if args.dry_run:
        return 0

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[build_rbxm] cli exited {e.returncode}", file=sys.stderr)
        return e.returncode
    return 0


def _shell_quote(s: str) -> str:
    if not s or any(c in s for c in " \t\"'\\$"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


if __name__ == "__main__":
    sys.exit(main())
