#!/usr/bin/env python3
"""Merge all per-clip r15.rbxm files into a single .rbxm for Studio import.

Usage:
    python3 python/merge_emotes.py --input work/emotes --output work/emotes/all_emotes.rbxm
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(HERE))
import build_rbxm  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=REPO_ROOT / "work" / "emotes",
                   help="Directory tree to scan for r15.rbxm files")
    p.add_argument("--output", type=Path, default=None,
                   help="Output .rbxm path (default: <input>/all_emotes.rbxm)")
    p.add_argument("--roblox-cli", type=str, default=None,
                   help="Path to roblox-cli")
    p.add_argument("--limit", type=int, default=None,
                   help="Max clips to include")
    args = p.parse_args(argv)

    input_dir = args.input.resolve()
    output_path = (args.output or (input_dir / "all_emotes.rbxm")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cli = build_rbxm.resolve_cli(args.roblox_cli)
    lua_script = REPO_ROOT / "lua" / "merge_rbxm.lua"

    globals_args = [
        f"MERGE_INPUT_DIR={input_dir}",
        f"MERGE_OUTPUT_PATH={output_path}",
    ]
    if args.limit:
        globals_args.append(f"MERGE_LIMIT={args.limit}")

    cmd = [
        cli, "run",
        "--run", str(lua_script),
        "--fs.readwrite", str(REPO_ROOT),
        "--fs.readwrite", str(input_dir),
        "--fs.readwrite", str(output_path.parent),
        "--load.asRobloxScript",
    ]
    for g in globals_args:
        cmd += ["--lua.globals", g]

    print(f"[merge_emotes] cli: {cli}")
    print(f"[merge_emotes] input: {input_dir}")
    print(f"[merge_emotes] output: {output_path}")
    print(f"[merge_emotes] cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"[merge_emotes] FAILED (exit {result.returncode})")
        return 1

    if output_path.is_file():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[merge_emotes] wrote {output_path} ({size_mb:.2f} MB)")
    else:
        print("[merge_emotes] WARNING: output file not found after run")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
