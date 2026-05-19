#!/usr/bin/env python3
"""Stage B of the FBX→Kimodo→R15 pipeline: invoke `kimodo_gen` on a clip
directory produced by `extract_constraints.py` (or `batch_extract.py`).

Reads <clip_dir>/meta.json for `duration_s`, calls `kimodo_gen` with the
constraints + prompt, writes <clip_dir>/<out_name>.bvh + .npz, then
updates meta.json with the run parameters.

Locating the kimodo_gen executable (in priority order):
  1. $KIMODO_GEN env var, if set, is used as-is.
  2. `kimodo_gen` on PATH.
  3. /Users/jrein/git/nv-tlabs/kimodo/.venv/bin/kimodo_gen (default
     install location used by this checkout).

This script intentionally only depends on the stdlib so it can be run
without `uv` / a virtualenv — the heavy work happens in the Kimodo venv
itself, invoked as a subprocess.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = "Kimodo-SOMA-RP"  # family alias; resolves to whichever
                                  # version is cached locally (e.g. v1.1)
DEFAULT_KIMODO_BIN = "/Users/jrein/git/nv-tlabs/kimodo/.venv/bin/kimodo_gen"


def resolve_kimodo_gen() -> str:
    env = os.environ.get("KIMODO_GEN")
    if env:
        return env
    found = shutil.which("kimodo_gen")
    if found:
        return found
    if Path(DEFAULT_KIMODO_BIN).is_file():
        return DEFAULT_KIMODO_BIN
    raise RuntimeError(
        "Could not locate kimodo_gen. Set $KIMODO_GEN, add it to PATH, "
        f"or install Kimodo at {DEFAULT_KIMODO_BIN}."
    )


def run_kimodo(
    clip_dir: Path,
    *,
    prompt: str,
    model: str = DEFAULT_MODEL,
    seed: int | None = None,
    diffusion_steps: int = 100,
    out_name: str = "generated",
    constraints_name: str = "constraints.json",
    extra_args: list[str] | None = None,
) -> dict:
    clip_dir = Path(clip_dir)
    meta_path = clip_dir / "meta.json"
    constraints_path = clip_dir / constraints_name
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if not constraints_path.is_file():
        raise FileNotFoundError(constraints_path)

    with meta_path.open() as f:
        meta = json.load(f)
    duration_s = float(meta["duration_s"])

    out_stem = clip_dir / out_name
    bin_path = resolve_kimodo_gen()

    cmd = [
        bin_path,
        prompt,
        "--model", model,
        "--duration", f"{duration_s}",
        "--constraints", str(constraints_path),
        "--output", str(out_stem),
        "--bvh",
        "--diffusion_steps", str(diffusion_steps),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if extra_args:
        cmd += list(extra_args)

    print(f"[run_kimodo] {' '.join(cmd[:1])} (prompt={prompt!r}) -> {out_stem}.bvh")
    sys.stdout.flush()
    # HF cache holds the LLaMA-3 / Kimodo weights; the sandbox blocks
    # huggingface.co so the metadata-refresh hit fails. Force offline mode
    # so the loader uses the cached snapshots without phoning home.
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    subprocess.run(cmd, check=True, env=env)

    bvh_path = Path(f"{out_stem}.bvh")
    npz_path = Path(f"{out_stem}.npz")
    if not bvh_path.is_file():
        raise RuntimeError(f"Expected {bvh_path} after kimodo_gen but it is missing")

    meta.update({
        "prompt": prompt,
        "kimodo_model": model,
        "kimodo_seed": seed,
        "kimodo_diffusion_steps": diffusion_steps,
        "kimodo_bvh": str(bvh_path),
        "kimodo_npz": str(npz_path) if npz_path.is_file() else None,
    })
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "clip_dir": str(clip_dir),
        "bvh": str(bvh_path),
        "npz": str(npz_path) if npz_path.is_file() else None,
        "duration_s": duration_s,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip-dir", type=Path, required=True,
                   help="Directory produced by extract_constraints.py "
                        "(must contain constraints.json + meta.json).")
    p.add_argument("--prompt", type=str, required=True,
                   help="Text prompt for Kimodo.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--out-name", type=str, default="generated",
                   help="Output stem inside --clip-dir (default: 'generated').")
    p.add_argument("--constraints-name", type=str, default="constraints.json",
                   help="Constraints JSON filename within --clip-dir. Use to "
                        "select between variants emitted by batch_constraints.py "
                        "(e.g. 'root_feet_hands.json').")
    p.add_argument("kimodo_args", nargs=argparse.REMAINDER,
                   help="Pass-through args after `--`, forwarded to kimodo_gen.")
    args = p.parse_args(argv)

    extra = list(args.kimodo_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    result = run_kimodo(
        args.clip_dir,
        prompt=args.prompt,
        model=args.model,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        out_name=args.out_name,
        constraints_name=args.constraints_name,
        extra_args=extra,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
