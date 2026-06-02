#!/usr/bin/env python3
"""Batch-generate UGC emotes: 20 prompts × 20 seeds each, parallelized.

Usage:
    python3 python/batch_emotes.py --out work/emotes --jobs 4
    python3 python/batch_emotes.py --out work/emotes --jobs 4 --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PIPELINE_SCRIPT = HERE / "prompt_pipeline.py"

SEEDS_PER_PROMPT = 20


@dataclass
class EmoteSpec:
    name: str
    prompt: str
    duration: str
    loop: bool = False
    loop_offset: float = 3.0
    inertial_blend_seconds: float = 0.4
    loop_cfg_constraint_weight: float = 2.0
    root_motion: bool = False
    notes: str = ""


EMOTES: list[EmoteSpec] = [
    # --- DANCES (looping) ---
    EmoteSpec(
        name="hip-hop-groove",
        prompt="A person is doing an energetic hip hop dance with bouncy movements and arm pumps",
        duration="6.0",
        loop=True,
        loop_offset=2.0,
        notes="Classic hip hop idle dance",
    ),
    EmoteSpec(
        name="smooth-shuffle",
        prompt="A person is doing a smooth shuffling dance with quick footwork and flowing arm movements",
        duration="6.0",
        loop=True,
        loop_offset=2.0,
        notes="Trendy shuffle dance",
    ),
    EmoteSpec(
        name="party-dance",
        prompt="A person is dancing happily at a party with energetic full body movements and head bobs",
        duration="7.0",
        loop=True,
        loop_offset=2.5,
        notes="Generic party dance",
    ),
    EmoteSpec(
        name="robot-dance",
        prompt="A person is doing a robotic dance with stiff mechanical movements and sharp isolated pops",
        duration="6.0",
        loop=True,
        loop_offset=2.0,
        notes="Robot/popping style",
    ),
    EmoteSpec(
        name="disco-fever",
        prompt="A person is dancing disco style with pointing arms and hip swaying side to side",
        duration="6.0",
        loop=True,
        loop_offset=2.0,
        notes="Retro disco",
    ),
    EmoteSpec(
        name="tiktok-dance",
        prompt="A person is doing a trendy dance with sharp arm choreography and body rolls",
        duration="5.0",
        loop=True,
        loop_offset=1.5,
        notes="Modern viral dance style",
    ),
    EmoteSpec(
        name="salsa-dance",
        prompt="A person is dancing salsa with rhythmic hip movements and smooth footwork in place",
        duration="6.0",
        loop=True,
        loop_offset=2.0,
        notes="Latin dance",
    ),
    # --- CELEBRATIONS (non-looping) ---
    EmoteSpec(
        name="victory-jump",
        prompt="A person jumps up excitedly with both fists raised in celebration then lands",
        duration="4.0",
        loop=False,
        notes="Victory celebration",
    ),
    EmoteSpec(
        name="fist-pump",
        prompt="A person pumps their fist in the air triumphantly with an energetic body motion",
        duration="3.0",
        loop=False,
        notes="Quick victory gesture",
    ),
    EmoteSpec(
        name="happy-clap",
        prompt="A person claps their hands together enthusiastically while bouncing with excitement",
        duration="4.0",
        loop=False,
        notes="Applause celebration",
    ),
    # --- GESTURES (non-looping) ---
    EmoteSpec(
        name="friendly-wave",
        prompt="A person waves hello enthusiastically with their right hand raised high",
        duration="3.5",
        loop=False,
        notes="Greeting wave",
    ),
    EmoteSpec(
        name="cool-shrug",
        prompt="A person shrugs their shoulders with both palms up in a casual whatever gesture",
        duration="3.0",
        loop=False,
        notes="Shrug emote",
    ),
    EmoteSpec(
        name="flex-pose",
        prompt="A person flexes both arms showing off their muscles with a confident stance",
        duration="4.0",
        loop=False,
        notes="Muscle flex",
    ),
    EmoteSpec(
        name="dramatic-bow",
        prompt="A person takes a deep theatrical bow with one arm sweeping forward gracefully",
        duration="4.0",
        loop=False,
        notes="Fancy bow",
    ),
    # --- IDLE/MOOD (looping) ---
    EmoteSpec(
        name="confident-idle",
        prompt="A person stands confidently with arms crossed occasionally shifting weight between feet",
        duration="6.0",
        loop=True,
        loop_offset=2.0,
        notes="Confident standing pose with subtle movement",
    ),
    EmoteSpec(
        name="impatient-wait",
        prompt="A person is standing impatiently tapping their foot and looking around restlessly",
        duration="6.0",
        loop=True,
        loop_offset=2.0,
        notes="Impatient idle",
    ),
    # --- ACTION (non-looping) ---
    EmoteSpec(
        name="martial-arts-kick",
        prompt="A person performs a powerful spinning kick with their right leg extended",
        duration="3.5",
        loop=False,
        notes="Combat kick",
    ),
    EmoteSpec(
        name="backflip",
        prompt="A person does a standing backflip and lands on both feet",
        duration="3.0",
        loop=False,
        notes="Acrobatic backflip",
    ),
    # --- MULTI-PROMPT (non-looping) ---
    EmoteSpec(
        name="dab-hit",
        prompt="A person quickly dabs with their head into their elbow and one arm extended",
        duration="3.0",
        loop=False,
        notes="Dab gesture",
    ),
    EmoteSpec(
        name="air-guitar",
        prompt="A person is playing air guitar with exaggerated strumming and head banging",
        duration="6.0",
        loop=True,
        loop_offset=2.0,
        notes="Rock out emote",
    ),
]


def _build_command(spec: EmoteSpec, out_dir: Path, seed: int, idx: int) -> list[str]:
    """Build the uv run command for one generation."""
    name = f"{spec.name}_v{idx:02d}"
    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        "--prompt", spec.prompt,
        "--out", str(out_dir),
        "--name", name,
        "--duration", spec.duration,
        "--seed", str(seed),
    ]
    if spec.loop:
        cmd += [
            "--loop",
            "--loop-offset", str(spec.loop_offset),
            "--inertial-blend-seconds", str(spec.inertial_blend_seconds),
            "--loop-cfg-constraint-weight", str(spec.loop_cfg_constraint_weight),
        ]
    if spec.root_motion:
        cmd.append("--root-motion")
    return cmd


def _run_one(cmd: list[str], name: str) -> dict:
    """Run a single pipeline invocation. Returns status dict."""
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(REPO_ROOT),
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            return {"name": name, "status": "ok", "elapsed": elapsed}
        else:
            return {
                "name": name,
                "status": "error",
                "returncode": result.returncode,
                "stderr": result.stderr[-500:] if result.stderr else "",
                "elapsed": elapsed,
            }
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "timeout", "elapsed": 600.0}
    except Exception as e:
        return {"name": name, "status": "exception", "error": str(e), "elapsed": 0.0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "work" / "emotes",
                   help="Output directory for all emote clips")
    p.add_argument("--jobs", "-j", type=int, default=4,
                   help="Max parallel generations")
    p.add_argument("--seeds-per-prompt", type=int, default=SEEDS_PER_PROMPT,
                   help="Number of seed variants per prompt")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--prompts", type=str, nargs="*", default=None,
                   help="Filter to specific prompt names (default: all)")
    p.add_argument("--start-seed", type=int, default=1000,
                   help="Base seed (increments per variant)")
    args = p.parse_args(argv)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = EMOTES
    if args.prompts:
        specs = [s for s in specs if s.name in args.prompts]
        if not specs:
            print(f"No matching prompts found. Available: {[s.name for s in EMOTES]}")
            return 1

    # Build all jobs
    jobs: list[tuple[list[str], str]] = []
    for spec in specs:
        for i in range(args.seeds_per_prompt):
            seed = args.start_seed + i
            name = f"{spec.name}_v{i:02d}"
            cmd = _build_command(spec, out_dir, seed, i)
            jobs.append((cmd, name))

    print(f"[batch_emotes] {len(specs)} prompts × {args.seeds_per_prompt} seeds = {len(jobs)} total generations")
    print(f"[batch_emotes] output: {out_dir}")
    print(f"[batch_emotes] parallelism: {args.jobs}")

    if args.dry_run:
        for cmd, name in jobs:
            print(f"  {name}: {' '.join(cmd)}")
        return 0

    # Save manifest
    manifest = {
        "prompts": [
            {
                "name": s.name,
                "prompt": s.prompt,
                "duration": s.duration,
                "loop": s.loop,
                "notes": s.notes,
            }
            for s in specs
        ],
        "seeds_per_prompt": args.seeds_per_prompt,
        "start_seed": args.start_seed,
        "total_jobs": len(jobs),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Run with process pool
    results: list[dict] = []
    completed = 0
    failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_one, cmd, name): name for cmd, name in jobs}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if res["status"] != "ok":
                failed += 1
                print(f"  [{completed}/{len(jobs)}] FAIL {res['name']}: {res.get('stderr', res.get('error', ''))[:200]}")
            else:
                print(f"  [{completed}/{len(jobs)}] OK {res['name']} ({res['elapsed']:.1f}s)")

    elapsed_total = time.time() - t_start
    print(f"\n[batch_emotes] Done: {completed - failed}/{len(jobs)} succeeded, "
          f"{failed} failed, {elapsed_total:.0f}s total")

    # Save results log
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
