#!/usr/bin/env python3
"""Batch-generate UGC emotes (wave 2): 20 new prompts × 20 seeds each.

Changes from v1:
  - Loop offset set to middle of duration for looping anims (better pivot pose)
  - Non-looping anims use --loop to constrain end pose back to origin (no pop)
  - New prompt set covering different motion styles

Usage:
    python3 python/batch_emotes_v2.py --out work/emotes_v2 --jobs 4
    python3 python/batch_emotes_v2.py --out work/emotes_v2 --jobs 4 --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
    loop_offset: float = 0.0
    inertial_blend_seconds: float = 0.4
    loop_cfg_constraint_weight: float = 2.0
    return_to_origin: bool = False
    notes: str = ""


EMOTES: list[EmoteSpec] = [
    # --- DANCES (looping, loop_offset at middle of duration) ---
    EmoteSpec(
        name="breakdance-toprock",
        prompt="A person is doing breakdance toprock moves with rhythmic stepping and arm swings",
        duration="7.0",
        loop=True,
        loop_offset=3.5,
        notes="Breakdance standing moves",
    ),
    EmoteSpec(
        name="pop-and-lock",
        prompt="A person is doing a popping and locking dance with sharp isolations and freezes",
        duration="6.0",
        loop=True,
        loop_offset=3.0,
        notes="Popping style dance",
    ),
    EmoteSpec(
        name="groove-bounce",
        prompt="A person is bouncing rhythmically with their whole body grooving to music",
        duration="5.0",
        loop=True,
        loop_offset=2.5,
        notes="Simple bounce groove",
    ),
    EmoteSpec(
        name="electric-slide",
        prompt="A person is doing a smooth side stepping dance with coordinated arm waves",
        duration="6.0",
        loop=True,
        loop_offset=3.0,
        notes="Line dance style",
    ),
    EmoteSpec(
        name="capoeira-ginga",
        prompt="A person is doing a capoeira ginga swaying back and forth with flowing arm guards",
        duration="6.0",
        loop=True,
        loop_offset=3.0,
        notes="Martial arts dance",
    ),
    EmoteSpec(
        name="head-bop-vibes",
        prompt="A person is nodding their head to music with subtle shoulder bounces and swaying",
        duration="5.0",
        loop=True,
        loop_offset=2.5,
        notes="Chill head nod idle",
    ),
    EmoteSpec(
        name="sway-dance",
        prompt="A person is swaying their hips and body side to side with raised arms dancing smoothly",
        duration="6.0",
        loop=True,
        loop_offset=3.0,
        notes="Smooth swaying dance",
    ),
    EmoteSpec(
        name="jump-dance",
        prompt="A person is jumping and dancing energetically with their arms pumping up and down",
        duration="5.0",
        loop=True,
        loop_offset=2.5,
        notes="High energy jumping dance",
    ),
    # --- CELEBRATIONS (non-looping, return to origin) ---
    EmoteSpec(
        name="mind-blown",
        prompt="A person raises both hands to the sides of their head then throws them outward in amazement",
        duration="4.0",
        return_to_origin=True,
        notes="Mind blown gesture",
    ),
    EmoteSpec(
        name="chest-bump",
        prompt="A person thumps their chest with their fist twice then points forward confidently",
        duration="3.5",
        return_to_origin=True,
        notes="Confident chest thump",
    ),
    EmoteSpec(
        name="happy-stomp",
        prompt="A person stamps their feet on the ground excitedly while pumping their fists with joy",
        duration="4.0",
        return_to_origin=True,
        notes="Excited stomping celebration",
    ),
    EmoteSpec(
        name="golf-clap",
        prompt="A person claps slowly and sarcastically with a subtle nod",
        duration="4.0",
        return_to_origin=True,
        notes="Sarcastic slow clap",
    ),
    # --- GESTURES (non-looping, return to origin) ---
    EmoteSpec(
        name="finger-wag",
        prompt="A person wags their index finger side to side disapprovingly with their other hand on hip",
        duration="3.5",
        return_to_origin=True,
        notes="Disapproval gesture",
    ),
    EmoteSpec(
        name="salute",
        prompt="A person snaps to attention and performs a sharp military salute then returns to rest",
        duration="3.5",
        return_to_origin=True,
        notes="Military salute",
    ),
    EmoteSpec(
        name="blow-kiss",
        prompt="A person blows a kiss with their right hand extending their arm outward gracefully",
        duration="3.5",
        return_to_origin=True,
        notes="Blow kiss emote",
    ),
    EmoteSpec(
        name="face-palm",
        prompt="A person slowly brings their hand up to cover their face and shakes their head",
        duration="4.0",
        return_to_origin=True,
        notes="Facepalm gesture",
    ),
    # --- ACTION (non-looping, return to origin) ---
    EmoteSpec(
        name="karate-combo",
        prompt="A person performs a quick punch punch kick martial arts combo then returns to a fighting stance",
        duration="4.0",
        return_to_origin=True,
        notes="Martial arts combo",
    ),
    EmoteSpec(
        name="spin-move",
        prompt="A person does a quick full body spin with arms extended then stops facing forward",
        duration="3.0",
        return_to_origin=True,
        notes="360 spin",
    ),
    EmoteSpec(
        name="ground-pound",
        prompt="A person jumps up slightly and slams both fists down toward the ground powerfully",
        duration="3.5",
        return_to_origin=True,
        notes="Ground slam attack",
    ),
    EmoteSpec(
        name="superhero-landing",
        prompt="A person drops into a crouching superhero landing pose with one fist on the ground then stands up",
        duration="4.5",
        return_to_origin=True,
        notes="Superhero landing",
    ),
]


def _build_command(spec: EmoteSpec, out_dir: Path, seed: int, idx: int) -> list[str]:
    """Build the command for one generation."""
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
    elif spec.return_to_origin:
        # Use the loop constraint mechanism to pin end pose == start pose,
        # but with loop_offset=0 (pin frame 0 at both endpoints).
        # This forces the animation to return to the starting pose at the end.
        cmd += [
            "--loop",
            "--loop-offset", "0.0",
            "--inertial-blend-seconds", str(spec.inertial_blend_seconds),
            "--loop-cfg-constraint-weight", str(spec.loop_cfg_constraint_weight),
        ]
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
    p.add_argument("--out", type=Path, default=REPO_ROOT / "work" / "emotes_v2",
                   help="Output directory for all emote clips")
    p.add_argument("--jobs", "-j", type=int, default=4,
                   help="Max parallel generations")
    p.add_argument("--seeds-per-prompt", type=int, default=SEEDS_PER_PROMPT,
                   help="Number of seed variants per prompt")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--prompts", type=str, nargs="*", default=None,
                   help="Filter to specific prompt names (default: all)")
    p.add_argument("--start-seed", type=int, default=2000,
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

    n_looping = sum(1 for s in specs if s.loop)
    n_return = sum(1 for s in specs if s.return_to_origin)
    print(f"[batch_emotes_v2] {len(specs)} prompts × {args.seeds_per_prompt} seeds = {len(jobs)} total")
    print(f"[batch_emotes_v2] {n_looping} looping (offset=mid), {n_return} return-to-origin")
    print(f"[batch_emotes_v2] output: {out_dir}")
    print(f"[batch_emotes_v2] parallelism: {args.jobs}")

    if args.dry_run:
        for cmd, name in jobs[:5]:
            print(f"  {name}: {' '.join(cmd)}")
        print(f"  ... ({len(jobs) - 5} more)")
        return 0

    # Save manifest
    manifest = {
        "prompts": [
            {
                "name": s.name,
                "prompt": s.prompt,
                "duration": s.duration,
                "loop": s.loop,
                "return_to_origin": s.return_to_origin,
                "loop_offset": s.loop_offset if s.loop else 0.0,
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
    print(f"\n[batch_emotes_v2] Done: {completed - failed}/{len(jobs)} succeeded, "
          f"{failed} failed, {elapsed_total:.0f}s total")

    # Save results log
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
