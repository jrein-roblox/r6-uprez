#!/usr/bin/env python3
"""Batch-generate looping latin dance emotes (wave 3), with quality gating.

Changes from v2:
  - Prompt set is all latin dance styles, all looping.
  - Every generated clip is scored by `loop_quality.py` before it counts.
    Failures (idle-start opens, open loop seam, collapsed-to-static) are moved
    aside and a FRESH SEED is generated in their place, so each style ends up
    with `--target-keeps` clips that actually loop.

Calibration note: on the v2 batch, looping dance prompts passed this gate at
about 41% (17% were idle-starts). Budget roughly 2.5 generations per keeper.

Do NOT run other roblox-cli commands (ugc_validate.lua, merge_emotes.py,
dump_r15_rig.lua) while a batch is live. Stage C shells out to roblox-cli once
per clip, and with --jobs 4 that is already 4 concurrent invocations; adding
more caused transient `_build_rbxm` failures that look like generation errors
but reproduce clean when run standalone.

Must run under kimodo's venv python — `prompt_pipeline.py` imports torch and
kimodo, and this script launches it with `sys.executable`:

    KP=/Users/jrein/git/nv-tlabs/kimodo/.venv/bin/python

    # pilot: one style, 10 keepers
    $KP python/batch_emotes_v3.py --prompts salsa --target-keeps 10 --jobs 4

    # full run: all 10 styles, 30 keepers each
    $KP python/batch_emotes_v3.py --target-keeps 30 --jobs 4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PIPELINE_SCRIPT = HERE / "prompt_pipeline.py"

sys.path.insert(0, str(HERE))
import loop_quality  # noqa: E402
import ugc_validate  # noqa: E402


@dataclass
class DanceSpec:
    name: str
    prompt: str
    duration: float
    inertial_blend_seconds: float = 0.4
    loop_cfg_constraint_weight: float = 2.0
    notes: str = ""

    @property
    def loop_offset(self) -> float:
        """Loop pivot at the middle of the clip.

        `_build_loop_constraints` samples pass 1 at this time and pins that
        pose at both ends of pass 2. Mid-clip is past the idle ramp-in, which
        is what keeps the loop from opening on a rest pose.
        """
        return round(self.duration / 2.0, 2)


# Ten latin styles. All looping; durations stay well under the 10 s UGC
# emote ceiling (`UGCValidationMaxAnimationLength`).
DANCES: list[DanceSpec] = [
    DanceSpec(
        name="salsa",
        prompt="A person is dancing salsa in place with quick three step footwork, "
               "rhythmic hip rotation and relaxed raised arms",
        duration=6.0,
        notes="Cuban salsa basic step",
    ),
    DanceSpec(
        name="bachata",
        prompt="A person is dancing bachata with side to side steps, a soft hip "
               "pop on every fourth beat and loose swaying shoulders",
        duration=6.0,
        notes="Bachata basic with hip pop",
    ),
    DanceSpec(
        name="merengue",
        prompt="A person is dancing merengue marching in place with small steps "
               "and continuous side to side hip sway",
        duration=5.0,
        notes="Merengue march",
    ),
    DanceSpec(
        name="cha-cha",
        prompt="A person is dancing cha cha with a quick triple step shuffle, "
               "sharp weight changes and crisp arm movements",
        duration=5.0,
        notes="Cha-cha triple step",
    ),
    DanceSpec(
        name="samba",
        prompt="A person is dancing samba with a bouncing bounce step, fast "
               "footwork and rolling hips while arms swing loosely",
        duration=6.0,
        notes="Samba no pe bounce",
    ),
    DanceSpec(
        name="rumba",
        prompt="A person is dancing a slow sensual rumba with deliberate weight "
               "shifts, figure eight hip motion and flowing arm lines",
        duration=6.0,
        notes="Slow rumba",
    ),
    DanceSpec(
        name="mambo",
        prompt="A person is dancing mambo stepping forward and back on the "
               "strong beat with energetic hip accents and expressive arms",
        duration=5.0,
        notes="Mambo forward-back basic",
    ),
    DanceSpec(
        name="reggaeton",
        prompt="A person is dancing reggaeton with a heavy grounded dembow "
               "bounce, rolling hips and low relaxed shoulders",
        duration=5.0,
        notes="Reggaeton dembow groove",
    ),
    DanceSpec(
        name="cumbia",
        prompt="A person is dancing cumbia with a dragging side step, gentle "
               "shoulder sway and one arm raised holding the rhythm",
        duration=6.0,
        notes="Cumbia side step",
    ),
    DanceSpec(
        name="tango",
        prompt="A person is dancing an argentine tango solo with sharp dramatic "
               "steps, a straight proud torso and precise staccato pauses",
        duration=6.0,
        notes="Dramatic solo tango",
    ),
]

DANCES_BY_NAME = {d.name: d for d in DANCES}


def _build_command(spec: DanceSpec, out_dir: Path, seed: int, clip_name: str,
                   loop_offset_mode: str = "auto",
                   cfg_weight: float | None = None) -> list[str]:
    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        "--prompt", spec.prompt,
        "--out", str(out_dir),
        "--name", clip_name,
        "--duration", str(spec.duration),
        "--seed", str(seed),
        "--loop",
        "--loop-offset", str(spec.loop_offset),
        "--loop-offset-mode", loop_offset_mode,
        "--inertial-blend-seconds", str(spec.inertial_blend_seconds),
        "--loop-cfg-constraint-weight",
        str(cfg_weight if cfg_weight is not None else spec.loop_cfg_constraint_weight),
    ]
    return cmd


def _quarantine(stage_clip: Path, reject_dir: Path, clip_name: str) -> None:
    """Move a failed/errored clip out of staging into the reject tree."""
    if not stage_clip.exists():
        return
    reject_dir.mkdir(parents=True, exist_ok=True)
    dest = reject_dir / clip_name
    if dest.exists():
        shutil.rmtree(dest)
    try:
        shutil.move(str(stage_clip), str(dest))
    except Exception as e:
        print(f"  WARN could not quarantine {clip_name}: {e}", file=sys.stderr)


@dataclass
class Attempt:
    style: str
    clip: str
    seed: int
    status: str          # ok | gate-failed | error | timeout | exception
    elapsed: float
    reasons: list[str]
    metrics: dict | None = None


def _generate_and_score(
    spec: DanceSpec,
    out_dir: Path,
    reject_dir: Path,
    seed: int,
    clip_name: str,
    gate_kwargs: dict,
    timeout: int,
    rig: "ugc_validate.Rig | None" = None,
    ugc_margin: float = 0.95,
    loop_offset_mode: str = "auto",
    cfg_weight: float | None = None,
) -> Attempt:
    """Run one generation, then gate it. Rejects are moved to reject_dir.

    Two gates, both must pass:
      1. loop_quality  — does it actually loop (no idle-start, closed seam)?
      2. ugc_validate  — would Roblox's UGC emote validation accept it?
    """
    t0 = time.time()
    # Generate into a staging directory and only PROMOTE into out_dir on a
    # pass. Writing straight into out_dir and moving failures out afterwards
    # meant the output dir transiently held in-flight clips, and any clip whose
    # pipeline errored was left behind entirely -- indistinguishable from a
    # keeper to anyone browsing the directory.
    stage_dir = out_dir / "_staging"
    stage_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_command(spec, stage_dir, seed, clip_name, loop_offset_mode,
                         cfg_weight)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, cwd=str(REPO_ROOT))
    except subprocess.TimeoutExpired:
        _quarantine(stage_dir / clip_name, reject_dir, clip_name)
        return Attempt(spec.name, clip_name, seed, "timeout", time.time() - t0,
                       [f"pipeline exceeded {timeout}s"])
    except Exception as e:
        _quarantine(stage_dir / clip_name, reject_dir, clip_name)
        return Attempt(spec.name, clip_name, seed, "exception", time.time() - t0,
                       [str(e)])

    elapsed = time.time() - t0
    if res.returncode != 0:
        # Keep enough stderr to actually diagnose; 400 chars truncated the
        # traceback to the point of uselessness.
        tail = (res.stderr or "")[-2000:]
        _quarantine(stage_dir / clip_name, reject_dir, clip_name)
        return Attempt(spec.name, clip_name, seed, "error", elapsed,
                       [f"exit {res.returncode}: {tail}"])

    clip_dir = stage_dir / clip_name
    try:
        m = loop_quality.score_clip(clip_dir, **gate_kwargs)
    except Exception as e:
        _quarantine(clip_dir, reject_dir, clip_name)
        return Attempt(spec.name, clip_name, seed, "error", elapsed,
                       [f"loop scoring failed: {e}"])

    metrics = asdict(m)
    reasons = list(m.reasons)

    # UGC gate. Runs even when the loop gate already failed so results.json
    # records every reason a seed was dropped, not just the first.
    if rig is not None:
        try:
            u = ugc_validate.validate_clip(clip_dir, rig, margin=ugc_margin)
            metrics["ugc"] = {
                "verdict": u.verdict,
                "length": u.length,
                "min_y": u.min_y, "min_y_part": u.min_y_part,
                "max_bounds": u.max_bounds, "max_bounds_part": u.max_bounds_part,
                "max_speed": u.max_speed, "max_speed_part": u.max_speed_part,
                "max_joint_move": u.max_joint_move,
            }
            reasons.extend(f"UGC {r}" for r in u.reasons())
        except Exception as e:
            reasons.append(f"UGC validation failed: {e}")

    if not reasons:
        # Promote: this is the ONLY path that puts a clip in out_dir.
        dest = out_dir / clip_name
        if dest.exists():
            shutil.rmtree(dest)
        try:
            shutil.move(str(clip_dir), str(dest))
        except Exception as e:
            _quarantine(clip_dir, reject_dir, clip_name)
            return Attempt(spec.name, clip_name, seed, "error", elapsed,
                           [f"promote failed: {e}"], metrics)
        return Attempt(spec.name, clip_name, seed, "ok", elapsed, [], metrics)

    _quarantine(clip_dir, reject_dir, clip_name)
    return Attempt(spec.name, clip_name, seed, "gate-failed", elapsed,
                   reasons, metrics)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "work" / "latin_v3",
                   help="Output directory for passing clips")
    p.add_argument("--reject-out", type=Path, default=None,
                   help="Where rejects go (default: <out>_rejected, kept OUT of "
                        "the merge path)")
    p.add_argument("--jobs", "-j", type=int, default=4, help="Max parallel generations")
    p.add_argument("--target-keeps", type=int, default=30,
                   help="Passing clips wanted per style")
    p.add_argument("--max-attempts-per-style", type=int, default=None,
                   help="Attempt cap per style (default: 3x target-keeps)")
    p.add_argument("--prompts", type=str, nargs="*", default=None,
                   help=f"Subset of styles (default: all). Available: "
                        f"{', '.join(d.name for d in DANCES)}")
    p.add_argument("--start-seed", type=int, default=3000, help="First seed")
    p.add_argument("--timeout", type=int, default=900, help="Per-generation timeout (s)")
    p.add_argument("--dry-run", action="store_true")
    # Gate thresholds — forwarded to loop_quality.score_clip.
    p.add_argument("--min-start-vs-idle", type=float,
                   default=loop_quality.DEFAULT_MIN_START_VS_IDLE)
    p.add_argument("--max-seam-pos", type=float, default=loop_quality.DEFAULT_MAX_SEAM_POS)
    p.add_argument("--max-seam-vel", type=float, default=loop_quality.DEFAULT_MAX_SEAM_VEL)
    p.add_argument("--max-root-path", type=float, default=loop_quality.DEFAULT_MAX_ROOT_PATH)
    p.add_argument("--max-root-disp", type=float, default=loop_quality.DEFAULT_MAX_ROOT_DISP)
    p.add_argument("--min-rest-collapse", type=float, default=loop_quality.DEFAULT_MIN_REST_COLLAPSE)
    p.add_argument("--min-energy", type=float, default=loop_quality.DEFAULT_MIN_ENERGY)
    p.add_argument("--loop-cfg-constraint-weight", type=float, default=None,
                   help="Override the per-spec loop constraint weight. Raising it "
                        "to ~8 removes the mid-clip idle stall but roughly triples "
                        "root travel (the character wanders); 2-4 keeps it in place "
                        "but stalls more often.")
    p.add_argument("--loop-offset-mode", choices=["fixed", "auto"], default="auto",
                   help="'auto' picks the loop pivot from the pass-1 motion so the loop does not open on a stand (default). 'fixed' uses spec.loop_offset.")
    p.add_argument("--no-gate", action="store_true",
                   help="Generate without the loop gate (debugging)")
    # UGC gate — ports Roblox's own CurveAnimation checks; see ugc_validate.py.
    p.add_argument("--rig-json", type=Path, default=ugc_validate.DEFAULT_RIG_JSON,
                   help="R15 rest geometry for the UGC gate's forward kinematics")
    p.add_argument("--ugc-margin", type=float, default=0.95,
                   help="Gate at this fraction of each UGC limit (default 0.95)")
    p.add_argument("--no-ugc-gate", action="store_true",
                   help="Skip the UGC validation gate (debugging)")
    args = p.parse_args(argv)

    specs = DANCES
    if args.prompts:
        unknown = [n for n in args.prompts if n not in DANCES_BY_NAME]
        if unknown:
            print(f"Unknown style(s): {unknown}. Available: "
                  f"{[d.name for d in DANCES]}")
            return 1
        specs = [DANCES_BY_NAME[n] for n in args.prompts]

    out_dir = args.out.resolve()
    reject_dir = (args.reject_out or Path(str(out_dir) + "_rejected")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target = args.target_keeps
    cap = args.max_attempts_per_style or (3 * target)
    gate_kwargs = dict(
        min_start_vs_idle=-1.0 if args.no_gate else args.min_start_vs_idle,
        max_seam_pos=1e9 if args.no_gate else args.max_seam_pos,
        max_seam_vel=1e9 if args.no_gate else args.max_seam_vel,
        min_energy=-1.0 if args.no_gate else args.min_energy,
        max_root_path=1e9 if args.no_gate else args.max_root_path,
        max_root_disp=1e9 if args.no_gate else args.max_root_disp,
        min_rest_collapse=-1.0 if args.no_gate else args.min_rest_collapse,
    )

    print(f"[v3] {len(specs)} style(s), target {target} keeper(s) each "
          f"(attempt cap {cap}/style)")
    print(f"[v3] out:     {out_dir}")
    print(f"[v3] rejects: {reject_dir}")
    print(f"[v3] loop gate: start_vs_idle>={gate_kwargs['min_start_vs_idle']} "
          f"seam_pos<={gate_kwargs['max_seam_pos']} "
          f"seam_vel<={gate_kwargs['max_seam_vel']} "
          f"energy>={gate_kwargs['min_energy']}")

    rig = None
    if args.no_ugc_gate:
        print("[v3] UGC gate:  DISABLED")
    else:
        rig = ugc_validate.Rig.load(args.rig_json)
        print(f"[v3] UGC gate:  margin={args.ugc_margin} "
              f"(len<={ugc_validate.MAX_ANIMATION_LENGTH}s, "
              f"Y>={ugc_validate.HEIGHT_TOL}, "
              f"dist<={ugc_validate.MAX_BOUNDS}, "
              f"speed<={ugc_validate.MAX_SPEED_STUDS_PER_SEC} studs/s, "
              f"jointMove<={ugc_validate.MAX_JOINT_MOVEMENT}) "
              f"rig={args.rig_json.name}")
    print(f"[v3] parallelism: {args.jobs}, python: {sys.executable}")

    if args.dry_run:
        for spec in specs:
            cmd = _build_command(spec, out_dir, args.start_seed, f"{spec.name}_v00",
                                 args.loop_offset_mode,
                                 args.loop_cfg_constraint_weight)
            print(f"\n  {spec.name} (loop_offset={spec.loop_offset}):")
            print("   ", " ".join(cmd))
        return 0

    (out_dir / "manifest.json").write_text(json.dumps({
        "wave": "v3-latin",
        "styles": [
            {**asdict(s), "loop_offset": s.loop_offset, "loop": True,
             # Record the EFFECTIVE weight, not the spec default -- a
             # --loop-cfg-constraint-weight override was silently absent from
             # the manifest, making a finished run impossible to audit.
             "loop_cfg_constraint_weight": (
                 args.loop_cfg_constraint_weight
                 if args.loop_cfg_constraint_weight is not None
                 else s.loop_cfg_constraint_weight)}
            for s in specs
        ],
        "loop_offset_mode": args.loop_offset_mode,
        "pin_root_every": 0,
        "lower_torso_highpass_sigma": "pipeline default (0 = off)",
        "target_keeps_per_style": target,
        "max_attempts_per_style": cap,
        "start_seed": args.start_seed,
        "loop_gate": {k: v for k, v in gate_kwargs.items()},
        "ugc_gate": None if rig is None else {
            "margin": args.ugc_margin,
            "rig_json": str(args.rig_json),
            "max_length_s": ugc_validate.MAX_ANIMATION_LENGTH,
            "height_tol": ugc_validate.HEIGHT_TOL,
            "max_bounds": ugc_validate.MAX_BOUNDS,
            "max_speed_studs_per_sec": ugc_validate.MAX_SPEED_STUDS_PER_SEC,
            "max_joint_movement": ugc_validate.MAX_JOINT_MOVEMENT,
        },
    }, indent=2))

    # Per-style counters guarded by a lock: worker threads only run
    # subprocesses, but the scheduler loop below mutates these.
    lock = threading.Lock()
    keeps: dict[str, int] = {s.name: 0 for s in specs}
    attempts: dict[str, int] = {s.name: 0 for s in specs}
    seed_cursor: dict[str, int] = {s.name: args.start_seed for s in specs}
    results: list[Attempt] = []
    t_start = time.time()

    def next_job(pool: ThreadPoolExecutor, spec: DanceSpec):
        """Submit one more generation for `spec`, or None if it's done."""
        with lock:
            if keeps[spec.name] >= target or attempts[spec.name] >= cap:
                return None
            seed = seed_cursor[spec.name]
            seed_cursor[spec.name] += 1
            idx = attempts[spec.name]
            attempts[spec.name] += 1
        clip_name = f"{spec.name}_v{idx:02d}"
        fut = pool.submit(_generate_and_score, spec, out_dir, reject_dir,
                          seed, clip_name, gate_kwargs, args.timeout,
                          rig, args.ugc_margin, args.loop_offset_mode,
                          args.loop_cfg_constraint_weight)
        return fut

    total_target = target * len(specs)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending: dict = {}
        # Prime the pool: round-robin across styles so every style starts
        # early rather than one style hogging all workers.
        while len(pending) < args.jobs:
            progressed = False
            for spec in specs:
                if len(pending) >= args.jobs:
                    break
                fut = next_job(pool, spec)
                if fut is not None:
                    pending[fut] = spec
                    progressed = True
            if not progressed:
                break

        while pending:
            done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                spec = pending.pop(fut)
                att = fut.result()
                results.append(att)

                if att.status == "ok":
                    with lock:
                        keeps[spec.name] += 1
                        k = keeps[spec.name]
                    print(f"  KEEP  {att.clip:<22} seed={att.seed} "
                          f"({att.elapsed:.0f}s)  [{spec.name} {k}/{target}]")
                elif att.status == "gate-failed":
                    print(f"  drop  {att.clip:<22} seed={att.seed} "
                          f"({att.elapsed:.0f}s)  {'; '.join(att.reasons)}")
                else:
                    print(f"  FAIL  {att.clip:<22} seed={att.seed} "
                          f"{att.status}: {'; '.join(att.reasons)[:200]}")

                # Backfill: one finished slot -> one new job for the style
                # that still needs clips most urgently.
                for cand in sorted(specs, key=lambda s: keeps[s.name]):
                    nxt = next_job(pool, cand)
                    if nxt is not None:
                        pending[nxt] = cand
                        break

    elapsed = time.time() - t_start
    kept_total = sum(keeps.values())
    print(f"\n[v3] done in {elapsed / 60:.1f} min — "
          f"{kept_total}/{total_target} keepers from {len(results)} generations")
    for s in specs:
        short = "" if keeps[s.name] >= target else "  <-- SHORT"
        print(f"    {s.name:<12} {keeps[s.name]:>3}/{target} keep "
              f"({attempts[s.name]} attempts){short}")

    tally: dict[str, int] = {}
    for r in results:
        for reason in r.reasons:
            key = reason.split(" (")[0]
            tally[key] = tally.get(key, 0) + 1
    if tally:
        print("\n  rejection reasons:")
        for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>4}  {k}")

    stage = out_dir / "_staging"
    if stage.is_dir():
        leftover = [p.name for p in stage.iterdir() if p.is_dir()]
        if leftover:
            print(f"\n[v3] WARNING {len(leftover)} clip(s) left in staging: {leftover}")
        else:
            stage.rmdir()

    (out_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2))
    print(f"\n[v3] wrote {out_dir / 'results.json'}")
    print(f"[v3] next: python3 python/merge_emotes.py --input {out_dir} "
          f"--output {out_dir / 'all_latin_v3.rbxm'}")

    return 0 if kept_total == total_target else 1


if __name__ == "__main__":
    sys.exit(main())
