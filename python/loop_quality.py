#!/usr/bin/env python3
"""Score a generated clip for loop quality — used to gate batch generation.

Two failure modes we care about for looping emotes:

  1. **Idle start.** `prompt_pipeline.py --loop --loop-offset N` pins the
     pass-1 pose at frame `N*30` to *both* endpoints of pass 2, specifically
     to skip the idle ramp-in (see `_build_loop_constraints`). When that pin
     is soft — or when pass 1 was still ramping in at `loop_offset` — pass 2
     opens in the arms-down rest pose. Looping such a clip reads as a hitch
     every cycle: the dance collapses back to idle and restarts.

  2. **Open seam.** Even with the pin honored, the last frame can drift from
     the first. Playing that on loop pops.

Both are measured in normalized pose space: joint positions made
root-relative and yaw-derotated, so the metrics are invariant to where the
character wandered and which way it faced.

The idle reference is per-clip: frame 0 of `generated_pass1.npz`, which is
the prompt's natural starting pose and is idle for essentially every prompt
we feed it. A clip whose pass-2 frame 0 sits close to that pose never left
idle. Pass `--idle-ref` to compare against a canonical pose instead (build
one with `--build-idle-ref`).

Usage:
    # score one clip
    python3 python/loop_quality.py work/emotes_v3/salsa-basic_v00

    # score a whole tree, table + JSON summary
    python3 python/loop_quality.py --report work/emotes_v3

    # calibrate: dump raw metrics for every clip as CSV
    python3 python/loop_quality.py --report work/emotes --csv /tmp/v1.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ joints --
# SOMASkeleton77 order, read off kimodo's BVH export (ROOT/JOINT lines).
# posed_joints[:, 0] == root_positions, confirming index 0 is Root; the BVH's
# trailing RightToeEnd has no posed_joints slot (78 BVH joints -> 77 here).
JOINT_NAMES = [
    "Root", "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "HeadEnd", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
]
# Indices we score on. Fingers (16-39, 44-67), eyes and jaw are excluded:
# they are 40+ of the 77 joints, so including them lets finger jitter
# dominate a whole-body pose distance.
MAJOR_JOINTS = [
    0,   # Root
    1,   # Hips
    2, 3, 4,      # Spine1, Spine2, Chest
    5, 6, 7, 8,   # Neck1, Neck2, Head, HeadEnd
    12, 13, 14, 15,   # Left shoulder -> hand
    40, 41, 42, 43,   # Right shoulder -> hand
    68, 69, 70, 71,   # Left leg -> toe base
    73, 74, 75, 76,   # Right leg -> toe base
]

FPS = 30.0

# ------------------------------------------------------------- thresholds --
# Calibrated against the 800 clips in work/emotes + work/emotes_v2; see
# `--report --csv` output. Units are meters (kimodo space) for the pose
# distances and meters/second for velocities.
#
# start_vs_idle: mean major-joint distance from pass-2 frame 0 to the idle
#   reference. Idle-start clips cluster near 0; clips that open mid-dance
#   sit well above. HIGHER IS BETTER.
DEFAULT_MIN_START_VS_IDLE = 0.10
# start_percentile: fraction of the clip's frames that are STILLER than frame 0.
# Low means the loop point is one of the clip's stillest poses -- the "dancer
# stops and restarts every cycle" artifact. Calibrated below.
DEFAULT_MIN_START_PERCENTILE = 0.35
# stall_ratio: slowest rolling-window speed / median speed. Reported as a
# diagnostic but NOT gated: against hand-labelled ground truth it produced
# false positives (two clips scoring 0.33 and 0.55 were still judged bad),
# because a clip can drift through the bind pose without its speed collapsing.
DEFAULT_MIN_STALL_RATIO = -1.0
# Root travel, metres in kimodo space (x3.0 for studs). These are in-place
# emotes, so wander is a defect. Hand-labelled good clips measured path
# 1.57-2.94 m and displacement 0.14-0.30 m. Raising the loop constraint weight
# to 8.0 removes the idle stall but pushes travel to ~6.9 m path / ~1.95 m
# displacement -- the character visibly runs around. Limits sit just above the
# labelled-good range.
DEFAULT_MAX_ROOT_PATH = 4.00
DEFAULT_MAX_ROOT_DISP = 0.60
# rest_collapse: closest approach to the rest/bind pose anywhere in the clip,
# over the median distance from it. THIS IS THE PRIMARY GATE. Calibrated on 20
# hand-labelled v1 salsa clips: the three judged good scored 0.568 / 0.605 /
# 0.674, every clip judged bad scored <= 0.435. A threshold of 0.50 reproduces
# the labels exactly with a 0.13 margin on either side.
#
# An endpoint-windowed variant (near-bind AND near-still in world space, within
# +/-20 frames of the loop point) was tried and separated the labels WORSE
# (two bad clips outscored a good one), so the whole-clip minimum is what ships.
DEFAULT_MIN_REST_COLLAPSE = 0.50
# seam_pos: mean major-joint distance frame 0 <-> frame T-1. LOWER IS BETTER.
# Raised from 0.06 after hand labelling: clips judged good reached 0.073, so
# 0.06 was rejecting good work. `_inertial_blend_loop_seam` cleans up residual
# of this size, which is why it is not visible.
DEFAULT_MAX_SEAM_POS = 0.09
# seam_vel: |velocity(frame 0) - velocity(frame T-1)|, mean over major
#   joints. Catches poses that match but move in opposite directions.
# Also raised from 1.20: a clip judged good measured 1.71.
DEFAULT_MAX_SEAM_VEL = 2.00
# energy: mean major-joint speed across the clip. Filters clips where the
#   endpoint pin collapsed the whole thing into a near-static pose (the
#   "trivial low-loss solution" prompt_pipeline warns about).
DEFAULT_MIN_ENERGY = 0.25


# ------------------------------------------------------------------ core ----
def normalized_pose(npz: dict, derotate: bool = True) -> np.ndarray:
    """(T, J, 3) joint positions, root-relative and (optionally) yaw-aligned.

    Removing root translation makes the metrics blind to world drift;
    removing yaw makes them blind to which way the character turned, so a
    dance that spins doesn't read as an open seam.
    """
    pj = np.asarray(npz["posed_joints"], dtype=np.float64)  # (T, 77, 3)
    pose = pj - pj[:, 0:1, :]
    if not derotate:
        return pose
    heading = np.asarray(npz["global_root_heading"], dtype=np.float64)  # (T,2)
    c = heading[:, 0:1]
    s = heading[:, 1:2]
    x, y, z = pose[..., 0], pose[..., 1], pose[..., 2]
    # Rotate about +Y by -yaw, where (c, s) = (cos yaw, sin yaw).
    return np.stack([x * c - z * s, y, x * s + z * c], axis=-1)


def _mean_joint_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean euclidean distance between two poses over MAJOR_JOINTS."""
    d = np.linalg.norm(a[MAJOR_JOINTS] - b[MAJOR_JOINTS], axis=-1)
    return float(d.mean())


@dataclass
class Metrics:
    name: str
    frames: int
    start_vs_idle: float
    start_percentile: float
    root_path: float
    root_disp: float
    stall_ratio: float
    rest_collapse: float
    seam_pos: float
    seam_vel: float
    energy: float
    ramp_ratio: float
    verdict: str
    reasons: list[str]

    @property
    def ok(self) -> bool:
        return self.verdict == "keep"


def score_clip(
    clip_dir: Path,
    *,
    idle_ref: np.ndarray | None = None,
    min_start_vs_idle: float = DEFAULT_MIN_START_VS_IDLE,
    min_start_percentile: float = DEFAULT_MIN_START_PERCENTILE,
    min_stall_ratio: float = DEFAULT_MIN_STALL_RATIO,
    max_root_path: float = DEFAULT_MAX_ROOT_PATH,
    max_root_disp: float = DEFAULT_MAX_ROOT_DISP,
    min_rest_collapse: float = DEFAULT_MIN_REST_COLLAPSE,
    max_seam_pos: float = DEFAULT_MAX_SEAM_POS,
    max_seam_vel: float = DEFAULT_MAX_SEAM_VEL,
    min_energy: float = DEFAULT_MIN_ENERGY,
    derotate: bool = True,
) -> Metrics:
    """Score one clip directory (expects generated.npz, generated_pass1.npz)."""
    clip_dir = Path(clip_dir)
    gen = clip_dir / "generated.npz"
    if not gen.is_file():
        raise FileNotFoundError(f"missing {gen}")

    with np.load(gen) as z:
        pose = normalized_pose(z, derotate=derotate)
        # Root travel in the ground plane. `normalized_pose` deliberately
        # removes root translation, so this has to be read separately. It
        # matters because these are in-place emotes: the pipeline folds root
        # motion into LowerTorso's Position curve, so a wandering clip both
        # looks wrong and inflates that translation.
        _rp = np.asarray(z["root_positions"], dtype=np.float64)[:, [0, 2]]
    root_path = float(np.linalg.norm(np.diff(_rp, axis=0), axis=1).sum())
    root_disp = float(np.linalg.norm(_rp - _rp[0], axis=1).max())

    # Idle reference: caller-supplied canonical pose, else this clip's own
    # pass-1 frame 0 (the prompt's natural, pre-loop starting pose).
    ref = idle_ref
    can_check_idle = True
    if ref is None:
        p1 = clip_dir / "generated_pass1.npz"
        if p1.is_file():
            with np.load(p1) as z1:
                ref = normalized_pose(z1, derotate=derotate)[0]
        else:
            # Non-looping clip (no pass 1) and no canonical reference. Frame 0
            # *is* the natural start, so there is nothing to compare against —
            # skip the idle check rather than fabricate a verdict.
            ref = pose[0]
            can_check_idle = False

    t = pose.shape[0]
    start_vs_idle = _mean_joint_dist(pose[0], ref)
    seam_pos = _mean_joint_dist(pose[0], pose[-1])

    # Where does the opening pose sit within THIS clip's own range of
    # distance-from-rest? `start_vs_idle` is an absolute distance, so its scale
    # tracks how big the prompt's motion is — measured on the shipped batches,
    # breakdance pass-1 reaches 0.32 from rest while salsa only reaches 0.25.
    # An absolute threshold therefore over-rejects the calmer styles. This
    # percentile is scale-free: it answers "is frame 0 one of the *stiller*
    # poses in this clip, or a real mid-motion one?" Since the loop pivot is
    # pinned at both ends, a low percentile is exactly the artifact of the
    # dancer settling to a stand at the loop point.
    dist_from_rest = np.array([_mean_joint_dist(pose[i], ref) for i in range(t)])
    start_percentile = float((dist_from_rest < dist_from_rest[0]).mean())

    # --- stall detection -------------------------------------------------
    # The endpoint-pin failure mode that endpoint metrics cannot see: kimodo
    # satisfies the pin by parking in a near-rest pose for a fraction of a
    # second somewhere mid-clip (typically just BEFORE the loop point), then
    # lunging back to the pinned pose. Measured on a clip that passed every
    # other check: frames 160-168 of 180 sat at dist_from_rest 0.053 against a
    # clip median of 0.185, with speed 0.03 against a median of 1.30. Looped,
    # that reads exactly as "the dancer stops and restarts".
    #
    # Clip-wide `energy` averages the dip away (a 15-frame stall in 180 frames
    # barely moves the mean), and `seam_*`/`start_*` only sample the endpoints.
    # These two ratios are scale-free and catch it wherever it happens.
    med_speed = float(np.median(speed_all := np.concatenate(
        [(sp := np.linalg.norm(np.diff(pose, axis=0) * FPS, axis=-1)[:, MAJOR_JOINTS]
          .mean(axis=1))[:1], sp])))
    win = max(3, min(9, t // 8))
    kern = np.ones(win) / win
    rolling = np.convolve(speed_all, kern, mode="valid")
    stall_ratio = float(rolling.min() / med_speed) if med_speed > 1e-9 else 0.0

    med_rest = float(np.median(dist_from_rest))
    rest_collapse = (float(dist_from_rest.min() / med_rest)
                     if med_rest > 1e-9 else 0.0)

    vel = np.diff(pose, axis=0) * FPS  # (T-1, J, 3) m/s
    seam_vel = float(
        np.linalg.norm(vel[0][MAJOR_JOINTS] - vel[-1][MAJOR_JOINTS], axis=-1).mean()
    )
    speed = np.linalg.norm(vel[:, MAJOR_JOINTS, :], axis=-1).mean(axis=1)  # (T-1,)
    energy = float(speed.mean())

    # Ramp: is the opening slice much slower than the clip as a whole? A
    # low ratio means an ease-in from stillness, which loops badly even when
    # the opening pose itself isn't literally the rest pose.
    head = max(1, int(0.15 * len(speed)))
    med = float(np.median(speed))
    ramp_ratio = float(speed[:head].mean() / med) if med > 1e-9 else 0.0

    reasons: list[str] = []
    if can_check_idle and start_vs_idle < min_start_vs_idle:
        reasons.append(
            f"idle-start (start_vs_idle {start_vs_idle:.3f} < {min_start_vs_idle:.3f})"
        )
    if can_check_idle and start_percentile < min_start_percentile:
        reasons.append(
            f"loop point settles to a stand (start_percentile "
            f"{start_percentile:.2f} < {min_start_percentile:.2f})"
        )
    if stall_ratio < min_stall_ratio:
        reasons.append(
            f"motion stalls mid-clip (stall_ratio {stall_ratio:.2f} "
            f"< {min_stall_ratio:.2f})"
        )
    if can_check_idle and rest_collapse < min_rest_collapse:
        reasons.append(
            f"returns to a stand mid-clip (rest_collapse {rest_collapse:.2f} "
            f"< {min_rest_collapse:.2f})"
        )
    if root_path > max_root_path:
        reasons.append(
            f"wanders (root_path {root_path:.2f} m > {max_root_path:.2f} m)"
        )
    if root_disp > max_root_disp:
        reasons.append(
            f"drifts off the spot (root_disp {root_disp:.2f} m "
            f"> {max_root_disp:.2f} m)"
        )
    if seam_pos > max_seam_pos:
        reasons.append(f"open seam (seam_pos {seam_pos:.3f} > {max_seam_pos:.3f})")
    if seam_vel > max_seam_vel:
        reasons.append(f"seam velocity jump (seam_vel {seam_vel:.2f} > {max_seam_vel:.2f})")
    if energy < min_energy:
        reasons.append(f"too static (energy {energy:.3f} < {min_energy:.3f})")

    return Metrics(
        name=clip_dir.name,
        frames=t,
        start_vs_idle=round(start_vs_idle, 4),
        start_percentile=round(start_percentile, 4),
        root_path=round(root_path, 3),
        root_disp=round(root_disp, 3),
        stall_ratio=round(stall_ratio, 4),
        rest_collapse=round(rest_collapse, 4),
        seam_pos=round(seam_pos, 4),
        seam_vel=round(seam_vel, 3),
        energy=round(energy, 4),
        ramp_ratio=round(ramp_ratio, 3),
        verdict="keep" if not reasons else "discard",
        reasons=reasons,
    )


def pick_mid_motion_frame(
    pass1_npz: Path,
    *,
    search_from: float = 0.25,
    search_to: float = 0.95,
    top_frac: float = 0.40,
    derotate: bool = True,
) -> tuple[int, dict]:
    """Choose a loop pivot frame that is genuinely mid-motion.

    A fixed `--loop-offset` (even at mid-duration) assumes the motion has
    developed by then. For dance prompts kimodo often ramps in slowly, so the
    pose at that time is still near the standing rest pose — and since
    `_build_loop_constraints` pins the pivot pose at BOTH ends of pass 2, the
    whole clip then opens and closes on a stand. Looped, it reads as the dancer
    stopping and restarting every cycle.

    So pick the frame from the data instead:
      1. Restrict to `[search_from, search_to]` of the clip, skipping the
         ramp-in and the last few frames.
      2. Keep the frames furthest from the rest pose (top `top_frac`) — those
         are unambiguously "in the dance".
      3. Among those, take the frame whose speed is closest to the clip's
         median speed. Picking peak distance alone tends to land on a
         momentary extreme (a full extension or a freeze); a median-speed
         frame is a *representative* moment of the motion, which makes a
         better loop pivot.

    Returns (frame_index, diagnostics).
    """
    with np.load(pass1_npz) as z:
        pose = normalized_pose(z, derotate=derotate)

    t = pose.shape[0]
    if t < 8:
        return 0, {"reason": "clip too short to search", "frames": t}

    rest = pose[0]
    dist = np.array([_mean_joint_dist(pose[i], rest) for i in range(t)])
    vel = np.diff(pose, axis=0) * FPS
    speed = np.linalg.norm(vel[:, MAJOR_JOINTS, :], axis=-1).mean(axis=1)
    speed = np.concatenate([speed[:1], speed])  # align length to t

    lo = max(1, int(search_from * t))
    hi = min(t - 1, int(search_to * t))
    if hi <= lo:
        lo, hi = 1, t - 1

    window = np.arange(lo, hi)
    d_win = dist[window]
    # Frames that are clearly out of the rest pose.
    cutoff = np.quantile(d_win, 1.0 - top_frac)
    candidates = window[d_win >= cutoff]
    if candidates.size == 0:
        candidates = window

    med_speed = float(np.median(speed[window]))
    best = int(candidates[np.argmin(np.abs(speed[candidates] - med_speed))])

    return best, {
        "frames": int(t),
        "search_window": [int(lo), int(hi)],
        "picked_frame": best,
        "picked_time_s": round(best / FPS, 3),
        "picked_dist_from_rest": round(float(dist[best]), 4),
        "max_dist_from_rest": round(float(dist.max()), 4),
        "median_dist_in_window": round(float(np.median(d_win)), 4),
        "picked_speed": round(float(speed[best]), 4),
        "median_speed_in_window": round(med_speed, 4),
    }


def build_idle_reference(roots: list[Path], derotate: bool = True) -> np.ndarray:
    """Median pass-1 frame-0 pose across many clips = canonical idle pose."""
    poses = []
    for root in roots:
        for p1 in sorted(Path(root).rglob("generated_pass1.npz")):
            try:
                with np.load(p1) as z:
                    poses.append(normalized_pose(z, derotate=derotate)[0])
            except Exception:
                continue
    if not poses:
        raise SystemExit("no generated_pass1.npz found to build an idle reference")
    print(f"[loop_quality] idle reference from {len(poses)} clips", file=sys.stderr)
    return np.median(np.stack(poses), axis=0)


# ------------------------------------------------------------------- cli ----
def _find_clips(root: Path) -> list[Path]:
    return sorted({p.parent for p in Path(root).rglob("generated.npz")})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", type=Path, nargs="?",
                   help="Clip directory to score (or tree root with --report)")
    p.add_argument("--report", action="store_true",
                   help="Score every clip under target and print a summary")
    p.add_argument("--csv", type=Path, default=None,
                   help="With --report: write raw per-clip metrics as CSV")
    p.add_argument("--json", type=Path, default=None,
                   help="With --report: write per-clip metrics as JSON")
    p.add_argument("--idle-ref", type=Path, default=None,
                   help="Canonical idle pose .npy (default: per-clip pass-1 frame 0)")
    p.add_argument("--build-idle-ref", type=Path, default=None,
                   help="Compute a canonical idle pose from target tree(s), write .npy, exit")
    p.add_argument("--extra-root", type=Path, action="append", default=[],
                   help="Additional tree(s) for --build-idle-ref")
    p.add_argument("--no-derotate", action="store_true",
                   help="Skip yaw alignment (debugging)")
    p.add_argument("--min-start-vs-idle", type=float, default=DEFAULT_MIN_START_VS_IDLE)
    p.add_argument("--min-start-percentile", type=float,
                   default=DEFAULT_MIN_START_PERCENTILE)
    p.add_argument("--min-stall-ratio", type=float, default=DEFAULT_MIN_STALL_RATIO)
    p.add_argument("--max-root-path", type=float, default=DEFAULT_MAX_ROOT_PATH)
    p.add_argument("--max-root-disp", type=float, default=DEFAULT_MAX_ROOT_DISP)
    p.add_argument("--min-rest-collapse", type=float,
                   default=DEFAULT_MIN_REST_COLLAPSE)
    p.add_argument("--max-seam-pos", type=float, default=DEFAULT_MAX_SEAM_POS)
    p.add_argument("--max-seam-vel", type=float, default=DEFAULT_MAX_SEAM_VEL)
    p.add_argument("--min-energy", type=float, default=DEFAULT_MIN_ENERGY)
    args = p.parse_args(argv)

    derotate = not args.no_derotate

    if args.build_idle_ref:
        roots = ([args.target] if args.target else []) + list(args.extra_root)
        ref = build_idle_reference(roots, derotate=derotate)
        args.build_idle_ref.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.build_idle_ref, ref)
        print(f"[loop_quality] wrote {args.build_idle_ref}")
        return 0

    if args.target is None:
        p.error("target is required")

    idle_ref = np.load(args.idle_ref) if args.idle_ref else None
    kw = dict(
        idle_ref=idle_ref,
        min_start_vs_idle=args.min_start_vs_idle,
        min_start_percentile=args.min_start_percentile,
        min_stall_ratio=args.min_stall_ratio,
        max_root_path=args.max_root_path,
        max_root_disp=args.max_root_disp,
        min_rest_collapse=args.min_rest_collapse,
        max_seam_pos=args.max_seam_pos,
        max_seam_vel=args.max_seam_vel,
        min_energy=args.min_energy,
        derotate=derotate,
    )

    if not args.report:
        m = score_clip(args.target, **kw)
        print(json.dumps(asdict(m), indent=2))
        return 0 if m.ok else 1

    clips = _find_clips(args.target)
    if not clips:
        print(f"[loop_quality] no clips under {args.target}")
        return 1

    rows: list[Metrics] = []
    for c in clips:
        try:
            rows.append(score_clip(c, **kw))
        except Exception as e:  # keep going; report at the end
            print(f"  ERR {c.name}: {e}", file=sys.stderr)

    kept = [r for r in rows if r.ok]
    print(f"\n[loop_quality] {len(kept)}/{len(rows)} keep "
          f"({100.0 * len(kept) / max(1, len(rows)):.1f}%)\n")

    hdr = (f"{'clip':<32} {'start_idle':>10} {'start_pct':>9} {'seam_pos':>9} "
           f"{'seam_vel':>9} {'energy':>7} {'ramp':>6}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r.name:<32} {r.start_vs_idle:>10.3f} {r.start_percentile:>9.2f} {r.seam_pos:>9.3f} "
              f"{r.seam_vel:>9.2f} {r.energy:>7.3f} {r.ramp_ratio:>6.2f}  "
              f"{r.verdict}{' — ' + '; '.join(r.reasons) if r.reasons else ''}")

    # Failure tally by reason category
    tally: dict[str, int] = {}
    for r in rows:
        for reason in r.reasons:
            key = reason.split(" (")[0]
            tally[key] = tally.get(key, 0) + 1
    if tally:
        print("\nfailure modes:")
        for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>4}  {k}")

    # Distribution summary — the numbers you tune thresholds from.
    print("\ndistributions (min / p10 / p50 / p90 / max):")
    for field in ("start_vs_idle", "start_percentile", "root_path",
                  "root_disp", "stall_ratio",
                  "rest_collapse", "seam_pos", "seam_vel", "energy",
                  "ramp_ratio"):
        vals = np.array([getattr(r, field) for r in rows], dtype=float)
        qs = np.percentile(vals, [0, 10, 50, 90, 100])
        print(f"  {field:<14} " + " / ".join(f"{q:7.3f}" for q in qs))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w") as f:
            f.write("clip,frames,start_vs_idle,start_percentile,seam_pos,seam_vel,energy,ramp_ratio,verdict\n")
            for r in rows:
                f.write(f"{r.name},{r.frames},{r.start_vs_idle},{r.start_percentile},{r.seam_pos},"
                        f"{r.seam_vel},{r.energy},{r.ramp_ratio},{r.verdict}\n")
        print(f"\n[loop_quality] wrote {args.csv}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(r) for r in rows], indent=2))
        print(f"[loop_quality] wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
