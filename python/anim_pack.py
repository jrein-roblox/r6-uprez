#!/usr/bin/env python3
"""Build a Roblox animation pack from a small number of generated clips.

A pack is 8 clips across 7 Marketplace asset types:

    IdleAnimation   idle x2 (the only slot taking two, weighted)
    WalkAnimation   walk
    RunAnimation    run
    JumpAnimation   jump      -- the ONLY slot allowed Loop = false
    FallAnimation   fall
    ClimbAnimation  climb
    SwimAnimation   swim + swimidle (two StringValues, one asset)

Kimodo reliably produces standing and locomotion motion from a prompt. It does
not reliably produce a themed ladder-climb or front-crawl, and even when it
does the result rarely matches the walk stylistically -- which is what makes a
pack read as one pack. So this module generates the three it can do (idle,
walk, run) and DERIVES the rest by transforming those curves. Every slot then
shares the same limb style by construction.

Two things drive the design:

1. **Cycle extraction, not endpoint pinning.** `prompt_pipeline --loop` pins one
   pose at both ends of a second diffusion pass. For locomotion that fights the
   translation, and it is what produced the mid-clip idle stall we spent a batch
   diagnosing. Instead: generate long and free, find where the motion NATURALLY
   returns to a matching pose and contact state, and cut there. Nothing is
   forced, so nothing stalls.

2. **Pack slots have stricter validation than emotes.** `CurveAnimLoopingRequired`
   fails any ANIMATION-category upload whose `Loop` is false (JumpAnimation
   exempt), and `CurveAnimPartsRotateOnlyIfBones` applies UNCONDITIONALLY for
   ANIMATION -- capping every part's Position drift at 0.5 studs with no
   LowerTorso exemption, unlike the emote path. Hence `--root-mode strip`
   upstream and the drift assertion here.

Usage:
    # inspect cycle detection on generated clips
    python3 python/anim_pack.py cycles work/zombie_raw

    # build the pack (cuts cycles, derives slots, writes r15.json per slot)
    python3 python/anim_pack.py build work/zombie_raw --out work/zombie_pack \
        --walk walk_s9000 --run run_s9000 --idle idle_s9000 --idle2 idle_s9001
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import loop_quality as lq  # noqa: E402

FPS = 30.0

# SOMASkeleton77 foot columns in the npz `foot_contacts` array. The local kimodo
# source documents 4 columns but 6 are emitted; columns 2 and 5 duplicate 1 and
# 4 (ToeEnd repeats ToeBase). Verified empirically, so score only these.
CONTACT_L_ANKLE, CONTACT_L_TOE = 0, 1
CONTACT_R_ANKLE, CONTACT_R_TOE = 3, 4
CONTACT_COLS = [CONTACT_L_ANKLE, CONTACT_L_TOE, CONTACT_R_ANKLE, CONTACT_R_TOE]

# R15 parts, and the left/right pairs used for mirroring.
MIRROR_PAIRS = [
    ("LeftUpperArm", "RightUpperArm"), ("LeftLowerArm", "RightLowerArm"),
    ("LeftHand", "RightHand"),
    ("LeftUpperLeg", "RightUpperLeg"), ("LeftLowerLeg", "RightLowerLeg"),
    ("LeftFoot", "RightFoot"),
]


# ------------------------------------------------------------ quaternions ----
def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product. Quaternions are (x, y, z, w) to match r15.json."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def _axis_angle_quat(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    v = np.asarray(axis, dtype=float)
    v = v / (np.linalg.norm(v) or 1.0)
    s = math.sin(angle / 2.0)
    return np.array([v[0] * s, v[1] * s, v[2] * s, math.cos(angle / 2.0)])


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Shortest-arc slerp. t may be an array; q0/q1 are single quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:  # nearly identical: lerp and renormalise
        out = q0[None, :] + t[:, None] * (q1 - q0)[None, :]
        return out / np.linalg.norm(out, axis=1, keepdims=True)
    theta = math.acos(dot)
    st = math.sin(theta)
    a = np.sin((1.0 - t) * theta) / st
    b = np.sin(t * theta) / st
    return a[:, None] * q0[None, :] + b[:, None] * q1[None, :]


# --------------------------------------------------------- r15.json access ---
def part_quats(part: dict) -> np.ndarray | None:
    if "rotX" not in part:
        return None
    return np.stack([np.asarray(part["rotX"], dtype=float),
                     np.asarray(part["rotY"], dtype=float),
                     np.asarray(part["rotZ"], dtype=float),
                     np.asarray(part["rotW"], dtype=float)], axis=1)


def _unroll(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions stay in the same hemisphere.

    q and -q are the same rotation, but `build_rbxm.lua` converts to Euler XYZ
    and the validator interpolates those three channels independently. Across a
    sign flip that takes the long way round, which shows up as an enormous
    one-frame velocity spike: an unrolled climb clip measured 231 studs/s on
    RightHand against the 45 studs/s `CurveAnimSpeedBounded` limit, purely from
    a flip introduced by pre-multiplying an offset.

    Applied inside `set_part_quats` so every transform in this module -- offset,
    retime, pitch, slice, blend -- gets it without having to remember.
    """
    q = np.array(q, dtype=float, copy=True)
    for i in range(1, q.shape[0]):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    return q


def set_part_quats(part: dict, q: np.ndarray) -> None:
    q = _unroll(q)
    part["rotX"] = q[:, 0].tolist()
    part["rotY"] = q[:, 1].tolist()
    part["rotZ"] = q[:, 2].tolist()
    part["rotW"] = q[:, 3].tolist()


def part_pos(part: dict) -> np.ndarray | None:
    if "posX" not in part:
        return None
    return np.stack([np.asarray(part["posX"], dtype=float),
                     np.asarray(part["posY"], dtype=float),
                     np.asarray(part["posZ"], dtype=float)], axis=1)


def set_part_pos(part: dict, p: np.ndarray) -> None:
    part["posX"] = p[:, 0].tolist()
    part["posY"] = p[:, 1].tolist()
    part["posZ"] = p[:, 2].tolist()


# -------------------------------------------------------- cycle detection ----
@dataclass
class Cycle:
    start: int
    length: int
    seam_pos: float
    seam_vel: float
    contact_match: bool
    stride_asym: float
    duty_l: float
    duty_r: float
    travel_m: float
    score: float

    @property
    def end(self) -> int:
        """Inclusive end frame. Shares its pose with `start`, so the emitted
        clip keeps it as the representable loop point (same convention as
        `_trim_middle_cycle` in pipeline.py)."""
        return self.start + self.length


def detect_cycle(
    clip_dir: Path,
    *,
    min_len: int = 20,
    max_len: int = 90,
    skip_head_frac: float = 0.20,
    require_contact_match: bool = True,
) -> Cycle | None:
    """Find the best single stride cycle in a free-running locomotion clip.

    Scored on four things, because seam distance alone is not enough:
      * pose + velocity match across the seam (it has to be continuous)
      * identical foot contact state at both ends -- a half-stride matches
        poses well but has the opposite foot forward, which reads as a limp
      * stride symmetry: the clip compared against its own mirrored,
        half-shifted self. This is what separates a real gait from "some leg
        movement that happens to repeat"
      * enough travel to be a real step rather than a shuffle in place
    """
    gen = clip_dir / "generated.npz"
    if not gen.is_file():
        return None
    with np.load(gen) as z:
        pose = lq.normalized_pose(z)
        contacts = np.asarray(z["foot_contacts"], dtype=bool)
        root = np.asarray(z["root_positions"], dtype=float)[:, [0, 2]]

    t = pose.shape[0]
    if contacts.shape[0] != t or not contacts.any():
        # An all-zero contact array means the detector found nothing; treating
        # that as "always airborne" would silently accept garbage.
        return None

    vel = np.diff(pose, axis=0) * FPS
    MJ = lq.MAJOR_JOINTS
    lo = int(skip_head_frac * t)          # skip the ramp-in from rest
    best: Cycle | None = None

    for start in range(lo, t - min_len - 1):
        for length in range(min_len, min(max_len, t - 1 - start) + 1):
            end = start + length
            if require_contact_match and not np.array_equal(
                    contacts[start, CONTACT_COLS], contacts[end, CONTACT_COLS]):
                continue
            seam_pos = float(np.linalg.norm(
                pose[start][MJ] - pose[end][MJ], axis=-1).mean())
            if seam_pos > 0.10:
                continue
            seam_vel = float(np.linalg.norm(
                vel[start][MJ] - vel[min(end, t - 2)][MJ], axis=-1).mean())

            seg = pose[start:end + 1]
            half = length // 2
            if half < 2:
                continue
            asym = _stride_asymmetry(seg, half)
            duty_l = float(contacts[start:end + 1, CONTACT_L_ANKLE].mean())
            duty_r = float(contacts[start:end + 1, CONTACT_R_ANKLE].mean())
            travel = float(np.linalg.norm(root[end] - root[start]))

            # Lower is better. Symmetry is weighted hardest because it is the
            # metric that actually distinguishes a gait cycle.
            score = seam_pos + 0.01 * seam_vel + 1.5 * asym
            if best is None or score < best.score:
                best = Cycle(start, length, round(seam_pos, 4),
                             round(seam_vel, 3), True, round(asym, 4),
                             round(duty_l, 3), round(duty_r, 3),
                             round(travel, 3), score)
    return best


def _stride_asymmetry(seg: np.ndarray, half: int) -> float:
    """Distance between the cycle and its own mirrored, half-shifted self.

    In a symmetric gait, shifting by half a stride and swapping left/right
    reproduces the original. A one-sided or limping motion does not.
    """
    # Mirror in normalized pose space: negate X and swap the left/right joint
    # index groups. lq.MAJOR_JOINTS is ordered spine, then L arm, R arm,
    # L leg, R leg -- so swap those two contiguous pairs.
    MJ = lq.MAJOR_JOINTS
    larm, rarm = slice(9, 13), slice(13, 17)
    lleg, rleg = slice(17, 21), slice(21, 25)
    a = seg[:, MJ, :].copy()
    m = a.copy()
    m[..., 0] *= -1.0                     # mirror across the sagittal plane
    m[:, larm, :], m[:, rarm, :] = a[:, rarm, :].copy(), a[:, larm, :].copy()
    m[:, lleg, :], m[:, rleg, :] = a[:, rleg, :].copy(), a[:, lleg, :].copy()
    m[:, larm, 0] *= -1.0
    m[:, rarm, 0] *= -1.0
    m[:, lleg, 0] *= -1.0
    m[:, rleg, 0] *= -1.0
    shifted = np.roll(m, half, axis=0)
    return float(np.linalg.norm(a - shifted, axis=-1).mean())


# ------------------------------------------------------------ cycle slicing --
def detect_idle_window(
    clip_dir: Path,
    *,
    min_sec: float = 1.5,
    max_sec: float = 4.0,
    skip_head_frac: float = 0.15,
    min_energy: float = 0.60,
    max_energy: float = 1e9,
    min_rest_collapse: float = 0.50,
    min_pose_offset: float = 0.06,
    energy_frac_of_best: float = 0.60,
    max_drift: float = 0.45,
) -> tuple[int, int] | None:
    """Find the idle window whose endpoints ALREADY match.

    An idle has no stride, so `detect_cycle`'s contact-state and symmetry tests
    do not apply -- but it still needs its start and end pose to agree, and
    picking an arbitrary window does not give you that.

    The failure this fixes: the builder used to slice a fixed 3 s from 25% in and
    rely on `blend_seam` to close it. `blend_seam` forces frame 0 to equal the
    last frame, so the seam measures as perfectly closed -- but when the natural
    endpoints are far apart that convergence happens over 4 frames, and the
    forced warp IS the visible snap. Choosing a window whose ends already agree
    leaves the blend with almost nothing to do.

    Scored on pose AND velocity match: matching poses moving in opposite
    directions still pop.

    CRITICAL: seam match alone is the wrong objective. The cheapest way to have
    matching endpoints is a window where NOTHING MOVES, and minimising seam
    without a motion floor picks exactly that -- it produced idles that stood at
    the bind pose while the source clip had plenty of motion available (window
    energy 0.008 against 1.117 available in the same clip).

    So two hard filters run before the seam is even scored, both calibrated
    against idles Jordan judged good (salsa) versus bad (static hip-hop):

        energy           good 1.01-1.31   bad 0.008-0.024   -> floor 0.60
        rest_collapse    (min distance from rest over the window, / median)

    `rest_collapse` is the metric Jordan's emote labels calibrated, and it is
    what catches the original complaint: a window that plays real motion but
    passes through the bind pose at its ends reads as "dances, then blends to
    bind pose at the seam". `energy_frac_of_best` additionally requires the
    window to be within a fraction of the liveliest window in the clip, so a
    uniformly sleepy clip cannot sneak through the absolute floor.
    """
    gen = clip_dir / "generated.npz"
    if not gen.is_file():
        return None
    with np.load(gen) as z:
        pose = lq.normalized_pose(z)

    # LowerTorso translation, so the search can respect the 0.5-stud Position
    # drift cap directly. Enforcing it here rather than discovering it after the
    # fact is what stops the search proposing windows that then fail validation:
    # three of four hip-hop idle sources produced 0.55-0.80 stud windows.
    lt_xyz = None
    r15_path = clip_dir / "r15.json"
    if r15_path.is_file():
        lt = json.loads(r15_path.read_text()).get("parts", {}).get("LowerTorso")
        if lt and "posX" in lt:
            lt_xyz = np.stack([np.asarray(lt["posX"], dtype=float),
                               np.asarray(lt["posY"], dtype=float),
                               np.asarray(lt["posZ"], dtype=float)], axis=1)

    t = pose.shape[0]
    vel = np.diff(pose, axis=0) * FPS
    MJ = lq.MAJOR_JOINTS
    speed = np.linalg.norm(vel[:, MJ, :], axis=-1).mean(axis=1)
    rest = pose[0]
    dist_rest = np.array([lq._mean_joint_dist(pose[i], rest) for i in range(t)])

    lo = int(skip_head_frac * t)
    min_len = max(4, int(min_sec * FPS))
    max_len = int(max_sec * FPS)

    # Liveliest window available, so the floor can adapt to the clip.
    best_energy = 0.0
    for start in range(lo, max(lo + 1, t - min_len - 1)):
        e = float(speed[start:start + min_len].mean())
        best_energy = max(best_energy, e)
    # The relative floor exists to stop the search picking the still parts of a
    # lively clip. But when a CEILING is set explicitly the caller is asking for
    # a calm window on purpose, and the relative floor would veto exactly what
    # they asked for -- 60% of a 0.5 best window is 0.3, which rejects
    # everything under it. So an explicit ceiling disables the relative floor and
    # `min_energy` alone guards against a frozen window.
    if max_energy < 1e8:
        energy_floor = min_energy
    else:
        energy_floor = max(min_energy, energy_frac_of_best * best_energy)

    best = None
    rejected_low = rejected_high = rejected_rest = rejected_drift = 0
    for start in range(lo, t - min_len - 1):
        for length in range(min_len, min(max_len, t - 1 - start) + 1):
            end = start + length
            win_energy = float(speed[start:end].mean())
            if win_energy < energy_floor:
                rejected_low += 1
                continue
            if win_energy > max_energy:
                rejected_high += 1
                continue
            if lt_xyz is not None and end < lt_xyz.shape[0]:
                seg_lt = lt_xyz[start:end + 1]
                if float(np.linalg.norm(seg_lt - seg_lt[0], axis=1).max()) > max_drift:
                    rejected_drift += 1
                    continue
            seg = dist_rest[start:end + 1]
            med = float(np.median(seg))
            # ABSOLUTE distance from the bind pose. `rest_collapse` below is a
            # RATIO of min to median within the window, so a window sitting
            # uniformly close to bind still scores high on it -- which is how a
            # 0.05-energy window with rest_collapse 0.80 got shipped as an idle
            # that read as "standing at bind with a little movement". Measured
            # median distance for that window was 0.012; a window that reads as a
            # held dance pose is 0.08+. This is the filter that catches it.
            if med < min_pose_offset:
                rejected_rest += 1
                continue
            if med <= 1e-9 or float(seg.min() / med) < min_rest_collapse:
                rejected_rest += 1
                continue
            dp = float(np.linalg.norm(pose[start][MJ] - pose[end][MJ], axis=-1).mean())
            dv = float(np.linalg.norm(
                vel[start][MJ] - vel[min(end, t - 2)][MJ], axis=-1).mean())
            cost = dp + 0.02 * dv
            if best is None or cost < best[0]:
                best = (cost, start, length, dp, dv,
                        float(speed[start:end].mean()), float(seg.min() / med))
    if best is None:
        # Distinguish the two energy rejections. Conflating them printed
        # "too static" for windows that were actually too ENERGETIC for an
        # explicit ceiling, which sent debugging the wrong way entirely.
        print(f"[pack] no idle window passed the filters "
              f"(energy in [{energy_floor:.2f}, {max_energy:.2f}], "
              f"pose_offset>={min_pose_offset:.2f}, "
              f"rest_collapse>={min_rest_collapse:.2f}, drift<={max_drift:.2f})\n"
              f"       rejected: {rejected_low} BELOW energy floor, "
              f"{rejected_high} ABOVE energy ceiling, "
              f"{rejected_rest} too near bind / collapsed, "
              f"{rejected_drift} over drift")
        return None
    _, start, length, dp, dv, en, rc = best
    print(f"[pack] idle window {start}..{start + length} ({length / FPS:.2f}s) "
          f"seam pos={dp:.4f} vel={dv:.3f} energy={en:.2f} rest_collapse={rc:.2f}")
    return start, length


def slice_cycle(r15: dict, start: int, length: int) -> dict:
    """Return a copy of `r15` containing only frames [start, start+length]."""
    out = copy.deepcopy(r15)
    end = start + length
    for part in out["parts"].values():
        q = part_quats(part)
        if q is not None:
            set_part_quats(part, q[start:end + 1])
        p = part_pos(part)
        if p is not None:
            set_part_pos(part, p[start:end + 1])
    root = out.get("root")
    if root:
        q = part_quats(root)
        if q is not None:
            set_part_quats(root, q[start:end + 1])
        p = part_pos(root)
        if p is not None:
            set_part_pos(root, p[start:end + 1])
    out["frameCount"] = length + 1
    return out


def _inertial_decay(n: int, x0: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """Inertialization decay curve (Bollo 2018), shape (n, D).

    True inertial blending, as distinct from the smoothstep offset-decay both
    `blend_seam` and `pipeline._inertial_blend_loop_seam` used to do. Those match
    POSITION at the seam and ignore velocity, so a slow clip whose two ends are
    moving differently still changes direction abruptly -- visible as the hitch
    in a graceful idle, where the seam velocity mismatch measured 0.43-0.53.

    Solves a quintic in the offset so that at the end of the blend the offset,
    its velocity, AND its acceleration are all zero:

        x(t) = A/2 t^5 + B t^4 + C t^3 + v0 t + x0
        x(T) = x'(T) = x''(T) = 0

    Carrying v0 is the whole point: the blend leaves the seam moving the way the
    previous frame was moving, then eases that away, instead of teleporting the
    pose and letting velocity jump.
    """
    # T = n-1, NOT n. The decay is APPLIED to frames 0..n-1, so it has to reach
    # zero at index n-1 -- if it only reaches zero at index n, the last blended
    # frame still carries a residual offset while frame n carries none, and that
    # step is a discontinuity. It cost a 200 studs/s spike at exactly the blend
    # boundary (t=0.46s for a 14-frame blend) against the 45 limit.
    T = float(max(1, n - 1))
    t = np.arange(n, dtype=float)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    v0 = np.asarray(v0, dtype=float).reshape(-1)

    # Solve A, B, C as a 3x3 system rather than by hand. TWO separate
    # hand-derived closed forms were wrong here in ways that still looked
    # plausible -- x(T) came out 0 while x'(T) and x''(T) did not -- so the
    # blend never actually landed smoothly. The system is tiny; solve it.
    #
    #   x(t) = A/2 t^5 + B t^4 + C t^3 + v0 t + x0
    #   x(T) = 0,  x'(T) = 0,  x''(T) = 0
    M = np.array([
        [T ** 5 / 2.0,       T ** 4,        T ** 3],
        [5.0 * T ** 4 / 2.0, 4.0 * T ** 3,  3.0 * T ** 2],
        [10.0 * T ** 3,      12.0 * T ** 2, 6.0 * T],
    ])
    out = np.empty((n, x0.shape[0]))
    for d in range(x0.shape[0]):
        A, B, C = np.linalg.solve(
            M, np.array([-(v0[d] * T + x0[d]), -v0[d], 0.0]))
        out[:, d] = (A / 2.0 * t ** 5 + B * t ** 4 + C * t ** 3
                     + v0[d] * t + x0[d])
    return out


def _quat_log(q: np.ndarray) -> np.ndarray:
    """Rotation -> axis*angle (3-vector), the linear space to blend rotations in."""
    q = q / (np.linalg.norm(q) or 1.0)
    if q[3] < 0.0:
        q = -q
    v = q[:3]
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3)
    return v / n * (2.0 * math.atan2(n, float(q[3])))


def _quat_exp(w: np.ndarray) -> np.ndarray:
    """axis*angle -> quaternion (x, y, z, w)."""
    ang = float(np.linalg.norm(w))
    if ang < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = w / ang
    s = math.sin(ang / 2.0)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, math.cos(ang / 2.0)])


def blend_seam_inertial(r15: dict, blend_frames: int = 8) -> None:
    """Close the loop seam with true inertialization.

    For each curve: measure the offset from frame 0 to the last frame, and the
    RATE that offset is changing, then decay both to zero across `blend_frames`
    with `_inertial_decay`. Rotations are handled in axis-angle log space so the
    quintic operates on a linear quantity.

    Longer blends suit slower motion -- a graceful idle needs more frames to
    absorb the same mismatch than a fast dance does, which is why idles get
    their own `--idle-blend-frames`.
    """
    n = r15["frameCount"]
    if blend_frames <= 1 or n < blend_frames + 3:
        return

    for part in r15["parts"].values():
        q = part_quats(part)
        if q is not None:
            # offset that takes frame 0 to the last frame, in log space
            off0 = _quat_log(_qmul(q[-1], _quat_conj(q[0])))
            # how fast that offset is changing across the seam
            off1 = _quat_log(_qmul(q[-2], _quat_conj(q[1])))
            v0 = off1 - off0
            dec = _inertial_decay(blend_frames, off0, v0)
            for i in range(blend_frames):
                q[i] = _qmul(_quat_exp(dec[i]), q[i])
            set_part_quats(part, q)

        p = part_pos(part)
        if p is not None:
            off0 = p[-1] - p[0]
            off1 = p[-2] - p[1]
            v0 = off1 - off0
            dec = _inertial_decay(blend_frames, off0, v0)
            p[:blend_frames] = p[:blend_frames] + dec
            set_part_pos(part, p)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def blend_seam(r15: dict, blend_frames: int = 4) -> None:
    """Ease the residual seam so frame 0 continues smoothly from the last frame.

    Same idea as `pipeline._inertial_blend_loop_seam`, applied here because a
    cut cycle is never perfectly closed. Rotations are slerped and positions
    lerped from the end pose toward the start pose over the first few frames,
    with a smoothstep decay so both value and velocity stay continuous.
    """
    n = r15["frameCount"]
    if blend_frames <= 0 or n < blend_frames + 2:
        return
    w = 1.0 - _smoothstep(np.arange(blend_frames) / float(blend_frames))
    for part in r15["parts"].values():
        q = part_quats(part)
        if q is not None:
            for i in range(blend_frames):
                q[i] = _slerp(q[i], q[-1], np.array([w[i]]))[0]
            set_part_quats(part, q)
        p = part_pos(part)
        if p is not None:
            off = p[-1] - p[0]
            for i in range(blend_frames):
                p[i] = p[i] + off * w[i]
            set_part_pos(part, p)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# --------------------------------------------------------- slot derivation ---
def retime(r15: dict, factor: float) -> dict:
    """Resample every curve to `factor` x the frame count (slerp rotations).

    Used to make a run out of a walk, or slow a walk into a climb. Resampling
    rather than relabelling frameRate keeps the emitted rate at 30 Hz, which
    matters: `CurveAnimNumericalDataValid` requires keys at least 1/70 x 0.85
    apart, so relabelling to a high rate can fail validation.
    """
    out = copy.deepcopy(r15)
    n = r15["frameCount"]
    m = max(2, int(round(n * factor)))
    src = np.linspace(0.0, n - 1, m)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = src - i0
    for part in out["parts"].values():
        q = part_quats(part)
        if q is not None:
            newq = np.empty((m, 4))
            for k in range(m):
                newq[k] = _slerp(q[i0[k]], q[i1[k]], np.array([frac[k]]))[0]
            set_part_quats(part, newq)
        p = part_pos(part)
        if p is not None:
            set_part_pos(part, p[i0] * (1 - frac)[:, None] + p[i1] * frac[:, None])
    out["frameCount"] = m
    return out


def pitch_body(r15: dict, degrees: float) -> dict:
    """Rotate the whole body about its lateral (X) axis.

    Swimming is the motivating case: a walk cycle pitched ~75 deg forward reads
    as a horizontal swimmer whose limbs are already moving in the pack's style.
    Applied to LowerTorso only -- every other part inherits it through the R15
    chain, so limb motion relative to the body is untouched.
    """
    out = copy.deepcopy(r15)
    lt = out["parts"].get("LowerTorso")
    if not lt:
        return out
    q = part_quats(lt)
    if q is None:
        return out
    rot = _axis_angle_quat((1.0, 0.0, 0.0), math.radians(degrees))
    set_part_quats(lt, _qmul(np.broadcast_to(rot, q.shape), q))
    return out


def add_joint_offset(r15: dict, part_name: str, axis: tuple[float, float, float],
                     degrees: float, *, ramp: bool = False) -> dict:
    """Pre-multiply a constant rotation onto one joint.

    Climbing is the motivating case: take the walk's leg cycle and raise both
    arms overhead so it reads as hauling up a ladder, without regenerating.
    `ramp` fades the offset in across the clip instead of applying it flat.
    """
    out = copy.deepcopy(r15)
    part = out["parts"].get(part_name)
    if not part:
        return out
    q = part_quats(part)
    if q is None:
        return out
    n = q.shape[0]
    if ramp:
        amounts = np.linspace(0.0, degrees, n)
        newq = np.stack([
            _qmul(_axis_angle_quat(axis, math.radians(a)), q[i])
            for i, a in enumerate(amounts)])
    else:
        rot = _axis_angle_quat(axis, math.radians(degrees))
        newq = _qmul(np.broadcast_to(rot, q.shape), q)
    set_part_quats(part, newq)
    return out


def phase_shift(r15: dict, frames: int) -> dict:
    """Roll every curve by `frames`, so a derived slot starts on the other foot."""
    out = copy.deepcopy(r15)
    for part in out["parts"].values():
        q = part_quats(part)
        if q is not None:
            set_part_quats(part, np.roll(q, frames, axis=0))
        p = part_pos(part)
        if p is not None:
            set_part_pos(part, np.roll(p, frames, axis=0))
    return out


def freeze_pose(r15: dict, frame: int, n_frames: int) -> dict:
    """Hold one frame's pose for `n_frames`. Base for a fall or a jump apex."""
    out = copy.deepcopy(r15)
    for part in out["parts"].values():
        q = part_quats(part)
        if q is not None:
            set_part_quats(part, np.repeat(q[frame:frame + 1], n_frames, axis=0))
        p = part_pos(part)
        if p is not None:
            set_part_pos(part, np.repeat(p[frame:frame + 1], n_frames, axis=0))
    out["frameCount"] = n_frames
    return out


def add_vertical_arc(r15: dict, peak_studs: float) -> dict:
    """Add a single sine hump to LowerTorso Y -- a hop for the jump slot.

    Kept small: `CurveAnimPartsRotateOnlyIfBones` caps Position drift from t=0
    at 0.5 studs for ANIMATION uploads, and LowerTorso is NOT exempt there.
    """
    out = copy.deepcopy(r15)
    lt = out["parts"].get("LowerTorso")
    if not lt:
        return out
    p = part_pos(lt)
    if p is None:
        return out
    n = p.shape[0]
    p[:, 1] = p[:, 1] + peak_studs * np.sin(np.linspace(0.0, math.pi, n))
    set_part_pos(lt, p)
    return out


def set_loop_priority(r15: dict, loop: bool, priority: str = "Core") -> dict:
    r15["loop"] = bool(loop)
    r15["priority"] = priority
    return r15


def max_position_drift(r15: dict) -> tuple[float, str]:
    """Worst Position-curve displacement from t=0, over all parts, in studs.

    `CurveAnimPartsRotateOnlyIfBones` caps this at 0.5 for ANIMATION uploads
    with NO LowerTorso exemption -- unlike the emote path, where the module
    skips boneless clips entirely. Derived slots add translation, so check it.
    """
    worst, worst_part = 0.0, "-"
    for name, part in r15["parts"].items():
        p = part_pos(part)
        if p is None:
            continue
        d = float(np.linalg.norm(p - p[0], axis=1).max())
        if d > worst:
            worst, worst_part = d, name
    return worst, worst_part


# ------------------------------------------------------------------- cli -----
def _cmd_cycles(args) -> int:
    root = Path(args.tree)
    clips = sorted({p.parent for p in root.rglob("generated.npz")})
    if not clips:
        print(f"no clips under {root}")
        return 1
    print(f"{'clip':<16} {'start':>5} {'len':>4} {'sec':>5} {'seam_p':>7} "
          f"{'seam_v':>7} {'asym':>6} {'dutyL':>6} {'dutyR':>6} {'travel':>7}")
    for c in clips:
        cy = detect_cycle(c, min_len=args.min_len, max_len=args.max_len)
        if cy is None:
            print(f"{c.name:<16}  no usable cycle")
            continue
        print(f"{c.name:<16} {cy.start:>5} {cy.length:>4} "
              f"{cy.length / FPS:>5.2f} {cy.seam_pos:>7.3f} {cy.seam_vel:>7.2f} "
              f"{cy.stride_asym:>6.3f} {cy.duty_l:>6.2f} {cy.duty_r:>6.2f} "
              f"{cy.travel_m:>7.2f}")
    return 0


# Slot table. `asset` is the Marketplace asset type; `sv` is the StringValue
# name the Animate script looks up. Only JumpAnimation may have Loop = false
# (CurveAnimLoopingRequired exempts it and nothing else).
SLOTS = [
    ("idle",     "IdleAnimation",  "idle",     True),
    ("idle2",    "IdleAnimation",  "idle",     True),
    ("walk",     "WalkAnimation",  "walk",     True),
    ("run",      "RunAnimation",   "run",      True),
    ("jump",     "JumpAnimation",  "jump",     False),
    ("fall",     "FallAnimation",  "fall",     True),
    ("climb",    "ClimbAnimation", "climb",    True),
    ("swim",     "SwimAnimation",  "swim",     True),
    ("swimidle", "SwimAnimation",  "swimidle", True),
]


def _cmd_build(args) -> int:
    """Cut cycles from the generated clips, derive the rest, write per-slot json."""
    src = Path(args.tree)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def load(name: str) -> tuple[dict, Path]:
        d = src / name
        f = d / "r15.json"
        if not f.is_file():
            raise SystemExit(f"missing {f}")
        return json.loads(f.read_text()), d

    # --- locomotion: cut one stride out of the free-running motion ---------
    built: dict[str, dict] = {}
    for slot, clip_name in (("walk", args.walk), ("run", args.run)):
        r15, d = load(clip_name)
        cy = detect_cycle(d, min_len=args.min_len, max_len=args.max_len)
        if cy is None:
            raise SystemExit(f"no usable cycle in {clip_name}")
        cut = slice_cycle(r15, cy.start, cy.length)
        blend_seam_inertial(cut, blend_frames=args.blend_frames)
        built[slot] = cut
        print(f"[pack] {slot:<9} <- {clip_name} frames {cy.start}..{cy.end} "
              f"({cy.length / FPS:.2f}s) seam={cy.seam_pos:.3f} "
              f"asym={cy.stride_asym:.3f} travel={cy.travel_m:.2f}m")

    # Walk and run are cross-faded CONCURRENTLY by the Animate script between
    # 6.4 and 12.8 studs/s, both Looped at the same Speed. If their lengths
    # differ their phases drift and you see four legs. So force a shared length.
    nw, nr = built["walk"]["frameCount"], built["run"]["frameCount"]
    if nw != nr:
        # Anchor on WALK's length by default. The blend is calibrated around
        # walk: at 6.4 studs/s runSpeed is 0.5 and timeWarp is 1.0, so walk is
        # the slot playing at its authored rate. Retiming to the SHORTER of the
        # two would speed both up, and these clips already overshoot target --
        # compressing makes skate worse, not better.
        target = {"walk": nw, "run": nr,
                  "min": min(nw, nr), "max": max(nw, nr)}[args.shared_length]
        for slot in ("walk", "run"):
            n = built[slot]["frameCount"]
            if n != target:
                built[slot] = retime(built[slot], target / n)
                print(f"[pack] retimed {slot} {n} -> "
                      f"{built[slot]['frameCount']} frames "
                      f"(shared-length={args.shared_length}) so walk/run "
                      f"phases stay locked")

    # --- idles ------------------------------------------------------------
    for slot, clip_name in (("idle", args.idle), ("idle2", args.idle2 or args.idle)):
        r15, d = load(clip_name)
        # Search for a window whose endpoints already agree, rather than
        # slicing arbitrarily and forcing them together with the blend.
        # idle1 can be capped calmer than idle2: the two play weighted 5/10, and
        # a busy primary idle reads as distracting for a slow, elegant style.
        cap = (args.idle_max_energy if slot == "idle"
               else args.idle2_max_energy or args.idle_max_energy)
        win = detect_idle_window(d, min_sec=args.idle_min_seconds,
                                 max_sec=args.idle_seconds,
                                 min_energy=args.idle_min_energy,
                                 max_energy=cap,
                                 min_pose_offset=args.idle_min_pose_offset)
        if win is None:
            # Falling back to an arbitrary window reintroduces exactly the bug
            # the search exists to prevent: blend_seam force-closes mismatched
            # endpoints over 4 frames, which reads as "dances, then snaps to
            # bind pose". Say so loudly rather than shipping it quietly.
            n = r15["frameCount"]
            start = int(0.25 * n)
            length = min(int(args.idle_seconds * FPS), n - start - 1)
            print(f"[pack] ** WARNING {slot}: no window passed the motion "
                  f"filters, falling back to an ARBITRARY {length / FPS:.1f}s "
                  f"window. This idle will likely snap to bind pose at the "
                  f"seam. Regenerate the idle with a prompt that says "
                  f"'dancing' rather than 'standing'.")
        else:
            start, length = win
        cut = slice_cycle(r15, start, length)
        blend_seam_inertial(cut, blend_frames=args.idle_blend_frames)
        if slot == "idle2" and args.idle2 is None:
            # Only one idle generated: make the second a slower variant so the
            # weighted pair does not look like the same clip twice.
            cut = retime(cut, 1.35)
        built[slot] = cut
        print(f"[pack] {slot:<9} <- {clip_name} {cut['frameCount'] / FPS:.2f}s")

    # --- derived slots ----------------------------------------------------
    # Each is the walk or idle restyled, so the whole pack shares limb style.
    walk, idle = built["walk"], built["idle"]

    # Climb: the leg cycle reads as hauling upward once both arms go overhead.
    #
    # Sign matters and is not obvious: POSITIVE rotation about the arm's local X
    # raises the hand, negative lowers it. Measured on this rig, with the base
    # walk's hands already at Y +1.13 (the zombie prompt raises one arm):
    #   -95 deg -> hand Y -0.60   (arms pushed DOWN -- the original bug)
    #   +30 deg -> +1.74
    #   +60 deg -> +2.17          (about where reach saturates)
    #
    # 30 deg is a CEILING, not a preference. Past ~35 the shoulder rotation
    # enters a region where the Euler XYZ decomposition build_rbxm.lua writes is
    # unstable, and the validator's 70 Hz cubic interpolation of those three
    # channels explodes between keys. Measured max part speed on this clip:
    #   +20 -> 8.5    +30 -> 8.8    +40 -> 232.4    +60 -> 231.7
    # against the 45 studs/s CurveAnimSpeedBounded limit. The keyframes
    # themselves are fine at every angle -- this only shows up at 70 Hz, which
    # is why lua/ugc_validate.lua has to be the gate and not the Python port.
    #
    # Keep the per-frame arm swing from the walk; this only biases the whole
    # cycle upward, so the alternation that sells a climb survives.
    climb = retime(walk, 1.25)
    climb = add_joint_offset(climb, "LeftUpperArm", (1, 0, 0), 30.0)
    climb = add_joint_offset(climb, "RightUpperArm", (1, 0, 0), 30.0)
    built["climb"] = climb

    # Swim: do NOT bake in a body pitch. The engine already rotates the
    # character ~90 deg to horizontal when swimming, so pitching the curves too
    # double-rotates and the swimmer ends up face-down or inverted. Ship the
    # limb cycle as-is and let the engine orient it. `pitch_body` is kept in the
    # module for other uses but is deliberately not applied here.
    built["swim"] = retime(walk, 1.15)
    built["swimidle"] = retime(idle, 1.0)

    # Fall: hold a mid-stride pose, arms up, so it reads as tumbling.
    fall = freeze_pose(walk, walk["frameCount"] // 2, int(1.0 * FPS))
    fall = add_joint_offset(fall, "LeftUpperArm", (1, 0, 0), 45.0)
    fall = add_joint_offset(fall, "RightUpperArm", (1, 0, 0), 45.0)
    built["fall"] = fall

    # Jump: short hop. JUMP_ANIM_DURATION is 0.31s for R15, which is the window
    # the engine expects before it transitions to `fall`.
    jump = retime(freeze_pose(walk, 0, int(0.34 * FPS)), 1.0)
    jump = add_joint_offset(jump, "LeftUpperArm", (1, 0, 0), 50.0)
    jump = add_joint_offset(jump, "RightUpperArm", (1, 0, 0), 50.0)
    built["jump"] = add_vertical_arc(jump, 0.25)

    # --- stamp Loop/Priority, check drift, write --------------------------
    manifest = []
    for slot, asset, sv, loop in SLOTS:
        r15 = built[slot]
        set_loop_priority(r15, loop, "Core")
        drift, part = max_position_drift(r15)
        warn = ""
        if drift > 0.5:
            warn = (f"  ** drift {drift:.3f} > 0.5 on {part}: "
                    f"CurveAnimPartsRotateOnlyIfBones applies to ANIMATION "
                    f"uploads with no LowerTorso exemption")
        d = out / slot
        d.mkdir(parents=True, exist_ok=True)
        (d / "r15.json").write_text(json.dumps(r15, separators=(",", ":")))
        manifest.append({
            "slot": slot, "asset_type": asset, "string_value": sv,
            "loop": loop, "priority": "Core",
            "frames": r15["frameCount"],
            "seconds": round(r15["frameCount"] / FPS, 3),
            "max_position_drift": round(drift, 4),
            "drift_part": part,
        })
        print(f"[pack] wrote {slot:<9} {r15['frameCount']:>3}f "
              f"{r15['frameCount'] / FPS:>5.2f}s loop={str(loop):<5} "
              f"drift={drift:.3f}{warn}")
    (out / "pack_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[pack] {len(manifest)} slots -> {out}")
    print(f"[pack] next: python3 python/build_rbxm.py --in {out.name} "
          f"--repo-root {out.parent} --per-clip --no-per-category --no-corpus")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cycles", help="report cycle detection over a tree")
    c.add_argument("tree", type=str)
    c.add_argument("--min-len", type=int, default=20)
    c.add_argument("--max-len", type=int, default=90)
    c.set_defaults(func=_cmd_cycles)

    b = sub.add_parser("build", help="cut cycles, derive slots, write per-slot json")
    b.add_argument("tree", type=str)
    b.add_argument("--out", type=str, required=True)
    b.add_argument("--walk", type=str, required=True)
    b.add_argument("--run", type=str, required=True)
    b.add_argument("--idle", type=str, required=True)
    b.add_argument("--idle2", type=str, default=None)
    b.add_argument("--shared-length", choices=["walk", "run", "min", "max"],
                   default="walk",
                   help="Which slot's cycle length walk and run are both "
                        "retimed to. They must match or their phases drift "
                        "during the concurrent cross-fade. Default 'walk' "
                        "because the blend is calibrated at walk's authored "
                        "rate; 'min' speeds both up and worsens skate.")
    b.add_argument("--idle-seconds", type=float, default=3.0,
                   help="Longest idle window to consider.")
    b.add_argument("--idle-blend-frames", type=int, default=10,
                   help="Seam blend length for idles, in frames. Idles use TRUE "
                        "inertialization (velocity-aware quintic decay), not the "
                        "smoothstep offset decay the locomotion slots use, "
                        "because slow motion needs velocity continuity to avoid "
                        "an abrupt direction change at the seam. Slower styles "
                        "want longer: 10 frames is a third of a second.")
    b.add_argument("--idle-min-pose-offset", type=float, default=0.06,
                   help="Minimum MEDIAN absolute distance from the bind pose "
                        "over the idle window. rest_collapse is a ratio and "
                        "cannot see this: a window hugging bind still scores 0.80. "
                        "Measured -- 0.012 reads as standing at bind, 0.08 reads "
                        "as a held dance pose, 0.15 is a full dance pose.")
    b.add_argument("--idle-max-energy", type=float, default=1e9,
                   help="Energy CEILING for the primary idle window. A busy idle "
                        "is distracting for a slow style -- cap it to force a "
                        "calmer window. Reference: elegant catwalk windows run "
                        "0.35-0.50, salsa/hip-hop 1.0-1.7.")
    b.add_argument("--idle2-max-energy", type=float, default=None,
                   help="Separate ceiling for idle2, so the pair can be calm + "
                        "livelier rather than two of the same.")
    b.add_argument("--idle-min-energy", type=float, default=0.60,
                   help="Absolute energy floor for an idle window. 0.60 was "
                        "calibrated on salsa and hip-hop, which are energetic "
                        "styles (0.93-0.98 whole-clip). A graceful style is "
                        "genuinely quieter -- an elegant catwalk idle measures "
                        "0.22-0.38 -- so lower this per style (~0.35). The "
                        "relative floor (60% of the clip's own liveliest "
                        "window) is the style-independent guard and still "
                        "prevents picking the static parts.")
    b.add_argument("--idle-min-seconds", type=float, default=1.5,
                   help="Shortest idle window to consider. The search picks the "
                        "window in [min, max] whose endpoints already match.")
    b.add_argument("--blend-frames", type=int, default=6,
                   help="Seam blend length for locomotion slots, in frames. "
                        "Inertial (velocity-aware) like the idles; raised from 4 "
                        "since the old smoothstep decay only matched position.")
    b.add_argument("--min-len", type=int, default=20)
    b.add_argument("--max-len", type=int, default=90)
    b.set_defaults(func=_cmd_build)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
