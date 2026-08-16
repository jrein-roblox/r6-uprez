#!/usr/bin/env python3
"""Reject clips that Roblox's UGC emote validation would fail.

This is a port of the numeric animation checks in Roblox's own validator, so
we stop paying the 80-Robux upload fee to discover a clip is out of bounds.
Source of truth (read, not guessed):

  game-engine/Client/RCCService/internalscripts/Packages/_Index/UGCValidation/
    UGCValidation/validationFolders/CurveAnimLengthBounded/...      -- duration
    UGCValidation/validationFolders/CurveAnimBoundsValid/...        -- too low / too far
    UGCValidation/validationFolders/CurveAnimSpeedBounded/...       -- teleport guard
    UGCValidation/validationFolders/CurveAnimPositionBounded/...    -- joint separation
    UGCValidation/util/CurveAnimationFrameCalculator.lua            -- frame sampling
    UGCValidation/util/AssetCalculator.lua                          -- the FK

Thresholds are the FFlag/FString defaults from UGCValidation/flags/:

  UGCValidationMaxAnimationLength      10      s     max clip length
  UGCValidateCurveAnimationMinLength   0       s     length must exceed this
  UGCValidateAnimationHeightTol        -3.1    studs min Y of ANY part vs HRP
  UGCValidationMaxAnimationBounds      25      studs max |position| of ANY part
  UGCValidationMaxAnimationDeltas      1.5     studs per 1/30 s  => 45 studs/s
  UGCValidateMaxAnimationMovement      0.3     studs max joint translation
  UGCValidateMaxAnimationFPS           70            validator sample rate

Two things to know about how this differs from the real validator:

  1. **Frame sampling.** Roblox resamples the curves at 1/70 s with the
     FloatCurves' Cubic interpolation. We evaluate at the clip's native 30 Hz
     keys, which is exact *at* keys but misses any inter-key cubic overshoot.
     `--margin` (default 0.95) gates at a fraction of each limit to absorb
     that. Speed is compared in studs/second so the sample rate cancels out.

  2. **Positions are HRP-relative.** `calculateAnimFramesAtOrigin` puts the
     HumanoidRootPart at the origin and runs FK down the R15 tree, so a part's
     `Position.Magnitude` is its distance from the HRP. That is why the height
     tolerance is -3.1: a standard R15 foot sits about 3 studs below the HRP,
     and anything lower means the animation drove the body through the floor.

**Use the right reference body — this is the single easiest thing to get
wrong.** `RigBuilder.createDefaultCharacter` calls
`Players:CreateHumanoidModelFromDescription(HumanoidDescription.new(), R15)`,
and that loads `rbxasset://avatar/characterCagedHSRV18.rbxm`, NOT the classic
blocky `characterR15.rbxm`. The difference is decisive:

    characterR15          rest foot Y = -2.20   -> 0.90 studs of headroom
    characterCagedHSRV18  rest foot Y = -2.85   -> 0.25 studs of headroom

Against the -3.1 floor that is the difference between a comfortable margin and
a razor's edge. Measured on the shipped v1 batch: the classic rig says 96.8%
of clips pass, the caged rig says 83.5%. Salsa clips land at -2.90..-2.96 —
inside the limit, but by less than 0.2 studs, which is why a knee bend a few
frames long is enough to fail an upload. `data/r15-rig-attachments-hsrv18.json`
is the correct default; `data/r15-rig-attachments.json` (classic) is kept only
for comparison.

Rest geometry is dumped by `lua/dump_r15_rig.lua`. `characterCagedHSRV18.rbxm`
ships in the built Studio app resources, not the source content tree:
`.../RobloxStudio.app/Contents/Resources/content/avatar/`. Regenerate with:

    roblox-cli run --run lua/dump_r15_rig.lua --load.asRobloxScript \
      --fs.readwrite <content/avatar> --fs.readwrite <out dir> \
      --lua.globals RIG_PATH=<...>/characterR15.rbxm \
      --lua.globals OUT_PATH=<...>/data/r15-rig-attachments.json

Usage:
    python3 python/ugc_validate.py work/latin_v3/salsa_v00
    python3 python/ugc_validate.py --report work/latin_v3
    python3 python/ugc_validate.py --report work/emotes --json /tmp/v1_ugc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
# The body the validator actually composes — see the module docstring.
DEFAULT_RIG_JSON = REPO_ROOT / "data" / "r15-rig-attachments-hsrv18.json"
CLASSIC_RIG_JSON = REPO_ROOT / "data" / "r15-rig-attachments.json"

# ---------------------------------------------------------- UGC thresholds --
MAX_ANIMATION_LENGTH = 10.0     # UGCValidationMaxAnimationLength
MIN_ANIMATION_LENGTH = 0.0      # UGCValidateCurveAnimationMinLength (exclusive)
HEIGHT_TOL = -3.1               # UGCValidateAnimationHeightTol
MAX_BOUNDS = 25.0               # UGCValidationMaxAnimationBounds
MAX_DELTA_PER_30HZ_FRAME = 1.5  # UGCValidationMaxAnimationDeltas
MAX_SPEED_STUDS_PER_SEC = MAX_DELTA_PER_30HZ_FRAME * 30.0   # 45
MAX_JOINT_MOVEMENT = 0.3        # UGCValidateMaxAnimationMovement
# UGCValidateBodyPartTranslationMaxDistanceHundredths = 50 -> 0.5 studs.
# Confirmed unset in production, so the compiled default applies.
# Enforced by CurveAnimPartsRotateOnlyIfBones via
# CurveAnimTranslationUtils.translationExceedsThreshold: max displacement of a
# part's Position curve FROM ITS t=0 VALUE, over every key. Unlike
# CurveAnimPositionBounded this does NOT exempt LowerTorso -- which is where
# `_fold_root_into_lower_torso` puts all of our root motion, so for this
# pipeline it is the binding constraint. It is the wall-clipping guard: it stops
# an animation displacing the body away from the HumanoidRootPart.
#
# Measured on shipped work: every clip in v1/v2/latin_v3 exceeds it
# (0.62-3.07 studs). A sigma-5 high-pass on the LowerTorso Position curve --
# keep per-step sway, drop slow drift -- brings them to 0.21-0.37.
# NOT ENFORCED in practice: clips measuring 0.78-1.33 studs passed real
# validation, which fits the code path -- CurveAnimPartsRotateOnlyIfBones
# `continue`s for EMOTE_ANIMATION uploads with no bone folders, which is us.
# Kept as a REPORTED metric (set --max-translation-drift to gate on it) so we
# notice if that ever changes. 0.5 is the source default of
# UGCValidateBodyPartTranslationMaxDistanceHundredths (confirmed unset in prod).
MAX_PART_TRANSLATION_DRIFT = 0.5
ENFORCE_TRANSLATION_DRIFT = False
VALIDATOR_FPS = 70              # UGCValidateMaxAnimationFPS

# CurveAnimPositionBounded skips LowerTorso: it is the one part allowed to
# carry translation, which is exactly where our pipeline folds root motion
# (`_fold_root_into_lower_torso` in export_r15.py).
POSITION_EXEMPT_PARTS = {"LowerTorso"}

# fullBodyFromHumanoidRootPartAssetHierarchy, AssetCalculator.lua:51
R15_PARENT: dict[str, str] = {
    "LowerTorso": "HumanoidRootPart",
    "UpperTorso": "LowerTorso",
    "Head": "UpperTorso",
    "LeftUpperArm": "UpperTorso",
    "LeftLowerArm": "LeftUpperArm",
    "LeftHand": "LeftLowerArm",
    "RightUpperArm": "UpperTorso",
    "RightLowerArm": "RightUpperArm",
    "RightHand": "RightLowerArm",
    "LeftUpperLeg": "LowerTorso",
    "LeftLowerLeg": "LeftUpperLeg",
    "LeftFoot": "LeftLowerLeg",
    "RightUpperLeg": "LowerTorso",
    "RightLowerLeg": "RightUpperLeg",
    "RightFoot": "RightLowerLeg",
}
ROOT_PART = "HumanoidRootPart"


# ------------------------------------------------------------- CFrame math --
# A CFrame is (position p, rotation R). Compose: A*B = (Ap + AR@Bp, AR@BR).
def _compose(ap, ar, bp, br):
    return ap + ar @ bp, ar @ br


def _inverse(p, r):
    rt = r.T
    return -(rt @ p), rt


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])


# ------------------------------------------------------------------- rig ----
@dataclass
class Rig:
    """Standard R15 rest geometry: per-part attachment CFrames."""
    attach_pos: dict[str, dict[str, np.ndarray]]
    attach_rot: dict[str, dict[str, np.ndarray]]
    # child part -> the RigAttachment name shared with its parent
    attach_to_parent: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "Rig":
        raw = json.loads(Path(path).read_text())
        pos: dict[str, dict[str, np.ndarray]] = {}
        rot: dict[str, dict[str, np.ndarray]] = {}
        names: dict[str, set[str]] = {}
        for part, info in raw.items():
            pos[part], rot[part], names[part] = {}, {}, set()
            for att, comps in info["attachments"].items():
                c = list(map(float, comps))
                pos[part][att] = np.array(c[0:3])
                # GetComponents() returns x,y,z then the rotation matrix
                # row-major: R00,R01,R02,R10,R11,R12,R20,R21,R22.
                rot[part][att] = np.array(c[3:12]).reshape(3, 3)
                if att.endswith("RigAttachment"):
                    names[part].add(att)

        # Derive child->parent attachment names by intersecting each pair's
        # RigAttachment sets. On the standard R15 rig the shared name is
        # unique, so this reproduces ConstantsInterface.getRigAttachmentToParent
        # without hardcoding a second table that could drift.
        to_parent: dict[str, str] = {}
        for child, parent in R15_PARENT.items():
            if child not in names or parent not in names:
                continue
            shared = names[child] & names[parent]
            if len(shared) != 1:
                raise ValueError(
                    f"expected exactly 1 shared RigAttachment between {child} "
                    f"and {parent}, got {sorted(shared)}"
                )
            to_parent[child] = shared.pop()
        missing = set(R15_PARENT) - set(to_parent)
        if missing:
            raise ValueError(f"rig json missing parts: {sorted(missing)}")
        return cls(pos, rot, to_parent)


# ---------------------------------------------------------------- checks ----
@dataclass
class Violation:
    check: str
    part: str
    time: float
    value: float
    limit: float

    def describe(self) -> str:
        return (f"{self.check}: {self.part} = {self.value:.3f} "
                f"(limit {self.limit:.3f}) at t={self.time:.2f}s")


@dataclass
class UgcResult:
    name: str
    frames: int
    length: float
    min_y: float
    min_y_part: str
    max_bounds: float
    max_bounds_part: str
    max_speed: float
    max_speed_part: str
    max_joint_move: float
    max_joint_move_part: str
    max_translation_drift: float
    max_translation_drift_part: str
    verdict: str
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == "pass"

    def reasons(self) -> list[str]:
        return [v.describe() for v in self.violations]


def forward_kinematics(r15: dict, rig: Rig) -> tuple[dict[str, np.ndarray], float]:
    """Per-part HRP-relative positions, shape (T, 3), plus clip length.

    Mirrors AssetCalculator.calculatePartTransformInHierarchy:
        child = parent * attach(parent) * animTransform(child) * attach(child)^-1
    with the HumanoidRootPart pinned at the origin.
    """
    parts = r15["parts"]
    n = int(r15["frameCount"])
    fps = float(r15["frameRate"])

    # Per-part animation transform per frame.
    anim_pos: dict[str, np.ndarray] = {}
    anim_rot: dict[str, np.ndarray] = {}
    for name, p in parts.items():
        if "rotX" in p:
            q = np.stack([np.asarray(p["rotX"], dtype=float),
                          np.asarray(p["rotY"], dtype=float),
                          np.asarray(p["rotZ"], dtype=float),
                          np.asarray(p["rotW"], dtype=float)], axis=1)
            anim_rot[name] = np.stack([_quat_to_matrix(*q[i]) for i in range(n)])
        else:
            anim_rot[name] = np.repeat(np.eye(3)[None], n, axis=0)
        if "posX" in p:
            anim_pos[name] = np.stack([np.asarray(p["posX"], dtype=float),
                                       np.asarray(p["posY"], dtype=float),
                                       np.asarray(p["posZ"], dtype=float)], axis=1)
        else:
            anim_pos[name] = np.zeros((n, 3))

    root = r15.get("root")
    if root and "posX" in root:
        anim_pos[ROOT_PART] = np.stack([np.asarray(root["posX"], dtype=float),
                                        np.asarray(root["posY"], dtype=float),
                                        np.asarray(root["posZ"], dtype=float)], axis=1)
    if root and "rotX" in root:
        q = np.stack([np.asarray(root["rotX"], dtype=float),
                      np.asarray(root["rotY"], dtype=float),
                      np.asarray(root["rotZ"], dtype=float),
                      np.asarray(root["rotW"], dtype=float)], axis=1)
        anim_rot[ROOT_PART] = np.stack([_quat_to_matrix(*q[i]) for i in range(n)])

    def anim(part: str, i: int):
        p = anim_pos.get(part)
        r = anim_rot.get(part)
        return (p[i] if p is not None else np.zeros(3),
                r[i] if r is not None else np.eye(3))

    # Walk the tree in dependency order (parents before children).
    order: list[str] = []
    def visit(part: str):
        if part in order:
            return
        parent = R15_PARENT.get(part)
        if parent and parent != ROOT_PART:
            visit(parent)
        order.append(part)
    for part in R15_PARENT:
        visit(part)

    out: dict[str, np.ndarray] = {p: np.zeros((n, 3)) for p in [ROOT_PART] + order}
    for i in range(n):
        cf: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        cf[ROOT_PART] = anim(ROOT_PART, i)
        out[ROOT_PART][i] = cf[ROOT_PART][0]
        for part in order:
            parent = R15_PARENT[part]
            att = rig.attach_to_parent[part]
            pp, pr = cf[parent]
            # parent * attach(parent)
            cp, cr = _compose(pp, pr, rig.attach_pos[parent][att],
                              rig.attach_rot[parent][att])
            # * animTransform(child)
            ap, ar = anim(part, i)
            cp, cr = _compose(cp, cr, ap, ar)
            # * attach(child)^-1
            ip, ir = _inverse(rig.attach_pos[part][att], rig.attach_rot[part][att])
            cp, cr = _compose(cp, cr, ip, ir)
            cf[part] = (cp, cr)
            out[part][i] = cp

    # animLength = max key time. Keys are written at (i-1)/fps in
    # build_rbxm.lua, so the last key sits at (n-1)/fps.
    length = (n - 1) / fps if n > 0 else 0.0
    return out, length


def validate_clip(clip_dir: Path, rig: Rig, margin: float = 0.95) -> UgcResult:
    """Apply the ported UGC animation checks to one clip directory."""
    clip_dir = Path(clip_dir)
    r15_path = clip_dir / "r15.json"
    if not r15_path.is_file():
        raise FileNotFoundError(f"missing {r15_path}")
    r15 = json.loads(r15_path.read_text())

    positions, length = forward_kinematics(r15, rig)
    n = int(r15["frameCount"])
    fps = float(r15["frameRate"])
    violations: list[Violation] = []

    # --- CurveAnimLengthBounded ---
    if length <= MIN_ANIMATION_LENGTH or length > MAX_ANIMATION_LENGTH * margin:
        violations.append(Violation("length", "-", length, length,
                                    MAX_ANIMATION_LENGTH * margin))

    # --- CurveAnimBoundsValid: min Y and max |position| over every part/frame ---
    min_y, min_y_part = float("inf"), "-"
    max_b, max_b_part = 0.0, "-"
    y_limit = HEIGHT_TOL * margin        # margin shrinks the allowed depth
    b_limit = MAX_BOUNDS * margin
    for part, pos in positions.items():
        ys = pos[:, 1]
        i = int(np.argmin(ys))
        if ys[i] < min_y:
            min_y, min_y_part = float(ys[i]), part
        if ys[i] < y_limit:
            violations.append(Violation("part too low", part, i / fps,
                                        float(ys[i]), y_limit))
        mags = np.linalg.norm(pos, axis=1)
        j = int(np.argmax(mags))
        if mags[j] > max_b:
            max_b, max_b_part = float(mags[j]), part
        if mags[j] > b_limit:
            violations.append(Violation("part too far", part, j / fps,
                                        float(mags[j]), b_limit))

    # --- CurveAnimSpeedBounded: per-part speed in studs/second ---
    max_s, max_s_part = 0.0, "-"
    s_limit = MAX_SPEED_STUDS_PER_SEC * margin
    for part, pos in positions.items():
        if n < 2:
            continue
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
        k = int(np.argmax(speed))
        if speed[k] > max_s:
            max_s, max_s_part = float(speed[k]), part
        if speed[k] > s_limit:
            violations.append(Violation("speed too fast", part, (k + 1) / fps,
                                        float(speed[k]), s_limit))

    # --- CurveAnimPositionBounded: joint translation, LowerTorso exempt ---
    max_m, max_m_part = 0.0, "-"
    m_limit = MAX_JOINT_MOVEMENT * margin
    for name, p in r15["parts"].items():
        if name in POSITION_EXEMPT_PARTS or "posX" not in p:
            continue
        mag = np.linalg.norm(np.stack([np.asarray(p["posX"], dtype=float),
                                       np.asarray(p["posY"], dtype=float),
                                       np.asarray(p["posZ"], dtype=float)], axis=1),
                             axis=1)
        k = int(np.argmax(mag))
        if mag[k] > max_m:
            max_m, max_m_part = float(mag[k]), name
        if mag[k] > m_limit:
            violations.append(Violation("joint separation", name, k / fps,
                                        float(mag[k]), m_limit))

    # --- CurveAnimPartsRotateOnlyIfBones: translation drift from t=0 ---
    # LowerTorso is NOT exempt here.
    max_d, max_d_part = 0.0, "-"
    d_limit = MAX_PART_TRANSLATION_DRIFT * margin
    for name, pp in r15["parts"].items():
        if "posX" not in pp:
            continue
        xyz = np.stack([np.asarray(pp["posX"], dtype=float),
                        np.asarray(pp["posY"], dtype=float),
                        np.asarray(pp["posZ"], dtype=float)], axis=1)
        drift = np.linalg.norm(xyz - xyz[0], axis=1)
        k = int(np.argmax(drift))
        if drift[k] > max_d:
            max_d, max_d_part = float(drift[k]), name
        if drift[k] > d_limit and ENFORCE_TRANSLATION_DRIFT:
            violations.append(Violation("translation drift", name, k / fps,
                                        float(drift[k]), d_limit))

    return UgcResult(
        name=clip_dir.name,
        frames=n,
        length=round(length, 3),
        min_y=round(min_y, 3), min_y_part=min_y_part,
        max_bounds=round(max_b, 3), max_bounds_part=max_b_part,
        max_speed=round(max_s, 2), max_speed_part=max_s_part,
        max_joint_move=round(max_m, 4), max_joint_move_part=max_m_part,
        max_translation_drift=round(max_d, 4),
        max_translation_drift_part=max_d_part,
        verdict="pass" if not violations else "fail",
        violations=violations,
    )


def lift_to_clear_floor(
    r15: dict,
    rig: Rig,
    *,
    safety: float = 0.015,
    max_lift: float = 0.35,
    margin: float = 1.0,
) -> tuple[float, float, float]:
    """Raise LowerTorso just enough to lift the lowest part above the UGC floor.

    `CurveAnimBoundsValid` rejects a clip if ANY part's HRP-relative Y drops
    below -3.1 studs. In practice clips miss by a hair -- a measured example
    failed at -3.221, i.e. by 0.121 studs, on 20 of 150 frames. Rejecting a
    whole generation over 3 cm is wasteful when a constant offset fixes it.

    Every part hangs off LowerTorso in the R15 chain, so adding a constant to
    `LowerTorso.posY` translates the entire body up by that amount and leaves
    the motion otherwise identical. It is applied as a CONSTANT (not per-frame)
    precisely so the animation is shifted, not reshaped -- a time-varying
    correction would flatten whatever crouch took it below the floor.

    The cost is that the character sits `lift` studs higher for the whole clip,
    so feet float by that much at their lowest. At ~0.1 studs (3 cm) that is
    invisible; `max_lift` refuses anything beyond 0.35 studs, where the clip is
    genuinely too low and should be rejected instead of fudged.

    Mutates `r15` in place. Returns (lift_applied, min_y_before, min_y_after).
    """
    pos, _ = forward_kinematics(r15, rig)
    min_y = min(float(p[:, 1].min()) for p in pos.values())
    # Clear the SAME floor the gate will check. Lifting to the raw -3.1 while
    # the gate runs at margin 0.95 (-2.945) leaves the clip still failing.
    target = HEIGHT_TOL * margin + safety
    if min_y >= target:
        return 0.0, min_y, min_y

    lift = target - min_y
    if lift > max_lift:
        # Too deep to fudge -- report without touching the clip so the caller
        # can reject it honestly.
        return -lift, min_y, min_y

    lt = r15.get("parts", {}).get("LowerTorso")
    if not lt or "posY" not in lt:
        return 0.0, min_y, min_y
    lt["posY"] = [float(v) + lift for v in lt["posY"]]
    root = r15.get("root")
    if root and "posY" in root:
        root["posY"] = [float(v) + lift for v in root["posY"]]
    return lift, min_y, min_y + lift


# ------------------------------------------------------------------- cli ----
def _find_clips(root: Path) -> list[Path]:
    return sorted({p.parent for p in Path(root).rglob("r15.json")})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", type=Path, help="Clip dir, or tree root with --report")
    p.add_argument("--report", action="store_true", help="Validate every clip under target")
    p.add_argument("--rig-json", type=Path, default=DEFAULT_RIG_JSON)
    p.add_argument("--margin", type=float, default=0.95,
                   help="Gate at this fraction of each limit (default 0.95) to "
                        "absorb the validator's 70 Hz cubic resampling")
    p.add_argument("--json", type=Path, default=None, help="Write results as JSON")
    args = p.parse_args(argv)

    rig = Rig.load(args.rig_json)

    if not args.report:
        r = validate_clip(args.target, rig, margin=args.margin)
        print(json.dumps({**asdict(r), "violations": r.reasons()}, indent=2))
        return 0 if r.ok else 1

    clips = _find_clips(args.target)
    if not clips:
        print(f"[ugc_validate] no clips under {args.target}")
        return 1

    rows: list[UgcResult] = []
    for c in clips:
        try:
            rows.append(validate_clip(c, rig, margin=args.margin))
        except Exception as e:
            print(f"  ERR {c.name}: {e}", file=sys.stderr)

    passed = [r for r in rows if r.ok]
    print(f"\n[ugc_validate] {len(passed)}/{len(rows)} pass "
          f"({100.0 * len(passed) / max(1, len(rows)):.1f}%)  margin={args.margin}\n")

    hdr = (f"{'clip':<30} {'len':>5} {'minY':>7} {'maxDist':>8} "
           f"{'maxSpd':>7} {'jointMv':>8} {'drift':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r.name:<30} {r.length:>5.2f} {r.min_y:>7.2f} {r.max_bounds:>8.2f} "
              f"{r.max_speed:>7.1f} {r.max_joint_move:>8.3f} "
              f"{r.max_translation_drift:>7.3f}  {r.verdict}"
              + (f" — {r.reasons()[0]}" if r.violations else ""))

    tally: dict[str, int] = {}
    for r in rows:
        for v in r.violations:
            tally[v.check] = tally.get(v.check, 0) + 1
    if tally:
        print("\nviolations by check:")
        for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>4}  {k}")

    print("\nheadroom (worst clip vs limit):")
    print(f"  length        {max(r.length for r in rows):7.2f} / {MAX_ANIMATION_LENGTH}")
    print(f"  min Y         {min(r.min_y for r in rows):7.2f} / {HEIGHT_TOL}")
    print(f"  max distance  {max(r.max_bounds for r in rows):7.2f} / {MAX_BOUNDS}")
    print(f"  max speed     {max(r.max_speed for r in rows):7.1f} / {MAX_SPEED_STUDS_PER_SEC}")
    print(f"  joint move    {max(r.max_joint_move for r in rows):7.3f} / {MAX_JOINT_MOVEMENT}")
    print(f"  transl drift  {max(r.max_translation_drift for r in rows):7.3f} / {MAX_PART_TRANSLATION_DRIFT}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            [{**asdict(r), "violations": r.reasons()} for r in rows], indent=2))
        print(f"\n[ugc_validate] wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
