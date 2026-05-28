# r6-uprez pipeline (working notes)

End-to-end: Roblox animation asset id → Kimodo full-body regeneration → R15 CurveAnimation rbxm. Auto-detects R6 vs R15 from the source clip's bone names. R6 animations get spine/clavicle/finger/toe articulation invented by Kimodo, anchored by sparse hand/foot effector constraints derived from the R6 limb tips.

## One command

```sh
python3 python/pipeline.py --asset-id <id> --out work --name <name> --prompt "<text>"
```

Outputs land in `work/<name>/`:
- `pose.json` — sampled per-frame world CFrames
- `constraints.json` — Kimodo input
- `meta.json` — bookkeeping
- `generated.{bvh,npz}` — Kimodo output
- `r15.json` — R15 retarget result
- `r15.rbxm` — final CurveAnimation

To re-run later stages without re-extracting / re-running Kimodo:

```sh
python3 python/pipeline.py --asset-id <id> --out work --name <name> \
    --skip pose --skip constraints --skip kimodo --prompt "..."
```

## Stage-by-stage

### Stage 1: pose extraction (`lua/extract_pose.lua`)
- `InsertService:LoadAsset(asset_id)`; if it returns a wrapper Animation, recurse on `AnimationId`.
- Bone-name discriminator picks R6 vs R15 (see `R15_DISCRIMINATING` / `R6_DISCRIMINATING` tables).
- Spawns a character via `Players:CreateHumanoidModelFromDescription(desc, RigType)` at HRP `(0, 5, 0)` — identity rotation.
- `KeyframeSequenceProvider:RegisterKeyframeSequence(clip)` then `Animator:LoadAnimation`. Works for both `KeyframeSequence` and `CurveAnimation`.
- For each output frame: `track.TimePosition = (i*dt) % source_duration`, `animator:StepAnimations(0)`, sample world CFrames.
- Reads `clip.Loop` (boolean inherited from `AnimationClip`); writes `looped`, `source_duration_s`, `source_n_frames`, `loop_passes` to pose.json.
- **Loop support:** when `LOOP_PASSES > 1` and the clip is looped, samples N consecutive cycles by wrapping `TimePosition` modulo `source_duration`. The Python side later trims back to the middle cycle.
- Bones sampled per frame: HRP, four effector tips (R6: synthetic from `Arm.CFrame * CFrame.new(0,-1,0)`; R15: direct LeftHand/RightHand/LeftFoot/RightFoot), and chain bones (R6: Torso, Left/Right Arm, Left/Right Leg; R15: LowerTorso + each arm/leg segment).

### Stage 2: pose → Kimodo constraints (`python/roblox_to_kimodo.py`)

#### Coord conversion (`_to_kimodo`)
- Subtract `y_offset_studs` from raw Y (defaults to lowest foot Y across the clip → lowest foot lands at Kimodo Y=0).
- Multiply by `stud_to_m` (default 0.30).
- 180° Y rotation: `pos.x→-x, pos.z→-z`; `quat (w,x,y,z) → (w,-x,y,-z)`. Roblox's −Z forward maps to Kimodo's +Z forward; left/right anatomical mapping preserved.

#### Keyframe selection (`effector_helpers.detect_velocity_extremes`)
- Per effector, picks both **valleys and peaks** of XZ-speed independently (NMS within each type so they don't compete; allows valley + peak as close as 1-2 frames apart).
- Adds a window `{0, 1, ..., W-1} ∪ {F-W, ..., F-1}` to every effector's keyframe set when the clip is looped (default `W=2`) so velocity matches at the loop seam.

#### Chain retarget (`_retarget_chain_quats`)
- `D[k] = source_world * inv(SOMA_BIND_CORRECTION[k])`; bind = identity for legs/torso, `Rz(±90°)` for arms (SOMA T-pose extends arms ±X but Roblox arms hang −Y at rest).
- `local_q[soma_j] = inv(D[parent_in_chain]) * D[child_in_chain]` per chain entry.
- Chain entries with `src_key=None` (R6 has no shin/forearm/hand) inherit `D[parent]` so the missing SOMA joint comes out identity → the upstream rigid limb propagates orientation through the implied subjoints.
- After collecting all frames' local quats per joint, `_unroll_quats_inplace` flips any quat with negative 4D dot vs. its predecessor — keeps the temporal path on a single hemisphere.
- `_quat_to_axis_angle(q, canonical=False)` does the final conversion. Canonical=False allows |aa| > π so the unroll is preserved through the conversion (Kimodo's `axis_angle_to_matrix` uses Rodrigues, which is correct for any θ).

#### Root: Torso (not HRP)
- `root_positions[t] = chain_converted[torso].pos[t]` (or `lower_torso` for R15). R6 keeps HRP anchored and moves all body motion through the Torso Motor6D — using HRP as root would discard dance bobs etc. Using Torso captures both locomotion and in-place body motion.

#### Loop endpoint match
- When `looped=True` and `F ≥ 2`, `_load_pose` overwrites the last frame's channel data with the first frame's data, so Kimodo gets matching position+rotation at frames 0 and F-1 by construction.

### Stage 3: Kimodo (`python/run_kimodo.py`)

`kimodo_gen "<prompt>" --model Kimodo-SOMA-RP --duration <secs> --constraints constraints.json --output generated --bvh --diffusion_steps 100 --cfg_type separated --cfg_weight <text> <constraint>`. Forces `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`. Kimodo's `motion_correction` C++ extension isn't built locally — produces an "uncorrected motion" warning and skips foot-skate cleanup; harmless.

### Stage 4: BVH → R15 (`pipeline.py:_retarget_bvh_to_r15_json`)

1. `export_r15.set_rig("soma")`, `export_r15.retarget(BIND_BVH, bvh_path, hardcoded_bind=True)` produces a dict with `root` (HRP world) and `parts.<name>` (Motor6D local) curves.
2. `--hrp-scale 0.72` scales root + LowerTorso XZ to compensate for SOMA-vs-R15 leg-length mismatch.
3. **Middle-cycle trim** (`_trim_middle_cycle`): if `loop_passes > 1`, slice arrays to indices `[(loop_passes//2)*s, (loop_passes//2)*s + s + 1)` where `s = source_n_frames - 1`. Includes the redundant final loop-point frame.
4. **Root-motion fold** (`_fold_root_into_lower_torso`, default ON): compute `delta_HRP[t] = inv(T_HRP[0]) * T_HRP[t]`, write `new_LT_local[t] = delta_HRP[t] * old_LT_local[t]`, **delete** `result["root"]` so the lua skips the HumanoidRootPart curve. Studio plays the rbxm at the spawn HRP pose. `--root-motion` keeps HRP curves for locomotion clips.
5. **Inertial blend at loop seam** (`_inertial_blend_loop_seam`, default 8 frames): for looped clips, fake an inertial blend over the first N frames using the clip's last frame as the "previous" pose. Position curves use `+offset * (1 - smoothstep(t))`. Rotation curves use SLERP-from-identity by `decay(t)` of the offset rotation, composed with the original frame's rotation. Hemisphere-aligns `last_q` to `first_q` before computing the offset to avoid long-arc paths. Walks every curve once afterwards (`_unroll`) negating any quat with a negative dot to its predecessor — keeps Roblox's RotationCurve interpolator on the short arc everywhere.

### Stage 5: rbxm (`python/build_rbxm.py` + `lua/build_rbxm.lua`)
- `FileSystemService:WriteInstances` writes a `CurveAnimation` Instance with one Folder per part containing **paired** `RotationCurve` + `Vector3Curve` (translation).
- **R15 hierarchy nesting**: folders are parented in rig order — LowerTorso under HumanoidRootPart, UpperTorso under LowerTorso, Head + UpperArms under UpperTorso, etc. Done via `PART_PARENT` map + a two-pass build (materialize then reparent). Unknown joint names fall back to direct children of the CurveAnimation root.
- **HumanoidRootPart folder always exists** (so child chains can hang off it) but only carries curves when `data.root` is non-nil. With root-motion fold on, HRP curves are skipped and Studio plays the rbxm at the spawn HRP pose.
- **Translation+Rotation always emitted together**: there is a downstream retargeting bug where a folder with only one of the two curves is treated as missing the joint entirely. We synthesize an identity Vector3Curve (0,0,0) when the JSON only has rotation, and identity RotationCurve (0,0,0,1) when it only has translation.
- **Embedded rig metadata**: `data/RigData.rbxm` is loaded once at startup via `FileSystemService:LoadInstances` and a `:Clone()` of every top-level instance is parented under each emitted CurveAnimation. The retarget runtime uses these instances to map curve names to live Motor6Ds; without them, downstream consumers like `AnimationClipProvider` have no way to resolve which joint a "LeftHand" folder drives.

## Prompt pipeline (`python/prompt_pipeline.py`)

Synthesize R15 motion from a text prompt — no source asset, no pose
extraction, no Stage 2 constraint synthesis. Reuses Stages 4 + 5 of the
asset-id pipeline (BVH→R15 retarget, rbxm build).

```sh
python3 python/prompt_pipeline.py --prompt "<text>" --out work --name <name> --duration <secs>
```

Outputs land in `work/<name>/`:
- `generated.bvh` / `generated.npz` — Kimodo output
- `meta.json` — bookkeeping (prompt, duration, seed, model, …)
- `r15.json` — R15 retarget result
- `r15.rbxm` — final CurveAnimation

### Stages

#### Stage A: Kimodo (`_run_kimodo_promptonly`)
Direct invocation of `kimodo_gen` with no constraints in the single-pass case. CFG defaults to `regular` with weight 5.0. The seed defaults to a fresh random int per run (printed for reproducibility).

#### Stage B: BVH → R15 (`pipeline._retarget_bvh_to_r15_json`)
Same as the asset-id pipeline, with two prompt-specific knobs:
- `--hrp-scale`: auto-derived from `target_hrp_to_ankle / soma_bind_hip_to_ankle`. For Rthro Rig (3.6693 stud) and the soma bind (2.643 stud) ⇒ ≈ 1.388. The historical 0.72 baseline (empirical, over-translated stock R15 by ~19%) was retired in favor of geometry.
- `--ground-y-mode` (default `first`): post-pass that shifts `LowerTorso.posY` so the rig's feet sit on the floor at the chosen anchor frame. Uses a proportional leg-chain model `R15_chain[i] = target_hrp_to_ankle × (soma_chain[i] / soma_bind_chain)`. Modes: `first` (frame 0), `min` (no penetration), `off`. Manual nudge via `--ground-y-bias`.

#### Stage C: rbxm
Identical to the asset-id pipeline's Stage 5.

### Two-pass loop synthesis (`--loop`)

Forces the start and end pose to match so the clip can loop cleanly. Doubles the kimodo wall-clock cost.

1. **Pass 1**: prompt-only generation → `generated_pass1.{bvh,npz}`.
2. **`_build_loop_constraints`** reads pass-1 NPZ at `--loop-offset` (default `0.0` s = frame 0) and emits five constraints pinning that pose at frames `[0, F-1]`:
   - `Root2D` — pins root XZ (zeroed, so pass 2 starts at the origin) at both endpoints.
   - Four `EndEffector` constraints (LeftHand / RightHand / LeftFoot / RightFoot) — share `local_joints_rot` (whole-body SOMASkeleton77 axis-angle) + `root_positions` (Y kept, XZ zeroed). Without all four, kimodo's per-effector world-position loss leaves the un-pinned limbs free to drift to a different pose at the loop point.
3. **Pass 2** re-runs kimodo with `--constraints constraints.json`. CFG is forced to `separated` so the constraint weight can be amplified independently of the text weight. Default `--loop-cfg-constraint-weight = 4.0` (kimodo default for separated constraint weight is 2.0). Pass 0 to skip the override and stay on the user's `--cfg-type`. The text weight reuses `--cfg-weight`.
4. **Inertial blend** (default 0.2 s = 6 frames at 30 fps; override with `--inertial-blend-seconds`): applied to the pass-2 retarget output to smooth the residual seam mismatch — kimodo's loss pulls toward the constraint but doesn't honor it perfectly.

Source frame for the loop pin: pass `--loop-offset 0.5` (seconds into pass 1) to skip past kimodo's idle ramp-in and pin on a mid-motion pose. Pass-1 root XZ at the sampled frame is zeroed regardless of how far the character had drifted by then.

NPZ rotations not BVH: pass-1 pose is sourced from `local_rot_mats` (kimodo's native `(F, 77, 3, 3)` SOMASkeleton77) via `scipy.Rotation.as_rotvec`. BVH local rotations use a different parent-frame convention; round-tripping through Euler→quat→axis-angle introduced a ~90° X-axis flip on the limbs.

### Multi-prompt (chained-segment generation)

Kimodo's CLI supports chained prompts via `.`-separation in `--prompt` and a space-separated per-segment duration list in `--duration`:

```
--prompt "a person waves hello. then they sit down." --duration "2.0 3.0"
```

The pipeline parses `--duration` (`_parse_durations`) into per-segment seconds, sums for total clip length, and forwards the raw string verbatim to `kimodo_gen` (which validates count vs. prompt segments). Single-token durations apply to every segment (kimodo handles the broadcast).

`--num-transition-frames` (default 5) controls the cross-fade window between segments. Higher = smoother blend at the cost of less time in each pose. Forwarded to both pass 1 and pass 2 (under `--loop`). Ignored for single-segment prompts.

### Prompt-pipeline flags summary

| Flag | Default | Purpose |
|---|---|---|
| `--prompt` | required | Text prompt; `.`-separated for chained segments. |
| `--duration` | `"3.0"` | Seconds per segment. Single value or quoted space-separated list. |
| `--num-transition-frames` | 5 | Cross-fade window (frames @ 30 fps) between chained segments. |
| `--seed` | random | Random by default; logged for reproducibility. Pass 1 + pass 2 share. |
| `--cfg-type` | `regular` | Kimodo CFG mode (`regular` / `separated` / `nocfg`). |
| `--cfg-weight` | 5.0 | Prompt weight (also pass-2 text weight under `--loop`). |
| `--cfg-text-weight` / `--cfg-constraint-weight` | 2.0 / 2.0 | Used only when `--cfg-type=separated`. |
| `--root-motion` | off | When off, fold HRP into LowerTorso (HRP stays at spawn). |
| `--inertial-blend` / `--inertial-blend-seconds` | 0 / — | Frames or seconds of seam smoothing. Seconds wins if both set. |
| `--looped` | off | Mark output as looping (required for `--inertial-blend` to take effect). |
| `--loop` | off | Two-pass loop synthesis (implies `--looped` + 0.2 s blend). |
| `--loop-cfg-constraint-weight` | 4.0 | Pass-2 separated constraint weight; 0 disables the override. |
| `--loop-offset` | 0.0 | Seconds into pass 1 to sample as the loop pivot pose. |
| `--ground-y-mode` | `first` | Foot-grounding post-pass anchor (`first` / `min` / `off`). |
| `--target-hrp-rest-y` / `--target-hrp-to-ankle` | 4.1197 / 3.6693 | Rthro Rig defaults; pass `2.0 / 1.6` for stock R15. |
| `--ground-y-bias` | 0.0 | Manual studs offset after the proportional grounding model. |
| `--hrp-scale` | auto | Geometric leg-length ratio (target_hrp_to_ankle / soma_bind). |
| `--skip` | — | `kimodo` / `retarget` / `rbxm`; reuse existing output. Repeatable. |

## Defaults that matter

| Knob | Default | Purpose |
|---|---|---|
| `--scale` | 0.30 | studs → meters; sized to land 5-stud-tall character at ~1.5 m for SOMA |
| `--y-offset-mode` | `auto` | shifts so lowest foot lands at Kimodo Y=0 |
| `--min-duration` | 0.0 | no looping by default |
| `--loop-passes` | 3 | for `Loop=true` clips, samples 3× then trims to middle cycle |
| `--loop-window` | 2 | pin first 2 + last 2 frames of looped sample to source pose for velocity match at seam |
| `--inertial-blend` | 8 | fake inertial blend over first 8 frames of trimmed cycle to mask residual seam mismatch |
| `--diffusion-steps` | 100 | Kimodo's iterations |
| `--cfg-weight` | 2.0 | constraint adherence (separated CFG) |
| `--root-motion` | off | when off, fold HRP into LowerTorso so character stays at spawn |
| `--foot-min-separation` | 8 | NMS frame distance for foot keyframe picks |
| `--hand-min-separation` | 8 | same for hands |
| `--dedupe` | off | shifts colliding effector frames; only needed for the Kimodo editor's hand-collision quirk |

## Test assets

| ID | Description | Loop | Notes |
|---|---|---|---|
| 180426354 | R6 walk | yes | leg swing peaks + valleys, minimal torso motion |
| 182435998 | R6 overhead wave/dance | yes | exercises torso bob (LT.posY captures it) and 180° arm singularity |
| 180436334 | R6 short dance | yes | 0.46 s source, 3-pass tests short-clip handling |
| 129423131 | R6 motion | no | Loop=false → single pass, no trim, no inertial blend |

## Known unsolved

**Dense-mode shoulder spin near 180° arm rotations** (e.g., 182435998 right arm overhead). Source quaternions smoothly cross `w=0` but the canonical axis-angle representation (`|aa| ≤ π`) flips axis directions there. Sparse-mode keyframes are far enough apart that Kimodo's diffusion smooths around it; dense mode forces faithful tracking of every frame's representation flicker, which Kimodo renders as visible spin.

What was tried (and removed):
- Swing-twist decomposition stripping rotation around `−Y` arm bone axis (`_remove_bone_twist`)
- Gaussian smoothing of source arm quats (`_smooth_quats`)
- Round-robin frame assignment across effectors (was a workaround for editor bug, not the spin)

What's still in the pipeline that helped (kept):
- Per-joint quaternion unroll across time + non-canonical axis-angle conversion (continuity through 180° at the cost of |aa| sometimes exceeding π)
- Hemisphere alignment in the inertial blend's offset computation

**v2 fix would distribute the rotation across more SOMA chain joints** (Hips swivel + Shoulder + Arm) instead of letting one joint sit at 180°. Splitting a 180° rotation into multiple smaller rotations keeps every joint far from the singularity.

## Files

```
r6-uprez/
├── data/
│   ├── soma_tpose.bvh          # SOMA T-pose bind (copied from motion-matching)
│   └── RigData.rbxm            # Joints/attachments/R15 reference rig embedded in every CurveAnimation
├── lua/
│   ├── extract_pose.lua        # Stage 1
│   └── build_rbxm.lua          # Stage 5 (R15 hierarchy nesting, paired curves, RigData embed)
└── python/
    ├── pipeline.py             # Asset-id orchestrator
    ├── prompt_pipeline.py      # Prompt-only orchestrator (Stages A/B/C; --loop two-pass; multi-prompt)
    ├── extract_pose.py         # Stage 1 driver (subprocesses roblox-cli)
    ├── roblox_to_kimodo.py     # Stage 2
    ├── run_kimodo.py           # Stage 3 (copied; supports duration_override)
    ├── export_r15.py           # Stage 4 retarget (copied)
    ├── soma_rig.py             # SOMA → R15 mapping + bind table (copied)
    ├── build_rbxm.py           # Stage 5 driver (copied)
    ├── effector_helpers.py     # detect_velocity_extremes, dedupe, SOMA30 tables
    └── vendor/quat.py          # quaternion ops (copied)
```

## Resume checklist

1. `cd /Users/jrein/git/roblox/jrein/r6-uprez/`
2. Pick a test asset from the table above (or a new one).
3. Run `python3 python/pipeline.py --asset-id <id> --out work --name test_<id> --prompt "<desc>"`.
4. Open `work/test_<id>/r15.rbxm` in Studio on an R15 character.
5. If results look wrong, check `work/test_<id>/{constraints.json, generated.bvh, r15.json}` for the failure stage.
