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
- `FileSystemService:WriteInstances` writes a `CurveAnimation` Instance with one Folder per part containing `RotationCurve` + (optional) `Vector3Curve`.
- Skips the HumanoidRootPart folder when `data.root` is nil (i.e., when fold-root is on).

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
│   └── soma_tpose.bvh          # SOMA T-pose bind (copied from motion-matching)
├── lua/
│   ├── extract_pose.lua        # Stage 1
│   └── build_rbxm.lua          # Stage 5 (copied from motion-matching, modified to skip HRP folder)
└── python/
    ├── pipeline.py             # Top-level orchestrator
    ├── extract_pose.py         # Stage 1 driver (subprocesses roblox-cli)
    ├── roblox_to_kimodo.py     # Stage 2
    ├── run_kimodo.py           # Stage 3 (copied)
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
