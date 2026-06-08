"""POST /generate - Submit a generation job."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

router = APIRouter()


class PromptSegment(BaseModel):
    text: str
    start_time: float
    end_time: float


class JointCFrame(BaseModel):
    name: str
    pos: List[float]   # position [x, y, z] in ground-centered Roblox studs
    quat: List[float]  # quaternion [qx, qy, qz, qw] in world space


class Constraint(BaseModel):
    effector: str
    time: float
    chain_world_cframes: Optional[List[JointCFrame]] = None


class GenerateRequest(BaseModel):
    prompts: List[PromptSegment]
    constraints: List[Constraint] = []
    duration: float = 3.0
    looped: bool = False
    loop_offset: float = 0.0  # seconds into pass 1 to use as the loop pivot pose
    seed: Optional[int] = None
    cfg_weight: float = 5.0
    diffusion_steps: int = 100
    target_rig: str = "r15"


class GenerateResponse(BaseModel):
    job_id: str
    status: str


# PascalCase (Roblox) → snake_case (pipeline chain defs)
PASCAL_TO_SNAKE = {
    "LowerTorso": "lower_torso", "UpperTorso": "upper_torso",
    "LeftUpperArm": "left_upper_arm", "LeftLowerArm": "left_lower_arm",
    "LeftHand": "left_hand", "RightUpperArm": "right_upper_arm",
    "RightLowerArm": "right_lower_arm", "RightHand": "right_hand",
    "LeftUpperLeg": "left_upper_leg", "LeftLowerLeg": "left_lower_leg",
    "LeftFoot": "left_foot", "RightUpperLeg": "right_upper_leg",
    "RightLowerLeg": "right_lower_leg", "RightFoot": "right_foot",
    "Head": "head",
}

# Plugin effector names → Kimodo constraint types
EFF_MAP = {
    "LeftFoot": ("left-foot", "LeftFoot"),
    "RightFoot": ("right-foot", "RightFoot"),
    "LeftHand": ("left-hand", "LeftHand"),
    "RightHand": ("right-hand", "RightHand"),
    "Hips": ("end-effector", "Hips"),  # Hips = SOMA root (joint 0)
}

# Position scale:
# XZ: HRP-relative, corrected for retarget hrp_scale
# Y: floor-relative, direct stud→meter (Kimodo Y=0 = floor)
CM_TO_STUD = 0.03
STUD_TO_METER = 0.30
SOMA_BIND_CHAIN = 2.643
TARGET_CHAIN = 3.6693
# Geometric value is TARGET_CHAIN / SOMA_BIND_CHAIN ≈ 1.388. Overridden to 1.1
# experimentally to reduce foot sliding (lower scale = less root XZ travel per
# leg swing). Used for BOTH the retarget AND the constraint XZ round-trip so
# constraints still land where placed.
HRP_SCALE = 1.1
STUD_TO_KIMODO_XZ = 1.0 / (100.0 * CM_TO_STUD * HRP_SCALE)  # ≈0.303 at scale 1.1
STUD_TO_KIMODO_Y = STUD_TO_METER  # 0.30


def build_kimodo_constraints(constraints: List[Constraint], total_duration: float):
    """Convert plugin constraints to Kimodo constraint format.

    Uses roblox_to_kimodo._retarget_chain_quats for proper R15→SOMA30 retargeting.
    """
    import numpy as np
    import roblox_to_kimodo as r2k
    from vendor.quat import to_scaled_angle_axis

    SOMA30_N_JOINTS = 30
    fps = 30
    # Kimodo produces round(duration*fps) frames, indices 0..n_frames-1.
    # (No +1: a 3s clip = 90 frames, last valid index is 89.)
    n_frames = int(round(total_duration * fps))

    kimodo_constraints = []
    for c in constraints:
        if c.effector not in EFF_MAP:
            print(f"[generate] Skipping unsupported effector: {c.effector}")
            continue

        cframes = c.chain_world_cframes
        if not cframes:
            print(f"[generate] No CFrame data for constraint, skipping")
            continue

        ctype, joint_name = EFF_MAP[c.effector]
        frame_idx = int(round(c.time * fps))
        frame_idx = max(0, min(frame_idx, n_frames - 1))

        # Convert CFrames to Kimodo space
        chain_pos_kimodo = {}
        chain_quat_kimodo = {}
        for cf in cframes:
            snake = PASCAL_TO_SNAKE.get(cf.name, cf.name)
            px, py, pz = cf.pos
            qx, qy, qz, qw = cf.quat
            chain_pos_kimodo[snake] = np.array([
                -px * STUD_TO_KIMODO_XZ,
                 py * STUD_TO_KIMODO_Y,
                -pz * STUD_TO_KIMODO_XZ,
            ])
            chain_quat_kimodo[snake] = np.array([qw, -qx, qy, -qz])  # wxyz

        # Hips → root2d constraint (XZ position + heading). This is the
        # canonical Kimodo way to constrain the root, and loads in the viewer.
        if c.effector == "Hips":
            lt = chain_pos_kimodo.get("lower_torso")
            if lt is None:
                continue
            smooth_2d = [[float(lt[0]), float(lt[2])]]
            entry = {
                "type": "root2d",
                "frame_indices": [frame_idx],
                "smooth_root_2d": smooth_2d,
            }
            # Heading from hips Y rotation (cos, sin of yaw)
            ltq = chain_quat_kimodo.get("lower_torso")
            if ltq is not None:
                w, x, y, z = ltq  # wxyz
                # yaw from quaternion (rotation about Y)
                yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + x * x))
                entry["global_root_heading"] = [[float(np.cos(yaw)), float(np.sin(yaw))]]
            print(f"[generate] Hips root2d at frame {frame_idx}: xz={smooth_2d}")
            kimodo_constraints.append(entry)
            continue

        # Retarget chain rotations to SOMA30 using pipeline's proven function
        local_rots = np.zeros((1, SOMA30_N_JOINTS, 3))

        R15_CHAINS = getattr(r2k, 'R15_CHAINS', None)
        if R15_CHAINS and ctype in R15_CHAINS:
            chain_def = R15_CHAINS[ctype]
            chain_rots = {
                src_key: chain_quat_kimodo[src_key]
                for _, src_key in chain_def
                if src_key and src_key in chain_quat_kimodo
            }
            if chain_rots:
                try:
                    quats = r2k._retarget_chain_quats(chain_def, chain_rots)
                    for soma_idx, q in quats.items():
                        if q[0] < 0:
                            q = -q
                        local_rots[0, soma_idx] = to_scaled_angle_axis(q)
                except Exception as e:
                    print(f"[generate] Retarget failed: {e}")

        # Root position from LowerTorso
        if "lower_torso" in chain_pos_kimodo:
            root_pos = chain_pos_kimodo["lower_torso"][None, :]
        else:
            root_pos = np.array([[0.0, 0.9, 0.0]])

        smooth_2d = root_pos[:, [0, 2]]

        print(f"[generate] frame={frame_idx}, non-zero rots={np.count_nonzero(local_rots)}, root_y={root_pos[0,1]:.3f}")
        kimodo_constraints.append({
            "type": ctype,
            "frame_indices": [frame_idx],
            "local_joints_rot": local_rots.tolist(),
            "root_positions": root_pos.tolist(),
            "smooth_root_2d": smooth_2d.tolist(),
            "joint_names": [joint_name],
        })

    return kimodo_constraints


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    from main import job_manager, WORK_DIR

    def run_generation(job):
        import prompt_pipeline
        import pipeline as parent_pipeline
        import run_kimodo
        import export_r15

        seed = req.seed if req.seed is not None else random.randrange(2**31 - 1)

        prompt_text = ". ".join(seg.text for seg in req.prompts)
        durations = [seg.end_time - seg.start_time for seg in req.prompts]
        duration_str = " ".join(f"{d:.2f}" for d in durations)
        total_duration = sum(durations)

        clip_dir = WORK_DIR / "_romotion"
        clip_dir.mkdir(parents=True, exist_ok=True)

        job.message = "Running Kimodo generation..."
        job.progress = 0.1

        r15_json_path = clip_dir / "r15.json"

        kimodo_constraints = build_kimodo_constraints(req.constraints, total_duration) if req.constraints else []

        # ── Stage A: produce generated.bvh ──
        # Call the kimodo helpers directly (NOT prompt_pipeline.main, which
        # also runs the rbxm build over the whole work/ folder — the plugin
        # builds the CurveAnimation itself from r15.json).
        import kimodo_warm

        def _run_pass(out_name: str, constraints, cfg_constraint_w: float):
            """One kimodo pass (warm, in-process). With constraints → separated CFG."""
            constraints_path = None
            if constraints:
                constraints_path = clip_dir / "constraints.json"
                constraints_path.write_text(json.dumps(constraints, indent=2))
                cfg_type = "separated"
                cfg_weight = [req.cfg_weight, cfg_constraint_w]
            else:
                cfg_type = "regular"
                cfg_weight = [req.cfg_weight]

            kimodo_warm.generate_bvh(
                clip_dir,
                prompt=prompt_text,
                duration_str=duration_str,
                seed=seed,
                diffusion_steps=req.diffusion_steps,
                cfg_type=cfg_type,
                cfg_weight=cfg_weight,
                num_transition_frames=5,
                out_name=out_name,
                constraints_path=constraints_path,
            )

        if req.looped:
            # Two-pass loop synthesis. Pass 1 (with any user constraints) gives
            # a pose to pin; pass 2 pins frame-0 pose at both endpoints AND keeps
            # the user constraints so looping + constraints work together.
            job.message = "Loop pass 1/2..."
            _run_pass("generated_pass1", kimodo_constraints, 2.0)

            n_loop = int(round(total_duration * 30))
            sample_frame = max(0, min(int(round(req.loop_offset * 30)), n_loop - 1))
            loop_constraints = prompt_pipeline._build_loop_constraints(
                clip_dir / "generated_pass1.npz", n_frames=n_loop, sample_frame=sample_frame,
            )
            merged = loop_constraints + kimodo_constraints  # pins + user constraints

            job.message = "Loop pass 2/2..."
            job.progress = 0.4
            _run_pass("generated", merged, 4.0)
        else:
            _run_pass("generated", kimodo_constraints, 2.0)

        # ── Stage B: retarget generated.bvh → r15.json + ground ──
        job.message = "Retargeting to R15..."
        job.progress = 0.7
        bvh_path = clip_dir / "generated.bvh"
        export_r15.set_rig(parent_pipeline.RIG if req.target_rig == "r15" else "r15plus")
        hrp_to_ankle = prompt_pipeline._RTHRO_HRP_TO_ANKLE
        parent_pipeline._retarget_bvh_to_r15_json(
            bvh_path, r15_json_path,
            root_motion=False, source_n_frames=0, loop_passes=1,
            looped=req.looped,
            inertial_blend_frames=6 if req.looped else 0,
            hrp_scale=HRP_SCALE,  # experimental override (see top of file)
            target_rig=req.target_rig,
        )
        result = json.loads(r15_json_path.read_text())
        offset = prompt_pipeline._ground_y(
            result, bvh_path, "first",
            target_hrp_rest_y=prompt_pipeline._RTHRO_HRP_REST_Y,
            target_hrp_to_ankle=hrp_to_ankle,
        )
        if offset != 0.0:
            r15_json_path.write_text(json.dumps(result, separators=(",", ":")))

        job.message = "Done"
        job.progress = 1.0
        job.result = {
            "animation": result,
            "seed": seed,
            "duration_s": total_duration,
        }
        return job.result

    job = job_manager.submit(run_generation)
    return GenerateResponse(job_id=job.id, status=job.status.value)
