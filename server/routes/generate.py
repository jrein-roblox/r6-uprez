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


class Constraint(BaseModel):
    """A single sparse constraint authored as a character-space effector target.

    All positions are ground/character-relative Roblox studs (HRP-centered,
    yaw-removed). `target` is the only thing the user authors (the gizmo);
    `root`/`hip_*` are captured automatically from the rig at the frame and
    are used to anchor the body (root y/xz) and derive heading.
    """
    effector: str
    time: float
    target: List[float]                          # effector gizmo position [x, y, z]
    target_rot: Optional[List[float]] = None     # effector rotation [qx, qy, qz, qw], char-relative
    root: Optional[List[float]] = None           # LowerTorso position [x, y, z]
    hip_l: Optional[List[float]] = None          # LeftUpperLeg position [x, y, z]
    hip_r: Optional[List[float]] = None          # RightUpperLeg position [x, y, z]


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


# Plugin effector → SOMA base-EE joint name (for expand_joint_names + bind).
# Limbs and Hips are full end-effector constraints (global pos + rot). "Root"
# is a special-cased root2d (XZ path + heading, hip height free).
EFF_TO_JOINT = {
    "LeftHand": "LeftHand",
    "RightHand": "RightHand",
    "LeftFoot": "LeftFoot",
    "RightFoot": "RightFoot",
    "Hips": "Hips",   # full 3D pelvis pin (joint 0 / root_idx)
    # "Root" handled separately as root2d
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
    """Convert plugin constraints to Kimodo constraint **objects**.

    Builds EndEffectorConstraintSet / Root2DConstraintSet directly with sparse
    global positions — no FK, no chain. Only the effector joint(+children),
    the root, and the two hips are filled in `global_joints_positions`; the
    constraint reads only those. The effector's SOMA global rotation comes from
    the gizmo's world quat via the per-joint bind correction (see
    roblox_to_kimodo._retarget_chain_quats: D[joint] = world * inv(bind)).
    """
    import numpy as np
    import torch
    import kimodo_warm
    import roblox_to_kimodo as r2k
    from vendor import quat
    from kimodo.geometry import axis_angle_to_matrix
    from kimodo.constraints import EndEffectorConstraintSet, Root2DConstraintSet

    skel = kimodo_warm.load_model_once().skeleton
    dev = getattr(skel, "device", "cpu")  # build on the model's device (e.g. mps)
    J = skel.nbjoints
    fps = 30
    # Kimodo produces round(duration*fps) frames, indices 0..n_frames-1.
    n_frames = int(round(total_duration * fps))

    def to_pos(p):
        """Char-space studs [x,y,z] → Kimodo meters (Y-up, 180°-Y flip)."""
        x, y, z = p
        return np.array([-x * STUD_TO_KIMODO_XZ, y * STUD_TO_KIMODO_Y, -z * STUD_TO_KIMODO_XZ])

    def to_rot_mat(quat_xyzw, joint_name):
        """Char-space quat [qx,qy,qz,qw] → SOMA global rotation matrix (3,3)."""
        qx, qy, qz, qw = quat_xyzw
        R = np.array([qw, qx, qy, qz])                  # wxyz
        Rk = r2k._quat_y180_conjugate(R)                # Roblox → Kimodo frame
        bind = r2k.SOMA_BIND_CORRECTION.get(joint_name, r2k._BIND_IDENTITY)
        soma_q = quat.mul(Rk, quat.inv(bind))           # D[joint] = global SOMA rot
        aa = r2k._quat_to_axis_angle(soma_q[None, :])[0]
        return axis_angle_to_matrix(torch.tensor(aa, dtype=torch.float32, device=dev))

    out = []
    for c in constraints:
        frame_idx = max(0, min(int(round(c.time * fps)), n_frames - 1))
        target_k = to_pos(c.target)
        root_k = to_pos(c.root) if c.root else target_k
        # Hips for heading; fall back to a forward-facing pair around the root.
        hip_r_k = to_pos(c.hip_r) if c.hip_r else root_k + np.array([0.1, 0.0, 0.0])
        hip_l_k = to_pos(c.hip_l) if c.hip_l else root_k - np.array([0.1, 0.0, 0.0])

        # ── Root (2D path): XZ + heading, hip height free ──
        if c.effector == "Root":
            diff = hip_r_k - hip_l_k
            yaw = float(np.arctan2(diff[2], -diff[0]))
            out.append(Root2DConstraintSet(
                skel,
                # frame_indices stays on CPU to match the index tensors the
                # constraint builds internally (mirrors from_dict); only the
                # data tensors live on the model device.
                frame_indices=torch.tensor([frame_idx]),
                smooth_root_2d=torch.tensor([[float(target_k[0]), float(target_k[2])]], dtype=torch.float32, device=dev),
                global_root_heading=torch.tensor([[float(np.cos(yaw)), float(np.sin(yaw))]], dtype=torch.float32, device=dev),
            ))
            print(f"[generate] Root2D @f{frame_idx}: xz=({target_k[0]:.2f},{target_k[2]:.2f}) yaw={yaw:.2f}")
            continue

        joint_name = EFF_TO_JOINT.get(c.effector)
        if joint_name is None:
            print(f"[generate] Skipping unsupported effector: {c.effector}")
            continue

        # ── Limb / Hips: full 3D end-effector (global pos + rot) ──
        pos = torch.zeros(1, J, 3, device=dev)
        rot = torch.eye(3, device=dev).reshape(1, 1, 3, 3).repeat(1, J, 1, 1)

        def fill(idx, vec):
            pos[0, idx] = torch.tensor(vec, dtype=torch.float32, device=dev)

        # Fill the effector's position joints (effector + leaf children).
        _, pos_names = skel.expand_joint_names([joint_name])
        for n in pos_names:
            fill(skel.bone_index[n], target_k)

        # Root index: the Hips gizmo IS the root; limbs use the captured body root.
        root_fill = target_k if c.effector == "Hips" else root_k
        fill(skel.root_idx, root_fill)

        # Hips (heading). hip_joint_idx is ordered [right, left].
        r_hip_idx, l_hip_idx = skel.hip_joint_idx
        fill(r_hip_idx, hip_r_k)
        fill(l_hip_idx, hip_l_k)

        # Effector rotation (only the effector joint's rotation is read).
        if c.target_rot:
            rot[0, skel.bone_index[joint_name]] = to_rot_mat(c.target_rot, joint_name)

        out.append(EndEffectorConstraintSet(
            skel,
            frame_indices=torch.tensor([frame_idx]),  # CPU (see Root2D note)
            global_joints_positions=pos,
            global_joints_rots=rot,
            smooth_root_2d=None,   # auto-derived from root index XZ
            joint_names=[joint_name],
        ))
        print(f"[generate] {c.effector} @f{frame_idx}: target={np.round(target_k,2)} root_y={root_fill[1]:.2f}")

    return out


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

        def _run_pass(out_name: str, constraint_lst, cfg_constraint_w: float):
            """One kimodo pass (warm, in-process). With constraints → separated CFG."""
            if constraint_lst:
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
                constraint_lst=constraint_lst,
            )

        if req.looped:
            # Two-pass loop synthesis. Pass 1 (with any user constraints) gives
            # a pose to pin; pass 2 pins frame-0 pose at both endpoints AND keeps
            # the user constraints so looping + constraints work together.
            job.message = "Loop pass 1/2..."
            _run_pass("generated_pass1", kimodo_constraints, 2.0)

            n_loop = int(round(total_duration * 30))
            sample_frame = max(0, min(int(round(req.loop_offset * 30)), n_loop - 1))
            # Loop pins come back as JSON-format dicts (whole-body pose from the
            # pass-1 NPZ); load them into constraint objects to merge with the
            # directly-built user objects.
            loop_dicts = prompt_pipeline._build_loop_constraints(
                clip_dir / "generated_pass1.npz", n_frames=n_loop, sample_frame=sample_frame,
            )
            from kimodo.constraints import load_constraints_lst
            loop_objs = load_constraints_lst(loop_dicts, kimodo_warm.load_model_once().skeleton)
            merged = loop_objs + kimodo_constraints  # pins + user constraints

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
