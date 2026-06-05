"""POST /auto-constraints - Detect velocity extrema for constraint placement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

router = APIRouter()


class AutoConstraintsRequest(BaseModel):
    job_id: str
    effectors: List[str] = ["left_foot", "right_foot", "left_hand", "right_hand"]
    min_separation_frames: int = 8


class ConstraintKeyframe(BaseModel):
    effector: str
    frame: int
    time: float
    position: List[float]


class AutoConstraintsResponse(BaseModel):
    constraints: List[ConstraintKeyframe]


@router.post("/auto-constraints", response_model=AutoConstraintsResponse)
def auto_constraints(req: AutoConstraintsRequest):
    from main import job_manager, WORK_DIR
    import numpy as np
    import export_r15
    import pipeline as parent_pipeline

    job = job_manager.get(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    clip_name = job.result["clip_name"]
    clip_dir = WORK_DIR / clip_name
    bvh_path = clip_dir / "generated.bvh"

    if not bvh_path.is_file():
        raise HTTPException(status_code=500, detail="BVH file not found for this job")

    import effector_helpers

    export_r15.set_rig(parent_pipeline.RIG)
    anim = export_r15._load_anim_any(bvh_path)
    names = list(anim["names"])
    world_pos = anim["world_pos"]
    fps = 30

    effector_map = {
        "left_hand": "LeftHand",
        "right_hand": "RightHand",
        "left_foot": "LeftFoot",
        "right_foot": "RightFoot",
    }

    from constraint_converter import STUD_TO_METER

    result_constraints: list[ConstraintKeyframe] = []

    for eff_name in req.effectors:
        soma_name = effector_map.get(eff_name)
        if not soma_name or soma_name not in names:
            continue

        joint_idx = names.index(soma_name)
        positions = world_pos[:, joint_idx, :]  # (F, 3) in cm, kimodo space

        frames = effector_helpers.detect_velocity_extremes(
            positions,
            min_separation=req.min_separation_frames,
        )

        for frame_idx in frames:
            pos_kimodo = positions[frame_idx]
            pos_roblox = [
                -float(pos_kimodo[0]) / (STUD_TO_METER * 100),
                float(pos_kimodo[1]) / (STUD_TO_METER * 100),
                -float(pos_kimodo[2]) / (STUD_TO_METER * 100),
            ]
            result_constraints.append(ConstraintKeyframe(
                effector=eff_name,
                frame=int(frame_idx),
                time=float(frame_idx) / fps,
                position=pos_roblox,
            ))

    return AutoConstraintsResponse(constraints=result_constraints)
