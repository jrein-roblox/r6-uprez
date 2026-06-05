"""POST /import-clip - Import an existing Roblox animation and extract constraints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

router = APIRouter()


class ImportClipRequest(BaseModel):
    asset_id: int
    sample_fps: int = 30
    max_duration: float = 10.0


class ImportConstraint(BaseModel):
    effector: str
    time: float
    position: list[float]


class ImportClipResponse(BaseModel):
    job_id: str
    status: str


@router.post("/import-clip", response_model=ImportClipResponse)
def import_clip(req: ImportClipRequest):
    from main import job_manager, WORK_DIR

    def run_import(job):
        import extract_pose
        import roblox_to_kimodo

        clip_name = f"import_{req.asset_id}"
        clip_dir = WORK_DIR / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        job.message = "Extracting poses from animation..."
        job.progress = 0.2

        pose_path = clip_dir / "pose.json"
        extract_pose.extract(
            asset_id=req.asset_id,
            out_path=pose_path,
            fps=req.sample_fps,
        )

        job.message = "Converting to constraints..."
        job.progress = 0.6

        pose_data = json.loads(pose_path.read_text())
        duration_s = pose_data.get("source_duration_s", 3.0)
        if duration_s > req.max_duration:
            duration_s = req.max_duration

        constraints_data = roblox_to_kimodo.convert(
            pose_path=pose_path,
            out_dir=clip_dir,
        )

        from constraint_converter import STUD_TO_METER

        plugin_constraints: list[dict] = []
        meta_path = clip_dir / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            for eff_name, eff_data in meta.get("effector_keyframes", {}).items():
                for frame_idx in eff_data.get("frames", []):
                    time = float(frame_idx) / req.sample_fps
                    if time > duration_s:
                        continue
                    plugin_constraints.append({
                        "effector": eff_name,
                        "time": time,
                        "position": [0, 0, 0],
                    })

        job.result = {
            "duration_s": duration_s,
            "rig_type": pose_data.get("rig_type", "unknown"),
            "constraints": plugin_constraints,
            "clip_name": clip_name,
        }
        return job.result

    job = job_manager.submit(run_import)
    return ImportClipResponse(job_id=job.id, status=job.status.value)
