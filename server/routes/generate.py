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
    effector: str
    time: float
    position: List[float]
    rotation: Optional[List[float]] = None


class GenerateRequest(BaseModel):
    prompts: List[PromptSegment]
    constraints: List[Constraint] = []
    duration: float = 3.0
    looped: bool = False
    seed: Optional[int] = None
    cfg_weight: float = 5.0
    diffusion_steps: int = 100
    target_rig: str = "r15"


class GenerateResponse(BaseModel):
    job_id: str
    status: str


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    from main import job_manager, WORK_DIR

    def run_generation(job):
        from constraint_converter import convert_constraints
        import prompt_pipeline
        import pipeline as parent_pipeline
        import run_kimodo
        import export_r15

        seed = req.seed if req.seed is not None else random.randrange(2**31 - 1)

        prompt_text = ". ".join(seg.text for seg in req.prompts)
        durations = [seg.end_time - seg.start_time for seg in req.prompts]
        duration_str = " ".join(f"{d:.2f}" for d in durations)
        total_duration = sum(durations)

        # Isolate this generation in its own directory so _build_rbxm
        # only processes this single clip (not the 800+ others in work/)
        iso_dir = WORK_DIR / f"_srv_{job.id}"
        clip_name = "clip"
        clip_dir = iso_dir / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        job.message = "Running Kimodo generation..."
        job.progress = 0.1

        has_constraints = len(req.constraints) > 0

        if has_constraints:
            kimodo_constraints = convert_constraints(
                [c.model_dump() for c in req.constraints],
                duration=total_duration,
            )
            constraints_path = clip_dir / "constraints.json"
            constraints_path.write_text(json.dumps(kimodo_constraints, indent=2))

            meta = {
                "source": "romotion_plugin",
                "prompt": prompt_text,
                "duration": duration_str,
                "duration_s": total_duration,
                "kimodo_model": run_kimodo.DEFAULT_MODEL,
                "kimodo_seed": seed,
                "kimodo_diffusion_steps": req.diffusion_steps,
                "looped": req.looped,
            }
            (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            run_kimodo.run_kimodo(
                clip_dir,
                prompt=prompt_text,
                model=run_kimodo.DEFAULT_MODEL,
                seed=seed,
                diffusion_steps=req.diffusion_steps,
                out_name="generated",
                extra_args=[
                    "--cfg_type", "separated",
                    "--cfg_weight", str(req.cfg_weight), "4.0",
                    "--num_transition_frames", "5",
                ],
                duration_override=duration_str,
            )
        else:
            cfg_weight = [req.cfg_weight]
            prompt_pipeline._run_kimodo_promptonly(
                clip_dir,
                prompt=prompt_text,
                duration=duration_str,
                model=run_kimodo.DEFAULT_MODEL,
                seed=seed,
                diffusion_steps=req.diffusion_steps,
                cfg_type="regular",
                cfg_weight=cfg_weight,
                num_transition_frames=5,
                out_name="generated",
            )

        job.message = "Retargeting to R15..."
        job.progress = 0.7

        bvh_path = clip_dir / "generated.bvh"
        r15_json_path = clip_dir / "r15.json"

        export_r15.set_rig(parent_pipeline.RIG if req.target_rig == "r15" else "r15plus")
        soma_bind_chain_studs = prompt_pipeline._soma_bind_hip_to_ankle_studs()
        target_hrp_to_ankle = prompt_pipeline._RTHRO_HRP_TO_ANKLE
        effective_hrp_scale = target_hrp_to_ankle / soma_bind_chain_studs

        parent_pipeline._retarget_bvh_to_r15_json(
            bvh_path, r15_json_path,
            root_motion=False,
            source_n_frames=0,
            loop_passes=1,
            looped=req.looped,
            inertial_blend_frames=6 if req.looped else 0,
            hrp_scale=effective_hrp_scale,
            target_rig=req.target_rig,
        )

        result = json.loads(r15_json_path.read_text())
        offset = prompt_pipeline._ground_y(
            result, bvh_path, "first",
            target_hrp_rest_y=prompt_pipeline._RTHRO_HRP_REST_Y,
            target_hrp_to_ankle=target_hrp_to_ankle,
        )
        if offset != 0.0:
            r15_json_path.write_text(json.dumps(result, separators=(",", ":")))

        job.message = "Done"
        job.progress = 1.0

        # Return the r15.json data directly — the plugin builds the
        # CurveAnimation instances in Lua (game:GetObjects doesn't work
        # with localhost URLs in Studio)
        job.result = {
            "animation": result,
            "seed": seed,
            "duration_s": total_duration,
        }
        return job.result

    job = job_manager.submit(run_generation)
    return GenerateResponse(job_id=job.id, status=job.status.value)
