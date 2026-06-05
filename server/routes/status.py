"""GET /status/{job_id} - Poll job status. GET /result/{job_id} - Get completed result."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


@router.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    from main import job_manager

    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Don't include the full animation data in status polls — only in
    # the dedicated /result endpoint (it can be 300KB+)
    return StatusResponse(
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        error=job.error,
        result={"seed": job.result["seed"], "duration_s": job.result["duration_s"]}
            if job.status.value == "completed" and job.result else None,
    )


@router.get("/result/{job_id}")
def get_result(job_id: str):
    """Return the full animation data (r15.json) for a completed job."""
    from main import job_manager

    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status.value != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    return job.result
