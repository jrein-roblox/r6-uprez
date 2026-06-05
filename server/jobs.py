"""Background job manager for animation generation tasks."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: Optional[str] = None
    output_path: Optional[Path] = None
    created_at: float = field(default_factory=time.time)


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[Job], Any]) -> Job:
        job_id = f"gen_{uuid.uuid4().hex[:8]}"
        job = Job(id=job_id)
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        job.status = JobStatus.RUNNING
        try:
            result = fn(job)
            job.result = result
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cleanup_old(self, max_age_s: float = 3600.0) -> None:
        now = time.time()
        with self._lock:
            expired = [
                jid
                for jid, j in self._jobs.items()
                if now - j.created_at > max_age_s
                and j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            ]
            for jid in expired:
                del self._jobs[jid]
