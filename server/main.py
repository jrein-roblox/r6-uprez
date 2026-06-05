"""RoMotion Backend Server - FastAPI app serving the Kimodo animation pipeline.

Install deps:
    pip3 install fastapi uvicorn

Run:
    cd r6-uprez/server && python3 main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from jobs import JobManager

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = REPO_ROOT / "work"
WORK_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "python"))

app = FastAPI(title="RoMotion Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

job_manager = JobManager(max_workers=2)

app.mount("/files", StaticFiles(directory=str(WORK_DIR)), name="files")

from routes.generate import router as generate_router  # noqa: E402
from routes.status import router as status_router  # noqa: E402
from routes.auto_constraints import router as auto_constraints_router  # noqa: E402
from routes.import_clip import router as import_clip_router  # noqa: E402

app.include_router(generate_router)
app.include_router(status_router)
app.include_router(auto_constraints_router)
app.include_router(import_clip_router)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8787)
