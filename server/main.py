"""RoMotion Backend Server - FastAPI app serving the Kimodo animation pipeline.

Run with KIMODO's venv python (it has torch + kimodo + fastapi + uvicorn),
so the model can be loaded in-process and kept warm between requests:

    /Users/jrein/git/nv-tlabs/kimodo/.venv/bin/python main.py

(Running under a different interpreter falls back to per-request model loads
only if torch/kimodo are importable there — normally they aren't.)
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


@app.on_event("startup")
def _prewarm_kimodo():
    """Load the Kimodo model in the background so the first generation is fast."""
    import threading
    def _load():
        try:
            import kimodo_warm
            kimodo_warm.load_model_once()
        except Exception as e:
            print(f"[main] Kimodo pre-warm failed (will load on first request): {e}", flush=True)
    threading.Thread(target=_load, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8787)
