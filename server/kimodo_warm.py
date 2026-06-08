"""Warm in-process Kimodo generator.

Loads the Kimodo model once and reuses it across requests, avoiding the
per-generation subprocess + model/text-encoder reload that `kimodo_gen` pays.

Produces the same <clip_dir>/<out_name>.bvh and .npz that
run_kimodo / _run_kimodo_promptonly produce, so the rest of the pipeline
(retarget, loop constraint building) is unchanged.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional

_model = None
_resolved = None
_lock = threading.Lock()
_gen_lock = threading.Lock()  # serialize inference (one GPU pass at a time)


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_once():
    """Load (and cache) the Kimodo model. First call is slow; rest are instant."""
    global _model, _resolved
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from kimodo import DEFAULT_MODEL, load_model
            device = _pick_device()
            print(f"[kimodo_warm] Loading model {DEFAULT_MODEL} on {device} (one-time)...", flush=True)
            m, r = load_model(
                DEFAULT_MODEL, device=device,
                default_family="Kimodo", return_resolved_name=True,
            )
            _model, _resolved = m, r
            print(f"[kimodo_warm] Model ready: {r}", flush=True)
    return _model


def generate_bvh(
    clip_dir: Path,
    *,
    prompt: str,
    duration_str: str,
    seed: Optional[int],
    diffusion_steps: int,
    cfg_type: str,            # "regular" or "separated"
    cfg_weight: List[float],  # [w] for regular, [text, constraint] for separated
    num_transition_frames: int,
    out_name: str,
    constraints_path: Optional[Path] = None,
) -> Path:
    """Run one warm generation, writing <clip_dir>/<out_name>.bvh (+ .npz).

    Returns the bvh path.
    """
    import torch
    from kimodo.tools import seed_everything
    from kimodo.constraints import load_constraints_lst
    from kimodo.exports.bvh import save_motion_bvh
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots
    from kimodo.scripts.generate import get_texts_and_num_frames_from_prompt

    model = load_model_once()
    device = model.device

    texts, num_frames = get_texts_and_num_frames_from_prompt(prompt, duration_str, model.fps)

    if seed is not None:
        seed_everything(seed)

    constraint_lst = []
    if constraints_path and Path(constraints_path).is_file():
        constraint_lst = load_constraints_lst(str(constraints_path), model.skeleton)

    cfg_kwargs = {}
    if cfg_type == "regular":
        cfg_kwargs = {"cfg_type": "regular", "cfg_weight": float(cfg_weight[0])}
    elif cfg_type == "separated":
        cfg_kwargs = {"cfg_type": "separated",
                      "cfg_weight": [float(cfg_weight[0]), float(cfg_weight[1])]}

    # Serialize inference: a single model can't run concurrent passes safely.
    with _gen_lock:
        output = model(
            texts, num_frames,
            constraint_lst=constraint_lst,
            num_denoising_steps=diffusion_steps,
            num_samples=1,
            multi_prompt=True,
            num_transition_frames=num_transition_frames,
            post_processing=False,  # motion_correction ext not installed; no-op anyway
            return_numpy=True,
            **cfg_kwargs,
        )

    clip_dir = Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    out_stem = clip_dir / out_name

    # NPZ (loop pass-1 reads local_rot_mats + root_positions from this)
    single = {
        k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v)
        for k, v in output.items()
    }
    save_kimodo_npz(str(out_stem) + ".npz", single)

    # BVH
    skeleton = model.skeleton
    if isinstance(skeleton, SOMASkeleton30):
        from kimodo.tools import to_device_mps_safe
        skeleton = to_device_mps_safe(skeleton.somaskel77, device)

    joints_pos = torch.from_numpy(output["posed_joints"][0]).to(device)
    joints_rot = torch.from_numpy(output["global_rot_mats"][0]).to(device)
    local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
    root_positions = joints_pos[:, skeleton.root_idx, :]
    bvh_path = Path(str(out_stem) + ".bvh")
    save_motion_bvh(
        str(bvh_path), local_rot_mats, root_positions,
        skeleton=skeleton, fps=model.fps,
    )
    return bvh_path
