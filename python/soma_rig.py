"""SOMA (Kimodo `somaskel77`) → R15 retarget mapping + bind-pose reference.

Kimodo's BVH export (`kimodo/exports/bvh.py`) writes the 77 SOMA bones
under a synthetic `Root` parent at world origin (positions on `Root` and
`Hips`, rotation channels in ZYX order). Source bone names follow
SMPL/X-style mocap conventions:

    Root → Hips → Spine1 → Spine2 → Chest
                                  → Neck1 → Neck2 → Head → (HeadEnd, Jaw, eyes)
                                  → LeftShoulder → LeftArm → LeftForeArm → LeftHand → (fingers)
                                  → RightShoulder → RightArm → RightForeArm → RightHand → (fingers)
         → LeftLeg → LeftShin → LeftFoot → LeftToeBase → LeftToeEnd
         → RightLeg → RightShin → RightFoot → RightToeBase → RightToeEnd

Naming quirk vs UE/MotusMan: SOMA's "LeftLeg" is the **upper** leg
(parented to Hips), then "LeftShin" then "LeftFoot". Don't confuse with
UE's `lowerleg`/`shin` semantics.

Coordinate system: Y-up, +Z forward, native units **meters** in Kimodo's
internal frame. The BVH exporter scales positions to **cm** for tool
compatibility, then `export_r15.py` applies `CM_TO_STUD = 0.03` as for
any other BVH.

Bind pose: the shipped `data/soma_tpose.bvh` (sourced from
`kimodo/assets/skeletons/somaskel77/somaskel77_standard_tpose.bvh`) is a
true T-pose with all local rotations = identity. World rotation per
joint is therefore identity (1, 0, 0, 0) for every SOMA bone — enabling
a trivial bind table. To regenerate from a Kimodo update, run
`kimodo_gen "T-pose" --duration 0.1 --bvh --bvh_standard_tpose --output
data/soma_tpose` and re-copy.
"""
from __future__ import annotations

import numpy as np


# R15 part, source SOMA bone (last in chain), R15 parent part, R15 Motor6D name.
# Spine and neck collapse the 3-segment / 3-segment SOMA chains to R15's
# single Waist / single Neck Motor6Ds.
SOMA_R15_JOINTS = [
    ("LowerTorso",    "Hips",         "HumanoidRootPart", "Root"),
    ("UpperTorso",    "Chest",        "LowerTorso",       "Waist"),
    ("Head",          "Head",         "UpperTorso",       "Neck"),
    ("LeftUpperArm",  "LeftArm",      "UpperTorso",       "LeftShoulder"),
    ("LeftLowerArm",  "LeftForeArm",  "LeftUpperArm",     "LeftElbow"),
    ("LeftHand",      "LeftHand",     "LeftLowerArm",     "LeftWrist"),
    ("RightUpperArm", "RightArm",     "UpperTorso",       "RightShoulder"),
    ("RightLowerArm", "RightForeArm", "RightUpperArm",    "RightElbow"),
    ("RightHand",     "RightHand",    "RightLowerArm",    "RightWrist"),
    ("LeftUpperLeg",  "LeftLeg",      "LowerTorso",       "LeftHip"),
    ("LeftLowerLeg",  "LeftShin",     "LeftUpperLeg",     "LeftKnee"),
    ("LeftFoot",      "LeftFoot",     "LeftLowerLeg",     "LeftAnkle"),
    ("RightUpperLeg", "RightLeg",     "LowerTorso",       "RightHip"),
    ("RightLowerLeg", "RightShin",    "RightUpperLeg",    "RightKnee"),
    ("RightFoot",     "RightFoot",    "RightLowerLeg",    "RightAnkle"),
]

# Chain of SOMA bones whose composite rotation feeds each R15 Motor6D.
# Spine: 3 SOMA segments (Spine1 → Spine2 → Chest) collapse into Waist.
# Neck:  Neck1 → Neck2 → Head collapse into the single R15 Neck.
SOMA_R15_CHAINS: dict[str, list[str]] = {
    "LowerTorso":    ["Hips"],
    "UpperTorso":    ["Spine1", "Spine2", "Chest"],
    "Head":          ["Neck1", "Neck2", "Head"],
    "LeftUpperArm":  ["LeftShoulder", "LeftArm"],
    "LeftLowerArm":  ["LeftForeArm"],
    "LeftHand":      ["LeftHand"],
    "RightUpperArm": ["RightShoulder", "RightArm"],
    "RightLowerArm": ["RightForeArm"],
    "RightHand":     ["RightHand"],
    "LeftUpperLeg":  ["LeftLeg"],
    "LeftLowerLeg":  ["LeftShin"],
    "LeftFoot":      ["LeftFoot"],
    "RightUpperLeg": ["RightLeg"],
    "RightLowerLeg": ["RightShin"],
    "RightFoot":     ["RightFoot"],
}

# Bind-pose world rotations for every SOMA bone the retarget reads.
#
# Source: `kimodo/assets/skeletons/somaskel77/standard_t_pose_global_offsets_rots.p`.
# That tensor is the rotation that takes a joint's BONES-SEED rest frame to
# the standard T-pose rest frame. By construction (see
# `kimodo/skeleton/transforms.py:change_tpose`), if you set
# `standard_global = identity` (true T-pose) then `seed_global = M[j]`. So
# `M[j]` is the joint's world rotation in the BONES-SEED rest pose — which
# is exactly what the BVH that `kimodo_gen` (without `--bvh_standard_tpose`)
# produces uses as its rest. That matches the convention of every animation
# we'll feed through Stage C, so this table is the right bind reference.
#
# Quats are (w, x, y, z) as everywhere else in this codebase. Symmetric L/R
# pairs verify by inspection (e.g. LeftLeg ↔ RightLeg differ only in sign on
# x and z components).
#
# To regenerate (e.g. after a Kimodo upgrade):
#   /Users/jrein/git/nv-tlabs/kimodo/.venv/bin/python -c "
#       import torch; m = torch.load('.../standard_t_pose_global_offsets_rots.p',
#       weights_only=False).squeeze().numpy()
#       # convert each (3,3) → (w,x,y,z), match against bone_order_names_with_parents
#   "
SOMA_BIND_WORLD: dict[str, np.ndarray] = {
    "Root":          np.array([1.0, 0.0, 0.0, 0.0]),
    "Hips":          np.array([+0.5008302927, +0.4991683662, +0.5008303523, +0.4991683662]),
    "Spine1":        np.array([+0.5010452867, +0.4989525676, +0.5010452867, +0.4989525974]),
    "Spine2":        np.array([+0.5261770487, +0.4723745883, +0.5261770487, +0.4723746479]),
    "Chest":         np.array([+0.5008835196, +0.4991148710, +0.5008836389, +0.4991149902]),
    "Neck1":         np.array([+0.4224390686, +0.5670495033, +0.4224392474, +0.5670496225]),
    "Neck2":         np.array([+0.4172193408, +0.5709007382, +0.4172196388, +0.5709012151]),
    "Head":          np.array([+0.5285517573, +0.4697152376, +0.5285521746, +0.4697162211]),
    # Arms: SEED rest is T-pose (arms extended horizontal), but R15's
    # natural rest is arms-hanging-down. We need a bind that maps SOMA's
    # bone-local frame to R15's part-local frame at rest.
    #
    # Empirically the diffusion-derived idle-pose bind (sampled from
    # `data/soma_idle.bvh` mid-frame, kimodo_gen "person standing still
    # with arms hanging at their sides") gave the closest match. Pre-
    # rotated by ±10° about world +Z (+10° for left, −10° for right) so
    # the resulting R15 arms hang slightly more tucked to the sides than
    # the raw idle pose (which had a small outward bias from diffusion
    # noise).
    "LeftShoulder":  np.array([+0.5682713583, +0.8131124468, +0.0758540700, -0.1008066078]),
    "LeftArm":       np.array([+0.3679025964, +0.7492056363, -0.4740354862, -0.2804085445]),
    "LeftForeArm":   np.array([+0.4234890003, +0.6141994414, -0.6402025056, -0.1831853287]),
    "LeftHand":      np.array([+0.3904089975, +0.6603874201, -0.6200352264, -0.1643946108]),
    "RightShoulder": np.array([-0.8089608867, +0.5732609532, +0.0757582633, +0.1059001838]),
    "RightArm":      np.array([-0.7388050964, +0.3689592147, +0.3075078234, -0.4727314946]),
    "RightForeArm":  np.array([-0.6116991674, +0.4275567037, +0.2168451903, -0.6292833679]),
    "RightHand":     np.array([-0.6597051074, +0.3886310816, +0.1398601338, -0.6278488645]),
    "LeftLeg":       np.array([+0.5046219826, -0.4953350127, +0.5046219826, -0.4953350127]),
    "LeftShin":      np.array([+0.5201702118, -0.4789812565, +0.5201702118, -0.4789812565]),
    "LeftFoot":      np.array([-0.1284100711, +0.6953494549, -0.1284100562, +0.6953495145]),
    "RightLeg":      np.array([+0.4953346848, +0.5046221018, +0.4953346848, +0.5046222210]),
    "RightShin":     np.array([+0.4789812565, +0.5201702118, +0.4789812565, +0.5201701522]),
    "RightFoot":     np.array([+0.6953494549, +0.1284099817, +0.6953494549, +0.1284100115]),
}
