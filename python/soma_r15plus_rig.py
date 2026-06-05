"""SOMA (Kimodo `somaskel77`) → R15-plus retarget mapping + bind-pose reference.

R15-plus extends the standard R15 rig with:
  - Spine bone (under UpperTorso)
  - Chest bone (under Spine)
  - Left/RightClavicle bones (under Chest)
  - 30 finger bones (5 fingers × 3 joints × 2 hands)
  - Left/RightToeBase bones (under feet)
  - HeadBase bone (under Head)

The SOMA skeleton maps cleanly to these additional joints since it already
has per-finger, per-spine-segment, and clavicle bones. The standard R15
retarget collapses these chains; R15-plus preserves them.

Hierarchy (showing new additions marked with *):

    HumanoidRootPart
      └─ LowerTorso (Root Motor6D)
           ├─ UpperTorso (Waist Motor6D)
           │    └─ *Spine (Bone)
           │         └─ *Chest (Bone)
           │              ├─ *LeftClavicle (Bone)
           │              │    └─ LeftUpperArm (LeftShoulder Motor6D)
           │              │         └─ LeftLowerArm (LeftElbow)
           │              │              └─ LeftHand (LeftWrist)
           │              │                   ├─ *LeftHandThumb1 > 2 > 3
           │              │                   ├─ *LeftHandIndex1 > 2 > 3
           │              │                   ├─ *LeftHandMiddle1 > 2 > 3
           │              │                   ├─ *LeftHandRing1 > 2 > 3
           │              │                   └─ *LeftHandPinky1 > 2 > 3
           │              ├─ *RightClavicle (Bone)
           │              │    └─ RightUpperArm → ...
           │              └─ Head (Neck Motor6D)
           │                   └─ *HeadBase (Bone)
           ├─ LeftUpperLeg (LeftHip)
           │    └─ LeftLowerLeg (LeftKnee)
           │         └─ LeftFoot (LeftAnkle)
           │              └─ *LeftToeBase (Bone)
           └─ RightUpperLeg → ...

SOMA bone mapping notes:
  - Waist: SOMA Spine1 only (no longer collapsing Spine1→Spine2→Chest)
  - Spine bone: SOMA Spine2
  - Chest bone: SOMA Chest
  - Clavicles: SOMA LeftShoulder / RightShoulder (1:1)
  - LeftShoulder Motor6D: SOMA LeftArm only (clavicle is now separate)
  - Neck: Neck1 → Neck2 → Head chain (parent is now Chest, not UpperTorso)
  - Fingers: SOMA LeftHandThumb1/2/3 → R15+ LeftHandThumb1/2/3 (direct 1:1)
             SOMA has 4-5 joints per finger (Index1→2→3→4→End), R15+ has 3;
             we map the first 3 directly and ignore the terminal joints.
  - Toes: SOMA LeftToeBase / RightToeBase (1:1)
  - HeadBase: SOMA HeadEnd
"""
from __future__ import annotations

import numpy as np


# R15-plus joint table: (R15+ part/bone name, source SOMA bone, R15+ parent, joint/bone name)
SOMA_R15PLUS_JOINTS = [
    # --- Core body (Motor6D joints) ---
    ("LowerTorso",     "Hips",            "HumanoidRootPart", "Root"),
    ("UpperTorso",     "Spine1",          "LowerTorso",       "Waist"),
    # --- Spine bones ---
    ("Spine",          "Spine2",          "UpperTorso",       "Spine"),
    ("Chest",          "Chest",           "Spine",            "Chest"),
    # --- Neck (parent is now Chest in the hierarchy) ---
    ("Head",           "Head",            "Chest",            "Neck"),
    # --- Left arm chain ---
    ("LeftClavicle",   "LeftShoulder",    "Chest",            "LeftClavicle"),
    ("LeftUpperArm",   "LeftArm",         "LeftClavicle",     "LeftShoulder"),
    ("LeftLowerArm",   "LeftForeArm",     "LeftUpperArm",     "LeftElbow"),
    ("LeftHand",       "LeftHand",        "LeftLowerArm",     "LeftWrist"),
    # --- Left hand fingers ---
    ("LeftHandThumb1",  "LeftHandThumb1",  "LeftHand",         "LeftHandThumb1"),
    ("LeftHandThumb2",  "LeftHandThumb2",  "LeftHandThumb1",   "LeftHandThumb2"),
    ("LeftHandThumb3",  "LeftHandThumb3",  "LeftHandThumb2",   "LeftHandThumb3"),
    ("LeftHandIndex1",  "LeftHandIndex1",  "LeftHand",         "LeftHandIndex1"),
    ("LeftHandIndex2",  "LeftHandIndex2",  "LeftHandIndex1",   "LeftHandIndex2"),
    ("LeftHandIndex3",  "LeftHandIndex3",  "LeftHandIndex2",   "LeftHandIndex3"),
    ("LeftHandMiddle1", "LeftHandMiddle1", "LeftHand",         "LeftHandMiddle1"),
    ("LeftHandMiddle2", "LeftHandMiddle2", "LeftHandMiddle1",  "LeftHandMiddle2"),
    ("LeftHandMiddle3", "LeftHandMiddle3", "LeftHandMiddle2",  "LeftHandMiddle3"),
    ("LeftHandRing1",   "LeftHandRing1",   "LeftHand",         "LeftHandRing1"),
    ("LeftHandRing2",   "LeftHandRing2",   "LeftHandRing1",    "LeftHandRing2"),
    ("LeftHandRing3",   "LeftHandRing3",   "LeftHandRing2",    "LeftHandRing3"),
    ("LeftHandPinky1",  "LeftHandPinky1",  "LeftHand",         "LeftHandPinky1"),
    ("LeftHandPinky2",  "LeftHandPinky2",  "LeftHandPinky1",   "LeftHandPinky2"),
    ("LeftHandPinky3",  "LeftHandPinky3",  "LeftHandPinky2",   "LeftHandPinky3"),
    # --- Right arm chain ---
    ("RightClavicle",  "RightShoulder",   "Chest",            "RightClavicle"),
    ("RightUpperArm",  "RightArm",        "RightClavicle",    "RightShoulder"),
    ("RightLowerArm",  "RightForeArm",    "RightUpperArm",    "RightElbow"),
    ("RightHand",      "RightHand",       "RightLowerArm",    "RightWrist"),
    # --- Right hand fingers ---
    ("RightHandThumb1",  "RightHandThumb1",  "RightHand",         "RightHandThumb1"),
    ("RightHandThumb2",  "RightHandThumb2",  "RightHandThumb1",   "RightHandThumb2"),
    ("RightHandThumb3",  "RightHandThumb3",  "RightHandThumb2",   "RightHandThumb3"),
    ("RightHandIndex1",  "RightHandIndex1",  "RightHand",         "RightHandIndex1"),
    ("RightHandIndex2",  "RightHandIndex2",  "RightHandIndex1",   "RightHandIndex2"),
    ("RightHandIndex3",  "RightHandIndex3",  "RightHandIndex2",   "RightHandIndex3"),
    ("RightHandMiddle1", "RightHandMiddle1", "RightHand",         "RightHandMiddle1"),
    ("RightHandMiddle2", "RightHandMiddle2", "RightHandMiddle1",  "RightHandMiddle2"),
    ("RightHandMiddle3", "RightHandMiddle3", "RightHandMiddle2",  "RightHandMiddle3"),
    ("RightHandRing1",   "RightHandRing1",   "RightHand",         "RightHandRing1"),
    ("RightHandRing2",   "RightHandRing2",   "RightHandRing1",    "RightHandRing2"),
    ("RightHandRing3",   "RightHandRing3",   "RightHandRing2",    "RightHandRing3"),
    ("RightHandPinky1",  "RightHandPinky1",  "RightHand",         "RightHandPinky1"),
    ("RightHandPinky2",  "RightHandPinky2",  "RightHandPinky1",   "RightHandPinky2"),
    ("RightHandPinky3",  "RightHandPinky3",  "RightHandPinky2",   "RightHandPinky3"),
    # --- Legs ---
    ("LeftUpperLeg",   "LeftLeg",         "LowerTorso",       "LeftHip"),
    ("LeftLowerLeg",   "LeftShin",        "LeftUpperLeg",     "LeftKnee"),
    ("LeftFoot",       "LeftFoot",        "LeftLowerLeg",     "LeftAnkle"),
    ("LeftToeBase",    "LeftToeBase",     "LeftFoot",         "LeftToeBase"),
    ("RightUpperLeg",  "RightLeg",        "LowerTorso",       "RightHip"),
    ("RightLowerLeg",  "RightShin",       "RightUpperLeg",    "RightKnee"),
    ("RightFoot",      "RightFoot",       "RightLowerLeg",    "RightAnkle"),
    ("RightToeBase",   "RightToeBase",    "RightFoot",        "RightToeBase"),
    # --- Head bone ---
    ("HeadBase",       "HeadEnd",         "Head",             "HeadBase"),
]

# Chain of SOMA bones whose composite local rotation feeds each R15+ joint.
# Unlike the standard R15 mapping which collapses multiple SOMA bones into
# single Motor6Ds, R15-plus preserves the granularity — most entries are
# single-bone chains.
SOMA_R15PLUS_CHAINS: dict[str, list[str]] = {
    "LowerTorso":     ["Hips"],
    "UpperTorso":     ["Spine1"],
    "Spine":          ["Spine2"],
    "Chest":          ["Chest"],
    "Head":           ["Neck1", "Neck2", "Head"],
    "LeftClavicle":   ["LeftShoulder"],
    "LeftUpperArm":   ["LeftArm"],
    "LeftLowerArm":   ["LeftForeArm"],
    "LeftHand":       ["LeftHand"],
    "LeftHandThumb1": ["LeftHandThumb1"],
    "LeftHandThumb2": ["LeftHandThumb2"],
    "LeftHandThumb3": ["LeftHandThumb3"],
    "LeftHandIndex1": ["LeftHandIndex1"],
    "LeftHandIndex2": ["LeftHandIndex2"],
    "LeftHandIndex3": ["LeftHandIndex3"],
    "LeftHandMiddle1": ["LeftHandMiddle1"],
    "LeftHandMiddle2": ["LeftHandMiddle2"],
    "LeftHandMiddle3": ["LeftHandMiddle3"],
    "LeftHandRing1":  ["LeftHandRing1"],
    "LeftHandRing2":  ["LeftHandRing2"],
    "LeftHandRing3":  ["LeftHandRing3"],
    "LeftHandPinky1": ["LeftHandPinky1"],
    "LeftHandPinky2": ["LeftHandPinky2"],
    "LeftHandPinky3": ["LeftHandPinky3"],
    "RightClavicle":  ["RightShoulder"],
    "RightUpperArm":  ["RightArm"],
    "RightLowerArm":  ["RightForeArm"],
    "RightHand":      ["RightHand"],
    "RightHandThumb1": ["RightHandThumb1"],
    "RightHandThumb2": ["RightHandThumb2"],
    "RightHandThumb3": ["RightHandThumb3"],
    "RightHandIndex1": ["RightHandIndex1"],
    "RightHandIndex2": ["RightHandIndex2"],
    "RightHandIndex3": ["RightHandIndex3"],
    "RightHandMiddle1": ["RightHandMiddle1"],
    "RightHandMiddle2": ["RightHandMiddle2"],
    "RightHandMiddle3": ["RightHandMiddle3"],
    "RightHandRing1":  ["RightHandRing1"],
    "RightHandRing2":  ["RightHandRing2"],
    "RightHandRing3":  ["RightHandRing3"],
    "RightHandPinky1": ["RightHandPinky1"],
    "RightHandPinky2": ["RightHandPinky2"],
    "RightHandPinky3": ["RightHandPinky3"],
    "LeftUpperLeg":   ["LeftLeg"],
    "LeftLowerLeg":   ["LeftShin"],
    "LeftFoot":       ["LeftFoot"],
    "LeftToeBase":    ["LeftToeBase"],
    "RightUpperLeg":  ["RightLeg"],
    "RightLowerLeg":  ["RightShin"],
    "RightFoot":      ["RightFoot"],
    "RightToeBase":   ["RightToeBase"],
    "HeadBase":       ["HeadEnd"],
}

# Bind-pose world rotations. Extends the base SOMA bind (from soma_rig.py)
# with entries for the new R15-plus joints. Joints not listed here default
# to whatever the bind BVH provides (identity for soma_tpose.bvh).
#
# The arm/shoulder bind values are reused from soma_rig.py since the same
# SOMA bones drive the same physical motion — they're just split across
# more target joints now. Finger, toe, and spine bones use the T-pose bind
# (identity) since:
#   - SOMA T-pose has fingers extended (identity local rotations)
#   - R15-plus rest pose also has fingers in a neutral extended position
#   - Spine segments at T-pose are near-identity
#
# The arm bind correction (T-pose → I-pose hang) matters for clavicles and
# upper arms because R15-plus at rest has arms down, not extended.
SOMA_R15PLUS_BIND_WORLD: dict[str, np.ndarray] = {
    # Core body (same as soma_rig.py)
    "Root":          np.array([1.0, 0.0, 0.0, 0.0]),
    "Hips":          np.array([+0.5008302927, +0.4991683662, +0.5008303523, +0.4991683662]),
    "Spine1":        np.array([+0.5010452867, +0.4989525676, +0.5010452867, +0.4989525974]),
    "Spine2":        np.array([+0.5261770487, +0.4723745883, +0.5261770487, +0.4723746479]),
    "Chest":         np.array([+0.5008835196, +0.4991148710, +0.5008836389, +0.4991149902]),
    "Neck1":         np.array([+0.4224390686, +0.5670495033, +0.4224392474, +0.5670496225]),
    "Neck2":         np.array([+0.4172193408, +0.5709007382, +0.4172196388, +0.5709012151]),
    "Head":          np.array([+0.5285517573, +0.4697152376, +0.5285521746, +0.4697162211]),
    "HeadEnd":       np.array([+0.5285517573, +0.4697152376, +0.5285521746, +0.4697162211]),
    # Arms — clavicle and upper arm now separate targets
    "LeftShoulder":  np.array([+0.5682713583, +0.8131124468, +0.0758540700, -0.1008066078]),
    "LeftArm":       np.array([+0.3679025964, +0.7492056363, -0.4740354862, -0.2804085445]),
    "LeftForeArm":   np.array([+0.4234890003, +0.6141994414, -0.6402025056, -0.1831853287]),
    "LeftHand":      np.array([+0.3904089975, +0.6603874201, -0.6200352264, -0.1643946108]),
    "RightShoulder": np.array([-0.8089608867, +0.5732609532, +0.0757582633, +0.1059001838]),
    "RightArm":      np.array([-0.7388050964, +0.3689592147, +0.3075078234, -0.4727314946]),
    "RightForeArm":  np.array([-0.6116991674, +0.4275567037, +0.2168451903, -0.6292833679]),
    "RightHand":     np.array([-0.6597051074, +0.3886310816, +0.1398601338, -0.6278488645]),
    # Legs (same as soma_rig.py)
    "LeftLeg":       np.array([+0.5046219826, -0.4953350127, +0.5046219826, -0.4953350127]),
    "LeftShin":      np.array([+0.5201702118, -0.4789812565, +0.5201702118, -0.4789812565]),
    "LeftFoot":      np.array([-0.1284100711, +0.6953494549, -0.1284100562, +0.6953495145]),
    "RightLeg":      np.array([+0.4953346848, +0.5046221018, +0.4953346848, +0.5046222210]),
    "RightShin":     np.array([+0.4789812565, +0.5201702118, +0.4789812565, +0.5201701522]),
    "RightFoot":     np.array([+0.6953494549, +0.1284099817, +0.6953494549, +0.1284100115]),
    # Finger and toe bones: identity bind (T-pose = rest for these joints).
    # Not listed here — they'll fall through to the BVH's frame-0 world
    # rotation which is identity for soma_tpose.bvh. Only override if you
    # need a non-identity bind for specific bones.
}
