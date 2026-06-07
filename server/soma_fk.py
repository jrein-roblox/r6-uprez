"""SOMA30 FK utility: compute joint world positions from local rotations.

Used to verify effector positions after retargeting and correct root offset.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# SOMA30 parent indices
SOMA30_PARENTS = [
    -1,  # 0 Hips (root)
    0, 1, 2,        # 1-3: Spine1, Spine2, Chest
    3, 4, 5,        # 4-6: Neck1, Neck2, Head
    6, 6, 6,        # 7-9: Jaw, LeftEye, RightEye
    3, 10, 11, 12,  # 10-13: LeftShoulder, LeftArm, LeftForeArm, LeftHand
    13, 13,         # 14-15: LeftThumbEnd, LeftMiddleEnd
    3, 16, 17, 18,  # 16-19: RightShoulder, RightArm, RightForeArm, RightHand
    19, 19,         # 20-21: RightThumbEnd, RightMiddleEnd
    0, 22, 23, 24,  # 22-25: LeftLeg, LeftShin, LeftFoot, LeftToeBase
    0, 26, 27, 28,  # 26-29: RightLeg, RightShin, RightFoot, RightToeBase
]

# SOMA30 neutral joint positions (T-pose, meters)
SOMA30_NEUTRAL = np.array([
    [+0.0000, +0.0000, +0.0000],  # 0 Hips
    [-0.0001, +0.0500, -0.0005],  # 1 Spine1
    [-0.0001, +0.1213, -0.0008],  # 2 Spine2
    [-0.0001, +0.1968, -0.0090],  # 3 Chest
    [-0.0020, +0.4599, -0.0145],  # 4 Neck1
    [-0.0020, +0.5370, +0.0085],  # 5 Neck2
    [-0.0020, +0.5983, +0.0280],  # 6 Head
    [-0.0019, +0.6030, +0.0590],  # 7 Jaw
    [+0.0301, +0.6521, +0.1039],  # 8 LeftEye
    [-0.0342, +0.6519, +0.1036],  # 9 RightEye
    [+0.0161, +0.4292, +0.0421],  # 10 LeftShoulder
    [+0.1653, +0.4292, -0.0129],  # 11 LeftArm
    [+0.4527, +0.4292, -0.0129],  # 12 LeftForeArm
    [+0.7236, +0.4292, -0.0129],  # 13 LeftHand
    [+0.8463, +0.3970, +0.0354],  # 14 LeftThumbEnd
    [+0.9137, +0.4260, -0.0132],  # 15 LeftMiddleEnd
    [-0.0139, +0.4286, +0.0431],  # 16 RightShoulder
    [-0.1643, +0.4286, -0.0123],  # 17 RightArm
    [-0.4517, +0.4286, -0.0123],  # 18 RightForeArm
    [-0.7230, +0.4286, -0.0123],  # 19 RightHand
    [-0.8457, +0.3965, +0.0357],  # 20 RightThumbEnd
    [-0.9130, +0.4255, -0.0126],  # 21 RightMiddleEnd
    [+0.1004, -0.0843, +0.0260],  # 22 LeftLeg
    [+0.1004, -0.5166, +0.0179],  # 23 LeftShin
    [+0.1004, -0.9381, -0.0169],  # 24 LeftFoot
    [+0.1004, -0.9887, +0.1154],  # 25 LeftToeBase
    [-0.1005, -0.0830, +0.0262],  # 26 RightLeg
    [-0.1005, -0.5166, +0.0181],  # 27 RightShin
    [-0.1005, -0.9377, -0.0166],  # 28 RightFoot
    [-0.1005, -0.9885, +0.1162],  # 29 RightToeBase
])

EFFECTOR_INDICES = {
    "LeftHand": 13,
    "RightHand": 19,
    "LeftFoot": 24,
    "RightFoot": 28,
}


def fk_joint_position(local_rots_aa, target_joint_idx, root_pos=None):
    """Forward-kinematic a single joint's world position.

    Args:
        local_rots_aa: (30, 3) axis-angle local rotations
        target_joint_idx: which joint to compute
        root_pos: (3,) root position, default [0,0,0]

    Returns:
        (3,) world position of the target joint
    """
    if root_pos is None:
        root_pos = np.zeros(3)

    # Build chain from root to target
    chain = []
    idx = target_joint_idx
    while idx >= 0:
        chain.append(idx)
        idx = SOMA30_PARENTS[idx]
    chain.reverse()

    # FK: accumulate world rotation and position
    world_pos = root_pos.copy()
    world_rot = R.identity()

    for i in range(1, len(chain)):
        parent_idx = chain[i - 1]
        child_idx = chain[i]
        # Local offset in parent's frame (from T-pose neutral)
        local_offset = SOMA30_NEUTRAL[child_idx] - SOMA30_NEUTRAL[parent_idx]
        # Apply parent's world rotation to offset
        world_pos = world_pos + world_rot.apply(local_offset)
        # Apply this joint's local rotation
        rot_aa = local_rots_aa[child_idx]
        if np.linalg.norm(rot_aa) > 1e-8:
            world_rot = world_rot * R.from_rotvec(rot_aa)

    return world_pos


def compute_root_correction(local_rots_aa, effector_name, desired_pos):
    """Compute root_positions so FK places the effector at desired_pos.

    Args:
        local_rots_aa: (30, 3) axis-angle rotations
        effector_name: "LeftHand", "RightHand", "LeftFoot", "RightFoot"
        desired_pos: (3,) desired effector world position in Kimodo meters

    Returns:
        (3,) corrected root_positions
    """
    eff_idx = EFFECTOR_INDICES[effector_name]
    # FK from origin to find the effector offset
    fk_offset = fk_joint_position(local_rots_aa, eff_idx, root_pos=np.zeros(3))
    # root_positions = desired - fk_offset ensures FK(root, rots) = desired
    return desired_pos - fk_offset


if __name__ == "__main__":
    # Quick sanity test
    identity_rots = np.zeros((30, 3))

    # With identity rotations, FK should give neutral positions
    for name, idx in EFFECTOR_INDICES.items():
        pos = fk_joint_position(identity_rots, idx)
        expected = SOMA30_NEUTRAL[idx]
        error = np.linalg.norm(pos - expected)
        print(f"{name} (idx={idx}): FK={pos}, neutral={expected}, error={error:.6f}")

    print("\nAll should be ~0 error (identity rots → T-pose positions)")
