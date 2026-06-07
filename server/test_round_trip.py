"""End-to-end round-trip test: Roblox hand position → Kimodo constraint → FK verify.

Tests that after retargeting rotations and computing root correction,
the SOMA FK places the effector at the expected Kimodo position.

Run: cd server && python3 test_round_trip.py
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from soma_fk import fk_joint_position, compute_root_correction, EFFECTOR_INDICES
import roblox_to_kimodo as r2k
from vendor.quat import to_scaled_angle_axis

# Scale constants
CM_TO_STUD = 0.03
STUD_TO_METER = 0.30
SOMA_BIND_CHAIN = 2.643
TARGET_CHAIN = 3.6693
HRP_SCALE = TARGET_CHAIN / SOMA_BIND_CHAIN
STUD_TO_KIMODO_XZ = 1.0 / (100.0 * CM_TO_STUD * HRP_SCALE)
STUD_TO_KIMODO_Y = STUD_TO_METER


def simulate_plugin_to_kimodo(chain_world_cframes, effector, rig_rest_ground_y=0.0, hrp_y=4.113):
    """Simulate the full plugin→server constraint building.

    Args:
        chain_world_cframes: list of (name, world_pos_studs, world_quat_xyzw)
        effector: "LeftHand", "RightHand", "LeftFoot", "RightFoot"
        rig_rest_ground_y: floor Y in world studs
        hrp_y: HRP world Y

    Returns:
        (root_positions, local_rots, effector_fk_pos) in Kimodo meters
    """
    # Pascal→snake mapping
    PASCAL_TO_SNAKE = {
        "LowerTorso": "lower_torso", "UpperTorso": "upper_torso",
        "LeftUpperArm": "left_upper_arm", "LeftLowerArm": "left_lower_arm",
        "LeftHand": "left_hand", "RightUpperArm": "right_upper_arm",
        "RightLowerArm": "right_lower_arm", "RightHand": "right_hand",
        "LeftUpperLeg": "left_upper_leg", "LeftLowerLeg": "left_lower_leg",
        "LeftFoot": "left_foot", "RightUpperLeg": "right_upper_leg",
        "RightLowerLeg": "right_lower_leg", "RightFoot": "right_foot",
    }

    EFF_MAP = {
        "LeftHand": "left-hand", "RightHand": "right-hand",
        "LeftFoot": "left-foot", "RightFoot": "right-foot",
    }

    # Convert positions (XZ: HRP-relative * 0.24, Y: floor-relative * 0.30)
    # and quaternions (180° Y conjugation)
    hrp_x, hrp_z = 0.0, 0.0  # assume HRP at origin XZ
    chain_pos_kimodo = {}
    chain_quat_kimodo = {}

    for name, world_pos, world_quat in chain_world_cframes:
        snake = PASCAL_TO_SNAKE[name]
        px = world_pos[0] - hrp_x  # HRP-relative X
        py = world_pos[1] - rig_rest_ground_y  # floor-relative Y
        pz = world_pos[2] - hrp_z  # HRP-relative Z
        chain_pos_kimodo[snake] = np.array([
            -px * STUD_TO_KIMODO_XZ,
             py * STUD_TO_KIMODO_Y,
            -pz * STUD_TO_KIMODO_XZ,
        ])
        qx, qy, qz, qw = world_quat
        chain_quat_kimodo[snake] = np.array([qw, -qx, qy, -qz])

    # Retarget rotations
    ctype = EFF_MAP[effector]
    R15_CHAINS = r2k.R15_CHAINS
    chain_def = R15_CHAINS[ctype]
    chain_rots = {
        src_key: chain_quat_kimodo[src_key]
        for _, src_key in chain_def
        if src_key and src_key in chain_quat_kimodo
    }

    local_rots = np.zeros((30, 3))
    if chain_rots:
        quats = r2k._retarget_chain_quats(chain_def, chain_rots)
        for soma_idx, q in quats.items():
            if q[0] < 0:
                q = -q
            local_rots[soma_idx] = to_scaled_angle_axis(q)

    # Root position from LowerTorso
    if "lower_torso" in chain_pos_kimodo:
        root_pos = chain_pos_kimodo["lower_torso"]
    else:
        root_pos = np.array([0.0, 0.9, 0.0])

    # FK to find where the effector lands
    eff_idx = EFFECTOR_INDICES[effector]
    fk_pos = fk_joint_position(local_rots, eff_idx, root_pos)

    # Desired effector position (from plugin chain data)
    desired_effector_snake = PASCAL_TO_SNAKE[effector]
    desired_pos = chain_pos_kimodo.get(desired_effector_snake)

    return root_pos, local_rots, fk_pos, desired_pos


def test_case(name, chain_data, effector, tolerance=0.15):
    """Run a single test case and report drift."""
    root_pos, local_rots, fk_pos, desired_pos = simulate_plugin_to_kimodo(
        chain_data, effector
    )

    if desired_pos is None:
        print(f"  [SKIP] {name}: no desired position for effector")
        return True

    error = fk_pos - desired_pos
    error_mag = np.linalg.norm(error)
    passed = error_mag < tolerance

    status = "✓" if passed else "✗"
    print(f"  [{status}] {name}")
    print(f"      desired: ({desired_pos[0]:+.4f}, {desired_pos[1]:+.4f}, {desired_pos[2]:+.4f})m")
    print(f"      FK got:  ({fk_pos[0]:+.4f}, {fk_pos[1]:+.4f}, {fk_pos[2]:+.4f})m")
    print(f"      error:   ({error[0]:+.4f}, {error[1]:+.4f}, {error[2]:+.4f}) mag={error_mag:.4f}m = {error_mag/STUD_TO_KIMODO_XZ:.2f} studs")
    if not passed:
        print(f"      EXCEEDS tolerance {tolerance}m!")

    return passed


def test_with_root_correction(name, chain_data, effector, tolerance=0.01):
    """Test with root correction applied."""
    root_pos, local_rots, fk_pos, desired_pos = simulate_plugin_to_kimodo(
        chain_data, effector
    )

    if desired_pos is None:
        return True

    # Apply root correction
    corrected_root = compute_root_correction(local_rots, effector, desired_pos)
    corrected_fk = fk_joint_position(local_rots, EFFECTOR_INDICES[effector], corrected_root)

    error = corrected_fk - desired_pos
    error_mag = np.linalg.norm(error)
    passed = error_mag < tolerance

    status = "✓" if passed else "✗"
    print(f"  [{status}] {name} (corrected)")
    print(f"      root before: ({root_pos[0]:+.4f}, {root_pos[1]:+.4f}, {root_pos[2]:+.4f})")
    print(f"      root after:  ({corrected_root[0]:+.4f}, {corrected_root[1]:+.4f}, {corrected_root[2]:+.4f})")
    print(f"      FK error:    {error_mag:.6f}m")

    return passed


def make_chain_data(lt_pos, ut_pos, ua_pos, la_pos, hand_pos,
                    lt_quat=(0,0,0,1), ut_quat=(0,0,0,1),
                    ua_quat=(0,0,0,1), la_quat=(0,0,0,1), hand_quat=(0,0,0,1),
                    effector="RightHand"):
    """Helper to build chain data for right hand tests."""
    if effector == "RightHand":
        return [
            ("LowerTorso", lt_pos, lt_quat),
            ("UpperTorso", ut_pos, ut_quat),
            ("RightUpperArm", ua_pos, ua_quat),
            ("RightLowerArm", la_pos, la_quat),
            ("RightHand", hand_pos, hand_quat),
        ]
    elif effector == "LeftHand":
        return [
            ("LowerTorso", lt_pos, lt_quat),
            ("UpperTorso", ut_pos, ut_quat),
            ("LeftUpperArm", ua_pos, ua_quat),
            ("LeftLowerArm", la_pos, la_quat),
            ("LeftHand", hand_pos, hand_quat),
        ]
    elif effector == "RightFoot":
        return [
            ("LowerTorso", lt_pos, lt_quat),
            ("RightUpperLeg", ua_pos, ua_quat),
            ("RightLowerLeg", la_pos, la_quat),
            ("RightFoot", hand_pos, hand_quat),
        ]
    elif effector == "LeftFoot":
        return [
            ("LowerTorso", lt_pos, lt_quat),
            ("LeftUpperLeg", ua_pos, ua_quat),
            ("LeftLowerLeg", la_pos, la_quat),
            ("LeftFoot", hand_pos, hand_quat),
        ]


def run_tests():
    print("=" * 70)
    print("Effector Position Round-Trip Tests")
    print("=" * 70)

    all_passed = True

    # Test 1: Rest pose (hand at side) - from real MCP data
    print("\n--- Without root correction (shows natural drift) ---")
    rest_chain = make_chain_data(
        lt_pos=(0, 3.58, 0), ut_pos=(0, 4.77, -0.2),
        ua_pos=(1.23, 5.67, -0.16), la_pos=(1.54, 4.98, -0.1),
        hand_pos=(1.37, 4.88, -0.05), effector="RightHand"
    )
    test_case("Rest pose hand at side", rest_chain, "RightHand", tolerance=0.3)

    # Test 2: Hand raised (waving)
    wave_chain = make_chain_data(
        lt_pos=(0, 3.58, 0), ut_pos=(0, 4.77, -0.2),
        ua_pos=(1.0, 6.5, -0.1), la_pos=(0.8, 7.2, -0.2),
        hand_pos=(0.5, 7.8, -0.3), effector="RightHand"
    )
    test_case("Hand raised (waving)", wave_chain, "RightHand", tolerance=0.3)

    # Test 3: Handstand - hand on floor
    handstand_chain = make_chain_data(
        lt_pos=(-0.45, 3.66, -7.78), ut_pos=(-0.21, 2.49, -7.72),
        ua_pos=(-0.22, 1.44, -6.78), la_pos=(-0.19, 0.53, -6.59),
        hand_pos=(0.19, -0.11, -6.44), effector="RightHand"
    )
    test_case("Handstand hand on floor", handstand_chain, "RightHand", tolerance=0.3)

    # --- Now with root correction ---
    print("\n--- With root correction (should be near-zero error) ---")
    all_passed &= test_with_root_correction("Rest pose", rest_chain, "RightHand")
    all_passed &= test_with_root_correction("Hand waving", wave_chain, "RightHand")
    all_passed &= test_with_root_correction("Handstand on floor", handstand_chain, "RightHand")

    # Test foot constraints too
    foot_chain = [
        ("LowerTorso", (0, 3.58, 0), (0, 0, 0, 1)),
        ("RightUpperLeg", (-0.5, 2.8, 0.1), (0, 0, 0, 1)),
        ("RightLowerLeg", (-0.5, 1.5, 0.3), (0, 0, 0, 1)),
        ("RightFoot", (-0.5, 0.3, 0.5), (0, 0, 0, 1)),
    ]
    all_passed &= test_with_root_correction("Foot on ground", foot_chain, "RightFoot")

    # Handstand feet in air
    handstand_foot = [
        ("LowerTorso", (-0.45, 3.66, -7.78), (0, 0, 0, 1)),
        ("LeftUpperLeg", (-0.29, 4.42, -8.46), (0, 0, 0, 1)),
        ("LeftLowerLeg", (-0.04, 5.62, -9.08), (0, 0, 0, 1)),
        ("LeftFoot", (0.04, 6.26, -9.79), (0, 0, 0, 1)),
    ]
    all_passed &= test_with_root_correction("Handstand foot in air", handstand_foot, "LeftFoot")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CORRECTED TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
