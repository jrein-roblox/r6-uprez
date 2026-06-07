"""Unit tests for Roblox ↔ Kimodo position round-trip.

Validates that positions placed in Studio come back at the same location
after going through: Plugin → Server → Kimodo → Retarget → Studio.

Run: cd server && python3 test_coordinates.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

# --- Constants (must match server/routes/generate.py) ---
CM_TO_STUD = 0.03
STUD_TO_METER = 0.30
SOMA_BIND_CHAIN = 2.643
TARGET_CHAIN = 3.6693
HRP_SCALE = TARGET_CHAIN / SOMA_BIND_CHAIN  # ~1.388
STUD_TO_KIMODO = 1.0 / (100.0 * CM_TO_STUD * HRP_SCALE)  # ~0.24

# SOMA reference
SOMA_FLOOR_Y = -0.938  # meters (LeftFoot neutral Y)
SOMA_HEAD_Y = 0.598
SOMA_HAND_Y = 0.429  # T-pose hand height


def roblox_to_kimodo(pos_hrp_relative_studs):
    """Plugin → Server conversion: HRP-relative studs to Kimodo meters."""
    x, y, z = pos_hrp_relative_studs
    return np.array([-x * STUD_TO_KIMODO, y * STUD_TO_KIMODO, -z * STUD_TO_KIMODO])


def kimodo_to_roblox(pos_kimodo_meters):
    """Retarget output → Studio: Kimodo meters to HRP-relative studs.

    The retarget applies: cm * CM_TO_STUD * HRP_SCALE to get studs.
    Kimodo meters → cm = *100. Axis flip reverses.
    """
    retarget_scale = 100.0 * CM_TO_STUD * HRP_SCALE  # 4.165 studs per meter
    kx, ky, kz = pos_kimodo_meters
    return np.array([-kx * retarget_scale, ky * retarget_scale, -kz * retarget_scale])


def test_round_trip(name, roblox_pos, tolerance=0.01):
    """Verify: Roblox → Kimodo → Retarget → Roblox matches original."""
    roblox_pos = np.array(roblox_pos, dtype=float)
    kimodo = roblox_to_kimodo(roblox_pos)
    roblox_back = kimodo_to_roblox(kimodo)
    error = np.max(np.abs(roblox_back - roblox_pos))
    passed = error < tolerance
    return passed, error, kimodo, roblox_back


def test_kimodo_sanity(name, roblox_pos, expected_kimodo_range):
    """Verify the Kimodo position is in a reasonable range for SOMA."""
    kimodo = roblox_to_kimodo(roblox_pos)
    ky = kimodo[1]
    y_min, y_max = expected_kimodo_range
    in_range = y_min <= ky <= y_max
    return in_range, ky


def run_all_tests():
    print("=" * 70)
    print("Roblox ↔ Kimodo Coordinate Round-Trip Tests")
    print("=" * 70)
    print(f"Scale: STUD_TO_KIMODO = {STUD_TO_KIMODO:.4f}")
    print(f"Retarget: 1 Kimodo meter → {100*CM_TO_STUD*HRP_SCALE:.3f} studs")
    print(f"Round-trip factor: {STUD_TO_KIMODO * 100 * CM_TO_STUD * HRP_SCALE:.6f} (should be 1.0)")
    print()

    all_passed = True
    fails = []

    # === SECTION 1: Round-trip accuracy ===
    print("--- Round-Trip Tests (should all be < 0.01 stud error) ---")
    cases = [
        ("Origin (at HRP)", [0, 0, 0]),
        ("Right hand at rest (hanging)", [1.5, -1.0, 0]),
        ("Left hand at rest", [-1.5, -1.0, 0]),
        ("Foot on ground", [0.5, -4.0, 0.5]),
        ("Foot on ground (left)", [-0.5, -4.0, 0.3]),
        ("Hand raised high (waving)", [2.0, 2.0, 0]),
        ("Hand overhead", [0.5, 3.0, -0.3]),
        ("Handstand: hand on floor", [0.5, -4.0, -0.5]),
        ("Handstand: hips", [0, -0.5, 0]),
        ("Handstand: feet in air", [-0.5, 3.0, 0]),
        ("Walking 5 studs forward", [0, -0.5, -5.0]),
        ("Walking 10 studs forward", [0, -0.5, -10.0]),
        ("Lateral movement", [3.0, -1.0, -2.0]),
        ("Far away", [5.0, -2.0, -15.0]),
        ("Behind character", [0, 0, 5.0]),
        ("Crouching (hips low)", [0, -2.0, 0]),
        ("Jumping (hips high)", [0, 2.0, -1.0]),
    ]

    for name, pos in cases:
        passed, error, kimodo, back = test_round_trip(name, pos)
        status = "✓" if passed else "✗"
        print(f"  [{status}] {name}: error={error:.6f}")
        if not passed:
            print(f"      in={pos} → kimodo={kimodo.tolist()} → out={back.tolist()}")
            all_passed = False
            fails.append(name)

    # === SECTION 2: Kimodo Y sanity (values should be in SOMA range) ===
    print()
    print("--- Kimodo Y Sanity Tests (positions make sense in SOMA space) ---")
    sanity_cases = [
        ("Standing hips (at HRP)", [0, 0, 0], (-0.2, 0.2)),  # near SOMA hips (Y=0)
        ("Foot on ground", [0.5, -4.0, 0], (-1.1, -0.8)),  # near SOMA floor (-0.938)
        ("Hand at rest", [1.5, -1.0, 0], (-0.4, 0.0)),  # between hips and floor
        ("Hand raised overhead", [0.5, 3.0, 0], (0.5, 1.0)),  # above head
        ("Handstand hand on floor", [0.5, -4.2, 0], (-1.2, -0.8)),  # near floor
        ("Handstand feet in air", [0, 2.5, 0], (0.4, 0.8)),  # above head
        ("Jumping hips", [0, 2.0, 0], (0.3, 0.7)),  # hips elevated
    ]

    for name, pos, (y_min, y_max) in sanity_cases:
        in_range, ky = test_kimodo_sanity(name, pos, (y_min, y_max))
        status = "✓" if in_range else "✗"
        print(f"  [{status}] {name}: Kimodo Y={ky:.3f}m (expected [{y_min:.1f}, {y_max:.1f}])")
        if not in_range:
            all_passed = False
            fails.append(f"Sanity: {name}")

    # === SECTION 3: Specific handstand/cartwheel case from user ===
    print()
    print("--- Handstand/Cartwheel Case (from real Studio data) ---")
    # Real data: HRP at Y=4.113. Chain parts at world positions:
    # LT: Y=3.659 → HRP-relative: 3.659-4.113 = -0.454
    # RightHand: Y=-0.109 → HRP-relative: -0.109-4.113 = -4.222
    # LeftFoot: Y=6.264 → HRP-relative: 6.264-4.113 = +2.151
    # RightFoot: Y=5.360 → HRP-relative: 5.360-4.113 = +1.247

    handstand_cases = [
        ("Hips (LT)", [-0.451 + 0, -0.454, -7.781 + 0]),  # XZ from HRP center
        ("RightHand (on floor)", [0.194 + 0, -4.222, -6.442 + 0]),
        ("LeftHand (on floor)", [0.673 + 0, -4.211, -8.183 + 0]),
        ("LeftFoot (in air)", [0.035 + 0, +2.151, -9.785 + 0]),
        ("RightFoot (in air)", [-1.595 + 0, +1.247, -5.639 + 0]),
    ]

    print("  (HRP-relative studs → Kimodo meters)")
    for name, pos in handstand_cases:
        kimodo = roblox_to_kimodo(pos)
        passed_rt, error, _, back = test_round_trip(name, pos)
        print(f"  {name}:")
        print(f"    Roblox (HRP-rel): ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) studs")
        print(f"    Kimodo:           ({kimodo[0]:+.4f}, {kimodo[1]:+.4f}, {kimodo[2]:+.4f}) meters")
        print(f"    Round-trip error:  {error:.6f} {'✓' if passed_rt else '✗'}")
        if not passed_rt:
            all_passed = False
            fails.append(f"Handstand: {name}")

    # Sanity check: hands should be near SOMA floor, feet above head
    hips_k = roblox_to_kimodo(handstand_cases[0][1])
    rhand_k = roblox_to_kimodo(handstand_cases[1][1])
    lfoot_k = roblox_to_kimodo(handstand_cases[3][1])

    print()
    print(f"  Hips Y in Kimodo: {hips_k[1]:+.3f}m (SOMA hips=0, floor=-0.938)")
    print(f"  RightHand Y:      {rhand_k[1]:+.3f}m (should be near floor)")
    print(f"  LeftFoot Y:       {lfoot_k[1]:+.3f}m (should be above head)")

    hands_ok = rhand_k[1] < -0.7  # hands near floor
    feet_ok = lfoot_k[1] > 0.3    # feet above head
    hips_ok = -0.3 < hips_k[1] < 0.3  # hips near neutral

    for check, label in [(hands_ok, "Hands near floor"), (feet_ok, "Feet above head"), (hips_ok, "Hips near neutral")]:
        status = "✓" if check else "✗"
        print(f"  [{status}] {label}")
        if not check:
            all_passed = False
            fails.append(f"Handstand sanity: {label}")

    # === Summary ===
    print()
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"FAILED ({len(fails)} failures):")
        for f in fails:
            print(f"  ✗ {f}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
