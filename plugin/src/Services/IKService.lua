--!strict
-- Analytic 2-bone IK for the limb chains, used by the AnimationBuilder bake to
-- make R15 end-effectors reach the model's (SOMA) output positions despite the
-- rotation-only retarget over a differently-proportioned skeleton.
--
-- It works in deltas: it rotates the current upper/lower bone directions to hit
-- the target, preserving each part's twist (so the retarget's wrist/elbow roll
-- survives) and only correcting reach. Reach is clamped so the arm never locks
-- straight or overreaches.

local IKService = {}

local EPS = 1e-5

-- Minimal rotation that takes unit-ish vector `a` onto `b` (world space).
local function rotationBetween(a: Vector3, b: Vector3): CFrame
	if a.Magnitude < EPS or b.Magnitude < EPS then
		return CFrame.identity
	end
	local u, v = a.Unit, b.Unit
	local d = math.clamp(u:Dot(v), -1, 1)
	if d > 0.999999 then
		return CFrame.identity
	end
	if d < -0.999999 then
		-- 180°: pick any axis perpendicular to u.
		local perp = u:Cross(Vector3.new(1, 0, 0))
		if perp.Magnitude < EPS then
			perp = u:Cross(Vector3.new(0, 0, 1))
		end
		return CFrame.fromAxisAngle(perp.Unit, math.pi)
	end
	return CFrame.fromAxisAngle(u:Cross(v).Unit, math.acos(d))
end

-- Rigidly rotate `cf` about world-space `pivot` by rotation `delta`.
local function rotateAbout(pivot: Vector3, delta: CFrame, cf: CFrame): CFrame
	return CFrame.new(pivot) * delta * CFrame.new(-pivot) * cf
end

-- Solve the chain so the wrist reaches `target`.
--   shoulderPos/elbowPos/wristPos : current joint world positions
--   target                        : desired wrist world position
--   upperCF/lowerCF               : current world CFrames of the upper/lower parts
--   poleHint (optional)           : world point biasing the bend plane;
--                                   defaults to the current elbow (preserve bend)
-- Returns the corrected upper/lower world CFrames and the achieved wrist pos.
function IKService.solve(
	shoulderPos: Vector3,
	elbowPos: Vector3,
	wristPos: Vector3,
	target: Vector3,
	upperCF: CFrame,
	lowerCF: CFrame,
	poleHint: Vector3?
): (CFrame, CFrame, Vector3)
	local l1 = (elbowPos - shoulderPos).Magnitude
	local l2 = (wristPos - elbowPos).Magnitude
	if l1 < EPS or l2 < EPS then
		return upperCF, lowerCF, wristPos
	end

	-- Clamp reach so we never lock straight or exceed the chain.
	local toTarget = target - shoulderPos
	local dist = toTarget.Magnitude
	if dist < EPS then
		return upperCF, lowerCF, wristPos
	end
	local minReach = math.abs(l1 - l2) + 1e-3
	local maxReach = (l1 + l2) - 1e-3
	dist = math.clamp(dist, minReach, maxReach)
	local forward = toTarget.Unit
	local clampedTarget = shoulderPos + forward * dist

	-- AIM FIRST: rigidly rotate the whole arm so the wrist points at the target.
	-- This preserves the arm's natural bend exactly, so the bend side can't flip
	-- (unlike reconstructing the plane against the new forward). The bend
	-- direction is then taken from the AIMED elbow's perpendicular offset.
	local aim = rotationBetween(wristPos - shoulderPos, clampedTarget - shoulderPos)
	local aimedElbow = shoulderPos + aim:VectorToWorldSpace(elbowPos - shoulderPos)
	local bendDir = (aimedElbow - shoulderPos) - forward * ((aimedElbow - shoulderPos):Dot(forward))
	if bendDir.Magnitude < EPS then
		-- Aimed arm is straight along forward: use a pole hint or a stable default.
		local ref = if poleHint then (poleHint - shoulderPos) else Vector3.new(0, -1, 0)
		bendDir = ref - forward * ref:Dot(forward)
		if bendDir.Magnitude < EPS then
			bendDir = forward:Cross(Vector3.new(1, 0, 0))
		end
	end
	bendDir = bendDir.Unit

	-- Law of cosines: angle at the shoulder between forward and the upper bone.
	local cos0 = math.clamp((l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist), -1, 1)
	local sin0 = math.sqrt(math.max(0, 1 - cos0 * cos0))
	local newElbow = shoulderPos + forward * (l1 * cos0) + bendDir * (l1 * sin0)

	-- Upper bone: rotate (curElbow→) onto (newElbow→) about the shoulder.
	local delta1 = rotationBetween(elbowPos - shoulderPos, newElbow - shoulderPos)
	local newUpperCF = rotateAbout(shoulderPos, delta1, upperCF)

	-- Lower bone: after delta1 the lower part has rotated rigidly about the
	-- shoulder; now rotate its wrist onto the target about the new elbow.
	local lowerAfter1 = rotateAbout(shoulderPos, delta1, lowerCF)
	local wristAfter1 = shoulderPos + delta1:VectorToWorldSpace(wristPos - shoulderPos)
	local delta2 = rotationBetween(wristAfter1 - newElbow, clampedTarget - newElbow)
	local newLowerCF = rotateAbout(newElbow, delta2, lowerAfter1)

	local achievedWrist = newElbow + delta2:VectorToWorldSpace(wristAfter1 - newElbow)
	return newUpperCF, newLowerCF, achievedWrist
end

return IKService
