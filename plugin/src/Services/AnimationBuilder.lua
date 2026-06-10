--!strict
-- Builds a CurveAnimation instance from r15.json data received from the backend.
-- This is the Lua equivalent of lua/build_rbxm.lua but runs inside the plugin.
--
-- When a rig + hand constraints are supplied, an IK bake pass drives each
-- constrained hand to its gizmo transform (position + orientation), blended in
-- and out over a few frames around the constraint, so a constrained hand lands
-- exactly where the user placed it. Everything else keeps the raw retarget.

local RigService = require(script.Parent.RigService)
local IKService = require(script.Parent.IKService)

local AnimationBuilder = {}
AnimationBuilder.DEBUG = false
-- Half-window (frames each side of a constraint) over which the IK correction
-- ramps from the raw retarget up to the full constraint and back.
AnimationBuilder.BLEND_FRAMES = 5

-- Limb chains the IK can drive: effector key, chain base, and the posed parts
-- (`hand` is the leaf end-effector part — hand or foot).
local LIMBS = {
	{ eff = "LeftHand",  base = "UpperTorso", upper = "LeftUpperArm",  lower = "LeftLowerArm",  hand = "LeftHand" },
	{ eff = "RightHand", base = "UpperTorso", upper = "RightUpperArm", lower = "RightLowerArm", hand = "RightHand" },
	{ eff = "LeftFoot",  base = "LowerTorso", upper = "LeftUpperLeg",  lower = "LeftLowerLeg",  hand = "LeftFoot" },
	{ eff = "RightFoot", base = "LowerTorso", upper = "RightUpperLeg", lower = "RightLowerLeg", hand = "RightFoot" },
}
local LIMB_EFF = { LeftHand = true, RightHand = true, LeftFoot = true, RightFoot = true }

-- R15 hierarchy: child -> parent
local PART_PARENT = {
	LowerTorso = "HumanoidRootPart",
	UpperTorso = "LowerTorso",
	Head = "UpperTorso",
	LeftUpperArm = "UpperTorso",
	LeftLowerArm = "LeftUpperArm",
	LeftHand = "LeftLowerArm",
	RightUpperArm = "UpperTorso",
	RightLowerArm = "RightUpperArm",
	RightHand = "RightLowerArm",
	LeftUpperLeg = "LowerTorso",
	LeftLowerLeg = "LeftUpperLeg",
	LeftFoot = "LeftLowerLeg",
	RightUpperLeg = "LowerTorso",
	RightLowerLeg = "RightUpperLeg",
	RightFoot = "RightLowerLeg",
}

function AnimationBuilder.build(data: { [string]: any }, rig: RigService.RigInfo?, constraints: { any }?): CurveAnimation
	local parts = data.parts or {}
	local root = data.root
	local frameRate = data.frameRate or 30
	local frameCount = data.frameCount or 0

	-- IK bake: drive each constrained hand to its gizmo transform (blended).
	-- Falls back to the raw retarget if there are no hand constraints.
	if rig and constraints then
		local ok, corrected = pcall(AnimationBuilder._bakeIK, data, rig, constraints)
		if ok and corrected then
			parts = corrected
			print("[RoMotion] IK bake applied (hard-pinned constraints)")
		elseif not ok then
			warn("[RoMotion] IK bake failed, using raw retarget: " .. tostring(corrected))
		end
	end

	local curveAnim = Instance.new("CurveAnimation")
	curveAnim.Name = "RoMotion_Generated"

	-- Build hierarchy of folders matching the R15 joint tree
	local folders: { [string]: Folder } = {}

	local function getOrCreateFolder(partName: string): Folder
		if folders[partName] then
			return folders[partName]
		end

		local parentName = PART_PARENT[partName]
		local parentFolder: Instance

		if parentName then
			parentFolder = getOrCreateFolder(parentName)
		else
			-- Top-level (HumanoidRootPart)
			parentFolder = curveAnim
		end

		local folder = Instance.new("Folder")
		folder.Name = partName
		folder.Parent = parentFolder
		folders[partName] = folder
		return folder
	end

	-- Ensure HumanoidRootPart folder exists
	local hrpFolder = Instance.new("Folder")
	hrpFolder.Name = "HumanoidRootPart"
	hrpFolder.Parent = curveAnim
	folders["HumanoidRootPart"] = hrpFolder

	-- Build curves for root (LowerTorso typically)
	if root then
		local ltFolder = getOrCreateFolder("LowerTorso")
		AnimationBuilder._buildCurvesForPart(ltFolder, root, frameRate, frameCount)
	end

	-- Build curves for each part
	for partName, partData in parts do
		local folder = getOrCreateFolder(partName)
		AnimationBuilder._buildCurvesForPart(folder, partData, frameRate, frameCount)
	end

	return curveAnim
end

function AnimationBuilder._buildCurvesForPart(
	folder: Folder,
	partData: { [string]: any },
	frameRate: number,
	frameCount: number
)
	local hasRot = partData.rotX ~= nil and #partData.rotX > 0
	local hasPos = partData.posX ~= nil and #partData.posX > 0
	if not (hasRot or hasPos) then return end

	local nFrames = 0
	if hasRot then
		nFrames = #partData.rotX
	elseif hasPos then
		nFrames = #partData.posX
	end
	if frameCount > 0 then
		nFrames = math.min(nFrames, frameCount)
	end

	-- Rotation: quaternion (rotX,rotY,rotZ,rotW) -> CFrame -> EulerXYZ
	local rotCurve = Instance.new("EulerRotationCurve")
	rotCurve.Name = "Rotation"
	rotCurve.RotationOrder = Enum.RotationOrder.XYZ
	rotCurve.Parent = folder
	local erx, ery, erz = rotCurve:X(), rotCurve:Y(), rotCurve:Z()

	-- Position: always emit (even if zero) for CurveAnimation compatibility
	local posCurve = Instance.new("Vector3Curve")
	posCurve.Name = "Position"
	posCurve.Parent = folder
	local px, py, pz = posCurve:X(), posCurve:Y(), posCurve:Z()

	local Cubic = Enum.KeyInterpolationMode.Cubic
	for i = 1, nFrames do
		local t = (i - 1) / frameRate

		-- Quaternion -> Euler
		local rx = hasRot and partData.rotX[i] or 0.0
		local ry = hasRot and partData.rotY[i] or 0.0
		local rz = hasRot and partData.rotZ[i] or 0.0
		local rw = hasRot and partData.rotW[i] or 1.0
		local cf = CFrame.new(0, 0, 0, rx, ry, rz, rw)
		local ex, ey, ez = cf:ToEulerAnglesXYZ()
		erx:InsertKey(FloatCurveKey.new(t, ex, Cubic))
		ery:InsertKey(FloatCurveKey.new(t, ey, Cubic))
		erz:InsertKey(FloatCurveKey.new(t, ez, Cubic))

		-- Position (delta from rest, identity = 0,0,0)
		local tx = hasPos and partData.posX[i] or 0.0
		local ty = hasPos and partData.posY[i] or 0.0
		local tz = hasPos and partData.posZ[i] or 0.0
		px:InsertKey(FloatCurveKey.new(t, tx, Cubic))
		py:InsertKey(FloatCurveKey.new(t, ty, Cubic))
		pz:InsertKey(FloatCurveKey.new(t, tz, Cubic))
	end
end

-- Transform CFrame for a part at frame i (1-indexed) from r15.json rot (+pos).
local function transformCF(pd: { [string]: any }?, i: number): CFrame
	if not pd then return CFrame.identity end
	local rx = (pd.rotX and pd.rotX[i]) or 0
	local ry = (pd.rotY and pd.rotY[i]) or 0
	local rz = (pd.rotZ and pd.rotZ[i]) or 0
	local rw = (pd.rotW and pd.rotW[i]) or 1
	local rot = CFrame.new(0, 0, 0, rx, ry, rz, rw)
	if pd.posX and pd.posX[i] then
		return CFrame.new(pd.posX[i], pd.posY[i], pd.posZ[i]) * rot
	end
	return rot
end

local function cfToQuat(cf: CFrame): (number, number, number, number)
	local axis, angle = cf.Rotation:ToAxisAngle()
	local s = math.sin(angle / 2)
	return axis.X * s, axis.Y * s, axis.Z * s, math.cos(angle / 2)
end

-- Replace limb (upper/lower) rotations with IK-corrected ones so each
-- Drive each constrained hand to its gizmo transform, blended over a window.
-- Returns a corrected `parts` table, or nil if there are no hand constraints.
function AnimationBuilder._bakeIK(data: { [string]: any }, rig: RigService.RigInfo, constraints: { any }): { [string]: any }?
	local geom = RigService.getRigGeometry(rig)
	local srcParts = data.parts or {}
	local fps = data.frameRate or 30
	local nFrames = data.frameCount or 0
	if nFrames <= 0 then
		local lt = srcParts.LowerTorso
		nFrames = (lt and lt.rotX and #lt.rotX) or 0
	end

	-- Collect HARD-PINNED constraints only (soft ones are model-influence only).
	-- Limb (hand/foot) pins -> IK; root (Hips/Root) pins -> LowerTorso override.
	local handCons: { [string]: { { frame: number, cf: CFrame } } } = {}
	local rootCons: { { frame: number, cf: CFrame, mode: string } } = {}
	for _, c in constraints do
		if c.pinned and c.gizmo and c.gizmo.effectorPart then
			local frame = math.floor((c.time or 0) * fps + 0.5)
			if LIMB_EFF[c.effector] then
				local list = handCons[c.effector] or {}
				table.insert(list, { frame = frame, cf = c.gizmo.effectorPart.CFrame })
				handCons[c.effector] = list
			elseif c.effector == "Hips" or c.effector == "Root" then
				table.insert(rootCons, { frame = frame, cf = c.gizmo.effectorPart.CFrame, mode = c.effector })
			end
		end
	end
	if not next(handCons) and #rootCons == 0 then return nil end

	-- Active limbs = constrained hands/feet; deep-copy their parts for rewriting.
	local active = {}
	for _, limb in LIMBS do
		if handCons[limb.eff] then table.insert(active, limb) end
	end
	local out: { [string]: any } = {}
	for name, pd in srcParts do out[name] = pd end
	for _, limb in active do
		for _, p in { limb.upper, limb.lower, limb.hand } do
			local src = srcParts[p]
			if src then
				local c: { [string]: any } = {}
				for k, v in src do c[k] = v end
				c.rotX, c.rotY, c.rotZ, c.rotW = {}, {}, {}, {}
				out[p] = c
			end
		end
	end
	-- Root pins override the LowerTorso curve (it carries the folded root
	-- motion), so deep-copy its rotation AND position tracks for rewriting.
	if #rootCons > 0 and geom.LowerTorso then
		local src = srcParts.LowerTorso
		if src then
			local c: { [string]: any } = {}
			for k, v in src do c[k] = v end
			c.rotX, c.rotY, c.rotZ, c.rotW = {}, {}, {}, {}
			if src.posX then c.posX, c.posY, c.posZ = {}, {}, {} end
			out.LowerTorso = c
		end
	end

	local BLEND = AnimationBuilder.BLEND_FRAMES
	local function jointWorld(parentCF: CFrame, partName: string, i: number): CFrame?
		local g = geom[partName]
		if not g then return nil end
		return parentCF * g.c0 * transformCF(srcParts[partName], i) * g.c1:Inverse()
	end

	-- Root motion is folded into LowerTorso, so the HRP stays at the rig's
	-- world CFrame — FK from there to get world-space poses matching playback
	-- (and the gizmos, which live in the same world).
	local hrp = rig.rootPart.CFrame

	local ltGeom = geom.LowerTorso
	for i = 1, nFrames do
		local rawLt = jointWorld(hrp, "LowerTorso", i)
		if not rawLt then break end

		-- ── Root pins (Hips / Root): override the LowerTorso world CFrame, then
		-- FK everything else from the corrected pelvis so limbs follow it. ──
		local ltCF = rawLt
		if #rootCons > 0 and ltGeom then
			local targetLt, sumW = nil, 0
			for _, con in rootCons do
				local cw = 1 - math.abs((i - 1) - con.frame) / (BLEND + 1)
				if cw > 0 then
					local desired: CFrame
					if con.mode == "Hips" then
						desired = con.cf -- full 3D pelvis pin
					else
						-- Root: pin XZ + yaw from the ground arrow, keep raw Y + lean.
						local _, arrowYaw = con.cf:ToEulerAnglesYXZ()
						local _, rawYaw = rawLt:ToEulerAnglesYXZ()
						desired = CFrame.new(con.cf.X, rawLt.Position.Y, con.cf.Z)
							* CFrame.Angles(0, arrowYaw - rawYaw, 0)
							* rawLt.Rotation
					end
					if not targetLt then
						targetLt, sumW = desired, cw
					else
						sumW += cw
						targetLt = targetLt:Lerp(desired, cw / sumW)
					end
				end
			end
			local w = math.clamp(sumW, 0, 1)
			if w > 0 and targetLt then
				ltCF = rawLt:Lerp(targetLt, w)
			end
			-- Write the (corrected or raw) LowerTorso transform back.
			local tLt = ltGeom.c0:Inverse() * hrp:Inverse() * ltCF * ltGeom.c1
			local o = out.LowerTorso
			o.rotX[i], o.rotY[i], o.rotZ[i], o.rotW[i] = cfToQuat(tLt)
			if o.posX then
				o.posX[i], o.posY[i], o.posZ[i] = tLt.Position.X, tLt.Position.Y, tLt.Position.Z
			end
		end

		local utCF = jointWorld(ltCF, "UpperTorso", i) or ltCF

		for _, limb in active do
			local gu, gl, gh = geom[limb.upper], geom[limb.lower], geom[limb.hand]
			local upperOut, lowerOut, handOut = out[limb.upper], out[limb.lower], out[limb.hand]
			local function keepRaw(partName: string, dst: { [string]: any }?)
				local s = srcParts[partName]
				if dst and s then
					dst.rotX[i], dst.rotY[i], dst.rotZ[i], dst.rotW[i] = s.rotX[i], s.rotY[i], s.rotZ[i], s.rotW[i]
				end
			end

			-- Gather ALL constraints influencing this frame and blend their gizmos
			-- into one weighted target (so dense constraints don't jerk as the
			-- "nearest" one switches). Each contributes weight by proximity; the
			-- combined weight `w` (capped at 1) blends raw → that target.
			local gizmoCF, sumW = nil, 0
			for _, con in handCons[limb.eff] do
				local cw = 1 - math.abs((i - 1) - con.frame) / (BLEND + 1)
				if cw > 0 then
					if not gizmoCF then
						gizmoCF, sumW = con.cf, cw
					else
						sumW += cw
						gizmoCF = gizmoCF:Lerp(con.cf, cw / sumW) -- incremental weighted avg
					end
				end
			end
			local w = math.clamp(sumW, 0, 1)

			local baseCF = if limb.base == "LowerTorso" then ltCF else utCF
			if w > 0 and gizmoCF and gu and gl and gh and upperOut and lowerOut then
				local upperCF = baseCF * gu.c0 * transformCF(srcParts[limb.upper], i) * gu.c1:Inverse()
				local lowerCF = upperCF * gl.c0 * transformCF(srcParts[limb.lower], i) * gl.c1:Inverse()
				local rawHandCF = lowerCF * gh.c0 * transformCF(srcParts[limb.hand], i) * gh.c1:Inverse()
				local shoulderPos = (baseCF * gu.c0).Position
				local elbowPos = (upperCF * gl.c0).Position
				local wristPos = (lowerCF * gh.c0).Position

				-- Blend the desired hand transform from raw → gizmo by the weight.
				local desiredHandWorld = rawHandCF:Lerp(gizmoCF, w)
				-- IK drives the wrist JOINT; back it off by the hand's wrist
				-- attachment (c1) so the hand-part CENTER lands on the target.
				local wristTarget = (desiredHandWorld * gh.c1).Position

				local newUpper, newLower = IKService.solve(
					shoulderPos, elbowPos, wristPos, wristTarget, upperCF, lowerCF
				)
				local tU = gu.c0:Inverse() * baseCF:Inverse() * newUpper * gu.c1
				local tL = gl.c0:Inverse() * newUpper:Inverse() * newLower * gl.c1
				upperOut.rotX[i], upperOut.rotY[i], upperOut.rotZ[i], upperOut.rotW[i] = cfToQuat(tU)
				lowerOut.rotX[i], lowerOut.rotY[i], lowerOut.rotZ[i], lowerOut.rotW[i] = cfToQuat(tL)
				if handOut then
					local tH = gh.c0:Inverse() * newLower:Inverse() * desiredHandWorld * gh.c1
					handOut.rotX[i], handOut.rotY[i], handOut.rotZ[i], handOut.rotW[i] = cfToQuat(tH)
				end
			else
				keepRaw(limb.upper, upperOut)
				keepRaw(limb.lower, lowerOut)
				keepRaw(limb.hand, handOut)
			end
		end
	end

	return out
end

return AnimationBuilder
