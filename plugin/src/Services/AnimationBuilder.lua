--!strict
-- Builds a CurveAnimation instance from r15.json data received from the backend.
-- This is the Lua equivalent of lua/build_rbxm.lua but runs inside the plugin.

local AnimationBuilder = {}

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

function AnimationBuilder.build(data: { [string]: any }): CurveAnimation
	local parts = data.parts or {}
	local root = data.root
	local frameRate = data.frameRate or 30
	local frameCount = data.frameCount or 0

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

return AnimationBuilder
