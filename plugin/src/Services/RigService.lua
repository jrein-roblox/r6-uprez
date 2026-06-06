--!strict
-- Rig detection and validation for RoMotion.

local RigService = {}

export type RigInfo = {
	model: Model,
	humanoid: Humanoid,
	animator: Animator,
	rootPart: BasePart,
	motors: { [string]: Motor6D },
}

function RigService.findRig(instance: Instance?): RigInfo?
	if not instance then
		return nil
	end

	local model: Model? = nil
	if instance:IsA("Model") then
		model = instance
	elseif instance:IsA("BasePart") then
		model = instance.Parent :: Model?
	end

	if not model or not model:IsA("Model") then
		return nil
	end

	local humanoid = model:FindFirstChildOfClass("Humanoid")
	if not humanoid then
		return nil
	end

	local rootPart = model:FindFirstChild("HumanoidRootPart") :: BasePart?
	if not rootPart or not rootPart:IsA("BasePart") then
		return nil
	end

	-- Find or create Animator
	local animator = humanoid:FindFirstChildOfClass("Animator")
	if not animator then
		animator = Instance.new("Animator")
		animator.Parent = humanoid
	end

	-- Enumerate joints (Motor6Ds or AnimationConstraints)
	local motors: { [string]: any } = {}
	for _, desc in model:GetDescendants() do
		if desc:IsA("Motor6D") and desc.Part1 then
			motors[desc.Part1.Name] = desc
		elseif desc.ClassName == "AnimationConstraint" then
			-- AnimationConstraint lives on the child part, uses Attachments
			motors[desc.Parent.Name] = desc
		end
	end

	return {
		model = model,
		humanoid = humanoid,
		animator = animator :: Animator,
		rootPart = rootPart,
		motors = motors,
	}
end

function RigService.isR15(rig: RigInfo): boolean
	return rig.motors["UpperTorso"] ~= nil and rig.motors["LowerTorso"] ~= nil
end

function RigService.getEffectorPart(rig: RigInfo, effector: string): BasePart?
	local partName = ({
		LeftHand = "LeftHand",
		RightHand = "RightHand",
		LeftFoot = "LeftFoot",
		RightFoot = "RightFoot",
		Hips = "LowerTorso",
		Root = "HumanoidRootPart",
	})[effector]

	if not partName then
		return nil
	end
	return rig.model:FindFirstChild(partName, true) :: BasePart?
end

-- Chain of part names from root to each effector
local EFFECTOR_CHAINS = {
	LeftHand = { "LowerTorso", "UpperTorso", "LeftUpperArm", "LeftLowerArm", "LeftHand" },
	RightHand = { "LowerTorso", "UpperTorso", "RightUpperArm", "RightLowerArm", "RightHand" },
	LeftFoot = { "LowerTorso", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot" },
	RightFoot = { "LowerTorso", "RightUpperLeg", "RightLowerLeg", "RightFoot" },
}

function RigService.getChainNames(effector: string): { string }
	return EFFECTOR_CHAINS[effector] or {}
end

export type ChainData = {
	model: Model,
	parts: { [string]: BasePart },
	joints: { { part0Name: string, part1Name: string, c0: CFrame, c1: CFrame } },
	connections: { RBXScriptConnection },
}

-- Clone the full chain as anchored parts with FK cascade on manipulation.
-- When user rotates any part, children re-position via FK.
function RigService.cloneChain(rig: RigInfo, effector: string, color: Color3): ChainData?
	local chainNames = EFFECTOR_CHAINS[effector]
	if not chainNames then return nil end

	local model = Instance.new("Model")
	model.Name = "RoMotion_Chain_" .. effector

	local clonedParts: { [string]: BasePart } = {}
	local joints: { { part0Name: string, part1Name: string, c0: CFrame, c1: CFrame } } = {}

	-- Capture joint data and compute animated CFrames via FK.
	-- In edit mode, parts stay at rest positions even when animated —
	-- only joint.Transform gets updated. We must FK manually.

	-- First: gather joint info for the full chain (including root→LowerTorso)
	type JointInfo = { part0Name: string, part1Name: string, c0: CFrame, c1: CFrame, transform: CFrame }
	local allJoints: { JointInfo } = {}

	for _, partName in chainNames do
		local joint = rig.motors[partName]
		if not joint then continue end

		local part0Name: string
		local c0: CFrame
		local c1: CFrame
		local transform: CFrame

		if joint:IsA("Motor6D") then
			part0Name = joint.Part0 and joint.Part0.Name or ""
			c0 = joint.C0
			c1 = joint.C1
			transform = joint.Transform
		elseif joint.ClassName == "AnimationConstraint" then
			part0Name = joint.Attachment0 and joint.Attachment0.Parent and joint.Attachment0.Parent.Name or ""
			c0 = joint.Attachment0 and joint.Attachment0.CFrame or CFrame.identity
			c1 = joint.Attachment1 and joint.Attachment1.CFrame or CFrame.identity
			transform = joint.Transform
		else
			continue
		end

		table.insert(allJoints, {
			part0Name = part0Name,
			part1Name = partName,
			c0 = c0,
			c1 = c1,
			transform = transform,
		})
	end

	-- FK from HumanoidRootPart through the chain to get animated CFrames
	local animatedCF: { [string]: CFrame } = {}
	animatedCF["HumanoidRootPart"] = rig.rootPart.CFrame

	-- Also grab CFrame of any parent part not in chain (as FK seed)
	for _, j in allJoints do
		if not animatedCF[j.part0Name] then
			local parentPart = rig.model:FindFirstChild(j.part0Name, true)
			if parentPart and parentPart:IsA("BasePart") then
				animatedCF[j.part0Name] = parentPart.CFrame
			end
		end
		-- FK: child = parent * C0 * Transform * C1:Inverse()
		local parentCF = animatedCF[j.part0Name] or CFrame.identity
		animatedCF[j.part1Name] = parentCF * j.c0 * j.transform * j.c1:Inverse()
	end

	-- Create cloned parts at animated positions
	for _, partName in chainNames do
		local srcPart = rig.model:FindFirstChild(partName, true) :: BasePart?
		if not srcPart or not srcPart:IsA("BasePart") then continue end

		local clone = Instance.new("Part")
		clone.Name = partName
		clone.Size = srcPart.Size
		clone.CFrame = animatedCF[partName] or srcPart.CFrame
		clone.Anchored = true
		clone.CanCollide = false
		clone.CanQuery = true
		clone.CanTouch = false
		clone.Transparency = 0.5
		clone.Color = color
		clone.Material = Enum.Material.ForceField
		clone.CastShadow = false
		clone.Parent = model
		clonedParts[partName] = clone
	end

	-- Store joints (only those with both parts in the chain) for readChainTransforms
	for _, j in allJoints do
		if clonedParts[j.part0Name] then
			table.insert(joints, {
				part0Name = j.part0Name,
				part1Name = j.part1Name,
				c0 = j.c0,
				c1 = j.c1,
			})
		end
	end

	model.Parent = workspace

	-- Set up FK cascade: when a part's CFrame changes, update all children below it
	local connections: { RBXScriptConnection } = {}
	local isUpdating = false

	local function fkFromPart(startIdx: number)
		if isUpdating then return end
		isUpdating = true
		-- Re-FK all joints from startIdx onward
		for i = startIdx, #joints do
			local j = joints[i]
			local part0 = clonedParts[j.part0Name]
			local part1 = clonedParts[j.part1Name]
			if part0 and part1 then
				-- Motor6D equation: Part1.CFrame = Part0.CFrame * C0 * Transform * C1:Inverse()
				-- Since we want to preserve current Transform, compute Part1 from Part0
				-- Transform = C0:Inv * Part0.CFrame:Inv * Part1.CFrame * C1 (current)
				-- But we want to CASCADE from parent change, so recompute Part1:
				-- We store the "local rotation" as the delta from rest
				local currentTransform = j.c0:Inverse() * part0.CFrame:Inverse() * part1.CFrame * j.c1
				part1.CFrame = part0.CFrame * j.c0 * currentTransform * j.c1:Inverse()
			end
		end
		isUpdating = false
	end

	-- Listen for CFrame changes on each part to cascade FK
	for i, partName in chainNames do
		local part = clonedParts[partName]
		if part then
			local partIdx = i
			local conn = part:GetPropertyChangedSignal("CFrame"):Connect(function()
				if isUpdating then return end
				-- Find which joint index this part is Part0 of, and FK from there
				for ji, j in joints do
					if j.part0Name == partName then
						fkFromPart(ji)
						break
					end
				end
			end)
			table.insert(connections, conn)
		end
	end

	return {
		model = model,
		parts = clonedParts,
		joints = joints,
		connections = connections,
	}
end

-- Read world CFrames from chain parts (user-poseable constraint visualization).
function RigService.readChainWorldCFrames(chain: ChainData, effector: string, groundCF: CFrame): { { name: string, pos: { number }, quat: { number } } }
	local chainNames = EFFECTOR_CHAINS[effector]
	if not chainNames then return {} end

	local result = {}
	for _, partName in chainNames do
		local part = chain.parts[partName]
		if part then
			local cf = part.CFrame
			local localPos = groundCF:PointToObjectSpace(cf.Position)
			local axis, angle = cf:ToAxisAngle()
			local halfAngle = angle / 2
			local sinHalf = math.sin(halfAngle)
			table.insert(result, {
				name = partName,
				pos = { localPos.X, localPos.Y, localPos.Z },
				quat = { axis.X * sinHalf, axis.Y * sinHalf, axis.Z * sinHalf, math.cos(halfAngle) },
			})
		end
	end
	return result
end


function RigService.destroyChain(chain: ChainData)
	for _, conn in chain.connections do
		conn:Disconnect()
	end
	chain.model:Destroy()
end

return RigService
