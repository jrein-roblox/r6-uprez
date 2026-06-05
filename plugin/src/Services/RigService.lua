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

	-- Enumerate Motor6Ds
	local motors: { [string]: Motor6D } = {}
	for _, desc in model:GetDescendants() do
		if desc:IsA("Motor6D") and desc.Part1 then
			motors[desc.Part1.Name] = desc
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

return RigService
