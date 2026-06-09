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
	-- Both Root (2D path) and Hips (3D pin) author the pelvis = LowerTorso.
	local partName = ({
		LeftHand = "LeftHand",
		RightHand = "RightHand",
		LeftFoot = "LeftFoot",
		RightFoot = "RightFoot",
		Root = "LowerTorso",
		Hips = "LowerTorso",
	})[effector]

	if not partName then
		return nil
	end
	return rig.model:FindFirstChild(partName, true) :: BasePart?
end

-- Capture the body anchor (root + hips) from the rig's current pose. The
-- server uses root for root_y/smooth_root_2d and the two hips for heading.
-- All are world-space positions; the caller converts to ground space.
export type BodyAnchor = { root: Vector3, hipL: Vector3, hipR: Vector3 }

function RigService.captureBody(rig: RigInfo): BodyAnchor
	local function pos(name: string, fallback: Vector3): Vector3
		local p = rig.model:FindFirstChild(name, true) :: BasePart?
		return if p and p:IsA("BasePart") then p.Position else fallback
	end
	local rootP = pos("LowerTorso", rig.rootPart.Position)
	return {
		root = rootP,
		hipL = pos("LeftUpperLeg", rootP - rig.rootPart.CFrame.RightVector * 0.5),
		hipR = pos("RightUpperLeg", rootP + rig.rootPart.CFrame.RightVector * 0.5),
	}
end

-- Detect velocity extrema (planted + swing moments) in a position track.
-- positions: array of Vector3 (one per frame). Returns array of frame indices.
-- Ported from python/effector_helpers.detect_velocity_extremes (XZ-speed).
function RigService.detectVelocityExtrema(positions: { Vector3 }, minSeparation: number): { number }
	local F = #positions
	if F < 3 then
		local all = {}
		for i = 1, F do all[i] = i end
		return all
	end

	-- XZ speed via central difference
	local speed = table.create(F, 0)
	for i = 2, F - 1 do
		local dx = positions[i + 1].X - positions[i - 1].X
		local dz = positions[i + 1].Z - positions[i - 1].Z
		speed[i] = math.sqrt(dx * dx + dz * dz)
	end
	speed[1] = speed[2]
	speed[F] = speed[F - 1]

	-- Gaussian smooth (sigma ~1.5, radius 3)
	local smooth = table.create(F, 0)
	local kernel = { 0.106, 0.141, 0.165, 0.176, 0.165, 0.141, 0.106 }
	for i = 1, F do
		local sum, wsum = 0, 0
		for k = -3, 3 do
			local j = math.clamp(i + k, 1, F)
			local w = kernel[k + 4]
			sum += speed[j] * w
			wsum += w
		end
		smooth[i] = sum / wsum
	end

	-- Find local minima (planted) and maxima (swing)
	local minima, maxima = {}, {}
	for i = 2, F - 1 do
		local s, sp, sn = smooth[i], smooth[i - 1], smooth[i + 1]
		if s <= sp and s <= sn then
			table.insert(minima, i)
		elseif s >= sp and s >= sn then
			table.insert(maxima, i)
		end
	end
	-- Classify the first and last frames by their single neighbor so the
	-- clip's start and end can also be captured.
	if smooth[1] <= smooth[2] then
		table.insert(minima, 1, 1)
	else
		table.insert(maxima, 1, 1)
	end
	if smooth[F] <= smooth[F - 1] then
		table.insert(minima, F)
	else
		table.insert(maxima, F)
	end

	-- Non-max suppression: minima slowest-first, maxima fastest-first
	local function nms(candidates: { number }, slowestFirst: boolean): { number }
		table.sort(candidates, function(a, b)
			if slowestFirst then return smooth[a] < smooth[b] else return smooth[a] > smooth[b] end
		end)
		local picked = {}
		for _, f in candidates do
			local ok = true
			for _, p in picked do
				if math.abs(f - p) < minSeparation then ok = false; break end
			end
			if ok then table.insert(picked, f) end
		end
		return picked
	end

	local pickedMin = nms(minima, true)
	local pickedMax = nms(maxima, false)

	-- Merge, dedupe, sort
	local seen = {}
	local combined = {}
	for _, f in pickedMin do if not seen[f] then seen[f] = true; table.insert(combined, f) end end
	for _, f in pickedMax do if not seen[f] then seen[f] = true; table.insert(combined, f) end end
	table.sort(combined)
	return combined
end

-- ════════════════════════════════════════════════════════════════════
-- Constraint gizmo: a single draggable/rotatable part marking one effector
-- target. The user moves it with Studio's native tools; on generate we read
-- its world CFrame in character (ground) space. No chain, no FK.
-- ════════════════════════════════════════════════════════════════════

export type Gizmo = {
	part: BasePart,
	effector: string,
}

local function quatFromCFrame(cf: CFrame): { number }
	-- Returns {qx, qy, qz, qw} from a CFrame's rotation via axis-angle.
	local axis, angle = cf:ToAxisAngle()
	local half = angle / 2
	local s = math.sin(half)
	return { axis.X * s, axis.Y * s, axis.Z * s, math.cos(half) }
end

function RigService.createConstraintGizmo(rig: RigInfo, effector: string, color: Color3, cframe: CFrame): Gizmo
	local part = Instance.new("Part")
	part.Name = "RoMotion_" .. effector
	part.Size = Vector3.new(0.6, 0.6, 0.6)
	part.CFrame = cframe
	part.Anchored = true
	part.CanCollide = false
	part.CanQuery = true
	part.CanTouch = false
	part.Transparency = 0.3
	part.Color = color
	part.Material = Enum.Material.Neon
	part.CastShadow = false
	part.Parent = workspace

	local sel = Instance.new("SelectionBox")
	sel.Adornee = part
	sel.Color3 = color
	sel.LineThickness = 0.03
	sel.Parent = part

	local billboard = Instance.new("BillboardGui")
	billboard.Name = "RoMotion_Label"
	billboard.Size = UDim2.new(0, 30, 0, 30)
	billboard.AlwaysOnTop = true
	billboard.Parent = part
	local lbl = Instance.new("TextLabel")
	lbl.Name = "Num"
	lbl.Size = UDim2.fromScale(1, 1)
	lbl.BackgroundTransparency = 1
	lbl.TextColor3 = Color3.new(1, 1, 1)
	lbl.TextStrokeTransparency = 0
	lbl.TextScaled = true
	lbl.Font = Enum.Font.SourceSansBold
	lbl.Parent = billboard

	return { part = part, effector = effector }
end

-- Read the gizmo target + captured body anchor, all in character (ground)
-- space. Returns the payload fields the server expects.
function RigService.readConstraintTarget(
	gizmo: Gizmo,
	groundCF: CFrame,
	body: BodyAnchor
): { target: { number }, target_rot: { number }, root: { number }, hip_l: { number }, hip_r: { number } }
	local function p(v: Vector3): { number }
		local l = groundCF:PointToObjectSpace(v)
		return { l.X, l.Y, l.Z }
	end
	-- Rotation expressed relative to groundCF (yaw removed) so the server's
	-- bind conversion is HRP-orientation independent.
	local localCF = groundCF:ToObjectSpace(gizmo.part.CFrame)
	return {
		target = p(gizmo.part.Position),
		target_rot = quatFromCFrame(localCF),
		root = p(body.root),
		hip_l = p(body.hipL),
		hip_r = p(body.hipR),
	}
end

function RigService.labelGizmo(gizmo: Gizmo, ordinal: number, color: Color3)
	gizmo.part.Name = "RoMotion_" .. gizmo.effector .. "_" .. tostring(ordinal)
	gizmo.part.Color = color
	local sel = gizmo.part:FindFirstChildOfClass("SelectionBox")
	if sel then sel.Color3 = color end
	local billboard = gizmo.part:FindFirstChild("RoMotion_Label") :: BillboardGui?
	local lbl = billboard and billboard:FindFirstChild("Num") :: TextLabel?
	if lbl then
		lbl.Text = gizmo.effector .. " " .. tostring(ordinal)
	end
end

function RigService.setGizmoVisible(gizmo: Gizmo, visible: boolean)
	gizmo.part.Transparency = if visible then 0.3 else 1
	local sel = gizmo.part:FindFirstChildOfClass("SelectionBox")
	if sel then sel.Visible = visible end
	local billboard = gizmo.part:FindFirstChild("RoMotion_Label") :: BillboardGui?
	if billboard then billboard.Enabled = visible end
end

function RigService.destroyGizmo(gizmo: Gizmo)
	gizmo.part:Destroy()
end

return RigService
