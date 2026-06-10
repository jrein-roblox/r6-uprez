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
		Look = "Head",
	})[effector]

	if not partName then
		return nil
	end
	return rig.model:FindFirstChild(partName, true) :: BasePart?
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
-- Constraint gizmo(s): a draggable/rotatable effector part. For limbs we also
-- spawn a second "root" part (sets root XZ/Y + heading via its orientation),
-- joined to the effector by a Beam so the effector→root relationship is
-- visible. The user positions both independently with Studio's native tools;
-- on generate we read their world CFrames in character (ground) space.
-- For Root/Hips the effector IS the pelvis, so there's no separate root part.
-- ════════════════════════════════════════════════════════════════════

export type Gizmo = {
	effectorPart: BasePart,
	rootPart: BasePart?, -- nil for Root/Hips/Look (effector is the root, or none)
	beam: Beam?,
	effector: string,
	connections: { RBXScriptConnection },
}

local HIP_HALF_WIDTH = 0.5 -- studs; only the left→right direction matters (heading)

local function isLimb(effector: string): boolean
	return effector ~= "Root" and effector ~= "Hips"
end

local function quatFromCFrame(cf: CFrame): { number }
	-- Returns {qx, qy, qz, qw} from a CFrame's rotation via axis-angle.
	local axis, angle = cf:ToAxisAngle()
	local half = angle / 2
	local s = math.sin(half)
	return { axis.X * s, axis.Y * s, axis.Z * s, math.cos(half) }
end

local function makeLabel(part: BasePart, text: string)
	local billboard = Instance.new("BillboardGui")
	billboard.Name = "RoMotion_Label"
	billboard.Size = UDim2.new(0, 36, 0, 22)
	billboard.AlwaysOnTop = true
	billboard.Parent = part
	local lbl = Instance.new("TextLabel")
	lbl.Name = "Num"
	lbl.Size = UDim2.fromScale(1, 1)
	lbl.BackgroundTransparency = 1
	lbl.Text = text
	lbl.TextColor3 = Color3.new(1, 1, 1)
	lbl.TextStrokeTransparency = 0
	lbl.TextScaled = true
	lbl.Font = Enum.Font.SourceSansBold
	lbl.Parent = billboard
end

local function addBoundingBox(part: BasePart, color: Color3)
	local sel = Instance.new("SelectionBox")
	sel.Adornee = part
	sel.Color3 = color
	sel.LineThickness = 0.03
	sel.Parent = part
end

local function makePart(name: string, shape: Enum.PartType, size: number, color: Color3, transparency: number, cframe: CFrame): BasePart
	local part = Instance.new("Part")
	part.Name = name
	part.Shape = shape
	part.Size = Vector3.new(size, size, size)
	part.CFrame = cframe
	part.Anchored = true
	part.CanCollide = false
	part.CanQuery = true
	part.CanTouch = false
	part.Transparency = transparency
	part:SetAttribute("RoMotionBase", transparency)
	part.Color = color
	part.Material = Enum.Material.Neon
	part.CastShadow = false
	part.Parent = workspace
	addBoundingBox(part, color)
	return part
end

-- Clone the rig's actual effector part so the gizmo has the real shape — much
-- easier to orient than a generic cube. Strips joints/attachments/surface
-- decals so it's a clean, tinted, translucent stand-in, plus a bounding box.
local function cloneEffectorPart(rig: RigInfo, effector: string, color: Color3, cframe: CFrame): BasePart?
	local src = RigService.getEffectorPart(rig, effector)
	if not src or not src:IsA("BasePart") then return nil end

	local clone = src:Clone()
	for _, ch in clone:GetDescendants() do
		if ch:IsA("JointInstance") or ch:IsA("Constraint") or ch:IsA("Attachment")
			or ch:IsA("Bone") or ch:IsA("SurfaceAppearance") or ch:IsA("Decal")
			or ch:IsA("Texture") or ch:IsA("BillboardGui") then
			ch:Destroy()
		end
	end
	clone.Name = "RoMotion_" .. effector
	clone.Anchored = true
	clone.CanCollide = false
	clone.CanQuery = true
	clone.CanTouch = false
	clone.Massless = true
	clone.CFrame = cframe
	clone.Color = color
	clone.Material = Enum.Material.SmoothPlastic -- keeps shape shading (Neon washes it out)
	clone.Transparency = 0.3
	clone:SetAttribute("RoMotionBase", 0.3)
	clone.CastShadow = false
	clone.Parent = workspace
	addBoundingBox(clone, color)
	return clone
end

-- A flat arrow lying on the ground, pointing +Z (the constrained facing). Used
-- for root anchors so they read as "stand here, face this way" and aren't
-- confused with the 3D Hips/torso gizmo. Returns the shaft (the part the user
-- moves) and a connection that keeps the two arrowhead segments locked to it —
-- welds don't resolve in Studio edit mode, so we reposition on CFrame change.
local function buildGroundArrow(name: string, color: Color3, cframe: CFrame): (BasePart, RBXScriptConnection)
	local shaft = Instance.new("Part")
	shaft.Name = name
	shaft.Size = Vector3.new(0.18, 0.08, 1.1)
	shaft.CFrame = cframe
	shaft.Anchored = true
	shaft.CanCollide = false
	shaft.CanQuery = true
	shaft.CanTouch = false
	shaft.Color = color
	shaft.Material = Enum.Material.Neon
	shaft.Transparency = 0.3
	shaft:SetAttribute("RoMotionBase", 0.3)
	shaft.CastShadow = false
	shaft.Parent = workspace
	addBoundingBox(shaft, color)

	-- Two angled segments at the -Z tip forming the arrowhead "V". -Z is the
	-- facing direction (a part's LookVector is -Z), so the arrow points where
	-- the character will face. Anchored and repositioned as the shaft moves.
	local offsets: { [BasePart]: CFrame } = {}
	for _, side in { -1, 1 } do
		local head = Instance.new("Part")
		head.Name = "Head"
		head.Size = Vector3.new(0.18, 0.08, 0.55)
		local localCF = CFrame.new(0, 0, -0.55) * CFrame.Angles(0, math.rad(side * 40), 0) * CFrame.new(0, 0, 0.27)
		head.CFrame = shaft.CFrame * localCF
		head.Anchored = true
		head.CanCollide = false
		head.CanQuery = true
		head.CanTouch = false
		head.Color = color
		head.Material = Enum.Material.Neon
		head.Transparency = 0.3
		head:SetAttribute("RoMotionBase", 0.3)
		head.CastShadow = false
		head.Parent = shaft
		offsets[head] = localCF
	end

	local conn = shaft:GetPropertyChangedSignal("CFrame"):Connect(function()
		for head, localCF in offsets do
			head.CFrame = shaft.CFrame * localCF
		end
	end)
	return shaft, conn
end

function RigService.createConstraintGizmo(
	rig: RigInfo,
	effector: string,
	color: Color3,
	effCFrame: CFrame,
	rootCFrame: CFrame,
	groundY: number
): Gizmo
	-- Look (head gaze): two gizmos joined by a beam (the gaze ray) — a target
	-- ball (where to look) and an origin ball (where the head is). The head's
	-- constrained orientation is lookAt(origin → target), so when you crouch you
	-- drop the origin and the aim tilts up to the target. effCFrame = Head CFrame.
	if effector == "Look" then
		local headPos = effCFrame.Position
		local target = makePart("RoMotion_Look", Enum.PartType.Ball, 0.5, color, 0.2, CFrame.new(headPos + effCFrame.LookVector * 3))
		makeLabel(target, "look")
		local origin = makePart("RoMotion_Look_From", Enum.PartType.Ball, 0.4, color:Lerp(Color3.new(0.7, 0.7, 0.7), 0.45), 0.5, CFrame.new(headPos))
		makeLabel(origin, "from")

		local a0 = Instance.new("Attachment"); a0.Name = "RoMotion_Link"; a0.Parent = target
		local a1 = Instance.new("Attachment"); a1.Name = "RoMotion_Link"; a1.Parent = origin
		local beam = Instance.new("Beam")
		beam.Name = "RoMotion_Link"
		beam.Attachment0 = a0
		beam.Attachment1 = a1
		beam.Color = ColorSequence.new(color)
		beam.Width0 = 0.06
		beam.Width1 = 0.06
		beam.FaceCamera = true
		beam.Segments = 1
		beam.Transparency = NumberSequence.new(0.2)
		beam.Parent = target

		return { effectorPart = target, rootPart = origin, beam = beam, effector = effector, connections = {} }
	end

	-- Root (2D path) effector: a flat ground arrow (position + facing). This is
	-- the ONLY ground gizmo; it pins root XZ + heading, hip height free.
	if effector == "Root" then
		local _, yaw = rootCFrame:ToEulerAnglesYXZ()
		local groundCF = CFrame.new(rootCFrame.X, groundY, rootCFrame.Z) * CFrame.fromEulerAnglesYXZ(0, yaw, 0)
		local arrow, conn = buildGroundArrow("RoMotion_Root", color, groundCF)
		makeLabel(arrow, "Root")
		return { effectorPart = arrow, effector = effector, connections = { conn } }
	end

	-- Effector gizmo = a clone of the rig's real part (clear orientation) +
	-- bounding box; falls back to a neon cube if the part can't be found.
	local effectorPart = cloneEffectorPart(rig, effector, color, effCFrame)
		or makePart("RoMotion_" .. effector, Enum.PartType.Block, 0.6, color, 0.3, effCFrame)
	makeLabel(effectorPart, effector)

	if not isLimb(effector) then
		-- Hips: the effector IS the 3D pelvis pin; single torso gizmo, no link.
		return { effectorPart = effectorPart, effector = effector, connections = {} }
	end

	-- Limb root anchor: a dimmed 3D LowerTorso clone at hip height. Hands/feet
	-- are pinned RELATIVE to the pelvis, so its height matters — it is NOT a
	-- ground marker. Its position sets root XZ/Y; its yaw sets heading.
	local rootColor = color:Lerp(Color3.new(0.7, 0.7, 0.7), 0.45)
	local rootPart = cloneEffectorPart(rig, "Root", rootColor, rootCFrame)
		or makePart("RoMotion_Root", Enum.PartType.Ball, 0.5, rootColor, 0.5, rootCFrame)
	rootPart.Name = "RoMotion_" .. effector .. "_Root"
	rootPart.Transparency = 0.55
	rootPart:SetAttribute("RoMotionBase", 0.55)
	makeLabel(rootPart, "root")

	-- Beam linking effector → root (auto-follows the parts as they move).
	local a0 = Instance.new("Attachment"); a0.Name = "RoMotion_Link"; a0.Parent = effectorPart
	local a1 = Instance.new("Attachment"); a1.Name = "RoMotion_Link"; a1.Parent = rootPart
	local beam = Instance.new("Beam")
	beam.Name = "RoMotion_Link"
	beam.Attachment0 = a0
	beam.Attachment1 = a1
	beam.Color = ColorSequence.new(color)
	beam.Width0 = 0.08
	beam.Width1 = 0.08
	beam.FaceCamera = true
	beam.Segments = 1
	beam.Transparency = NumberSequence.new(0.2)
	beam.Parent = effectorPart

	return {
		effectorPart = effectorPart,
		rootPart = rootPart,
		beam = beam,
		effector = effector,
		connections = {},
	}
end

-- Read the effector target + root anchor, all in character (ground) space.
-- Root + hips come from the root gizmo (or the effector gizmo itself for
-- Root/Hips); hips are synthesized from the root gizmo's orientation so its
-- yaw drives the constrained heading.
function RigService.readConstraintTarget(
	gizmo: Gizmo,
	groundCF: CFrame
): { target: { number }, target_rot: { number }, root: { number }, hip_l: { number }, hip_r: { number } }
	local function p(v: Vector3): { number }
		local l = groundCF:PointToObjectSpace(v)
		return { l.X, l.Y, l.Z }
	end

	-- Look: target_rot is the head orientation facing the target from the
	-- user-placed gaze origin (the "from" gizmo). No root/hips (rotation-only).
	if gizmo.effector == "Look" then
		local from = if gizmo.rootPart then gizmo.rootPart.Position else gizmo.effectorPart.Position
		local to = gizmo.effectorPart.Position
		local lookWorld = if (to - from).Magnitude > 1e-3
			then CFrame.lookAt(from, to)
			else CFrame.new(from) -- origin == target: no meaningful direction
		local localCF = groundCF:ToObjectSpace(lookWorld)
		return {
			target = p(to),
			target_rot = quatFromCFrame(localCF),
		}
	end

	local rootCF = (gizmo.rootPart or gizmo.effectorPart).CFrame
	-- Rotation expressed relative to groundCF (yaw removed) so the server's
	-- bind conversion is HRP-orientation independent.
	local localCF = groundCF:ToObjectSpace(gizmo.effectorPart.CFrame)
	return {
		target = p(gizmo.effectorPart.Position),
		target_rot = quatFromCFrame(localCF),
		root = p(rootCF.Position),
		hip_r = p((rootCF * CFrame.new(HIP_HALF_WIDTH, 0, 0)).Position),
		hip_l = p((rootCF * CFrame.new(-HIP_HALF_WIDTH, 0, 0)).Position),
	}
end

function RigService.labelGizmo(gizmo: Gizmo, ordinal: number, color: Color3)
	gizmo.effectorPart.Name = "RoMotion_" .. gizmo.effector .. "_" .. tostring(ordinal)
	gizmo.effectorPart.Color = color
	local sel = gizmo.effectorPart:FindFirstChildOfClass("SelectionBox")
	if sel then sel.Color3 = color end
	local billboard = gizmo.effectorPart:FindFirstChild("RoMotion_Label") :: BillboardGui?
	local lbl = billboard and billboard:FindFirstChild("Num") :: TextLabel?
	if lbl then
		lbl.Text = gizmo.effector .. " " .. tostring(ordinal)
	end
	if gizmo.rootPart then
		local rl = (gizmo.rootPart:FindFirstChild("RoMotion_Label") :: BillboardGui?)
		local rlbl = rl and rl:FindFirstChild("Num") :: TextLabel?
		local subName = if gizmo.effector == "Look" then "from" else "root"
		if rlbl then rlbl.Text = subName .. " " .. tostring(ordinal) end
	end
	if gizmo.beam then gizmo.beam.Color = ColorSequence.new(color) end
end

-- Fade a gizmo by alpha (1 = fully visible at its base transparency, 0 = gone).
-- Each part stores its shown transparency in the "RoMotionBase" attribute so we
-- can fade toward fully transparent linearly. Used for the "local" focus mode.
local function partAlpha(part: BasePart, alpha: number)
	local base = part:GetAttribute("RoMotionBase")
	if typeof(base) ~= "number" then base = 0.3 end
	part.Transparency = 1 - alpha * (1 - base)
end

function RigService.setGizmoAlpha(gizmo: Gizmo, alpha: number)
	local on = alpha > 0.02
	local function apply(part: BasePart?)
		if not part then return end
		partAlpha(part, alpha)
		for _, d in part:GetDescendants() do
			if d:IsA("BasePart") then
				partAlpha(d, alpha)
			elseif d:IsA("SelectionBox") then
				d.Visible = on
				d.Transparency = 1 - alpha -- fade the bounding-box lines
			elseif d:IsA("BillboardGui") then
				d.Enabled = on
			elseif d:IsA("TextLabel") then
				d.TextTransparency = 1 - alpha
				d.TextStrokeTransparency = 1 - alpha
			end
		end
	end
	apply(gizmo.effectorPart)
	apply(gizmo.rootPart)
	if gizmo.beam then
		gizmo.beam.Enabled = on
		gizmo.beam.Transparency = NumberSequence.new(1 - alpha * (1 - 0.2))
	end
end

function RigService.setGizmoVisible(gizmo: Gizmo, visible: boolean)
	RigService.setGizmoAlpha(gizmo, if visible then 1 else 0)
end

-- Mirror a constraint's hard/soft pin state on its gizmo: hard pins are opaque,
-- soft pins keep their translucent default. We stash the original (soft) base
-- transparency in "RoMotionSoftBase" so toggling back restores the exact look.
-- Callers should re-run their visibility pass afterward (it reads RoMotionBase).
function RigService.setGizmoPinned(gizmo: Gizmo, pinned: boolean)
	local function one(p: BasePart)
		local soft = p:GetAttribute("RoMotionSoftBase")
		if typeof(soft) ~= "number" then
			local cur = p:GetAttribute("RoMotionBase")
			soft = if typeof(cur) == "number" then cur else 0.3
			p:SetAttribute("RoMotionSoftBase", soft)
		end
		p:SetAttribute("RoMotionBase", if pinned then 0 else soft)
	end
	local function apply(part: BasePart?)
		if not part then return end
		one(part)
		for _, d in part:GetDescendants() do
			if d:IsA("BasePart") then one(d) end
		end
	end
	apply(gizmo.effectorPart)
	apply(gizmo.rootPart)
end

function RigService.destroyGizmo(gizmo: Gizmo)
	for _, conn in gizmo.connections do
		conn:Disconnect()
	end
	gizmo.effectorPart:Destroy()
	if gizmo.rootPart then gizmo.rootPart:Destroy() end
end

-- ════════════════════════════════════════════════════════════════════
-- Rig rest geometry for FK/IK (the IK bake in AnimationBuilder).
-- For each child part: its parent's name and the joint's C0/C1 (attachment
-- CFrames). FK identity (same as the old cloneChain): for a joint on the
-- child, childWorld = parentWorld * C0 * Transform * C1:Inverse().
-- ════════════════════════════════════════════════════════════════════

export type JointGeom = { parentName: string, c0: CFrame, c1: CFrame }

function RigService.getRigGeometry(rig: RigInfo): { [string]: JointGeom }
	local geom: { [string]: JointGeom } = {}
	for partName, joint in rig.motors do
		local parentName: string?
		local c0, c1: CFrame, CFrame
		if joint:IsA("Motor6D") then
			parentName = joint.Part0 and joint.Part0.Name or nil
			c0, c1 = joint.C0, joint.C1
		elseif joint.ClassName == "AnimationConstraint" then
			local a0, a1 = joint.Attachment0, joint.Attachment1
			parentName = a0 and a0.Parent and a0.Parent.Name or nil
			c0 = a0 and a0.CFrame or CFrame.identity
			c1 = a1 and a1.CFrame or CFrame.identity
		else
			continue
		end
		if parentName then
			geom[partName] = { parentName = parentName, c0 = c0, c1 = c1 }
		end
	end
	return geom
end

return RigService
