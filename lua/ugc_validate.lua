--!strict
-- Faithful port of Roblox's UGC CurveAnimation checks, run under roblox-cli.
--
-- `python/ugc_validate.py` evaluates the curves at their native 30 Hz keys.
-- The real validator resamples at 1/70 s using the FloatCurves' own Cubic
-- interpolation, which can overshoot BETWEEN keys — so a clip can be inside
-- the limits at every key and still spike past one for a frame or two. This
-- script closes that gap by using Roblox's actual curve evaluation
-- (`EulerRotationCurve:GetRotationAtTime`, `Vector3Curve:GetValueAtTime`)
-- and the actual R15 rig geometry.
--
-- Ported from (read, not guessed):
--   UGCValidation/util/CurveAnimationFrameCalculator.lua  calculateAnimFramesAtOrigin
--   UGCValidation/util/AssetCalculator.lua                calculatePartTransformInHierarchy
--   UGCValidation/validationFolders/CurveAnimBoundsValid
--   UGCValidation/validationFolders/CurveAnimSpeedBounded
--   UGCValidation/validationFolders/CurveAnimPositionBounded
--   UGCValidation/validationFolders/CurveAnimLengthBounded
--   UGCValidation/validationFolders/CurveAnimRigDataPresent
--
-- Globals:
--   _G.UGC_INPUT_DIR   directory tree to scan for r15.rbxm
--   _G.UGC_RIG_PATH    the reference body. Must be characterCagedHSRV18.rbxm —
--                      that is what CreateHumanoidModelFromDescription loads,
--                      and its rest foot sits at -2.85 vs characterR15's
--                      -2.20, i.e. 0.25 studs of headroom to the -3.1 floor
--                      instead of 0.90. It ships in the built Studio app
--                      resources, not the source content tree:
--                      .../RobloxStudio.app/Contents/Resources/content/avatar/
--   _G.UGC_OUT_PATH    JSON results output
--   _G.UGC_MARGIN      optional; gate at this fraction of each limit (default 1.0)

local FileSystemService = game:GetService("FileSystemService")

local INPUT_DIR: string = (_G :: any).UGC_INPUT_DIR or error("UGC_INPUT_DIR required")
local RIG_PATH: string = (_G :: any).UGC_RIG_PATH or error("UGC_RIG_PATH required")
local OUT_PATH: string = (_G :: any).UGC_OUT_PATH or error("UGC_OUT_PATH required")
local MARGIN: number = tonumber((_G :: any).UGC_MARGIN or "1.0") :: number

-- ------------------------------------------------------ UGC thresholds ----
-- FFlag/FString defaults from UGCValidation/flags/.
local MAX_ANIMATION_LENGTH = 10.0    -- UGCValidationMaxAnimationLength
local MIN_ANIMATION_LENGTH = 0.0     -- UGCValidateCurveAnimationMinLength
local HEIGHT_TOL = -3.1              -- UGCValidateAnimationHeightTol
local MAX_BOUNDS = 25.0              -- UGCValidationMaxAnimationBounds
local MAX_DELTAS = 1.5               -- UGCValidationMaxAnimationDeltas (per 1/30 s)
local MAX_JOINT_MOVEMENT = 0.3       -- UGCValidateMaxAnimationMovement
local VALIDATOR_FPS = 70             -- UGCValidateMaxAnimationFPS

local FRAME_DELTA = 1.0 / VALIDATOR_FPS
local DEFAULT_FRAME_TIME = 1.0 / 30.0
-- CurveAnimSpeedBounded.lua:33-35
local MAX_MOVEMENT_MULTIPLIER = FRAME_DELTA / DEFAULT_FRAME_TIME
local MAX_ALLOWED_MOVEMENT = MAX_DELTAS * MAX_MOVEMENT_MULTIPLIER
local MAX_STUDS_PER_SEC = MAX_DELTAS * (1.0 / DEFAULT_FRAME_TIME)

-- CurveAnimPositionBounded.lua:33-35 — LowerTorso is the one part allowed
-- to carry translation, which is where our pipeline folds root motion.
local POSITION_EXEMPT = { LowerTorso = true }

-- AssetCalculator.lua:36-56, fullBodyFromHumanoidRootPartAssetHierarchy
local PARENT: { [string]: string } = {
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
-- Parents before children, so FK can walk this in one pass.
local ORDER = {
	"LowerTorso", "UpperTorso", "Head",
	"LeftUpperArm", "LeftLowerArm", "LeftHand",
	"RightUpperArm", "RightLowerArm", "RightHand",
	"LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
	"RightUpperLeg", "RightLowerLeg", "RightFoot",
}
local ROOT_PART = "HumanoidRootPart"

-- ------------------------------------------------------------------ rig ----
local rigParts: { [string]: BasePart } = {}
do
	local loaded = FileSystemService:LoadInstances(RIG_PATH)
	local model: Instance? = nil
	for _, inst in loaded do
		if inst:IsA("Model") then model = inst break end
	end
	assert(model, "no Model in " .. RIG_PATH)
	for _, d in (model :: Instance):GetDescendants() do
		if d:IsA("BasePart") then rigParts[d.Name] = d end
	end
	print(string.format("[ugc_validate] rig %s: %d parts", RIG_PATH, #ORDER + 1))
end

-- The RigAttachment shared by a part and its parent. Derived rather than
-- hardcoded so it can't drift from the rig asset.
local attachToParent: { [string]: string } = {}
for child, parent in PARENT do
	local cp, pp = rigParts[child], rigParts[parent]
	assert(cp and pp, "rig missing " .. child .. " or " .. parent)
	local found: string? = nil
	for _, a in cp:GetChildren() do
		if a:IsA("Attachment") and string.match(a.Name, "RigAttachment$") then
			if pp:FindFirstChild(a.Name) then
				assert(found == nil, "ambiguous attachment for " .. child)
				found = a.Name
			end
		end
	end
	assert(found, "no shared RigAttachment for " .. child)
	attachToParent[child] = found :: string
end

local function attCF(partName: string, attName: string): CFrame
	local part = rigParts[partName]
	local att = part:FindFirstChild(attName) :: Attachment
	assert(att, attName .. " not on " .. partName)
	return att.CFrame
end

-- -------------------------------------------------------------- helpers ----
local function jsonStr(s: string): string
	return '"' .. (s:gsub('\\', '\\\\'):gsub('"', '\\"')) .. '"'
end

local function num(x: number): string
	return string.format("%.5f", x)
end

-- --------------------------------------------------------- clip checking ---
type Track = { pos: Vector3Curve?, rot: EulerRotationCurve? }

local function hasKeys(container: Instance?): boolean
	if not container then return false end
	for _, axis in (container :: Instance):GetChildren() do
		if axis:IsA("FloatCurve") and #(axis :: FloatCurve):GetKeys() > 0 then
			return true
		end
	end
	return false
end

local function collectTracks(curveAnim: Instance): { [string]: Track }
	local tracks: { [string]: Track } = {}
	for _, desc in curveAnim:GetDescendants() do
		if not desc:IsA("Folder") then continue end
		if desc.Name ~= ROOT_PART and PARENT[desc.Name] == nil then continue end
		local pos = desc:FindFirstChild("Position")
		local rot = desc:FindFirstChild("Rotation")
		tracks[desc.Name] = {
			pos = if hasKeys(pos) then (pos :: Vector3Curve) else nil,
			rot = if hasKeys(rot) then (rot :: EulerRotationCurve) else nil,
		}
	end
	return tracks
end

local function animLengthOf(tracks: { [string]: Track }): number
	local maxTime = -1
	local function scan(container: Instance?)
		if not container then return end
		for _, fc in (container :: Instance):GetChildren() do
			if not fc:IsA("FloatCurve") then continue end
			for _, k in (fc :: FloatCurve):GetKeys() do
				maxTime = math.max(maxTime, k.Time)
			end
		end
	end
	for _, t in tracks do
		scan(t.pos)
		scan(t.rot)
	end
	return maxTime
end

-- CurveAnimationFrameCalculator.calculateTransformsAtTime + AssetCalculator FK
local function framePositions(tracks: { [string]: Track }, t: number): { [string]: Vector3 }
	local function animTrans(name: string): CFrame
		local tr = tracks[name]
		if not tr then return CFrame.new() end
		local rot = if tr.rot then (tr.rot :: EulerRotationCurve):GetRotationAtTime(t) else CFrame.new()
		local pos = if tr.pos
			then Vector3.new(unpack((tr.pos :: Vector3Curve):GetValueAtTime(t)))
			else Vector3.zero
		return rot + pos
	end

	local cf: { [string]: CFrame } = {}
	cf[ROOT_PART] = animTrans(ROOT_PART)
	local out: { [string]: Vector3 } = { [ROOT_PART] = cf[ROOT_PART].Position }
	for _, part in ORDER do
		local parent = PARENT[part]
		local att = attachToParent[part]
		local c = (cf[parent] * attCF(parent, att)) * animTrans(part) * attCF(part, att):Inverse()
		cf[part] = c
		out[part] = c.Position
	end
	return out
end

type Violation = { check: string, part: string, time: number, value: number, limit: number }

local function validateClip(curveAnim: Instance): string
	local tracks = collectTracks(curveAnim)
	local animLength = animLengthOf(tracks)
	local violations: { Violation } = {}

	local function add(check: string, part: string, time: number, value: number, limit: number)
		table.insert(violations, { check = check, part = part, time = time, value = value, limit = limit })
	end

	-- CurveAnimRigDataPresent
	local nRigData = 0
	local rigDataValid = true
	for _, child in curveAnim:GetChildren() do
		if child:IsA("AnimationRigData") then
			nRigData += 1
			local ok, valid = pcall(function() return (child :: any):IsValidR15() end)
			if not ok or not valid then rigDataValid = false end
		end
	end
	if nRigData ~= 1 then
		add("rigdata count", "-", 0, nRigData, 1)
	elseif not rigDataValid then
		add("rigdata IsValidR15", "-", 0, 0, 1)
	end

	-- CurveAnimLengthBounded
	if animLength <= MIN_ANIMATION_LENGTH or animLength > MAX_ANIMATION_LENGTH * MARGIN then
		add("length", "-", animLength, animLength, MAX_ANIMATION_LENGTH * MARGIN)
	end

	-- CurveAnimPositionBounded — raw local translation, LowerTorso exempt
	local maxMove, maxMovePart = 0.0, "-"
	do
		local limit = MAX_JOINT_MOVEMENT * MARGIN
		local t = 0.0
		while t <= animLength do
			for name, tr in tracks do
				if POSITION_EXEMPT[name] or not tr.pos then continue end
				local mag = Vector3.new(unpack((tr.pos :: Vector3Curve):GetValueAtTime(t))).Magnitude
				if mag > maxMove then maxMove, maxMovePart = mag, name end
				if mag > limit then add("joint separation", name, t, mag, limit) end
			end
			t += FRAME_DELTA
		end
	end

	-- CurveAnimBoundsValid + CurveAnimSpeedBounded over the 1/70 s sampling
	local minY, minYPart = math.huge, "-"
	local maxDist, maxDistPart = 0.0, "-"
	local maxSpeed, maxSpeedPart = 0.0, "-"
	do
		local yLimit = HEIGHT_TOL * MARGIN
		local dLimit = MAX_BOUNDS * MARGIN
		local sLimit = MAX_ALLOWED_MOVEMENT * MARGIN
		local prev: { [string]: Vector3 }? = nil
		local t = 0.0
		while t <= animLength do
			local pos = framePositions(tracks, t)
			for part, p in pos do
				if p.Y < minY then minY, minYPart = p.Y, part end
				if p.Y < yLimit then add("part too low", part, t, p.Y, yLimit) end
				local m = p.Magnitude
				if m > maxDist then maxDist, maxDistPart = m, part end
				if m > dLimit then add("part too far", part, t, m, dLimit) end
				if prev then
					local delta = (p - (prev :: any)[part]).Magnitude
					local studsPerSec = (delta / MAX_MOVEMENT_MULTIPLIER) * (1.0 / DEFAULT_FRAME_TIME)
					if studsPerSec > maxSpeed then maxSpeed, maxSpeedPart = studsPerSec, part end
					if delta > sLimit then
						add("speed too fast", part, t, studsPerSec, MAX_STUDS_PER_SEC * MARGIN)
					end
				end
			end
			prev = pos
			t += FRAME_DELTA
		end
	end

	-- Report only the worst instance of each check to keep the JSON small.
	local worst: { [string]: Violation } = {}
	for _, v in violations do
		local cur = worst[v.check]
		if not cur or math.abs(v.value - v.limit) > math.abs(cur.value - cur.limit) then
			worst[v.check] = v
		end
	end
	local vparts: { string } = {}
	for check, v in worst do
		table.insert(vparts, string.format(
			'{"check":%s,"part":%s,"time":%s,"value":%s,"limit":%s}',
			jsonStr(check), jsonStr(v.part), num(v.time), num(v.value), num(v.limit)))
	end

	return string.format(
		'{"name":%s,"length":%s,"min_y":%s,"min_y_part":%s,"max_bounds":%s,'
			.. '"max_bounds_part":%s,"max_speed":%s,"max_speed_part":%s,'
			.. '"max_joint_move":%s,"max_joint_move_part":%s,"rig_data":%d,'
			.. '"verdict":%s,"violations":[%s]}',
		jsonStr(curveAnim.Name), num(animLength),
		num(minY), jsonStr(minYPart),
		num(maxDist), jsonStr(maxDistPart),
		num(maxSpeed), jsonStr(maxSpeedPart),
		num(maxMove), jsonStr(maxMovePart),
		nRigData,
		jsonStr(if #vparts == 0 then "pass" else "fail"),
		table.concat(vparts, ","))
end

-- ------------------------------------------------------------------ main ---
local rbxmFiles: { string } = {}
for fileData in FileSystemService:Walk(INPUT_DIR, Enum.FileSystemWalkMode.Recursive) do
	if fileData.Path:match("/r15%.rbxm$") then
		table.insert(rbxmFiles, fileData.Path)
	end
end
table.sort(rbxmFiles)
print(string.format("[ugc_validate] %d clip(s), margin=%.2f, sampling %d Hz",
	#rbxmFiles, MARGIN, VALIDATOR_FPS))

local results: { string } = {}
local nPass, nFail = 0, 0
for i, path in rbxmFiles do
	local loaded = FileSystemService:LoadInstances(path)
	for _, inst in loaded do
		if not inst:IsA("CurveAnimation") then continue end
		local ok, json = pcall(validateClip, inst)
		if ok then
			table.insert(results, json :: string)
			if string.match(json :: string, '"verdict":"pass"') then
				nPass += 1
			else
				nFail += 1
				print(string.format("  FAIL %s", inst.Name))
			end
		else
			print(string.format("  ERR  %s: %s", inst.Name, tostring(json)))
		end
	end
	if i % 25 == 0 then
		print(string.format("  ... %d/%d", i, #rbxmFiles))
	end
end

FileSystemService:WriteFile(OUT_PATH, "[" .. table.concat(results, ",") .. "]")
print(string.format("[ugc_validate] %d pass / %d fail -> %s", nPass, nFail, OUT_PATH))
