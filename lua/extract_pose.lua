--!strict
--[[
  extract_pose.lua — Stage 1 of the r6-uprez pipeline.

  Reads <work>/_extract_config.json:
    { asset_id: number, out_path: string, sample_fps: number = 30 }
  Rig type (R6 or R15) is auto-detected from the clip's bone names — see
  detectRigType().

  Loads the animation asset via InsertService:LoadAsset, follows the
  Animation→AnimationId indirection if needed, registers the clip with
  KeyframeSequenceProvider (works for both KeyframeSequence and
  CurveAnimation), plays it on a spawned R6 or R15 character via Animator,
  scrubs through TimePosition at fixed 1/fps intervals, and samples world
  CFrames for HumanoidRootPart + four effectors (hands & feet).

  R6 has no Hand/Foot parts, so effector positions are computed by applying
  a 1-stud Y-down offset to the Arm/Leg part CFrame (default R6 part size
  is 1x2x1 centered, so part_center * CFrame.new(0, -1, 0) lands at the
  limb tip).

  Output JSON to <out_path>:
    { asset_id, clip_id, rig_type, fps, n_frames, duration_s, frames: [
        { t, hrp:{pos,rot}, left_hand:{...}, right_hand:{...},
          left_foot:{...}, right_foot:{...} }, ...
    ]}

  Run (driven by python/extract_pose.py):
    roblox-cli run --run extract_pose.lua \
      --fs.readwrite <work_dir> \
      --load.asRobloxScript
]]

local FileSystemService = game:GetService("FileSystemService")
local HttpService = game:GetService("HttpService")
local InsertService = game:GetService("InsertService")
local KeyframeSequenceProvider = game:GetService("KeyframeSequenceProvider")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")

RunService:Pause()

-- =============================================================================
-- Locate config. The Python driver places `_extract_config.json` in the
-- --fs.readwrite root. We probe a small set of known paths.
-- =============================================================================
local CONFIG_CANDIDATES = {
	"_extract_config.json",
	"work/_extract_config.json",
	"/Users/jrein/git/roblox/jrein/r6-uprez/work/_extract_config.json",
}

local function readConfig()
	for _, candidate in ipairs(CONFIG_CANDIDATES) do
		local ok, content = pcall(function()
			return FileSystemService:ReadFile(candidate, Enum.FileMode.Text)
		end)
		if ok and content and #content > 0 then
			print("[extract_pose] config:", candidate)
			return HttpService:JSONDecode(content), candidate
		end
	end
	error("extract_pose: could not locate _extract_config.json. Looked in: "
		.. table.concat(CONFIG_CANDIDATES, ", "))
end

local config = readConfig()
local ASSET_ID = assert(tonumber(config.asset_id), "asset_id required")
local OUT_PATH = assert(config.out_path, "out_path required")
local FPS = tonumber(config.sample_fps) or 30
local MIN_DURATION = tonumber(config.min_duration_s) or 0.0  -- 0 = no looping
local LOOP_PASSES = math.max(1, tonumber(config.loop_passes) or 1)

print(string.format("[extract_pose] asset=%d fps=%d min_duration=%.2fs loop_passes=%d -> %s",
	ASSET_ID, FPS, MIN_DURATION, LOOP_PASSES, OUT_PATH))

-- =============================================================================
-- Resolve a wrapper Animation asset to its inner clip (KeyframeSequence /
-- CurveAnimation). Mirrors the pattern in download_assets.lua:163-196.
-- =============================================================================
local function getAssetIdFromUrl(url: string): number?
	if not url then return nil end
	return tonumber(url:match("(%d+)"))
end

local function loadClip(assetId: number): Instance?
	local model = InsertService:LoadAsset(assetId)
	if not model then return nil end
	local first = model:GetChildren()[1]
	if not first then return nil end

	-- Direct hit: the asset itself is a clip.
	if first:IsA("KeyframeSequence") or first:IsA("CurveAnimation") then
		return first
	end

	-- Wrapper Animation → recurse on AnimationId.
	if first:IsA("Animation") and (first :: Animation).AnimationId then
		local innerId = getAssetIdFromUrl((first :: Animation).AnimationId)
		if innerId and innerId ~= assetId then
			return loadClip(innerId)
		end
	end

	-- Last-ditch: search descendants for a clip.
	for _, desc in ipairs(model:GetDescendants()) do
		if desc:IsA("KeyframeSequence") or desc:IsA("CurveAnimation") then
			return desc
		end
	end
	return nil
end

local clip = loadClip(ASSET_ID)
assert(clip, "extract_pose: could not load clip for asset " .. ASSET_ID)
print(string.format("[extract_pose] loaded %s '%s'", clip.ClassName, clip.Name))

-- Both KeyframeSequence and CurveAnimation inherit AnimationClip.Loop. We
-- propagate it so the Python side can constrain frame 0 == frame F-1 for
-- looped clips, ensuring the regenerated motion can loop seamlessly.
local IS_LOOPED = (clip :: any).Loop or false
print(string.format("[extract_pose] clip.Loop=%s", tostring(IS_LOOPED)))

-- =============================================================================
-- Rig detection from bone names in the clip.
--
-- KeyframeSequence: Keyframe → Pose ("Name" = bone name; recursive children).
-- CurveAnimation:   Folder ("Name" = bone name; contains Position/Rotation
--                   curves).
--
-- R15 has uniquely-named bones LowerTorso/UpperTorso/Left{Upper,Lower}Arm/
-- Hand/etc. R6 has the disjoint set Torso/"Left Arm"/"Right Arm"/"Left Leg"/
-- "Right Leg". Head and HumanoidRootPart appear in both, so we look for the
-- discriminating names. R15 wins on tie (well-formed R15 clips dominate).
-- =============================================================================
local R15_DISCRIMINATING = {
	LowerTorso = true, UpperTorso = true,
	LeftUpperArm = true, LeftLowerArm = true, LeftHand = true,
	RightUpperArm = true, RightLowerArm = true, RightHand = true,
	LeftUpperLeg = true, LeftLowerLeg = true, LeftFoot = true,
	RightUpperLeg = true, RightLowerLeg = true, RightFoot = true,
}
local R6_DISCRIMINATING = {
	Torso = true,
	["Left Arm"] = true, ["Right Arm"] = true,
	["Left Leg"] = true, ["Right Leg"] = true,
}

local function detectRigType(rootClip: Instance): string
	local r15Hits, r6Hits = 0, 0
	for _, desc in ipairs(rootClip:GetDescendants()) do
		local n = desc.Name
		if R15_DISCRIMINATING[n] then r15Hits += 1 end
		if R6_DISCRIMINATING[n] then r6Hits += 1 end
	end
	print(string.format("[extract_pose] bone-name hits: R15=%d R6=%d",
		r15Hits, r6Hits))
	if r15Hits > 0 then return "R15" end
	if r6Hits > 0 then return "R6" end
	-- Empty descendant scan (e.g. a clip without authored bones at all):
	-- default to R15 since modern uploads target it.
	return "R15"
end

local RIG_TYPE = detectRigType(clip)
print(string.format("[extract_pose] detected rig: %s", RIG_TYPE))

-- =============================================================================
-- Spawn character. CreateHumanoidModelFromDescription gives a clean rig at
-- the requested HumanoidRigType (R6: Torso/Head/Arms/Legs; R15: full chain).
-- We set Anchored on HRP so root motion shows up in CFrame samples instead
-- of being absorbed by physics drift.
-- =============================================================================
local desc = Instance.new("HumanoidDescription")
local rigEnum = if RIG_TYPE == "R6" then Enum.HumanoidRigType.R6 else Enum.HumanoidRigType.R15
local character = Players:CreateHumanoidModelFromDescription(desc, rigEnum)
character.Name = "ExtractPoseChar"
character.Parent = workspace

local hrp = character:FindFirstChild("HumanoidRootPart") :: BasePart
assert(hrp, "spawned character has no HumanoidRootPart")
hrp.CFrame = CFrame.new(0, 5, 0)

local humanoid = character:FindFirstChildOfClass("Humanoid") :: Humanoid
assert(humanoid, "spawned character has no Humanoid")
local animator = humanoid:FindFirstChildOfClass("Animator") :: Animator
if not animator then
	animator = Instance.new("Animator")
	animator.Parent = humanoid
end

-- Disable retargeting so what we sample matches the source clip exactly.
workspace.Retargeting = Enum.AnimatorRetargetingMode.Disabled

-- =============================================================================
-- Register clip + load track. RegisterKeyframeSequence accepts both
-- KeyframeSequence and CurveAnimation and returns a content id usable as
-- Animation.AnimationId.
-- =============================================================================
local contentId = KeyframeSequenceProvider:RegisterKeyframeSequence(clip)
assert(contentId, "RegisterKeyframeSequence returned nil")
local animationInst = Instance.new("Animation")
animationInst.AnimationId = contentId

local track = animator:LoadAnimation(animationInst)
track.Looped = false
track:Play(0)
-- Length is populated asynchronously after Play; spin until it's set.
local spinFor = 0.0
while track.Length == 0.0 and spinFor < 5.0 do
	wait(0)
	spinFor += 1/60
end
assert(track.Length > 0, "track.Length still 0 after 5s — clip may be empty")
local duration = track.Length
print(string.format("[extract_pose] duration=%.4fs", duration))

-- =============================================================================
-- Sampling layout.
--
-- Two parallel streams per frame:
--   1. Effector tips: world CFrames of hand/foot tips. Used by the Python
--      side for keyframe selection (velocity-valley detection on tip XZ)
--      and for diagnostic comparison. R6 derives tips from Arm/Leg CFrames
--      via a 1-stud Y offset (default R6 part is 1x2x1 centered).
--   2. Chain bones: world CFrames of every Roblox bone that maps to a
--      SOMA chain joint (Hips/Leg/Shin/Foot for the foot constraint;
--      Hips/Arm/ForeArm/Hand for the hand constraint). The Python side
--      runs the world-delta retarget on these to fill the SOMA chain
--      local rotations.
-- =============================================================================
local R6_TIP_OFFSET = CFrame.new(0, -1, 0)

local function findPart(name: string): BasePart
	local p = character:FindFirstChild(name)
	assert(p and p:IsA("BasePart"), "missing BasePart " .. name)
	return p :: BasePart
end

-- key → (() -> CFrame). Effector tip samplers (4 of them) and chain bone
-- samplers (rig-dependent count).
local effectorSamplers: { [string]: () -> CFrame } = {}
local chainSamplers: { [string]: () -> CFrame } = {}

if RIG_TYPE == "R15" then
	local lh = findPart("LeftHand")
	local rh = findPart("RightHand")
	local lf = findPart("LeftFoot")
	local rf = findPart("RightFoot")
	effectorSamplers.left_hand  = function() return lh.CFrame end
	effectorSamplers.right_hand = function() return rh.CFrame end
	effectorSamplers.left_foot  = function() return lf.CFrame end
	effectorSamplers.right_foot = function() return rf.CFrame end

	local lt = findPart("LowerTorso")
	local lua_part = findPart("LeftUpperArm"); local lla = findPart("LeftLowerArm")
	local rua = findPart("RightUpperArm");     local rla = findPart("RightLowerArm")
	local lul = findPart("LeftUpperLeg");      local lll = findPart("LeftLowerLeg")
	local rul = findPart("RightUpperLeg");     local rll = findPart("RightLowerLeg")
	chainSamplers.lower_torso     = function() return lt.CFrame end
	chainSamplers.left_upper_arm  = function() return lua_part.CFrame end
	chainSamplers.left_lower_arm  = function() return lla.CFrame end
	chainSamplers.left_hand       = function() return lh.CFrame end
	chainSamplers.right_upper_arm = function() return rua.CFrame end
	chainSamplers.right_lower_arm = function() return rla.CFrame end
	chainSamplers.right_hand      = function() return rh.CFrame end
	chainSamplers.left_upper_leg  = function() return lul.CFrame end
	chainSamplers.left_lower_leg  = function() return lll.CFrame end
	chainSamplers.left_foot       = function() return lf.CFrame end
	chainSamplers.right_upper_leg = function() return rul.CFrame end
	chainSamplers.right_lower_leg = function() return rll.CFrame end
	chainSamplers.right_foot      = function() return rf.CFrame end
else
	local la = findPart("Left Arm")
	local ra = findPart("Right Arm")
	local ll = findPart("Left Leg")
	local rl = findPart("Right Leg")
	local torso = findPart("Torso")
	print(string.format("[extract_pose] R6 part sizes: LA=%s RA=%s LL=%s RL=%s",
		tostring(la.Size), tostring(ra.Size), tostring(ll.Size), tostring(rl.Size)))
	effectorSamplers.left_hand  = function() return la.CFrame * R6_TIP_OFFSET end
	effectorSamplers.right_hand = function() return ra.CFrame * R6_TIP_OFFSET end
	effectorSamplers.left_foot  = function() return ll.CFrame * R6_TIP_OFFSET end
	effectorSamplers.right_foot = function() return rl.CFrame * R6_TIP_OFFSET end

	-- R6 has no shoulder/clavicle, no shin, no separate hand/foot — the
	-- whole limb is one rigid part. We sample only torso + the 4 limbs.
	-- The Python chain retargeter handles missing intermediate bones by
	-- inheriting D[parent] (identity local rotation on the missing SOMA
	-- joint), which keeps the limb straight as it should be for R6.
	chainSamplers.torso     = function() return torso.CFrame end
	chainSamplers.left_arm  = function() return la.CFrame end
	chainSamplers.right_arm = function() return ra.CFrame end
	chainSamplers.left_leg  = function() return ll.CFrame end
	chainSamplers.right_leg = function() return rl.CFrame end
end

-- =============================================================================
-- CFrame -> normalized quaternion (x, y, z, w).
-- =============================================================================
local function cframeToQuat(cf: CFrame): (number, number, number, number)
	local _, _, _, m00, m01, m02, m10, m11, m12, m20, m21, m22 = cf:GetComponents()
	local trace = m00 + m11 + m22
	local x, y, z, w
	if trace > 0 then
		local s = math.sqrt(trace + 1.0) * 2
		w = 0.25 * s
		x = (m21 - m12) / s
		y = (m02 - m20) / s
		z = (m10 - m01) / s
	elseif (m00 > m11) and (m00 > m22) then
		local s = math.sqrt(1.0 + m00 - m11 - m22) * 2
		w = (m21 - m12) / s
		x = 0.25 * s
		y = (m01 + m10) / s
		z = (m02 + m20) / s
	elseif m11 > m22 then
		local s = math.sqrt(1.0 + m11 - m00 - m22) * 2
		w = (m02 - m20) / s
		x = (m01 + m10) / s
		y = 0.25 * s
		z = (m12 + m21) / s
	else
		local s = math.sqrt(1.0 + m22 - m00 - m11) * 2
		w = (m10 - m01) / s
		x = (m02 + m20) / s
		y = (m12 + m21) / s
		z = 0.25 * s
	end
	-- Normalize to unit length.
	local n = math.sqrt(x*x + y*y + z*z + w*w)
	if n < 1e-9 then return 0, 0, 0, 1 end
	return x/n, y/n, z/n, w/n
end

local function sampleTransform(cf: CFrame): { pos: { number }, rot: { number } }
	local p = cf.Position
	local qx, qy, qz, qw = cframeToQuat(cf)
	return { pos = { p.X, p.Y, p.Z }, rot = { qx, qy, qz, qw } }
end

-- =============================================================================
-- Step through the clip and sample. If MIN_DURATION exceeds the source
-- length we loop the clip — the OUTPUT duration is `outDuration` and we
-- wrap the sample time back into [0, duration) when seeking TimePosition.
-- LOOP_PASSES>1 only applies to clips marked Loop and overrides
-- MIN_DURATION; the Python side later trims to the middle cycle so the
-- final exported clip is one cycle long with smooth boundaries.
-- =============================================================================
local effectiveLoopPasses = LOOP_PASSES
if effectiveLoopPasses > 1 and not IS_LOOPED then
	print("[extract_pose] loop_passes>1 ignored (clip.Loop is false)")
	effectiveLoopPasses = 1
end
local outDuration = math.max(duration, MIN_DURATION)
if effectiveLoopPasses > 1 then
	outDuration = duration * effectiveLoopPasses
end
if outDuration > duration then
	track.Looped = true
	print(string.format("[extract_pose] looping %.3fs source to fill %.2fs output (passes=%d)",
		duration, outDuration, effectiveLoopPasses))
end
local nFrames = math.max(2, math.floor(outDuration * FPS + 0.5) + 1)
local dt = outDuration / (nFrames - 1)

local frames = table.create(nFrames)
for i = 0, nFrames - 1 do
	local t = math.min(i * dt, outDuration)
	-- Wrap into the source clip's own duration so TimePosition stays valid
	-- even when we're sampling beyond the source's native length.
	local srcT = duration > 0 and (t % duration) or t
	track.TimePosition = srcT
	animator:StepAnimations(0)

	local sample: { [string]: any } = { t = t }
	sample.hrp = sampleTransform(hrp.CFrame)
	for name, fn in pairs(effectorSamplers) do
		sample[name] = sampleTransform(fn())
	end
	local chain: { [string]: any } = {}
	for name, fn in pairs(chainSamplers) do
		chain[name] = sampleTransform(fn())
	end
	sample.chain = chain
	frames[i + 1] = sample
end

track:Stop(0)
track:Destroy()
character:Destroy()

-- =============================================================================
-- Write output JSON.
-- =============================================================================
-- Native frame count of one source cycle, useful for the Python trim step.
local sourceNFrames = math.max(2, math.floor(duration * FPS + 0.5) + 1)
local payload = {
	asset_id = ASSET_ID,
	clip_name = clip.Name,
	clip_class = clip.ClassName,
	rig_type = RIG_TYPE,
	fps = FPS,
	n_frames = nFrames,
	duration_s = outDuration,
	source_duration_s = duration,
	source_n_frames = sourceNFrames,
	min_duration_s = MIN_DURATION,
	loop_passes = effectiveLoopPasses,
	looped = IS_LOOPED,
	frames = frames,
}

local encoded = HttpService:JSONEncode(payload)
FileSystemService:WriteFile(OUT_PATH, encoded, Enum.FileMode.Text)
print(string.format("[extract_pose] wrote %d frames to %s (%d bytes)",
	nFrames, OUT_PATH, #encoded))
