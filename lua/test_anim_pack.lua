--!strict
-- Wire a generated animation pack onto the local character and drive it through
-- every state, so you can actually see the slots the way players will.
--
-- A grid preview (lua/play_emotes.lua) is not enough for a pack: foot skate, the
-- walk<->run cross-fade, and state-transition pops only exist when the ENGINE is
-- moving the character. This drives the real Humanoid.
--
-- Put this in ServerScriptService (or paste in the Command Bar during Play) with
-- the pack's CurveAnimations somewhere in the DataModel -- import each slot's
-- r15.rbxm, or the merged pack rbxm. Clip names must match the slot names:
--   idle, idle2, walk, run, jump, fall, climb, swim, swimidle
--
-- Requires StarterPlayer.AllowCustomAnimations = true, otherwise the Animate
-- script ignores overrides entirely (humanoidAnimateSAuth.lua:648-651).
--
-- The speed sweep is the point: the Animate script loads BOTH the walk and run
-- tracks whenever the pose is Running and cross-fades them between roughly 6.4
-- and 12.8 studs/s. If walk and run have different Length their phases drift and
-- you see four legs. Watch the band, not the endpoints.

local Players = game:GetService("Players")
local AnimationClipProvider = game:GetService("AnimationClipProvider")
local StarterPlayer = game:GetService("StarterPlayer")

local SLOT_TO_ANIMATE_PATH: { [string]: { string } } = {
	-- slot -> { child of the Animate script, child Animation name }
	idle     = { "idle", "Animation1" },
	idle2    = { "idle", "Animation2" },
	walk     = { "walk", "WalkAnim" },
	run      = { "run", "RunAnim" },
	jump     = { "jump", "JumpAnim" },
	fall     = { "fall", "FallAnim" },
	climb    = { "climb", "ClimbAnim" },
	swim     = { "swim", "Swim" },
	swimidle = { "swimidle", "SwimIdle" },
}

-- Docs' example weighting for the two idles: 5 / 10 => one third, two thirds.
local IDLE_WEIGHTS: { [string]: number } = { idle = 5, idle2 = 10 }

local SPEED_SWEEP = { 2, 4, 6, 6.4, 8, 10, 12, 12.8, 16, 20 }
local SECONDS_PER_SPEED = 3

local function findClips(): { [string]: AnimationClip }
	local found: { [string]: AnimationClip } = {}
	for _, d in game:GetDescendants() do
		if d:IsA("AnimationClip") and SLOT_TO_ANIMATE_PATH[d.Name] then
			found[d.Name] = d :: AnimationClip
		end
	end
	return found
end

local function registerAll(clips: { [string]: AnimationClip }): { [string]: string }
	local ids: { [string]: string } = {}
	for slot, clip in clips do
		local ok, id = pcall(function()
			return AnimationClipProvider:RegisterAnimationClip(clip)
		end)
		if ok and id then
			ids[slot] = id :: string
			print(string.format("  registered %-9s Loop=%s Priority=%s",
				slot, tostring((clip :: any).Loop), tostring((clip :: any).Priority)))
		else
			warn(string.format("  FAILED to register %s: %s", slot, tostring(id)))
		end
	end
	return ids
end

local function applyTo(character: Model, ids: { [string]: string })
	local animate = character:FindFirstChild("Animate")
	if not animate then
		warn("[test_anim_pack] no Animate script in character. Is "
			.. "AllowCustomAnimations enabled and the character loaded?")
		return
	end

	for slot, path in SLOT_TO_ANIMATE_PATH do
		local id = ids[slot]
		if not id then continue end
		local group = animate:FindFirstChild(path[1])
		if not group then
			warn(string.format("[test_anim_pack] Animate has no %q group", path[1]))
			continue
		end
		local anim = group:FindFirstChild(path[2])
		if not anim then
			-- Child names are not load-bearing: configureAnimationSet iterates
			-- whatever Animation children exist. Create one if the expected
			-- name is absent.
			anim = Instance.new("Animation")
			anim.Name = path[2]
			anim.Parent = group
		end
		;(anim :: Animation).AnimationId = id

		local weight = IDLE_WEIGHTS[slot]
		if weight then
			local wv = anim:FindFirstChild("Weight")
			if not wv then
				wv = Instance.new("NumberValue")
				wv.Name = "Weight"
				wv.Parent = anim
			end
			;(wv :: NumberValue).Value = weight
		end
		print(string.format("  %-9s -> Animate.%s.%s", slot, path[1], path[2]))
	end
end

local function sweepSpeed(humanoid: Humanoid)
	print("[test_anim_pack] speed sweep -- watch 6.4..12.8 for the walk/run blend")
	for _, speed in SPEED_SWEEP do
		humanoid.WalkSpeed = speed
		local marker = if speed >= 6.4 and speed <= 12.8 then "  <-- BLEND BAND" else ""
		print(string.format("  WalkSpeed = %.1f%s", speed, marker))
		-- Walk a straight line so foot skate is visible.
		local root = humanoid.RootPart
		if root then
			humanoid:MoveTo(root.Position + root.CFrame.LookVector * (speed * SECONDS_PER_SPEED))
		end
		task.wait(SECONDS_PER_SPEED)
	end
	humanoid.WalkSpeed = 16
	print("[test_anim_pack] sweep done, WalkSpeed restored to 16")
end

-- ------------------------------------------------------------------ main ----
if not StarterPlayer.AllowCustomAnimations then
	warn("[test_anim_pack] StarterPlayer.AllowCustomAnimations is FALSE -- the "
		.. "Animate script will ignore every override. Set it true and retry.")
end

local clips = findClips()
local n = 0
for _ in clips do n += 1 end
print(string.format("[test_anim_pack] found %d/9 pack clips", n))
if n == 0 then
	error("No pack clips found. Import the slot rbxms; clip names must be "
		.. "idle, idle2, walk, run, jump, fall, climb, swim, swimidle.")
end
for slot in SLOT_TO_ANIMATE_PATH do
	if not clips[slot] then
		warn(string.format("  missing slot: %s", slot))
	end
end

local ids = registerAll(clips)

local function onCharacter(character: Model)
	local humanoid = character:WaitForChild("Humanoid") :: Humanoid
	-- Stop whatever the default pack started before swapping ids in.
	local animator = humanoid:FindFirstChildOfClass("Animator")
	if animator then
		for _, track in animator:GetPlayingAnimationTracks() do
			track:Stop(0)
		end
	end
	applyTo(character, ids)
	task.wait(1)
	sweepSpeed(humanoid)
end

Players.PlayerAdded:Connect(function(player)
	player.CharacterAppearanceLoaded:Connect(onCharacter)
	if player.Character then
		onCharacter(player.Character)
	end
end)
for _, player in Players:GetPlayers() do
	player.CharacterAppearanceLoaded:Connect(onCharacter)
	if player.Character then
		task.spawn(onCharacter, player.Character)
	end
end
