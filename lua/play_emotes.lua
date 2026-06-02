--!strict
-- Studio playback script: finds all AnimationClip instances in Workspace,
-- registers them via AnimationClipProvider:RegisterClipContent, clones the
-- "Rig" template for each, and plays the animations in a grid layout.
--
-- Usage: paste into Studio Command Bar or run as a Script in ServerScriptService.
-- Prerequisites:
--   - A Humanoid rig named "Rig" in Workspace (used as the template).
--   - AnimationClip instances (CurveAnimation/KeyframeSequence) anywhere
--     in Workspace (import the merged all_emotes.rbxm into Workspace).

local AnimationClipProvider = game:GetService("AnimationClipProvider")

-- Configuration
local GRID_SPACING = 8         -- studs between rigs
local GRID_COLUMNS = 10        -- rigs per row
local START_POSITION = Vector3.new(0, 0, 0)

-- Find the template rig
local templateRig = workspace:FindFirstChild("Rig")
if not templateRig then
	error("No 'Rig' found in Workspace. Please add an R15 humanoid rig named 'Rig'.")
end

-- Recursively find all AnimationClip instances in Workspace
local function findClips(parent: Instance): { AnimationClip }
	local clips: { AnimationClip } = {}
	for _, child in parent:GetDescendants() do
		if child:IsA("AnimationClip") then
			table.insert(clips, child :: AnimationClip)
		end
	end
	return clips
end

local clips = findClips(workspace)
print(string.format("[play_emotes] Found %d AnimationClip instances", #clips))

if #clips == 0 then
	warn("[play_emotes] No AnimationClips found in Workspace. Import your emotes first.")
	return
end

-- Clean up any previously spawned playback rigs
local playbackFolder = workspace:FindFirstChild("EmotePlayback")
if playbackFolder then
	playbackFolder:Destroy()
end
playbackFolder = Instance.new("Folder")
playbackFolder.Name = "EmotePlayback"
playbackFolder.Parent = workspace

-- Hide the template rig
templateRig.PrimaryPart = templateRig:FindFirstChild("HumanoidRootPart") :: BasePart
local origPosition = (templateRig :: Model).PrimaryPart.Position

for i, clip in clips do
	-- Grid position
	local col = (i - 1) % GRID_COLUMNS
	local row = math.floor((i - 1) / GRID_COLUMNS)
	local pos = START_POSITION + Vector3.new(col * GRID_SPACING, 0, row * GRID_SPACING)

	-- Clone the template rig
	local rig = (templateRig :: Model):Clone()
	rig.Name = clip.Name
	rig.Parent = playbackFolder

	-- Position it
	local hrp = rig:FindFirstChild("HumanoidRootPart") :: BasePart
	if hrp then
		local offset = pos - origPosition
		rig:PivotTo(rig:GetPivot() + offset)
	end

	-- Add a BillboardGui label above the rig
	local billboard = Instance.new("BillboardGui")
	billboard.Name = "Label"
	billboard.Size = UDim2.new(0, 200, 0, 50)
	billboard.StudsOffset = Vector3.new(0, 6, 0)
	billboard.AlwaysOnTop = true
	billboard.Adornee = hrp
	billboard.Parent = rig

	local label = Instance.new("TextLabel")
	label.Size = UDim2.new(1, 0, 1, 0)
	label.BackgroundTransparency = 0.5
	label.BackgroundColor3 = Color3.new(0, 0, 0)
	label.TextColor3 = Color3.new(1, 1, 1)
	label.TextScaled = true
	label.Text = clip.Name
	label.Parent = billboard

	-- Register the clip and play it
	local humanoid = rig:FindFirstChildOfClass("Humanoid")
	if not humanoid then
		warn(string.format("[play_emotes] No Humanoid in rig clone for %s", clip.Name))
		continue
	end

	local animator = humanoid:FindFirstChildOfClass("Animator")
	if not animator then
		animator = Instance.new("Animator")
		animator.Parent = humanoid
	end

	-- RegisterClipContent returns a content URI we can use to load the animation
	local ok, contentId = pcall(function()
		return AnimationClipProvider:RegisterClipContent(clip)
	end)
	if not ok then
		warn(string.format("[play_emotes] Failed to register %s: %s", clip.Name, tostring(contentId)))
		continue
	end

	-- Create an Animation instance with the registered content
	local anim = Instance.new("Animation")
	anim.AnimationId = contentId
	anim.Parent = rig

	-- Determine if this clip should loop. Use the clip's Loop property if set,
	-- otherwise check if the clip's parent folder has a "looping" BoolValue,
	-- otherwise check clip.Loop property directly.
	local shouldLoop = false
	if (clip :: any).Loop ~= nil then
		shouldLoop = (clip :: any).Loop
	end

	-- Load and play
	local track = (animator :: Animator):LoadAnimation(anim)
	track.Looped = shouldLoop
	track:Play()

	print(string.format("[play_emotes] [%d/%d] Playing: %s (looped=%s)",
		i, #clips, clip.Name, tostring(track.Looped)))
end

print(string.format("[play_emotes] Done! %d rigs spawned in Workspace.EmotePlayback", #clips))
print("[play_emotes] To clean up: workspace.EmotePlayback:Destroy()")
