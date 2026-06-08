--!strict
--[[
	RoMotion - Generative Animation Authoring Plugin for Roblox Studio

	Uses Kimodo motion synthesis via a local backend server to generate
	animations from text prompts + sparse effector constraints.
]]

local Selection = game:GetService("Selection")
local ChangeHistoryService = game:GetService("ChangeHistoryService")

local src = script.Parent.src
local Constants = require(src.Utils.Constants)
local State = require(src.State)
local Signal = require(src.Signal)
local RigService = require(src.Services.RigService)
local BackendService = require(src.Services.BackendService)
local PlaybackService = require(src.Services.PlaybackService)
local DataModelService = require(src.Services.DataModelService)
local AnimationBuilder = require(src.Services.AnimationBuilder)

-- ════════════════════════════════════════════════════════════════════
-- Plugin Setup
-- ════════════════════════════════════════════════════════════════════

local toolbar = plugin:CreateToolbar(Constants.PLUGIN_NAME)
local toggleButton = toolbar:CreateButton(
	Constants.PLUGIN_ID,
	"Open RoMotion - Generative Animation Editor",
	"rbxassetid://6031068420", -- placeholder icon
	"RoMotion"
)

local widgetInfo = DockWidgetPluginGuiInfo.new(
	Enum.InitialDockState.Bottom,
	false, -- initially disabled
	false, -- override previous state
	800,   -- default width
	400,   -- default height
	400,   -- min width
	200    -- min height
)

local widget = plugin:CreateDockWidgetPluginGui(Constants.WIDGET_ID, widgetInfo)
widget.Title = "RoMotion"
widget.Name = Constants.WIDGET_ID

-- ════════════════════════════════════════════════════════════════════
-- App State
-- ════════════════════════════════════════════════════════════════════

-- Ground Y from rig's REST pose (computed once on rig selection, never changes)
local rigRestGroundY = 0

local appState = {
	rig = State.new(nil :: RigService.RigInfo?),
	projectName = State.new("Untitled"),
	duration = State.new(Constants.DEFAULT_DURATION),
	prompts = State.new({} :: { DataModelService.PromptData }),
	constraints = State.new({} :: { DataModelService.ConstraintData }),
	playbackTime = State.new(0),
	playbackState = State.new("stopped" :: PlaybackService.PlaybackState),
	generationStatus = State.new("idle" :: string), -- idle | generating | completed | failed
	generationProgress = State.new(0),
	generationMessage = State.new(""),
	currentJobId = State.new(nil :: string?),
	scrollOffset = State.new(0),
	pixelsPerSecond = State.new(100),
	selectedConstraints = State.new({} :: { number }), -- indices into constraints
	looped = State.new(false),
	seed = State.new(0),
	serverConnected = State.new(false),
}

local playbackSvc: typeof(PlaybackService.new(nil :: any))? = nil

-- Constraint color: base effector hue shifted subtly by timeline position
-- so multiple constraints of the same effector are visually distinguishable.
local function constraintColor(effector: string, timeFrac: number): Color3
	local base = Constants.EFFECTOR_COLORS[effector] or Color3.new(1, 1, 1)
	local h, s, v = base:ToHSV()
	h = (h + (timeFrac - 0.5) * 0.16) % 1 -- ±0.08 hue shift across the timeline
	return Color3.fromHSV(h, s, v)
end

-- Authoritative max timeline time: longer of the prompt timeline and the
-- loaded animation track. Used by scrub + all constraint/prompt drag clamps.
local function getMaxTime(): number
	local prompts = appState.prompts:get()
	local promptEnd = if #prompts > 0 then prompts[#prompts].endTime else 0
	local trackEnd = if playbackSvc then playbackSvc:getDuration() else 0
	return math.max(promptEnd, trackEnd, 1)
end

-- Time of the LAST valid frame. A clip of duration D has frames 0..N-1 where
-- N = round(D*fps); the last frame sits at (N-1)/fps, NOT D. Scrubbing or
-- constraining to D would land on frame N (one past the end) which Kimodo drops
-- and which wraps the AnimationTrack to the start.
local function getLastFrameTime(): number
	local n = math.max(1, math.floor(getMaxTime() * Constants.FPS + 0.5))
	return (n - 1) / Constants.FPS
end

-- Place a constraint for `effector` at `time`, capturing the rig's current
-- pose. Seeks playback to `time` first so the chain is captured at that frame.
-- Shared by manual placement (label click) and auto-constraint.
local function placeConstraint(effector: string, time: number)
	local rig = appState.rig:get()
	if not rig then return end

	-- Snap to nearest frame at FPS
	time = math.floor(time * Constants.FPS + 0.5) / Constants.FPS

	if playbackSvc then
		playbackSvc:seekTo(time)
		playbackSvc:ensureStepped()
	end

	local effPart = RigService.getEffectorPart(rig, effector)
	if not effPart then return end

	local color = Constants.EFFECTOR_COLORS[effector] or Color3.new(1, 1, 1)
	local chainModel = RigService.cloneChain(rig, effector, color)
	if not chainModel then return end

	local constraints_val = table.clone(appState.constraints:get())
	table.insert(constraints_val, {
		effector = effector,
		time = time,
		cframe = effPart.CFrame,
		chain = chainModel,
	})
	table.sort(constraints_val, function(a, b) return a.time < b.time end)
	appState.constraints:set(constraints_val)
end

-- ════════════════════════════════════════════════════════════════════
-- UI Construction
-- ════════════════════════════════════════════════════════════════════

local function createThemeColor(element: string): Color3
	local theme = settings().Studio.Theme
	return theme:GetColor(Enum.StudioStyleGuideColor.MainBackground)
end

local function buildUI()
	local GUTTER = Constants.GUTTER

	-- Timeline coordinate conversion (shared everywhere). time=0 maps to x=GUTTER
	-- so the left gutter holds effector labels/+buttons without overlapping content.
	local function timeToPx(t: number): number
		return GUTTER + (t - appState.scrollOffset:get()) * appState.pixelsPerSecond:get()
	end
	-- relX is relative to a frame whose left edge is the timeline left edge.
	local function pxToTime(relX: number): number
		return (relX - GUTTER) / appState.pixelsPerSecond:get() + appState.scrollOffset:get()
	end
	-- Snap a time to the nearest frame at FPS.
	local function snapToFrame(t: number): number
		return math.floor(t * Constants.FPS + 0.5) / Constants.FPS
	end

	-- Main container
	local mainFrame = Instance.new("Frame")
	mainFrame.Name = "MainFrame"
	mainFrame.Size = UDim2.fromScale(1, 1)
	mainFrame.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainBackground)
	mainFrame.BorderSizePixel = 0
	mainFrame.Parent = widget

	local layout = Instance.new("UIListLayout")
	layout.FillDirection = Enum.FillDirection.Vertical
	layout.SortOrder = Enum.SortOrder.LayoutOrder
	layout.Padding = UDim.new(0, 0)
	layout.Parent = mainFrame

	-- ─── Top Bar (Rig selection + project name + server status) ───
	local topBar = Instance.new("Frame")
	topBar.Name = "TopBar"
	topBar.Size = UDim2.new(1, 0, 0, 32)
	topBar.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Titlebar)
	topBar.BorderSizePixel = 0
	topBar.LayoutOrder = 0
	topBar.Parent = mainFrame

	local topLayout = Instance.new("UIListLayout")
	topLayout.FillDirection = Enum.FillDirection.Horizontal
	topLayout.VerticalAlignment = Enum.VerticalAlignment.Center
	topLayout.SortOrder = Enum.SortOrder.LayoutOrder
	topLayout.Padding = UDim.new(0, 8)
	topLayout.Parent = topBar

	local topPad = Instance.new("UIPadding")
	topPad.PaddingLeft = UDim.new(0, 8)
	topPad.Parent = topBar

	local rigLabel = Instance.new("TextLabel")
	rigLabel.Name = "RigLabel"
	rigLabel.Size = UDim2.new(0, 150, 1, 0)
	rigLabel.BackgroundTransparency = 1
	rigLabel.Text = "No rig selected"
	rigLabel.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainText)
	rigLabel.TextSize = 13
	rigLabel.Font = Enum.Font.SourceSans
	rigLabel.TextXAlignment = Enum.TextXAlignment.Left
	rigLabel.LayoutOrder = 1
	rigLabel.Parent = topBar

	-- Duration input
	local durLabel = Instance.new("TextLabel")
	durLabel.Size = UDim2.new(0, 50, 1, 0)
	durLabel.BackgroundTransparency = 1
	durLabel.Text = "Duration:"
	durLabel.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.DimmedText)
	durLabel.TextSize = 12
	durLabel.Font = Enum.Font.SourceSans
	durLabel.TextXAlignment = Enum.TextXAlignment.Right
	durLabel.LayoutOrder = 2
	durLabel.Parent = topBar

	local durInput = Instance.new("TextBox")
	durInput.Name = "DurationInput"
	durInput.Size = UDim2.new(0, 40, 0, 20)
	durInput.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.InputFieldBackground)
	durInput.BorderColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Border)
	durInput.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainText)
	durInput.Text = tostring(Constants.DEFAULT_DURATION)
	durInput.TextSize = 12
	durInput.Font = Enum.Font.Code
	durInput.ClearTextOnFocus = false
	durInput.LayoutOrder = 3
	durInput.Parent = topBar
	local durCorner = Instance.new("UICorner")
	durCorner.CornerRadius = UDim.new(0, 3)
	durCorner.Parent = durInput

	local durSuffix = Instance.new("TextLabel")
	durSuffix.Size = UDim2.new(0, 12, 1, 0)
	durSuffix.BackgroundTransparency = 1
	durSuffix.Text = "s"
	durSuffix.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.DimmedText)
	durSuffix.TextSize = 12
	durSuffix.Font = Enum.Font.SourceSans
	durSuffix.LayoutOrder = 4
	durSuffix.Parent = topBar

	durInput.FocusLost:Connect(function()
		local val = tonumber(durInput.Text)
		if val and val > 0.5 and val <= 30 then
			appState.duration:set(val)
			-- Also stretch the last prompt block to fill
			local prompts_val = appState.prompts:get()
			if #prompts_val > 0 then
				local updated = table.clone(prompts_val)
				updated[#updated] = table.clone(updated[#updated])
				updated[#updated].endTime = val
				appState.prompts:set(updated)
			end
		else
			durInput.Text = string.format("%.1f", appState.duration:get())
		end
	end)

	appState.duration:subscribe(function(d)
		durInput.Text = string.format("%.1f", d)
	end)

	-- Seed input (0 = random each generation)
	local seedLabel = Instance.new("TextLabel")
	seedLabel.Size = UDim2.new(0, 36, 1, 0)
	seedLabel.BackgroundTransparency = 1
	seedLabel.Text = "Seed:"
	seedLabel.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.DimmedText)
	seedLabel.TextSize = 12
	seedLabel.Font = Enum.Font.SourceSans
	seedLabel.TextXAlignment = Enum.TextXAlignment.Right
	seedLabel.LayoutOrder = 5
	seedLabel.Parent = topBar

	local seedInput = Instance.new("TextBox")
	seedInput.Name = "SeedInput"
	seedInput.Size = UDim2.new(0, 70, 0, 20)
	seedInput.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.InputFieldBackground)
	seedInput.BorderColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Border)
	seedInput.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainText)
	seedInput.Text = "0"
	seedInput.PlaceholderText = "0 = random"
	seedInput.TextSize = 12
	seedInput.Font = Enum.Font.Code
	seedInput.ClearTextOnFocus = false
	seedInput.LayoutOrder = 6
	seedInput.Parent = topBar
	local seedCorner = Instance.new("UICorner")
	seedCorner.CornerRadius = UDim.new(0, 3)
	seedCorner.Parent = seedInput

	seedInput.FocusLost:Connect(function()
		local val = tonumber(seedInput.Text)
		if val and val >= 0 then
			appState.seed:set(math.floor(val))
		else
			seedInput.Text = tostring(appState.seed:get())
		end
	end)

	appState.seed:subscribe(function(s)
		seedInput.Text = tostring(s)
	end)

	local serverDot = Instance.new("Frame")
	serverDot.Name = "ServerDot"
	serverDot.Size = UDim2.new(0, 8, 0, 8)
	serverDot.BackgroundColor3 = Color3.fromRGB(244, 67, 54)
	serverDot.LayoutOrder = 7
	serverDot.Parent = topBar
	local corner = Instance.new("UICorner")
	corner.CornerRadius = UDim.new(1, 0)
	corner.Parent = serverDot

	local serverLabel = Instance.new("TextLabel")
	serverLabel.Name = "ServerLabel"
	serverLabel.Size = UDim2.new(0, 50, 1, 0)
	serverLabel.BackgroundTransparency = 1
	serverLabel.Text = "Server"
	serverLabel.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.DimmedText)
	serverLabel.TextSize = 11
	serverLabel.Font = Enum.Font.SourceSans
	serverLabel.TextXAlignment = Enum.TextXAlignment.Left
	serverLabel.LayoutOrder = 8
	serverLabel.Parent = topBar

	-- ─── Toolbar (Transport + Generate + Actions) ───
	local toolBar = Instance.new("Frame")
	toolBar.Name = "ToolBar"
	toolBar.Size = UDim2.new(1, 0, 0, 36)
	toolBar.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.RibbonTab)
	toolBar.BorderSizePixel = 0
	toolBar.LayoutOrder = 1
	toolBar.Parent = mainFrame

	local toolLayout = Instance.new("UIListLayout")
	toolLayout.FillDirection = Enum.FillDirection.Horizontal
	toolLayout.VerticalAlignment = Enum.VerticalAlignment.Center
	toolLayout.Padding = UDim.new(0, 4)
	toolLayout.Parent = toolBar

	local toolPad = Instance.new("UIPadding")
	toolPad.PaddingLeft = UDim.new(0, 8)
	toolPad.Parent = toolBar

	-- Transport buttons
	local function makeButton(name: string, text: string, color: Color3?): TextButton
		local btn = Instance.new("TextButton")
		btn.Name = name
		btn.Size = UDim2.new(0, 32, 0, 26)
		btn.BackgroundColor3 = color or settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Button)
		btn.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.ButtonText)
		btn.Text = text
		btn.TextSize = 14
		btn.Font = Enum.Font.SourceSansBold
		btn.AutoButtonColor = true
		btn.Parent = toolBar
		local c = Instance.new("UICorner")
		c.CornerRadius = UDim.new(0, 4)
		c.Parent = btn
		return btn
	end

	local playBtn = makeButton("Play", "▶")
	local pauseBtn = makeButton("Pause", "⏸")
	local stopBtn = makeButton("Stop", "⏹")

	-- Time display
	local timeLabel = Instance.new("TextLabel")
	timeLabel.Name = "TimeLabel"
	timeLabel.Size = UDim2.new(0, 80, 0, 26)
	timeLabel.BackgroundTransparency = 1
	timeLabel.Text = "0.000s"
	timeLabel.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainText)
	timeLabel.TextSize = 13
	timeLabel.Font = Enum.Font.Code
	timeLabel.TextXAlignment = Enum.TextXAlignment.Center
	timeLabel.Parent = toolBar

	-- Separator
	local sep = Instance.new("Frame")
	sep.Name = "Sep"
	sep.Size = UDim2.new(0, 1, 0, 20)
	sep.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Border)
	sep.BorderSizePixel = 0
	sep.Parent = toolBar

	-- Generate button
	local generateBtn = makeButton("Generate", "Generate", Color3.fromRGB(76, 175, 80))
	generateBtn.Size = UDim2.new(0, 80, 0, 26)
	generateBtn.TextColor3 = Color3.new(1, 1, 1)

	-- Progress label
	local progressLabel = Instance.new("TextLabel")
	progressLabel.Name = "ProgressLabel"
	progressLabel.Size = UDim2.new(0, 120, 0, 26)
	progressLabel.BackgroundTransparency = 1
	progressLabel.Text = ""
	progressLabel.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.DimmedText)
	progressLabel.TextSize = 11
	progressLabel.Font = Enum.Font.SourceSans
	progressLabel.TextXAlignment = Enum.TextXAlignment.Left
	progressLabel.Parent = toolBar

	-- Separator 2
	local sep2 = Instance.new("Frame")
	sep2.Name = "Sep2"
	sep2.Size = UDim2.new(0, 1, 0, 20)
	sep2.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Border)
	sep2.BorderSizePixel = 0
	sep2.Parent = toolBar

	-- Action buttons
	local autoConstrainBtn = makeButton("AutoConstrain", "Auto-C")
	autoConstrainBtn.Size = UDim2.new(0, 56, 0, 26)

	local clearBtn = makeButton("ClearConstraints", "Clear")
	clearBtn.Size = UDim2.new(0, 48, 0, 26)

	local importBtn = makeButton("Import", "Import")
	importBtn.Size = UDim2.new(0, 56, 0, 26)

	-- ─── Timeline Area ───
	local timelineFrame = Instance.new("Frame")
	timelineFrame.Name = "Timeline"
	timelineFrame.Size = UDim2.new(1, 0, 1, -68) -- fill remaining
	timelineFrame.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainBackground)
	timelineFrame.BorderSizePixel = 0
	timelineFrame.LayoutOrder = 2
	timelineFrame.ClipsDescendants = true
	timelineFrame.Parent = mainFrame

	-- Time ruler
	local ruler = Instance.new("Frame")
	ruler.Name = "Ruler"
	ruler.Size = UDim2.new(1, 0, 0, Constants.RULER_HEIGHT)
	ruler.Position = UDim2.new(0, 0, 0, 0)
	ruler.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Titlebar)
	ruler.BorderSizePixel = 0
	ruler.Parent = timelineFrame

	-- Prompt track
	local promptTrack = Instance.new("Frame")
	promptTrack.Name = "PromptTrack"
	promptTrack.Size = UDim2.new(1, 0, 0, Constants.PROMPT_TRACK_HEIGHT)
	promptTrack.Position = UDim2.new(0, 0, 0, Constants.RULER_HEIGHT)
	promptTrack.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainBackground)
	promptTrack.BorderColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Border)
	promptTrack.BorderSizePixel = 1
	promptTrack.ClipsDescendants = true
	promptTrack.Parent = timelineFrame

	-- Constraint tracks
	local constraintArea = Instance.new("Frame")
	constraintArea.Name = "ConstraintTracks"
	constraintArea.Size = UDim2.new(1, 0, 0, Constants.CONSTRAINT_TRACK_HEIGHT * #Constants.EFFECTORS)
	constraintArea.Position = UDim2.new(0, 0, 0, Constants.RULER_HEIGHT + Constants.PROMPT_TRACK_HEIGHT)
	constraintArea.BackgroundTransparency = 1
	constraintArea.Parent = timelineFrame

	local constraintLayout = Instance.new("UIListLayout")
	constraintLayout.FillDirection = Enum.FillDirection.Vertical
	constraintLayout.SortOrder = Enum.SortOrder.LayoutOrder
	constraintLayout.Parent = constraintArea

	-- Forward declarations for constraint drag system
	local constraintDiamonds: { Frame } = {}
	local draggingConstraint: number? = nil
	local selectedConstraint: number? = nil
	local renderConstraints: () -> ()

	for i, effName in Constants.EFFECTORS do
		local track = Instance.new("Frame")
		track.Name = effName
		track.Size = UDim2.new(1, 0, 0, Constants.CONSTRAINT_TRACK_HEIGHT)
		track.BackgroundColor3 = if i % 2 == 0
			then settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainBackground)
			else settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.TableItem)
		track.BorderSizePixel = 0
		track.LayoutOrder = i
		track.ClipsDescendants = true
		track.Active = true
		track.Parent = constraintArea

		local eName = effName

		-- "+" button to add a constraint for this effector at the current time
		local addBtn = Instance.new("TextButton")
		addBtn.Name = "Add"
		addBtn.Size = UDim2.new(0, 18, 0, 18)
		addBtn.Position = UDim2.new(0, 2, 0.5, -9)
		addBtn.BackgroundColor3 = Constants.EFFECTOR_COLORS[effName] or Color3.new(1, 1, 1)
		addBtn.BackgroundTransparency = 0.3
		addBtn.Text = "+"
		addBtn.TextColor3 = Color3.new(0, 0, 0)
		addBtn.TextSize = 14
		addBtn.Font = Enum.Font.SourceSansBold
		addBtn.AutoButtonColor = true
		addBtn.ZIndex = 6
		addBtn.Parent = track
		local addCorner = Instance.new("UICorner")
		addCorner.CornerRadius = UDim.new(0, 3)
		addCorner.Parent = addBtn
		addBtn.MouseButton1Click:Connect(function()
			placeConstraint(eName, appState.playbackTime:get())
		end)

		local label = Instance.new("TextLabel")
		label.Size = UDim2.new(0, 60, 1, 0)
		label.Position = UDim2.new(0, 24, 0, 0)
		label.BackgroundTransparency = 1
		label.Text = effName
		label.TextColor3 = Constants.EFFECTOR_COLORS[effName] or Color3.new(1, 1, 1)
		label.TextSize = 10
		label.Font = Enum.Font.SourceSans
		label.TextXAlignment = Enum.TextXAlignment.Left
		label.Parent = track

		-- Drag movement on this track
		track.InputChanged:Connect(function(input)
			if input.UserInputType == Enum.UserInputType.MouseMovement and draggingConstraint then
				local relX = input.Position.X - constraintArea.AbsolutePosition.X
				local newTime = math.clamp(pxToTime(relX), 0, getLastFrameTime())

				-- Move the diamond directly (don't trigger renderConstraints)
				if constraintDiamonds[draggingConstraint] then
					constraintDiamonds[draggingConstraint].Position = UDim2.new(0, math.floor(timeToPx(newTime)) - 8, 0.5, -8)
				end

				-- Update state silently (renderConstraints skips during drag)
				local updated = table.clone(appState.constraints:get())
				if draggingConstraint <= #updated then
					updated[draggingConstraint] = table.clone(updated[draggingConstraint])
					updated[draggingConstraint].time = newTime
					appState.constraints:set(updated)
				end
			end
		end)
		track.InputEnded:Connect(function(input)
			if input.UserInputType == Enum.UserInputType.MouseButton1 then
				-- Snap the dragged constraint to the nearest frame
				if draggingConstraint then
					local updated = table.clone(appState.constraints:get())
					if draggingConstraint <= #updated then
						updated[draggingConstraint] = table.clone(updated[draggingConstraint])
						updated[draggingConstraint].time = snapToFrame(updated[draggingConstraint].time)
						appState.constraints:set(updated)
					end
				end
				draggingConstraint = nil
				renderConstraints() -- rebuild now that drag is over
			end
		end)
	end

	-- Playhead
	local playhead = Instance.new("Frame")
	playhead.Name = "Playhead"
	playhead.Size = UDim2.new(0, 2, 1, 0)
	playhead.Position = UDim2.new(0, 0, 0, 0)
	playhead.BackgroundColor3 = Constants.PLAYHEAD_COLOR
	playhead.BorderSizePixel = 0
	playhead.ZIndex = 10
	playhead.Parent = timelineFrame

	-- ─── State→UI bindings ───

	appState.rig:subscribe(function(rig)
		if rig then
			rigLabel.Text = "Rig: " .. rig.model.Name
		else
			rigLabel.Text = "No rig selected"
		end
	end)

	appState.serverConnected:subscribe(function(connected)
		serverDot.BackgroundColor3 = if connected
			then Color3.fromRGB(76, 175, 80)
			else Color3.fromRGB(244, 67, 54)
	end)

	appState.playbackTime:subscribe(function(t)
		timeLabel.Text = string.format("%.3fs", t)
		playhead.Position = UDim2.new(0, math.floor(timeToPx(t)), 0, 0)
	end)

	appState.generationStatus:subscribe(function(status)
		if status == "generating" then
			generateBtn.Text = "..."
			generateBtn.BackgroundColor3 = Color3.fromRGB(255, 152, 0)
		elseif status == "completed" then
			generateBtn.Text = "Generate"
			generateBtn.BackgroundColor3 = Color3.fromRGB(76, 175, 80)
			progressLabel.Text = "Done!"
		elseif status == "failed" then
			generateBtn.Text = "Generate"
			generateBtn.BackgroundColor3 = Color3.fromRGB(76, 175, 80)
		else
			generateBtn.Text = "Generate"
			generateBtn.BackgroundColor3 = Color3.fromRGB(76, 175, 80)
			progressLabel.Text = ""
		end
	end)

	appState.generationMessage:subscribe(function(msg)
		progressLabel.Text = msg
	end)

	-- ─── Button Handlers ───

	playBtn.MouseButton1Click:Connect(function()
		if playbackSvc then
			playbackSvc:play()
		end
	end)

	pauseBtn.MouseButton1Click:Connect(function()
		if playbackSvc then
			playbackSvc:pause()
		end
	end)

	stopBtn.MouseButton1Click:Connect(function()
		if playbackSvc then
			playbackSvc:stop()
		end
	end)

	generateBtn.MouseButton1Click:Connect(function()
		local rig = appState.rig:get()
		if not rig then
			warn("[RoMotion] No rig selected")
			return
		end

		local prompts = appState.prompts:get()
		if #prompts == 0 then
			warn("[RoMotion] No prompts defined")
			return
		end

		appState.generationStatus:set("generating")
		appState.generationProgress:set(0)
		appState.generationMessage:set("Submitting...")

		task.spawn(function()
			local ok, result = pcall(function()
				-- Build request
				local promptSegments = {}
				for _, p in prompts do
					table.insert(promptSegments, {
						text = p.text,
						start_time = p.startTime,
						end_time = p.endTime,
					})
				end

				-- XZ: HRP center (stable). Y: floor (rigRestGroundY, stable).
				local hrpPos = rig.rootPart.Position
				local _, hrpYaw, _ = rig.rootPart.CFrame:ToEulerAnglesYXZ()
				local groundCF = CFrame.new(hrpPos.X, rigRestGroundY, hrpPos.Z)
					* CFrame.fromEulerAnglesYXZ(0, hrpYaw, 0)

				-- Root position is no longer used separately — it comes from the
				-- chain's LowerTorso position (relative to HRP) via chain_world_cframes

				-- Ensure animation is active so rig parts are at animated positions
				if playbackSvc then
					playbackSvc:ensureStepped()
				end

				local constraintsList = {}
				for _, c in appState.constraints:get() do
					local chainCFrames = {}
					if c.chain and type(c.chain) == "table" and c.chain.parts then
						chainCFrames = RigService.readChainWorldCFrames(c.chain, c.effector, groundCF)
					end

					table.insert(constraintsList, {
						effector = c.effector,
						time = c.time,
						chain_world_cframes = chainCFrames,
					})
				end

				-- Compute duration from prompt segments (authoritative)
				local totalDuration = 0
				for _, seg in promptSegments do
					totalDuration += (seg.end_time - seg.start_time)
				end
				appState.duration:set(totalDuration)

				local resp = BackendService.generate({
					prompts = promptSegments,
					constraints = constraintsList,
					duration = totalDuration,
					looped = appState.looped:get(),
					seed = if appState.seed:get() > 0 then appState.seed:get() else nil,
				})

				appState.currentJobId:set(resp.job_id)

				-- Poll for completion
				while true do
					task.wait(Constants.POLL_INTERVAL)
					local status = BackendService.getStatus(resp.job_id)
					appState.generationProgress:set(status.progress)
					appState.generationMessage:set(status.message)

					if status.status == "completed" then
						-- Fetch the full animation data
						appState.generationMessage:set("Loading animation...")
						local resultData = BackendService.getResult(resp.job_id)

						-- Build CurveAnimation from the JSON data
						local curveAnim = AnimationBuilder.build(resultData.animation)

						if playbackSvc then
							playbackSvc:destroy()
						end
						playbackSvc = PlaybackService.new(rig)
						playbackSvc.TimeChanged:Connect(function(t)
							appState.playbackTime:set(t)
						end)
						playbackSvc.StateChanged:Connect(function(s)
							appState.playbackState:set(s)
						end)

						local loaded = playbackSvc:loadFromCurveAnimation(curveAnim)
						if loaded then
							appState.generationStatus:set("completed")
							appState.duration:set(playbackSvc:getDuration())
							-- Show the seed that was used (don't overwrite the input,
							-- so seed=0 keeps generating fresh random results)
							if resultData.seed then
								appState.generationMessage:set("Done (seed " .. tostring(resultData.seed) .. ")")
							end
						else
							appState.generationStatus:set("failed")
							appState.generationMessage:set("Failed to load animation")
						end
						break
					elseif status.status == "failed" then
						appState.generationStatus:set("failed")
						appState.generationMessage:set(status.error or "Generation failed")
						break
					end
				end
			end)

			if not ok then
				warn("[RoMotion] Generation error:", tostring(result))
				appState.generationStatus:set("failed")
				appState.generationMessage:set("Error (see Output)")
			end
		end)
	end)

	autoConstrainBtn.MouseButton1Click:Connect(function()
		local rig = appState.rig:get()
		if not rig or not playbackSvc then
			warn("[RoMotion] Select a rig and generate/import an animation first")
			return
		end

		task.spawn(function()
			local fps = Constants.FPS
			local duration = playbackSvc:getDuration()
			local nFrames = math.max(3, math.floor(duration * fps))
			local effectors = { "LeftHand", "RightHand", "LeftFoot", "RightFoot" }
			local hrpInv = rig.rootPart.CFrame:Inverse()

			appState.generationMessage:set("Analyzing motion...")

			-- Sample each effector's HRP-local position at every frame
			local samples: { [string]: { Vector3 } } = {}
			for _, e in effectors do samples[e] = {} end

			for f = 0, nFrames - 1 do
				local t = f / fps
				playbackSvc:seekTo(t)
				playbackSvc:ensureStepped()
				for _, e in effectors do
					local part = RigService.getEffectorPart(rig, e)
					if part then
						-- HRP-local so locomotion doesn't dominate the velocity
						samples[e][f + 1] = hrpInv * part.Position
					else
						samples[e][f + 1] = Vector3.zero
					end
				end
			end

			-- Detect extrema per effector, collect (effector, time) pairs
			local picks: { { effector: string, time: number } } = {}
			for _, e in effectors do
				local frames = RigService.detectVelocityExtrema(samples[e], 8)
				for _, f in frames do
					table.insert(picks, { effector = e, time = (f - 1) / fps })
				end
			end

			-- Place a constraint at each detected pose
			for _, p in picks do
				placeConstraint(p.effector, p.time)
				task.wait()
			end
			appState.generationMessage:set(string.format("Auto-placed %d constraints", #picks))
		end)
	end)

	-- Clear all constraints (and their workspace chain models)
	clearBtn.MouseButton1Click:Connect(function()
		for _, c in appState.constraints:get() do
			if c.chain and type(c.chain) == "table" and c.chain.model then
				RigService.destroyChain(c.chain)
			end
		end
		appState.constraints:set({})
	end)

	-- Import: prompt for an asset ID, load it onto the rig for scrub/constrain
	importBtn.MouseButton1Click:Connect(function()
		local rig = appState.rig:get()
		if not rig then
			warn("[RoMotion] Select a rig before importing")
			return
		end

		-- Small centered input dialog
		local dialog = Instance.new("Frame")
		dialog.Size = UDim2.new(0, 260, 0, 70)
		dialog.Position = UDim2.new(0.5, -130, 0.5, -35)
		dialog.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Titlebar)
		dialog.BorderColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Border)
		dialog.BorderSizePixel = 1
		dialog.ZIndex = 200
		dialog.Parent = mainFrame

		local prompt = Instance.new("TextLabel")
		prompt.Size = UDim2.new(1, -16, 0, 20)
		prompt.Position = UDim2.new(0, 8, 0, 6)
		prompt.BackgroundTransparency = 1
		prompt.Text = "Animation Asset ID:"
		prompt.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainText)
		prompt.TextSize = 12
		prompt.Font = Enum.Font.SourceSans
		prompt.TextXAlignment = Enum.TextXAlignment.Left
		prompt.ZIndex = 201
		prompt.Parent = dialog

		local input = Instance.new("TextBox")
		input.Size = UDim2.new(1, -16, 0, 24)
		input.Position = UDim2.new(0, 8, 0, 30)
		input.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.InputFieldBackground)
		input.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainText)
		input.PlaceholderText = "e.g. 507771019"
		input.Text = ""
		input.TextSize = 13
		input.Font = Enum.Font.Code
		input.ClearTextOnFocus = false
		input.ZIndex = 201
		input.Parent = dialog
		input:CaptureFocus()

		input.FocusLost:Connect(function(enterPressed)
			if enterPressed then
				local id = tonumber(input.Text)
				if id and id > 0 then
					task.spawn(function()
						appState.generationMessage:set("Loading asset " .. id .. "...")
						if playbackSvc then
							local loaded = playbackSvc:loadFromAssetId(id)
							if loaded then
								local dur = playbackSvc:getDuration()
								appState.duration:set(dur)
								-- Resize the prompt timeline to match the imported clip
								local prompts_val = appState.prompts:get()
								if #prompts_val == 0 then
									appState.prompts:set({
										{ text = "imported motion", startTime = 0, endTime = dur },
									})
								else
									-- Stretch last block so the timeline ends at `dur`
									local updated = table.clone(prompts_val)
									updated[#updated] = table.clone(updated[#updated])
									updated[#updated].endTime = math.max(
										dur, updated[#updated].startTime + 0.3
									)
									appState.prompts:set(updated)
								end
								appState.generationMessage:set("Imported " .. id .. string.format(" (%.1fs)", dur))
							else
								appState.generationMessage:set("Import failed")
							end
						end
					end)
				end
			end
			dialog:Destroy()
		end)
	end)

	-- Timeline scrub interaction (click + drag)
	local isScrubbing = false

	local function scrubToInput(input: InputObject)
		local relX = input.Position.X - timelineFrame.AbsolutePosition.X
		local time = math.clamp(pxToTime(relX), 0, getLastFrameTime())
		appState.playbackTime:set(time)
		if playbackSvc then
			playbackSvc:seekTo(time)
		end
	end

	-- Drag handling
	local function endDrag()
		draggingConstraint = nil
		isScrubbing = false
	end

	-- Movement handler: scrubbing only. Constraint/prompt drags are owned by
	-- their per-track InputChanged handlers (correct gutter math).
	local function onDragMove(input: InputObject)
		if input.UserInputType ~= Enum.UserInputType.MouseMovement then return end
		if isScrubbing then
			scrubToInput(input)
		end
	end

	-- Ensure all timeline frames receive input events
	timelineFrame.Active = true
	promptTrack.Active = true
	constraintArea.Active = true
	mainFrame.Active = true

	-- Listen on ALL relevant areas for movement
	ruler.InputChanged:Connect(onDragMove)
	timelineFrame.InputChanged:Connect(onDragMove)
	promptTrack.InputChanged:Connect(onDragMove)
	constraintArea.InputChanged:Connect(onDragMove)

	-- Release: listen on multiple elements for redundancy
	local function onInputEnded(input: InputObject)
		if input.UserInputType == Enum.UserInputType.MouseButton1 then
			endDrag()
		end
	end
	mainFrame.InputEnded:Connect(onInputEnded)
	timelineFrame.InputEnded:Connect(onInputEnded)
	promptTrack.InputEnded:Connect(onInputEnded)
	constraintArea.InputEnded:Connect(onInputEnded)

	-- Timeline scrub — only from the ruler (top bar with time labels)
	ruler.Active = true
	ruler.InputBegan:Connect(function(input)
		if input.UserInputType == Enum.UserInputType.MouseButton1 then
			isScrubbing = true
			scrubToInput(input)
		end
	end)

	-- Zoom only (scrub movement handled by drag overlay)
	timelineFrame.InputChanged:Connect(function(input)
		if input.UserInputType == Enum.UserInputType.MouseWheel then
			local current = appState.pixelsPerSecond:get()
			local factor = if input.Position.Z > 0 then 1.2 else 1 / 1.2
			appState.pixelsPerSecond:set(math.clamp(current * factor, 20, 500))
		end
	end)


	-- Prompt editing popup (hidden until double-click)
	local editPopup = Instance.new("Frame")
	editPopup.Name = "EditPopup"
	editPopup.Size = UDim2.new(0, 250, 0, 30)
	editPopup.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.InputFieldBackground)
	editPopup.BorderColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Border)
	editPopup.BorderSizePixel = 1
	editPopup.Visible = false
	editPopup.ZIndex = 20
	editPopup.Parent = timelineFrame
	local editCorner = Instance.new("UICorner")
	editCorner.CornerRadius = UDim.new(0, 4)
	editCorner.Parent = editPopup

	local editBox = Instance.new("TextBox")
	editBox.Size = UDim2.new(1, -8, 1, -4)
	editBox.Position = UDim2.new(0, 4, 0, 2)
	editBox.BackgroundTransparency = 1
	editBox.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainText)
	editBox.TextSize = 12
	editBox.Font = Enum.Font.SourceSans
	editBox.ClearTextOnFocus = false
	editBox.TextXAlignment = Enum.TextXAlignment.Left
	editBox.ZIndex = 21
	editBox.Parent = editPopup

	local editingIndex: number? = nil

	editBox.FocusLost:Connect(function(enterPressed)
		if editingIndex and enterPressed then
			local prompts_val = appState.prompts:get()
			if editingIndex <= #prompts_val then
				local updated = table.clone(prompts_val)
				updated[editingIndex] = table.clone(updated[editingIndex])
				updated[editingIndex].text = editBox.Text
				appState.prompts:set(updated)
			end
		end
		editPopup.Visible = false
		editingIndex = nil
	end)

	-- Contiguous prompt blocks: no gaps, no overlaps. Drag boundaries to resize.
	local promptElements: { Instance } = {}
	local lastClickTime = 0
	local lastClickBlock = 0
	local draggingBoundary: number? = nil -- index of left block at this boundary

	local function renderPromptBlocks()
		if draggingBoundary then return end
		for _, el in promptElements do
			el:Destroy()
		end
		table.clear(promptElements)

		local prompts_val = appState.prompts:get()
		local pps = appState.pixelsPerSecond:get()
		local scroll = appState.scrollOffset:get()

		-- Render blocks
		local x = 0
		for i, prompt in prompts_val do
			local dur = prompt.endTime - prompt.startTime
			local startPx = timeToPx(x)
			local w = dur * pps

			local block = Instance.new("Frame")
			block.Name = "Prompt_" .. tostring(i)
			block.Size = UDim2.new(0, math.floor(w), 1, -4)
			block.Position = UDim2.new(0, math.floor(startPx), 0, 2)
			block.BackgroundColor3 = Constants.PROMPT_COLORS[((i - 1) % #Constants.PROMPT_COLORS) + 1]
			block.BackgroundTransparency = 0.3
			block.BorderSizePixel = 0
			block.Active = true
			block.Parent = promptTrack

			local rc = Instance.new("UICorner")
			rc.CornerRadius = UDim.new(0, 4)
			rc.Parent = block

			local lbl = Instance.new("TextLabel")
			lbl.Size = UDim2.fromScale(1, 1)
			lbl.BackgroundTransparency = 1
			lbl.Text = prompt.text
			lbl.TextColor3 = Color3.new(1, 1, 1)
			lbl.TextSize = 11
			lbl.Font = Enum.Font.SourceSans
			lbl.TextTruncate = Enum.TextTruncate.AtEnd
			lbl.TextXAlignment = Enum.TextXAlignment.Center
			lbl.Parent = block

			-- Double-click to edit text, right-click to delete
			local idx = i
			block.InputBegan:Connect(function(input)
				if input.UserInputType == Enum.UserInputType.MouseButton1 then
					local now = os.clock()
					if lastClickBlock == idx and (now - lastClickTime) < 0.4 then
						editBox.Text = prompt.text
						editPopup.Position = UDim2.new(0, math.floor(startPx), 0, Constants.RULER_HEIGHT - 2)
						editPopup.Size = UDim2.new(0, math.max(200, math.floor(w)), 0, 30)
						editPopup.Visible = true
						editingIndex = idx
						editBox:CaptureFocus()
					end
					lastClickTime = now
					lastClickBlock = idx
				elseif input.UserInputType == Enum.UserInputType.MouseButton2 then
					if #prompts_val > 1 then
						local updated = table.clone(prompts_val)
						table.remove(updated, idx)
						-- Recalculate times to stay contiguous
						local t = 0
						for j, p in updated do
							local d = p.endTime - p.startTime
							updated[j] = table.clone(p)
							updated[j].startTime = t
							updated[j].endTime = t + d
							t += d
						end
						appState.prompts:set(updated)
						appState.duration:set(t)
					end
				end
			end)

			table.insert(promptElements, block)
			x += dur
		end

		-- Render boundaries between blocks
		x = 0
		for i = 1, #prompts_val do
			x += prompts_val[i].endTime - prompts_val[i].startTime
			-- Boundary after each block (internal dividers + last edge)
			local isLast = (i == #prompts_val)
			local boundary = Instance.new("Frame")
			boundary.Name = "Boundary_" .. tostring(i)
			boundary.Size = UDim2.new(0, 8, 1, 0)
			boundary.Position = UDim2.new(0, math.floor(timeToPx(x)) - 4, 0, 0)
			boundary.BackgroundColor3 = if isLast then Color3.fromRGB(255, 120, 120) else Color3.new(1, 1, 1)
			boundary.BackgroundTransparency = 0.5
			boundary.BorderSizePixel = 0
			boundary.Active = true
			boundary.ZIndex = 5
			boundary.Parent = promptTrack

			local bIdx = i
			boundary.InputBegan:Connect(function(input)
				if input.UserInputType == Enum.UserInputType.MouseButton1 then
					draggingBoundary = bIdx
					boundary.BackgroundTransparency = 0
				end
			end)

			table.insert(promptElements, boundary)
		end
	end

	-- Drag movement for prompt boundaries
	promptTrack.InputChanged:Connect(function(input)
		if input.UserInputType ~= Enum.UserInputType.MouseMovement then return end
		if not draggingBoundary then return end

		local relX = input.Position.X - promptTrack.AbsolutePosition.X
		local pps = appState.pixelsPerSecond:get()
		local time = pxToTime(relX)

		local prompts_val = appState.prompts:get()
		local isLast = (draggingBoundary == #prompts_val)

		if isLast then
			-- Drag last edge: extend/shrink last block
			local beforeStart = prompts_val[#prompts_val].startTime
			local newDur = math.max(0.3, time - beforeStart)
			local lastBlock = promptElements[#prompts_val]
			local lastBoundary = promptElements[#promptElements]
			if lastBlock and lastBlock:IsA("Frame") then
				lastBlock.Size = UDim2.new(0, math.floor(newDur * pps), 1, -4)
			end
			if lastBoundary and lastBoundary:IsA("Frame") then
				lastBoundary.Position = UDim2.new(0, math.floor(timeToPx(beforeStart + newDur)) - 4, 0, 0)
			end
		else
			-- Internal boundary: resize blocks[idx] and blocks[idx+1]
			local beforeStart = prompts_val[draggingBoundary].startTime
			local afterEnd = prompts_val[draggingBoundary + 1].endTime
			time = math.clamp(time, beforeStart + 0.3, afterEnd - 0.3)

			local newDur1 = time - beforeStart
			local newDur2 = afterEnd - time

			local block1 = promptElements[draggingBoundary]
			local block2 = promptElements[draggingBoundary + 1]
			local boundary = promptElements[#appState.prompts:get() + draggingBoundary]
			if block1 and block1:IsA("Frame") then
				block1.Size = UDim2.new(0, math.floor(newDur1 * pps), 1, -4)
			end
			if block2 and block2:IsA("Frame") then
				block2.Position = UDim2.new(0, math.floor(timeToPx(time)), 0, 2)
				block2.Size = UDim2.new(0, math.floor(newDur2 * pps), 1, -4)
			end
			if boundary and boundary:IsA("Frame") then
				boundary.Position = UDim2.new(0, math.floor(timeToPx(time)) - 4, 0, 0)
			end
		end
	end)

	-- Release: update state and re-render
	promptTrack.InputEnded:Connect(function(input)
		if input.UserInputType == Enum.UserInputType.MouseButton1 and draggingBoundary then
			local relX = input.Position.X - promptTrack.AbsolutePosition.X
			local time = snapToFrame(pxToTime(relX))

			local prompts_val = appState.prompts:get()
			local updated = table.clone(prompts_val)
			local isLast = (draggingBoundary == #prompts_val)

			if isLast then
				local beforeStart = updated[#updated].startTime
				local newDur = math.max(0.3, time - beforeStart)
				updated[#updated] = table.clone(updated[#updated])
				updated[#updated].endTime = beforeStart + newDur
				appState.duration:set(beforeStart + newDur)
			else
				local beforeStart = updated[draggingBoundary].startTime
				local afterEnd = updated[draggingBoundary + 1].endTime
				time = math.clamp(time, beforeStart + 0.3, afterEnd - 0.3)
				updated[draggingBoundary] = table.clone(updated[draggingBoundary])
				updated[draggingBoundary].endTime = time
				updated[draggingBoundary + 1] = table.clone(updated[draggingBoundary + 1])
				updated[draggingBoundary + 1].startTime = time
			end

			draggingBoundary = nil
			appState.prompts:set(updated)
		end
	end)

	-- Add block button: click to append a new 2s prompt segment
	local addBlockBtn = Instance.new("TextButton")
	addBlockBtn.Name = "AddBlock"
	addBlockBtn.Size = UDim2.new(0, 24, 0, 24)
	addBlockBtn.Position = UDim2.new(0, 0, 0, 0) -- updated by renderPromptBlocks
	addBlockBtn.BackgroundColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.Button)
	addBlockBtn.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.ButtonText)
	addBlockBtn.Text = "+"
	addBlockBtn.TextSize = 16
	addBlockBtn.Font = Enum.Font.SourceSansBold
	addBlockBtn.ZIndex = 5
	addBlockBtn.Parent = promptTrack
	local addCorner = Instance.new("UICorner")
	addCorner.CornerRadius = UDim.new(0, 4)
	addCorner.Parent = addBlockBtn

	addBlockBtn.MouseButton1Click:Connect(function()
		local prompts_val = appState.prompts:get()
		local updated = table.clone(prompts_val)
		local lastEnd = if #updated > 0 then updated[#updated].endTime else 0
		table.insert(updated, {
			text = "describe motion",
			startTime = lastEnd,
			endTime = lastEnd + 2.0,
		})
		appState.duration:set(lastEnd + 2.0)
		appState.prompts:set(updated)
	end)

	-- Position the + button after rendering
	local origRender = renderPromptBlocks
	renderPromptBlocks = function()
		origRender()
		local prompts_val = appState.prompts:get()
		local totalDur = if #prompts_val > 0 then prompts_val[#prompts_val].endTime else 0
		addBlockBtn.Position = UDim2.new(0, math.floor(timeToPx(totalDur)) + 8, 0.5, -12)
	end

	appState.prompts:subscribe(renderPromptBlocks)
	appState.pixelsPerSecond:subscribe(renderPromptBlocks)
	appState.scrollOffset:subscribe(renderPromptBlocks)

	-- Constraint diamond rendering on timeline (draggable to change time)
	renderConstraints = function()
		if draggingConstraint then return end -- don't rebuild while dragging
		for _, d in constraintDiamonds do
			d:Destroy()
		end
		table.clear(constraintDiamonds)

		local constraints_val = appState.constraints:get()
		local pps = appState.pixelsPerSecond:get()
		local scroll = appState.scrollOffset:get()
		local maxT = getMaxTime()

		-- Per-effector ordinal (1st, 2nd, ... of each effector by time order)
		local effCount: { [string]: number } = {}

		for i, constraint in constraints_val do
			local effTrack = constraintArea:FindFirstChild(constraint.effector)
			if not effTrack then continue end

			effCount[constraint.effector] = (effCount[constraint.effector] or 0) + 1
			local ordinal = effCount[constraint.effector]
			local timeFrac = math.clamp(constraint.time / maxT, 0, 1)
			local color = constraintColor(constraint.effector, timeFrac)

			local px = timeToPx(constraint.time)
			local diamond = Instance.new("Frame")
			diamond.Name = "C_" .. constraint.effector .. "_" .. tostring(i)
			diamond.Size = UDim2.new(0, 16, 0, 16)
			diamond.Position = UDim2.new(0, math.floor(px) - 8, 0.5, -8)
			diamond.BackgroundColor3 = color
			diamond.Rotation = 0
			diamond.Active = true
			diamond.ZIndex = 5
			diamond.BorderSizePixel = if i == selectedConstraint then 2 else 0
			diamond.BorderColor3 = Color3.new(1, 1, 1)
			diamond.Parent = effTrack
			local diamondCorner = Instance.new("UICorner")
			diamondCorner.CornerRadius = UDim.new(0, 3)
			diamondCorner.Parent = diamond

			-- Number label on the diamond
			local numLbl = Instance.new("TextLabel")
			numLbl.Size = UDim2.fromScale(1, 1)
			numLbl.BackgroundTransparency = 1
			numLbl.Text = tostring(ordinal)
			numLbl.TextColor3 = Color3.new(0, 0, 0)
			numLbl.TextSize = 11
			numLbl.Font = Enum.Font.SourceSansBold
			numLbl.ZIndex = 6
			numLbl.Parent = diamond

			-- Update the matching world chain marker (number + hue)
			if constraint.chain and type(constraint.chain) == "table" and constraint.chain.parts then
				RigService.labelChain(constraint.chain, constraint.effector, ordinal, color)
			end

			local idx = i

			-- Left-click: select (UI + DataModel) and start drag
			diamond.InputBegan:Connect(function(input)
				if input.UserInputType == Enum.UserInputType.MouseButton1 then
					draggingConstraint = idx
					selectedConstraint = idx
					-- Select the chain model in the explorer/viewport
					local c = appState.constraints:get()[idx]
					if c and c.chain and type(c.chain) == "table" and c.chain.model then
						Selection:Set({ c.chain.model })
					end
					-- Re-highlight without full rebuild (we're mid-drag)
					for di, d in constraintDiamonds do
						d.BorderSizePixel = if di == idx then 2 else 0
						d.BorderColor3 = Color3.new(1, 1, 1)
					end
				end
			end)

			-- Right-click to delete constraint (and its chain)
			diamond.InputBegan:Connect(function(input)
				if input.UserInputType == Enum.UserInputType.MouseButton2 then
					local updated = table.clone(appState.constraints:get())
					local removed = table.remove(updated, idx)
					if removed then
						if removed.chain and type(removed.chain) == "table" and removed.chain.model then
							RigService.destroyChain(removed.chain)
						end
					end
					appState.constraints:set(updated)
				end
			end)

			table.insert(constraintDiamonds, diamond)
		end
	end

	-- Handle constraint diamond dragging


	appState.constraints:subscribe(renderConstraints)
	appState.pixelsPerSecond:subscribe(renderConstraints)
	appState.scrollOffset:subscribe(renderConstraints)

	-- Time ruler ticks rendering
	local TimelineLayout = require(src.Utils.TimelineLayout)
	local rulerTicks: { Instance } = {}

	local function renderRuler()
		for _, t in rulerTicks do
			t:Destroy()
		end
		table.clear(rulerTicks)

		local pps = appState.pixelsPerSecond:get()
		local scroll = appState.scrollOffset:get()
		local width = ruler.AbsoluteSize.X
		local interval = TimelineLayout.getTickInterval(pps)
		local startTime = math.floor(scroll / interval) * interval
		local endTime = scroll + width / pps

		local time = startTime
		while time <= endTime do
			local px = timeToPx(time)
			if px >= GUTTER then
				local tick = Instance.new("TextLabel")
				tick.Size = UDim2.new(0, 40, 1, 0)
				tick.Position = UDim2.new(0, math.floor(px), 0, 0)
				tick.BackgroundTransparency = 1
				tick.Text = string.format("%.1fs", time)
				tick.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.DimmedText)
				tick.TextSize = 10
				tick.Font = Enum.Font.Code
				tick.TextXAlignment = Enum.TextXAlignment.Left
				tick.Parent = ruler
				table.insert(rulerTicks, tick)
			end
			time += interval
		end
	end

	appState.pixelsPerSecond:subscribe(renderRuler)
	appState.scrollOffset:subscribe(renderRuler)
	task.defer(renderRuler)

	-- Initial render (state may already be populated before subscribers connected)
	renderPromptBlocks()
	renderConstraints()
	renderRuler()

	return mainFrame
end

-- ════════════════════════════════════════════════════════════════════
-- Rig Selection
-- ════════════════════════════════════════════════════════════════════

local function onSelectionChanged()
	local selected = Selection:Get()
	if #selected == 1 then
		local rig = RigService.findRig(selected[1])
		if rig then
			-- Don't reset if it's the same rig we already have
			local currentRig = appState.rig:get()
			if currentRig and currentRig.model == rig.model then
				return
			end
			appState.rig:set(rig)
			-- Compute rest-pose ground Y (rig is at rest when first selected)
			local lf = rig.model:FindFirstChild("LeftFoot", true) :: BasePart?
			local rf = rig.model:FindFirstChild("RightFoot", true) :: BasePart?
			if lf and rf then
				rigRestGroundY = math.min(lf.Position.Y - lf.Size.Y/2, rf.Position.Y - rf.Size.Y/2)
			elseif lf then
				rigRestGroundY = lf.Position.Y - lf.Size.Y/2
			end
			if playbackSvc then
				playbackSvc:destroy()
			end
			playbackSvc = PlaybackService.new(rig)
			playbackSvc.TimeChanged:Connect(function(t)
				appState.playbackTime:set(t)
			end)
			playbackSvc.StateChanged:Connect(function(s)
				appState.playbackState:set(s)
			end)
		end
	end
end

-- ════════════════════════════════════════════════════════════════════
-- Server Health Check
-- ════════════════════════════════════════════════════════════════════

local function checkServer()
	while true do
		local connected = BackendService.healthCheck()
		appState.serverConnected:set(connected)
		task.wait(5)
	end
end

-- ════════════════════════════════════════════════════════════════════
-- Plugin Lifecycle
-- ════════════════════════════════════════════════════════════════════

local mainFrame: Frame? = nil
local selectionConn: RBXScriptConnection? = nil

local function onWidgetEnabled()
	if widget.Enabled then
		if not mainFrame then
			mainFrame = buildUI()
		end
		selectionConn = Selection.SelectionChanged:Connect(onSelectionChanged)
		onSelectionChanged() -- pick up current selection
		task.spawn(checkServer)
	else
		if selectionConn then
			selectionConn:Disconnect()
			selectionConn = nil
		end
	end
end

toggleButton.Click:Connect(function()
	widget.Enabled = not widget.Enabled
	toggleButton:SetActive(widget.Enabled)
end)

widget:GetPropertyChangedSignal("Enabled"):Connect(function()
	toggleButton:SetActive(widget.Enabled)
	onWidgetEnabled()
end)

-- Seed a default prompt BEFORE building UI so the initial render shows it
if #appState.prompts:get() == 0 then
	appState.prompts:set({
		{
			text = "a person waving",
			startTime = 0,
			endTime = appState.duration:get(),
		},
	})
end

-- If the widget was left enabled across a place reload, the Enabled-changed
-- signal won't fire — build the UI now so it isn't blank.
if widget.Enabled then
	onWidgetEnabled()
end

print("[RoMotion] Plugin loaded")
