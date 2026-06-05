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

-- ════════════════════════════════════════════════════════════════════
-- UI Construction
-- ════════════════════════════════════════════════════════════════════

local function createThemeColor(element: string): Color3
	local theme = settings().Studio.Theme
	return theme:GetColor(Enum.StudioStyleGuideColor.MainBackground)
end

local function buildUI()
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
	topLayout.Padding = UDim.new(0, 8)
	topLayout.Parent = topBar

	local topPad = Instance.new("UIPadding")
	topPad.PaddingLeft = UDim.new(0, 8)
	topPad.Parent = topBar

	local rigLabel = Instance.new("TextLabel")
	rigLabel.Name = "RigLabel"
	rigLabel.Size = UDim2.new(0, 200, 1, 0)
	rigLabel.BackgroundTransparency = 1
	rigLabel.Text = "No rig selected"
	rigLabel.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainText)
	rigLabel.TextSize = 13
	rigLabel.Font = Enum.Font.SourceSans
	rigLabel.TextXAlignment = Enum.TextXAlignment.Left
	rigLabel.Parent = topBar

	local serverDot = Instance.new("Frame")
	serverDot.Name = "ServerDot"
	serverDot.Size = UDim2.new(0, 8, 0, 8)
	serverDot.BackgroundColor3 = Color3.fromRGB(244, 67, 54) -- red = disconnected
	serverDot.Parent = topBar
	local corner = Instance.new("UICorner")
	corner.CornerRadius = UDim.new(1, 0)
	corner.Parent = serverDot

	local serverLabel = Instance.new("TextLabel")
	serverLabel.Name = "ServerLabel"
	serverLabel.Size = UDim2.new(0, 80, 1, 0)
	serverLabel.BackgroundTransparency = 1
	serverLabel.Text = "Server"
	serverLabel.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.DimmedText)
	serverLabel.TextSize = 11
	serverLabel.Font = Enum.Font.SourceSans
	serverLabel.TextXAlignment = Enum.TextXAlignment.Left
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
		track.Parent = constraintArea

		local label = Instance.new("TextLabel")
		label.Size = UDim2.new(0, 70, 1, 0)
		label.BackgroundTransparency = 1
		label.Text = effName
		label.TextColor3 = Constants.EFFECTOR_COLORS[effName] or Color3.new(1, 1, 1)
		label.TextSize = 10
		label.Font = Enum.Font.SourceSans
		label.TextXAlignment = Enum.TextXAlignment.Left
		label.Parent = track

		local pad = Instance.new("UIPadding")
		pad.PaddingLeft = UDim.new(0, 4)
		pad.Parent = label
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
		local px = (t - appState.scrollOffset:get()) * appState.pixelsPerSecond:get()
		playhead.Position = UDim2.new(0, math.floor(px), 0, 0)
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

				local constraintsList = {}
				for _, c in appState.constraints:get() do
					table.insert(constraintsList, {
						effector = c.effector,
						time = c.time,
						position = { c.cframe.Position.X, c.cframe.Position.Y, c.cframe.Position.Z },
						rotation = nil, -- TODO: extract from CFrame
					})
				end

				local resp = BackendService.generate({
					prompts = promptSegments,
					constraints = constraintsList,
					duration = appState.duration:get(),
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
							if resultData.seed then
								appState.seed:set(resultData.seed)
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
				appState.generationStatus:set("failed")
				appState.generationMessage:set(tostring(result))
			end
		end)
	end)

	autoConstrainBtn.MouseButton1Click:Connect(function()
		local jobId = appState.currentJobId:get()
		if not jobId then
			warn("[RoMotion] No generation to analyze")
			return
		end

		task.spawn(function()
			local ok, result = pcall(function()
				return BackendService.autoConstraints({ job_id = jobId })
			end)
			if ok and result then
				local newConstraints: { DataModelService.ConstraintData } = {}
				for _, c in result.constraints do
					table.insert(newConstraints, {
						effector = c.effector,
						time = c.time,
						cframe = CFrame.new(c.position[1], c.position[2], c.position[3]),
					})
				end
				appState.constraints:set(newConstraints)
			else
				warn("[RoMotion] Auto-constraint failed:", result)
			end
		end)
	end)

	-- Timeline scrub interaction
	timelineFrame.InputBegan:Connect(function(input)
		if input.UserInputType == Enum.UserInputType.MouseButton1 then
			local relX = input.Position.X - timelineFrame.AbsolutePosition.X
			local time = relX / appState.pixelsPerSecond:get() + appState.scrollOffset:get()
			time = math.max(0, math.min(time, appState.duration:get()))
			appState.playbackTime:set(time)
			if playbackSvc then
				playbackSvc:seekTo(time)
			end
		end
	end)

	-- Zoom with scroll wheel
	timelineFrame.InputChanged:Connect(function(input)
		if input.UserInputType == Enum.UserInputType.MouseWheel then
			local current = appState.pixelsPerSecond:get()
			local factor = if input.Position.Z > 0 then 1.2 else 1 / 1.2
			appState.pixelsPerSecond:set(math.clamp(current * factor, 20, 500))
		end
	end)

	-- Prompt block rendering
	local promptBlocks: { Frame } = {}

	local function renderPromptBlocks()
		for _, block in promptBlocks do
			block:Destroy()
		end
		table.clear(promptBlocks)

		local prompts_val = appState.prompts:get()
		local pps = appState.pixelsPerSecond:get()
		local scroll = appState.scrollOffset:get()

		for i, prompt in prompts_val do
			local startPx = (prompt.startTime - scroll) * pps
			local endPx = (prompt.endTime - scroll) * pps
			local width = endPx - startPx

			if width < 2 then continue end

			local block = Instance.new("Frame")
			block.Name = "Prompt_" .. tostring(i)
			block.Size = UDim2.new(0, math.floor(width), 1, -4)
			block.Position = UDim2.new(0, math.floor(startPx), 0, 2)
			block.BackgroundColor3 = Constants.PROMPT_COLORS[((i - 1) % #Constants.PROMPT_COLORS) + 1]
			block.BackgroundTransparency = 0.3
			block.BorderSizePixel = 0
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
			lbl.Parent = block

			local lblPad = Instance.new("UIPadding")
			lblPad.PaddingLeft = UDim.new(0, 4)
			lblPad.PaddingRight = UDim.new(0, 4)
			lblPad.Parent = lbl

			table.insert(promptBlocks, block)
		end
	end

	appState.prompts:subscribe(renderPromptBlocks)
	appState.pixelsPerSecond:subscribe(renderPromptBlocks)
	appState.scrollOffset:subscribe(renderPromptBlocks)

	-- Constraint diamond rendering
	local constraintDiamonds: { Frame } = {}

	local function renderConstraints()
		for _, d in constraintDiamonds do
			d:Destroy()
		end
		table.clear(constraintDiamonds)

		local constraints_val = appState.constraints:get()
		local pps = appState.pixelsPerSecond:get()
		local scroll = appState.scrollOffset:get()

		for _, constraint in constraints_val do
			local effTrack = constraintArea:FindFirstChild(constraint.effector)
			if not effTrack then continue end

			local px = (constraint.time - scroll) * pps
			local diamond = Instance.new("Frame")
			diamond.Name = "C_" .. constraint.effector
			diamond.Size = UDim2.new(0, 10, 0, 10)
			diamond.Position = UDim2.new(0, math.floor(px) - 5 + 70, 0.5, -5)
			diamond.BackgroundColor3 = Constants.EFFECTOR_COLORS[constraint.effector] or Color3.new(1, 1, 1)
			diamond.Rotation = 45
			diamond.BorderSizePixel = 0
			diamond.Parent = effTrack

			table.insert(constraintDiamonds, diamond)
		end
	end

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
			local px = (time - scroll) * pps
			if px >= 0 then
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

	-- Double-click prompt track to add a prompt
	promptTrack.InputBegan:Connect(function(input)
		if input.UserInputType == Enum.UserInputType.MouseButton1 then
			-- Check if double-click (simplified: just add on click for now)
			local relX = input.Position.X - promptTrack.AbsolutePosition.X
			local time = relX / appState.pixelsPerSecond:get() + appState.scrollOffset:get()
			time = math.max(0, time)

			-- Find gap at this time
			local prompts_val = appState.prompts:get()
			local inExisting = false
			for _, p in prompts_val do
				if time >= p.startTime and time < p.endTime then
					inExisting = true
					break
				end
			end

			if not inExisting then
				local endTime = math.min(time + 2.0, appState.duration:get())
				-- Check for next block overlap
				for _, p in prompts_val do
					if p.startTime > time and p.startTime < endTime then
						endTime = p.startTime
					end
				end
				if endTime - time < 0.5 then return end

				local newPrompts = table.clone(prompts_val)
				table.insert(newPrompts, {
					text = "describe motion",
					startTime = time,
					endTime = endTime,
				})
				table.sort(newPrompts, function(a, b) return a.startTime < b.startTime end)
				appState.prompts:set(newPrompts)
			end
		end
	end)

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
			appState.rig:set(rig)
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

-- Start with a default prompt if none
task.defer(function()
	if #appState.prompts:get() == 0 then
		appState.prompts:set({
			{
				text = "a person walking forwards",
				startTime = 0,
				endTime = Constants.DEFAULT_DURATION,
			},
		})
	end
end)

print("[RoMotion] Plugin loaded")
