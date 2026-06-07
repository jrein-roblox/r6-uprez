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
	rigLabel.Size = UDim2.new(0, 150, 1, 0)
	rigLabel.BackgroundTransparency = 1
	rigLabel.Text = "No rig selected"
	rigLabel.TextColor3 = settings().Studio.Theme:GetColor(Enum.StudioStyleGuideColor.MainText)
	rigLabel.TextSize = 13
	rigLabel.Font = Enum.Font.SourceSans
	rigLabel.TextXAlignment = Enum.TextXAlignment.Left
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

	local serverDot = Instance.new("Frame")
	serverDot.Name = "ServerDot"
	serverDot.Size = UDim2.new(0, 8, 0, 8)
	serverDot.BackgroundColor3 = Color3.fromRGB(244, 67, 54)
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

	-- Forward declarations for constraint drag system
	local constraintDiamonds: { Frame } = {}
	local draggingConstraint: number? = nil
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

		-- Click to place a constraint at the current playback time.
		-- Clones the full effector chain as a poseable ghost.
		-- Label click to add constraint (not the whole track)
		local eName = effName
		label.InputBegan:Connect(function(input)
			if input.UserInputType ~= Enum.UserInputType.MouseButton1 then return end
			local rig = appState.rig:get()
			if not rig then return end

			-- Ensure animation is stepped at current time so transforms are populated
			if playbackSvc then
				playbackSvc:ensureStepped()
			end

			local time = appState.playbackTime:get()
			local effPart = RigService.getEffectorPart(rig, eName)
			if not effPart then return end

			local color = Constants.EFFECTOR_COLORS[eName] or Color3.new(1, 1, 1)
			local chainModel = RigService.cloneChain(rig, eName, color)
			if not chainModel then return end

			local constraints_val = table.clone(appState.constraints:get())
			table.insert(constraints_val, {
				effector = eName,
				time = time,
				cframe = effPart.CFrame,
				chain = chainModel, -- ChainData from RigService.cloneChain
			})
			table.sort(constraints_val, function(a, b) return a.time < b.time end)
			appState.constraints:set(constraints_val)
		end)

		-- Drag movement on this track
		track.InputChanged:Connect(function(input)
			if input.UserInputType == Enum.UserInputType.MouseMovement and draggingConstraint then
				local relX = input.Position.X - constraintArea.AbsolutePosition.X
				local newTime = relX / appState.pixelsPerSecond:get() + appState.scrollOffset:get()
				local maxT = appState.duration:get()
				if maxT <= 0 and playbackSvc then maxT = playbackSvc:getDuration() end
				if maxT <= 0 then maxT = 10 end
				newTime = math.clamp(newTime, 0, maxT)

				-- Move the diamond directly (don't trigger renderConstraints)
				local pps = appState.pixelsPerSecond:get()
				local scroll = appState.scrollOffset:get()
				local px = (newTime - scroll) * pps
				if constraintDiamonds[draggingConstraint] then
					constraintDiamonds[draggingConstraint].Position = UDim2.new(0, math.floor(px) - 8, 0.5, -8)
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

				-- Compute ground level from the rig's actual foot positions
				local lFoot = rig.model:FindFirstChild("LeftFoot", true) :: BasePart?
				local rFoot = rig.model:FindFirstChild("RightFoot", true) :: BasePart?
				local groundY = 0
				if lFoot and rFoot then
					groundY = math.min(
						lFoot.Position.Y - lFoot.Size.Y / 2,
						rFoot.Position.Y - rFoot.Size.Y / 2
					)
				elseif lFoot then
					groundY = lFoot.Position.Y - lFoot.Size.Y / 2
				end

				local hrpPos = rig.rootPart.Position
				local _, hrpYaw, _ = rig.rootPart.CFrame:ToEulerAnglesYXZ()
				local groundCF = CFrame.new(hrpPos.X, groundY, hrpPos.Z)
					* CFrame.fromEulerAnglesYXZ(0, hrpYaw, 0)

				-- Root position: HRP XZ (stable anchor) + LowerTorso Y (height).
				-- HRP doesn't move with animation, prevents drift across iterations.
				local ltPart = rig.model:FindFirstChild("LowerTorso", true) :: BasePart?
				local rootLocalPos = { 0, 0, 0 }
				if ltPart then
					local ltY = ltPart.Position.Y - groundY
					-- XZ = 0 relative to character center (HRP is the anchor)
					rootLocalPos = { 0, ltY, 0 }
				end

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
				warn("[RoMotion] Generation error:", tostring(result))
				appState.generationStatus:set("failed")
				appState.generationMessage:set("Error (see Output)")
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

	-- Timeline scrub interaction (click + drag)
	local isScrubbing = false

	local function scrubToInput(input: InputObject)
		local relX = input.Position.X - timelineFrame.AbsolutePosition.X
		local time = relX / appState.pixelsPerSecond:get() + appState.scrollOffset:get()
		local maxTime = appState.duration:get()
		if maxTime <= 0 and playbackSvc then
			maxTime = playbackSvc:getDuration()
		end
		if maxTime <= 0 then maxTime = 10 end
		time = math.clamp(time, 0, maxTime)
		appState.playbackTime:set(time)
		if playbackSvc then
			playbackSvc:seekTo(time)
		end
	end

	-- Drag handling
	local function endDrag()
		draggingPromptIdx = nil
		draggingPromptEdge = nil
		draggingConstraint = nil
		isScrubbing = false
	end

	-- Movement handler shared by all drag-sensitive areas
	local function onDragMove(input: InputObject)
		if input.UserInputType ~= Enum.UserInputType.MouseMovement then return end

		if isScrubbing then
			scrubToInput(input)
		end

		if draggingPromptIdx then
			local relX = input.Position.X - promptTrack.AbsolutePosition.X
			local time = relX / appState.pixelsPerSecond:get() + appState.scrollOffset:get()
			time = math.clamp(time, 0, appState.duration:get())
			local updated = table.clone(appState.prompts:get())
			if draggingPromptIdx <= #updated then
				updated[draggingPromptIdx] = table.clone(updated[draggingPromptIdx])
				if draggingPromptEdge == "right" then
					updated[draggingPromptIdx].endTime = math.max(time, updated[draggingPromptIdx].startTime + 0.2)
				elseif draggingPromptEdge == "left" then
					updated[draggingPromptIdx].startTime = math.min(time, updated[draggingPromptIdx].endTime - 0.2)
				end
				appState.prompts:set(updated)
			end
		end

		if draggingConstraint then
			local relX = input.Position.X - constraintArea.AbsolutePosition.X
			local newTime = relX / appState.pixelsPerSecond:get() + appState.scrollOffset:get()
			newTime = math.clamp(newTime, 0, appState.duration:get())
			local updated = table.clone(appState.constraints:get())
			if draggingConstraint <= #updated then
				updated[draggingConstraint] = table.clone(updated[draggingConstraint])
				updated[draggingConstraint].time = newTime
				appState.constraints:set(updated)
			end
		end
	end

	-- Ensure all timeline frames receive input events
	timelineFrame.Active = true
	promptTrack.Active = true
	constraintArea.Active = true
	mainFrame.Active = true

	-- Listen on ALL relevant areas for movement
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

	-- Prompt block rendering
	local promptBlocks: { Frame } = {}
	local lastClickTime = 0
	local lastClickIndex = 0
	local draggingPromptIdx: number? = nil
	local draggingPromptEdge: string? = nil -- "left" or "right"

	local function renderPromptBlocks()
		if draggingPromptIdx then return end -- don't rebuild while dragging
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
			block.Active = true
			block.Parent = promptTrack

			local rc = Instance.new("UICorner")
			rc.CornerRadius = UDim.new(0, 4)
			rc.Parent = block

			local lbl = Instance.new("TextLabel")
			lbl.Size = UDim2.new(1, -12, 1, 0)
			lbl.Position = UDim2.new(0, 6, 0, 0)
			lbl.BackgroundTransparency = 1
			lbl.Text = prompt.text
			lbl.TextColor3 = Color3.new(1, 1, 1)
			lbl.TextSize = 11
			lbl.Font = Enum.Font.SourceSans
			lbl.TextTruncate = Enum.TextTruncate.AtEnd
			lbl.TextXAlignment = Enum.TextXAlignment.Left
			lbl.Parent = block

			-- Right edge handle for resizing
			local rightHandle = Instance.new("Frame")
			rightHandle.Size = UDim2.new(0, 10, 1, 0)
			rightHandle.Position = UDim2.new(1, -10, 0, 0)
			rightHandle.BackgroundColor3 = Color3.new(1, 1, 1)
			rightHandle.BackgroundTransparency = 0.6
			rightHandle.BorderSizePixel = 0
			rightHandle.Active = true
			rightHandle.ZIndex = 5
			rightHandle.Parent = block

			local leftHandle = Instance.new("Frame")
			leftHandle.Size = UDim2.new(0, 10, 1, 0)
			leftHandle.Position = UDim2.new(0, 0, 0, 0)
			leftHandle.BackgroundColor3 = Color3.new(1, 1, 1)
			leftHandle.BackgroundTransparency = 0.6
			leftHandle.BorderSizePixel = 0
			leftHandle.Active = true
			leftHandle.ZIndex = 5
			leftHandle.Parent = block

			local idx = i

			-- Drag edges (parent block handles move+release)
			rightHandle.InputBegan:Connect(function(input)
				if input.UserInputType == Enum.UserInputType.MouseButton1 then
					draggingPromptIdx = idx
					draggingPromptEdge = "right"
				end
			end)

			leftHandle.InputBegan:Connect(function(input)
				if input.UserInputType == Enum.UserInputType.MouseButton1 then
					draggingPromptIdx = idx
					draggingPromptEdge = "left"
				end
			end)

			-- Handle drag movement on block (handles disable Active, so block gets events)
			block.InputChanged:Connect(function(input)
				if input.UserInputType == Enum.UserInputType.MouseMovement and draggingPromptIdx then
					local relX = input.Position.X - promptTrack.AbsolutePosition.X
					local time = relX / appState.pixelsPerSecond:get() + appState.scrollOffset:get()
					local maxT = appState.duration:get()
					if maxT <= 0 and playbackSvc then maxT = playbackSvc:getDuration() end
					if maxT <= 0 then maxT = 10 end
					time = math.clamp(time, 0, maxT)
					local updated = table.clone(appState.prompts:get())
					if draggingPromptIdx <= #updated then
						updated[draggingPromptIdx] = table.clone(updated[draggingPromptIdx])
						if draggingPromptEdge == "right" then
							updated[draggingPromptIdx].endTime = math.max(time, updated[draggingPromptIdx].startTime + 0.2)
						elseif draggingPromptEdge == "left" then
							updated[draggingPromptIdx].startTime = math.min(time, updated[draggingPromptIdx].endTime - 0.2)
						end
						appState.prompts:set(updated)
					end
				end
			end)

			block.InputEnded:Connect(function(input)
				if input.UserInputType == Enum.UserInputType.MouseButton1 then
					draggingPromptIdx = nil
					draggingPromptEdge = nil
					renderPromptBlocks() -- rebuild now that drag is over
				end
			end)

			-- Double-click body to edit, right-click to delete
			block.InputBegan:Connect(function(input)
				if input.UserInputType == Enum.UserInputType.MouseButton1 then
					local now = os.clock()
					if lastClickIndex == idx and (now - lastClickTime) < 0.4 then
						editBox.Text = prompt.text
						editPopup.Position = UDim2.new(0, math.floor(startPx), 0, Constants.RULER_HEIGHT - 2)
						editPopup.Size = UDim2.new(0, math.max(200, math.floor(width)), 0, 30)
						editPopup.Visible = true
						editingIndex = idx
						editBox:CaptureFocus()
					end
					lastClickTime = now
					lastClickIndex = idx
				elseif input.UserInputType == Enum.UserInputType.MouseButton2 then
					local updated = table.clone(appState.prompts:get())
					table.remove(updated, idx)
					appState.prompts:set(updated)
				end
			end)

			table.insert(promptBlocks, block)
		end
	end


	-- Drag overlay: full-size invisible button that captures input while dragging.
	-- Only visible during active drags — ensures mouse-up is always caught.
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

		for i, constraint in constraints_val do
			local effTrack = constraintArea:FindFirstChild(constraint.effector)
			if not effTrack then continue end

			local px = (constraint.time - scroll) * pps
			local diamond = Instance.new("Frame")
			diamond.Name = "C_" .. constraint.effector .. "_" .. tostring(i)
			diamond.Size = UDim2.new(0, 16, 0, 16)
			diamond.Position = UDim2.new(0, math.floor(px) - 8, 0.5, -8)
			diamond.BackgroundColor3 = Constants.EFFECTOR_COLORS[constraint.effector] or Color3.new(1, 1, 1)
			diamond.Rotation = 0
			diamond.Active = true
			diamond.ZIndex = 5
			diamond.Parent = effTrack
			local diamondCorner = Instance.new("UICorner")
			diamondCorner.CornerRadius = UDim.new(0, 3)
			diamondCorner.Parent = diamond

			local idx = i

			-- Left-click to start drag (parent track handles move+release)
			diamond.InputBegan:Connect(function(input)
				if input.UserInputType == Enum.UserInputType.MouseButton1 then
					draggingConstraint = idx
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
			-- Don't reset if it's the same rig we already have
			local currentRig = appState.rig:get()
			if currentRig and currentRig.model == rig.model then
				return
			end
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
				text = "a person walking forward",
				startTime = 0,
				endTime = appState.duration:get(),
			},
		})
	end
end)

print("[RoMotion] Plugin loaded")
