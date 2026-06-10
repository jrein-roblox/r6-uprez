--!strict
-- Save/load RoMotion projects to ServerStorage/RBX_ANIMSAVES/<Rig>/RoMotion_<Name>

local ServerStorage = game:GetService("ServerStorage")
local Constants = require(script.Parent.Parent.Utils.Constants)

local DataModelService = {}

export type PromptData = {
	text: string,
	startTime: number,
	endTime: number,
}

export type ConstraintData = {
	effector: string,
	time: number,
	cframe: CFrame,
	pinned: boolean?,
}

export type ProjectData = {
	name: string,
	duration: number,
	looped: boolean,
	seed: number,
	prompts: { PromptData },
	constraints: { ConstraintData },
}

local function getOrCreateFolder(parent: Instance, name: string): Folder
	local existing = parent:FindFirstChild(name)
	if existing and existing:IsA("Folder") then
		return existing
	end
	local folder = Instance.new("Folder")
	folder.Name = name
	folder.Parent = parent
	return folder
end

local function getAnimSavesFolder(rigName: string): Folder
	local saves = getOrCreateFolder(ServerStorage, Constants.SAVE_FOLDER_NAME)
	return getOrCreateFolder(saves, rigName)
end

function DataModelService.save(rigName: string, project: ProjectData)
	local rigFolder = getAnimSavesFolder(rigName)
	local projName = "RoMotion_" .. project.name

	-- Remove existing if present
	local existing = rigFolder:FindFirstChild(projName)
	if existing then
		existing:Destroy()
	end

	local projFolder = Instance.new("Folder")
	projFolder.Name = projName
	projFolder.Parent = rigFolder

	-- Settings
	local settings = Instance.new("Configuration")
	settings.Name = "Settings"
	settings.Parent = projFolder

	local dur = Instance.new("NumberValue")
	dur.Name = "Duration"
	dur.Value = project.duration
	dur.Parent = settings

	local looped = Instance.new("BoolValue")
	looped.Name = "Looped"
	looped.Value = project.looped
	looped.Parent = settings

	local seed = Instance.new("IntValue")
	seed.Name = "Seed"
	seed.Value = project.seed
	seed.Parent = settings

	-- Prompts
	local promptsFolder = Instance.new("Folder")
	promptsFolder.Name = "Prompts"
	promptsFolder.Parent = projFolder

	for i, prompt in project.prompts do
		local cfg = Instance.new("Configuration")
		cfg.Name = tostring(i)
		cfg.Parent = promptsFolder

		local text = Instance.new("StringValue")
		text.Name = "Text"
		text.Value = prompt.text
		text.Parent = cfg

		local startTime = Instance.new("NumberValue")
		startTime.Name = "StartTime"
		startTime.Value = prompt.startTime
		startTime.Parent = cfg

		local endTime = Instance.new("NumberValue")
		endTime.Name = "EndTime"
		endTime.Value = prompt.endTime
		endTime.Parent = cfg
	end

	-- Constraints
	local constraintsFolder = Instance.new("Folder")
	constraintsFolder.Name = "Constraints"
	constraintsFolder.Parent = projFolder

	for _, effName in Constants.EFFECTORS do
		local effFolder = Instance.new("Folder")
		effFolder.Name = effName
		effFolder.Parent = constraintsFolder
	end

	for _, constraint in project.constraints do
		local effFolder = constraintsFolder:FindFirstChild(constraint.effector)
		if effFolder then
			local cfVal = Instance.new("CFrameValue")
			cfVal.Name = string.format("%.3f", constraint.time)
			cfVal.Value = constraint.cframe
			cfVal:SetAttribute("Pinned", constraint.pinned == true)
			cfVal.Parent = effFolder
		end
	end
end

function DataModelService.load(rigName: string, projectName: string): ProjectData?
	local saves = ServerStorage:FindFirstChild(Constants.SAVE_FOLDER_NAME)
	if not saves then
		return nil
	end
	local rigFolder = saves:FindFirstChild(rigName)
	if not rigFolder then
		return nil
	end
	local projFolder = rigFolder:FindFirstChild("RoMotion_" .. projectName)
	if not projFolder then
		return nil
	end

	local settings = projFolder:FindFirstChild("Settings") :: Configuration?
	local duration = 3.0
	local looped = false
	local seed = 0
	if settings then
		local durVal = settings:FindFirstChild("Duration") :: NumberValue?
		if durVal then duration = durVal.Value end
		local loopVal = settings:FindFirstChild("Looped") :: BoolValue?
		if loopVal then looped = loopVal.Value end
		local seedVal = settings:FindFirstChild("Seed") :: IntValue?
		if seedVal then seed = seedVal.Value end
	end

	-- Load prompts
	local prompts: { PromptData } = {}
	local promptsFolder = projFolder:FindFirstChild("Prompts")
	if promptsFolder then
		local children = promptsFolder:GetChildren()
		table.sort(children, function(a, b)
			return tonumber(a.Name) or 0 < tonumber(b.Name) or 0
		end)
		for _, cfg in children do
			if cfg:IsA("Configuration") then
				local text = (cfg:FindFirstChild("Text") :: StringValue?)
				local startTime = (cfg:FindFirstChild("StartTime") :: NumberValue?)
				local endTime = (cfg:FindFirstChild("EndTime") :: NumberValue?)
				if text and startTime and endTime then
					table.insert(prompts, {
						text = text.Value,
						startTime = startTime.Value,
						endTime = endTime.Value,
					})
				end
			end
		end
	end

	-- Load constraints
	local constraints: { ConstraintData } = {}
	local constraintsFolder = projFolder:FindFirstChild("Constraints")
	if constraintsFolder then
		for _, effFolder in constraintsFolder:GetChildren() do
			if effFolder:IsA("Folder") then
				for _, cfVal in effFolder:GetChildren() do
					if cfVal:IsA("CFrameValue") then
						local time = tonumber(cfVal.Name) or 0
						table.insert(constraints, {
							effector = effFolder.Name,
							time = time,
							cframe = cfVal.Value,
							pinned = cfVal:GetAttribute("Pinned") == true,
						})
					end
				end
			end
		end
	end
	table.sort(constraints, function(a, b) return a.time < b.time end)

	return {
		name = projectName,
		duration = duration,
		looped = looped,
		seed = seed,
		prompts = prompts,
		constraints = constraints,
	}
end

function DataModelService.listProjects(rigName: string): { string }
	local saves = ServerStorage:FindFirstChild(Constants.SAVE_FOLDER_NAME)
	if not saves then
		return {}
	end
	local rigFolder = saves:FindFirstChild(rigName)
	if not rigFolder then
		return {}
	end
	local projects: { string } = {}
	for _, child in rigFolder:GetChildren() do
		if child:IsA("Folder") and child.Name:sub(1, 9) == "RoMotion_" then
			table.insert(projects, child.Name:sub(10))
		end
	end
	return projects
end

function DataModelService.storeGeneratedAnimation(rigName: string, projectName: string, curveAnim: Instance)
	local saves = ServerStorage:FindFirstChild(Constants.SAVE_FOLDER_NAME)
	if not saves then return end
	local rigFolder = saves:FindFirstChild(rigName)
	if not rigFolder then return end
	local projFolder = rigFolder:FindFirstChild("RoMotion_" .. projectName)
	if not projFolder then return end

	local existing = projFolder:FindFirstChild("GeneratedAnimation")
	if existing then
		existing:Destroy()
	end
	local clone = curveAnim:Clone()
	clone.Name = "GeneratedAnimation"
	clone.Parent = projFolder
end

return DataModelService
