--!strict
-- Merge all per-clip r15.rbxm files from a directory tree into a single
-- rbxm containing a Folder of uniquely-named CurveAnimations.
--
-- Run via roblox-cli:
--   roblox-cli run --run lua/merge_rbxm.lua --fs.readwrite <repo_root> --load.asRobloxScript
--
-- Globals:
--   _G.MERGE_INPUT_DIR   -- absolute path to scan for r15.rbxm files
--   _G.MERGE_OUTPUT_PATH -- absolute path for the output .rbxm
--   _G.MERGE_LIMIT       -- optional max clips (string-encoded int)

local FileSystemService = game:GetService("FileSystemService")

local INPUT_DIR: string = (_G :: any).MERGE_INPUT_DIR or error("MERGE_INPUT_DIR required")
local OUTPUT_PATH: string = (_G :: any).MERGE_OUTPUT_PATH or error("MERGE_OUTPUT_PATH required")
local LIMIT_STR: string? = (_G :: any).MERGE_LIMIT
local LIMIT: number? = LIMIT_STR and tonumber(LIMIT_STR) or nil

print(string.format("[merge_rbxm] input=%s output=%s limit=%s", INPUT_DIR, OUTPUT_PATH, tostring(LIMIT)))

-- Find all r15.rbxm files using Walk (same API as build_rbxm.lua)
local rbxmFiles: { string } = {}
for fileData in FileSystemService:Walk(INPUT_DIR, Enum.FileSystemWalkMode.Recursive) do
	local path = fileData.Path
	if path:match("/r15%.rbxm$") then
		table.insert(rbxmFiles, path)
	end
end
table.sort(rbxmFiles)

if LIMIT and #rbxmFiles > LIMIT then
	local trimmed = {}
	for i = 1, LIMIT do
		trimmed[i] = rbxmFiles[i]
	end
	rbxmFiles = trimmed
end

print(string.format("[merge_rbxm] found %d rbxm files", #rbxmFiles))

-- Load each rbxm and collect CurveAnimations into a single Folder
local root = Instance.new("Folder")
root.Name = "UGC_Emotes"

local nameCount: { [string]: number } = {}

for _, path in rbxmFiles do
	local ok2, instances = pcall(function()
		return FileSystemService:LoadInstances(path)
	end)
	if not ok2 or not instances or #instances == 0 then
		warn(string.format("[merge_rbxm] failed to load: %s", path))
		continue
	end

	-- Derive a unique name from the directory structure:
	-- e.g., .../emotes/hip-hop-groove_v03/r15.rbxm -> "hip-hop-groove_v03"
	local dirName = path:match("([^/]+)/r15%.rbxm$") or ("clip_" .. tostring(#root:GetChildren() + 1))

	-- Ensure uniqueness
	if nameCount[dirName] then
		nameCount[dirName] += 1
		dirName = dirName .. "_" .. tostring(nameCount[dirName])
	else
		nameCount[dirName] = 1
	end

	for _, inst in instances do
		if inst:IsA("AnimationClip") then
			inst.Name = dirName
			inst.Parent = root
		else
			-- Some rbxm files wrap in a Folder; grab children
			for _, child in inst:GetChildren() do
				if child:IsA("AnimationClip") then
					child.Name = dirName
					child.Parent = root
				end
			end
		end
	end
end

print(string.format("[merge_rbxm] merged %d clips into %s", #root:GetChildren(), OUTPUT_PATH))

-- Write the merged file
FileSystemService:WriteInstances(OUTPUT_PATH, { root })
print("[merge_rbxm] done")
