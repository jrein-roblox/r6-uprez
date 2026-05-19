--!strict
-- Headless rbxm builder. Reads r15.json files produced by Stage C
-- (`python/batch_retarget.py`) and writes CurveAnimation .rbxm files
-- via `FileSystemService:WriteInstances` — no Studio in the loop.
--
-- Run via the Python wrapper (`python/build_rbxm.py`) which sets globals
-- via `--lua.globals` and invokes:
--   robloxdev-cli run \
--       --run lua/build_rbxm.lua \
--       --fs.readwrite <repo_root> \
--       --load.asRobloxScript
--
-- Globals consumed (all optional, sensible defaults):
--   _G.BUILD_RBXM_REPO_ROOT      -- repo root absolute path (required for output paths)
--   _G.BUILD_RBXM_PATTERN        -- glob filter on relative clip path (e.g. "Crouch/*")
--   _G.BUILD_RBXM_PER_CLIP       -- "1"/"0" (default 1)
--   _G.BUILD_RBXM_PER_CATEGORY   -- "1"/"0" (default 1)
--   _G.BUILD_RBXM_CORPUS         -- "1"/"0" (default 1)
--   _G.BUILD_RBXM_LIMIT          -- max clips after filter (string-encoded int)

local FileSystemService = game:GetService("FileSystemService")
local HttpService = game:GetService("HttpService")

-- ---------------------------------------------------------- config -------

local function flag(name: string, default: boolean): boolean
	local v = (_G :: any)[name]
	if v == nil then return default end
	if type(v) == "boolean" then return v end
	if v == "1" or v == "true" or v == "yes" then return true end
	if v == "0" or v == "false" or v == "no" then return false end
	return default
end

local REPO_ROOT = (_G :: any).BUILD_RBXM_REPO_ROOT
	or "/Users/jrein/git/roblox/jrein/motion-matching"
local PATTERN: string? = (_G :: any).BUILD_RBXM_PATTERN
local PER_CLIP = flag("BUILD_RBXM_PER_CLIP", true)
local PER_CATEGORY = flag("BUILD_RBXM_PER_CATEGORY", true)
local CORPUS = flag("BUILD_RBXM_CORPUS", true)
local LIMIT_STR: string? = (_G :: any).BUILD_RBXM_LIMIT
local LIMIT: number? = LIMIT_STR and tonumber(LIMIT_STR) or nil

-- Default Stage A source dir is the mined corpus (Kimodo_Constraints).
-- For procedural clips, set BUILD_RBXM_IN_DIR=data/Kimodo_Procedural and
-- BUILD_RBXM_OUT_NAME=Kimodo_Procedural so output rbxms land alongside.
local IN_DIR_REL = (_G :: any).BUILD_RBXM_IN_DIR or "data/Kimodo_Constraints"
local OUT_DIR_NAME = (_G :: any).BUILD_RBXM_OUT_NAME or "Kimodo_Animations"

local CONSTRAINTS_DIR = REPO_ROOT .. "/" .. IN_DIR_REL
local OUT_DIR = REPO_ROOT .. "/data/" .. OUT_DIR_NAME
local R15_JSON_NAME = "r15.json"
local PER_CLIP_RBXM_NAME = "r15.rbxm"

print(string.format(
	"[build_rbxm] repo_root=%s in=%s out=%s\n  per-clip=%s per-category=%s corpus=%s pattern=%s limit=%s",
	REPO_ROOT, IN_DIR_REL, OUT_DIR_NAME,
	tostring(PER_CLIP), tostring(PER_CATEGORY), tostring(CORPUS),
	tostring(PATTERN), tostring(LIMIT)
))

-- --------------------------------------------------------- helpers -------

-- Convert a glob (just `*` and `?`) to a Lua pattern. Anchored.
local function globToLuaPattern(glob: string): string
	local lua = "^"
	for c in glob:gmatch(".") do
		if c == "*" then
			lua ..= ".*"
		elseif c == "?" then
			lua ..= "."
		elseif c:match("[%^%$%(%)%%%.%[%]%+%-]") then
			lua ..= "%" .. c
		else
			lua ..= c
		end
	end
	return lua .. "$"
end

local PATTERN_LUA: string? = PATTERN and globToLuaPattern(PATTERN) or nil

-- Path relative to CONSTRAINTS_DIR (assumes `path` starts with that prefix).
local function relTo(path: string, base: string): string
	if path:sub(1, #base) == base then
		local rel = path:sub(#base + 1)
		if rel:sub(1, 1) == "/" then rel = rel:sub(2) end
		return rel
	end
	return path
end

-- "Crouch/M_Neutral_Crouch_Diamond_BR_FR_Lfoot/r15.json" -> ("Crouch", "M_Neutral_Crouch_Diamond_BR_FR_Lfoot")
-- "Traversal/Climb/<Clip>/r15.json"                       -> ("Traversal", "<Clip>")
-- "Walk_F/r15.json"                                        -> ("Walk", "Walk_F")
-- Two-or-more-level paths: top dir = category, leaf = clip. One-level
-- paths (procedural clips, no category subdir): synthesize category from
-- the clip's first underscore token ("Walk_F" → "Walk", "Idle" → "Idle").
local function categoryAndClip(rel: string): (string?, string?)
	-- Try ≥2-level path first.
	local cat, after = rel:match("^([^/]+)/(.+)/" .. R15_JSON_NAME .. "$")
	if cat and after then
		local clip = (after :: any):match("([^/]+)$")
		return cat, clip
	end
	-- 1-level fallback: <Clip>/r15.json
	local clip = rel:match("^([^/]+)/" .. R15_JSON_NAME .. "$")
	if clip then
		local synthCat = clip:match("^([^_]+)") or clip
		return synthCat, clip
	end
	return nil, nil
end

local function ensureDir(path: string)
	local ok, err = pcall(function()
		FileSystemService:CreateDirectories(path)
	end)
	if not ok then warn("CreateDirectories failed", path, err) end
end

-- --------------------------------------------- CurveAnimation builder ----
-- Direct port of `studio/load_curve_animation.server.luau:27` with
-- `:RegisterAnimationClip` removed — we emit Instances to disk instead of
-- registering them with AnimationClipProvider.
local function buildCurveAnimation(data: any, clipName: string): Instance
	local Cubic = Enum.KeyInterpolationMode.Cubic
	local nFrames = data.frameCount
	local frameHz = data.frameRate

	local ca = Instance.new("CurveAnimation")
	ca.Name = clipName

	for partName, p in pairs(data.parts) do
		local f = Instance.new("Folder", ca); f.Name = partName
		local rc = Instance.new("RotationCurve", f); rc.Name = "Rotation"
		for i = 1, nFrames do
			local cf = CFrame.new(0, 0, 0, p.rotX[i], p.rotY[i], p.rotZ[i], p.rotW[i])
			rc:InsertKey(RotationCurveKey.new((i - 1) / frameHz, cf, Cubic))
		end
		if p.posX then
			local pc = Instance.new("Vector3Curve", f); pc.Name = "Position"
			local px, py, pz = pc:X(), pc:Y(), pc:Z()
			for i = 1, nFrames do
				local ti = (i - 1) / frameHz
				px:InsertKey(FloatCurveKey.new(ti, p.posX[i], Cubic))
				py:InsertKey(FloatCurveKey.new(ti, p.posY[i], Cubic))
				pz:InsertKey(FloatCurveKey.new(ti, p.posZ[i], Cubic))
			end
		end
	end

	-- HumanoidRootPart curves are emitted only when r15.json includes a
	-- `root` block. The pipeline's `_fold_root_into_lower_torso` strips it
	-- on purpose so Studio uses the character's spawn HRP pose (feet on
	-- ground) instead of overriding it to world origin.
	if data.root ~= nil then
		local hrpF = Instance.new("Folder", ca); hrpF.Name = "HumanoidRootPart"
		local hrpRot = Instance.new("RotationCurve", hrpF); hrpRot.Name = "Rotation"
		local hrpPos = Instance.new("Vector3Curve", hrpF); hrpPos.Name = "Position"
		local cx, cy, cz = hrpPos:X(), hrpPos:Y(), hrpPos:Z()
		for i = 1, nFrames do
			local ti = (i - 1) / frameHz
			local r = data.root
			local cf = CFrame.new(0, 0, 0, r.rotX[i], r.rotY[i], r.rotZ[i], r.rotW[i])
			hrpRot:InsertKey(RotationCurveKey.new(ti, cf, Cubic))
			cx:InsertKey(FloatCurveKey.new(ti, r.posX[i], Cubic))
			cy:InsertKey(FloatCurveKey.new(ti, r.posY[i], Cubic))
			cz:InsertKey(FloatCurveKey.new(ti, r.posZ[i], Cubic))
		end
	end

	return ca
end

-- ---------------------------------------------------- enumerate clips ----

local clipJobs: { { rel: string, clipDir: string, jsonPath: string, category: string, clipName: string } } = {}

for fileData in FileSystemService:Walk(CONSTRAINTS_DIR, Enum.FileSystemWalkMode.Recursive) do
	local path = fileData.Path
	-- We're hunting for r15.json files
	if not path:match("/" .. R15_JSON_NAME .. "$") then continue end
	local rel = relTo(path, CONSTRAINTS_DIR)
	local cat, clipName = categoryAndClip(rel)
	if not cat or not clipName then
		print(string.format("[build_rbxm] skip (unexpected layout): %s", rel))
		continue
	end
	if PATTERN_LUA then
		local clipRel = cat .. "/" .. clipName
		-- Match against either "<category>/<clip>" or the bare clip name.
		-- The latter is convenient for procedural clips (synthetic category)
		-- where the user just wants `--pattern 'Walk_F'`.
		if not clipRel:match(PATTERN_LUA) and not clipName:match(PATTERN_LUA) then
			continue
		end
	end
	local clipDir = path:gsub("/" .. R15_JSON_NAME .. "$", "")
	table.insert(clipJobs, {
		rel = rel,
		clipDir = clipDir,
		jsonPath = path,
		category = cat,
		clipName = clipName,
	})
end

if LIMIT and #clipJobs > LIMIT then
	for i = LIMIT + 1, #clipJobs do clipJobs[i] = nil end
end

print(string.format("[build_rbxm] %d clips matched", #clipJobs))
if #clipJobs == 0 then return end

ensureDir(OUT_DIR)

-- ---------------------------------------- build + write per-clip rbxms ---

-- Aggregators for category + corpus passes. We hold one canonical
-- CurveAnimation per clip; per-category and corpus use :Clone() since
-- a single Instance can only have one parent.
local perCategory: { [string]: { Instance } } = {}
local clipCAs: { Instance } = {}

for i, job in ipairs(clipJobs) do
	local raw = FileSystemService:ReadFile(job.jsonPath, Enum.FileMode.Text)
	local data = HttpService:JSONDecode(raw)
	local ca = buildCurveAnimation(data, job.clipName)

	if PER_CLIP then
		local outPath = job.clipDir .. "/" .. PER_CLIP_RBXM_NAME
		FileSystemService:WriteInstances(outPath, { ca })
	end

	if PER_CATEGORY or CORPUS then
		-- Keep canonical for category; clone for corpus too
		if PER_CATEGORY then
			local list = perCategory[job.category]
			if not list then list = {}; perCategory[job.category] = list end
			table.insert(list, ca)
			table.insert(clipCAs, ca)
		else
			-- Corpus only — still need to retain the instance
			table.insert(clipCAs, ca)
		end
	else
		ca:Destroy()
	end

	if i % 50 == 0 or i == #clipJobs then
		print(string.format("[build_rbxm] [%d/%d] built %s/%s", i, #clipJobs, job.category, job.clipName))
	end
end

-- ----------------------------------- write per-category and corpus rbxms -

if PER_CATEGORY then
	for category, clips in pairs(perCategory) do
		local catFolder = Instance.new("Folder")
		catFolder.Name = category
		-- Move the canonical CurveAnimations under the category folder.
		-- (For CORPUS, we re-clone these out of the folder before destroying it.)
		for _, ca in ipairs(clips) do
			ca.Parent = catFolder
		end

		local outPath = OUT_DIR .. "/" .. category .. ".rbxm"
		FileSystemService:WriteInstances(outPath, { catFolder })
		print(string.format("[build_rbxm] wrote %s (%d clips)", outPath, #clips))

		if CORPUS then
			-- Re-parent into per-category folders for the corpus aggregate.
			-- We need to keep the cat folder alive so it can be parented
			-- under the corpus root; clone is simpler.
			-- But the canonical clips are still inside this catFolder; we'll
			-- handle corpus next using a Clone().
		end
	end
end

if CORPUS then
	local corpusRoot = Instance.new("Folder")
	corpusRoot.Name = "Kimodo_Animations"
	-- For each category, clone each clip under a category folder.
	-- This works whether PER_CATEGORY was true or false.
	if PER_CATEGORY then
		for category, _ in pairs(perCategory) do
			-- clips were re-parented under catFolder above; gather clones now
			-- by walking via the original perCategory listings (Instances live)
			local catClone = Instance.new("Folder")
			catClone.Name = category
			catClone.Parent = corpusRoot
			for _, ca in ipairs(perCategory[category]) do
				if ca and ca.Parent then ca:Clone().Parent = catClone end
			end
		end
	else
		-- No per-category; group by category from job records
		local byCat: { [string]: { Instance } } = {}
		for i, ca in ipairs(clipCAs) do
			local job = clipJobs[i]
			if job then
				local list = byCat[job.category]
				if not list then list = {}; byCat[job.category] = list end
				table.insert(list, ca)
			end
		end
		for category, clips in pairs(byCat) do
			local catClone = Instance.new("Folder")
			catClone.Name = category
			catClone.Parent = corpusRoot
			for _, ca in ipairs(clips) do
				ca:Clone().Parent = catClone
			end
		end
	end

	local outPath = OUT_DIR .. "/all.rbxm"
	FileSystemService:WriteInstances(outPath, { corpusRoot })
	print(string.format("[build_rbxm] wrote %s", outPath))
end

print("[build_rbxm] done")
