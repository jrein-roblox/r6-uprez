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
-- Embedded rig metadata (joints, attachments, R15 reference rig) parented
-- under every CurveAnimation we emit. The retarget runtime uses these
-- instances to map curve names to live Motor6Ds; without them, downstream
-- consumers like AnimationClipProvider have no way to resolve which joint
-- a "LeftHand" folder drives. Authored once in Studio and committed at
-- data/RigData.rbxm.
local RIG_DATA_PATH = REPO_ROOT .. "/data/RigData.rbxm"

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

-- Load the rig-data template once at startup. Each CurveAnimation gets a
-- :Clone() of every top-level instance — Instance can only have one parent,
-- so cloning per-clip is required even though the source data is identical.
-- We tolerate a missing file (warn + proceed) so this script keeps working
-- on machines that haven't authored the rbxm yet.
local rigDataTemplates: { Instance } = {}
do
	local ok, loaded = pcall(function()
		return FileSystemService:LoadInstances(RIG_DATA_PATH)
	end)
	if ok and loaded then
		for _, inst in ipairs(loaded) do
			table.insert(rigDataTemplates, inst)
		end
		print(string.format(
			"[build_rbxm] loaded %d rig-data instance(s) from %s",
			#rigDataTemplates, RIG_DATA_PATH))
	else
		warn(string.format(
			"[build_rbxm] could not load %s (%s); proceeding without rig data",
			RIG_DATA_PATH, tostring(loaded)))
	end
end

-- --------------------------------------------- CurveAnimation builder ----
-- Direct port of `studio/load_curve_animation.server.luau:27` with
-- `:RegisterAnimationClip` removed — we emit Instances to disk instead of
-- registering them with AnimationClipProvider.

-- R15 rig parenting. Folders inside the CurveAnimation are nested to match
-- the actual Motor6D chain (LowerTorso under HumanoidRootPart, UpperTorso
-- under LowerTorso, ...). CurveAnimation playback walks descendants by
-- name, so the layout is purely organizational — but downstream tooling
-- (e.g. animation editors that expect rig-shaped trees) reads it. HRP is
-- the root and has no parent in this map (parented to the CurveAnimation
-- itself).
local PART_PARENT: { [string]: string } = {
	LowerTorso     = "HumanoidRootPart",
	UpperTorso     = "LowerTorso",
	Head           = "UpperTorso",
	LeftUpperArm   = "UpperTorso",
	LeftLowerArm   = "LeftUpperArm",
	LeftHand       = "LeftLowerArm",
	RightUpperArm  = "UpperTorso",
	RightLowerArm  = "RightUpperArm",
	RightHand      = "RightLowerArm",
	LeftUpperLeg   = "LowerTorso",
	LeftLowerLeg   = "LeftUpperLeg",
	LeftFoot       = "LeftLowerLeg",
	RightUpperLeg  = "LowerTorso",
	RightLowerLeg  = "RightUpperLeg",
	RightFoot      = "RightLowerLeg",
}

local function buildCurveAnimation(data: any, clipName: string): Instance
	local Cubic = Enum.KeyInterpolationMode.Cubic
	local nFrames = data.frameCount
	local frameHz = data.frameRate

	local ca = Instance.new("CurveAnimation")
	ca.Name = clipName

	-- Embed rig metadata at the root so downstream retargeting tooling can
	-- resolve curve names to live joints. We clone the cached template
	-- because each CurveAnimation needs its own copy.
	for _, template in ipairs(rigDataTemplates) do
		template:Clone().Parent = ca
	end

	-- Pass 1: materialize a folder for every part we'll touch (every key
	-- of data.parts plus HumanoidRootPart, which always exists so child
	-- folders can hang off it even when no root curves are emitted —
	-- e.g. when `_fold_root_into_lower_torso` stripped data.root). We
	-- deliberately do not parent here; that's pass 2.
	local folders: { [string]: Folder } = {}
	local function ensureFolder(name: string): Folder
		local existing = folders[name]
		if existing then return existing end
		local f = Instance.new("Folder")
		f.Name = name
		folders[name] = f
		return f
	end
	ensureFolder("HumanoidRootPart")
	for partName, _ in pairs(data.parts) do
		ensureFolder(partName)
	end

	-- Pass 2: parent each folder by its R15 chain ancestor. Unknown names
	-- (anything outside PART_PARENT and not HRP) fall back to direct
	-- children of the CurveAnimation so we don't silently drop curves.
	for name, folder in pairs(folders) do
		local parentName = PART_PARENT[name]
		if parentName and folders[parentName] then
			folder.Parent = folders[parentName]
		else
			folder.Parent = ca
		end
	end

	-- Always emit Rotation and Position together. The retargeting tool
	-- has a bug where a folder with only one of the two curves is treated
	-- as missing the joint entirely, so when r15.json has rotation but
	-- no translation (the common case for non-root joints) we synthesize
	-- an identity-translation Vector3Curve. Identity in Motor6D / C1
	-- space is (0, 0, 0) — translation here is the per-frame delta from
	-- rest, not the world position.
	local function writeCurves(folder: Folder, p: any)
		local hasRot = p.rotX ~= nil
		local hasPos = p.posX ~= nil
		if not (hasRot or hasPos) then return end

		local rc = Instance.new("RotationCurve", folder); rc.Name = "Rotation"
		local pc = Instance.new("Vector3Curve", folder); pc.Name = "Position"
		local px, py, pz = pc:X(), pc:Y(), pc:Z()
		for i = 1, nFrames do
			local ti = (i - 1) / frameHz
			local rx = hasRot and p.rotX[i] or 0.0
			local ry = hasRot and p.rotY[i] or 0.0
			local rz = hasRot and p.rotZ[i] or 0.0
			local rw = hasRot and p.rotW[i] or 1.0
			local cf = CFrame.new(0, 0, 0, rx, ry, rz, rw)
			rc:InsertKey(RotationCurveKey.new(ti, cf, Cubic))
			local tx = hasPos and p.posX[i] or 0.0
			local ty = hasPos and p.posY[i] or 0.0
			local tz = hasPos and p.posZ[i] or 0.0
			px:InsertKey(FloatCurveKey.new(ti, tx, Cubic))
			py:InsertKey(FloatCurveKey.new(ti, ty, Cubic))
			pz:InsertKey(FloatCurveKey.new(ti, tz, Cubic))
		end
	end

	for partName, p in pairs(data.parts) do
		writeCurves(folders[partName], p)
	end

	-- HumanoidRootPart curves are emitted only when r15.json includes a
	-- `root` block. The pipeline's `_fold_root_into_lower_torso` strips
	-- it on purpose so Studio uses the character's spawn HRP pose (feet
	-- on ground) instead of overriding it to world origin. Either way
	-- the HRP folder still exists from pass 1, so LowerTorso et al. are
	-- correctly nested under it.
	if data.root ~= nil then
		writeCurves(folders["HumanoidRootPart"], data.root)
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
