--!strict
local FileSystemService = game:GetService("FileSystemService")
local RIG_PATH: string = (_G :: any).RIG_PATH
local OUT_PATH: string = (_G :: any).OUT_PATH

local loaded = FileSystemService:LoadInstances(RIG_PATH)
print("[dump_rig] loaded " .. tostring(#loaded) .. " top-level instance(s)")

local model: Instance? = nil
for _, inst in loaded do
	if inst:IsA("Model") then model = inst break end
end
if not model then model = loaded[1] end
assert(model, "no instance loaded")
print("[dump_rig] root: " .. model.Name .. " [" .. model.ClassName .. "]")

local function fmt(cf: CFrame): string
	local c = { cf:GetComponents() }
	local parts = {}
	for _, v in c do table.insert(parts, string.format("%.6f", v)) end
	return "[" .. table.concat(parts, ",") .. "]"
end

local lines: { string } = {}
local nParts, nAtt = 0, 0
for _, d in (model :: Instance):GetDescendants() do
	if not d:IsA("BasePart") then continue end
	nParts += 1
	local atts: { string } = {}
	for _, a in d:GetChildren() do
		if a:IsA("Attachment") then
			nAtt += 1
			table.insert(atts, string.format('"%s":%s', a.Name, fmt((a :: Attachment).CFrame)))
		end
	end
	table.insert(lines, string.format('"%s":{"size":[%.6f,%.6f,%.6f],"attachments":{%s}}',
		d.Name, d.Size.X, d.Size.Y, d.Size.Z, table.concat(atts, ",")))
end
print(string.format("[dump_rig] parts=%d attachments=%d", nParts, nAtt))
FileSystemService:WriteFile(OUT_PATH, "{" .. table.concat(lines, ",") .. "}")
print("[dump_rig] wrote " .. OUT_PATH)
