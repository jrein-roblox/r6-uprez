--!strict
-- Convert all RotationCurve instances to EulerRotationCurve (XYZ) in place.
-- Run in Studio Command Bar after importing CurveAnimations.
--
-- Finds all RotationCurve descendants in Workspace, converts each quaternion
-- key to Euler angles (XYZ order), creates a replacement EulerRotationCurve,
-- and destroys the original.

local converted = 0
local failed = 0

for _, desc in workspace:GetDescendants() do
	if not desc:IsA("RotationCurve") then continue end

	local parent = desc.Parent
	if not parent then continue end

	local name = desc.Name

	-- Read all keys from the RotationCurve
	local keys = desc:GetKeys()
	if #keys == 0 then
		desc:Destroy()
		continue
	end

	-- Create replacement EulerRotationCurve
	local euler = Instance.new("EulerRotationCurve")
	euler.Name = name
	euler.RotationOrder = Enum.RotationOrder.XYZ
	local ex, ey, ez = euler:X(), euler:Y(), euler:Z()

	local ok = true
	for _, key in keys do
		local t = key.Time
		local cf = key.Value -- CFrame
		local interpMode = key.Interpolation

		local success, rx, ry, rz = pcall(function()
			return cf:ToEulerAnglesXYZ()
		end)
		if not success then
			ok = false
			break
		end

		ex:InsertKey(FloatCurveKey.new(t, rx, interpMode))
		ey:InsertKey(FloatCurveKey.new(t, ry, interpMode))
		ez:InsertKey(FloatCurveKey.new(t, rz, interpMode))
	end

	if ok then
		euler.Parent = parent
		desc:Destroy()
		converted += 1
	else
		euler:Destroy()
		failed += 1
		warn(string.format("[convert] Failed to convert: %s", desc:GetFullName()))
	end
end

print(string.format("[convert_rotation_curves] Done: %d converted, %d failed", converted, failed))
