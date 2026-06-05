--!strict
-- Visual markers (spheres) at constraint positions in the 3D viewport.

local Constants = require(script.Parent.Parent.Utils.Constants)

local ConstraintGizmo = {}
ConstraintGizmo.__index = ConstraintGizmo

export type Gizmo = typeof(setmetatable({} :: {
	_parts: { BasePart },
	_folder: Folder,
}, ConstraintGizmo))

function ConstraintGizmo.new(): Gizmo
	local folder = Instance.new("Folder")
	folder.Name = "RoMotion_Gizmos"
	folder.Parent = workspace

	return setmetatable({
		_parts = {},
		_folder = folder,
	}, ConstraintGizmo) :: any
end

function ConstraintGizmo:clear()
	for _, part in self._parts do
		part:Destroy()
	end
	table.clear(self._parts)
end

function ConstraintGizmo:render(constraints: { { effector: string, time: number, cframe: CFrame } }, currentTime: number)
	self:clear()

	for _, c in constraints do
		local color = Constants.EFFECTOR_COLORS[c.effector] or Color3.new(1, 1, 1)
		local isAtCurrentTime = math.abs(c.time - currentTime) < 0.05

		local part = Instance.new("Part")
		part.Name = c.effector .. "_gizmo"
		part.Shape = Enum.PartType.Ball
		part.Size = Vector3.new(0.3, 0.3, 0.3)
		part.CFrame = c.cframe
		part.Anchored = true
		part.CanCollide = false
		part.CanQuery = false
		part.CanTouch = false
		part.Color = color
		part.Material = Enum.Material.Neon
		part.Transparency = if isAtCurrentTime then 0 else 0.6
		part.CastShadow = false
		part.Parent = self._folder

		table.insert(self._parts, part)
	end
end

function ConstraintGizmo:destroy()
	self:clear()
	self._folder:Destroy()
end

return ConstraintGizmo
