--!strict
-- 3D handles for manipulating selected constraint positions.
-- Uses Handles (move) and ArcHandles (rotate) available to external plugins.

local Signal = require(script.Parent.Parent.Signal)

local ConstraintDragger = {}
ConstraintDragger.__index = ConstraintDragger

export type Dragger = typeof(setmetatable({} :: {
	_handles: Handles?,
	_adornee: BasePart?,
	_active: boolean,
	CFrameChanged: typeof(Signal.new()),
}, ConstraintDragger))

function ConstraintDragger.new(plugin: Plugin): Dragger
	local self = setmetatable({
		_handles = nil,
		_adornee = nil,
		_active = false,
		_initialCFrame = CFrame.identity,
		CFrameChanged = Signal.new(),
	}, ConstraintDragger) :: any

	return self
end

function ConstraintDragger:show(cframe: CFrame)
	self:hide()

	-- Create a tiny invisible part as adornee for the handles
	local adornee = Instance.new("Part")
	adornee.Name = "RoMotion_DragAdornee"
	adornee.Size = Vector3.new(0.1, 0.1, 0.1)
	adornee.CFrame = cframe
	adornee.Anchored = true
	adornee.CanCollide = false
	adornee.CanQuery = false
	adornee.CanTouch = false
	adornee.Transparency = 1
	adornee.CastShadow = false
	adornee.Parent = workspace

	local handles = Instance.new("Handles")
	handles.Adornee = adornee
	handles.Style = Enum.HandlesStyle.Movement
	handles.Color3 = Color3.fromRGB(255, 200, 50)
	handles.Parent = adornee

	handles.MouseDrag:Connect(function(face: Enum.NormalId, distance: number)
		local axis = Vector3.FromNormalId(face)
		adornee.CFrame = adornee.CFrame + axis * distance
		self.CFrameChanged:Fire(adornee.CFrame)
	end)

	self._handles = handles
	self._adornee = adornee
	self._active = true
	self._initialCFrame = cframe
end

function ConstraintDragger:hide()
	if self._handles then
		self._handles:Destroy()
		self._handles = nil
	end
	if self._adornee then
		self._adornee:Destroy()
		self._adornee = nil
	end
	self._active = false
end

function ConstraintDragger:isActive(): boolean
	return self._active
end

function ConstraintDragger:getCFrame(): CFrame
	if self._adornee then
		return self._adornee.CFrame
	end
	return CFrame.identity
end

function ConstraintDragger:setCFrame(cframe: CFrame)
	if self._adornee then
		self._adornee.CFrame = cframe
	end
end

function ConstraintDragger:destroy()
	self:hide()
	self.CFrameChanged:Destroy()
end

return ConstraintDragger
