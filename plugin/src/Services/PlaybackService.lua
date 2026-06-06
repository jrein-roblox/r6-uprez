--!strict
-- PlaybackService: loads CurveAnimation from backend, plays via Animator.

local RunService = game:GetService("RunService")
local AnimationClipProvider = game:GetService("AnimationClipProvider")

local Signal = require(script.Parent.Parent.Signal)
local RigService = require(script.Parent.RigService)

local PlaybackService = {}
PlaybackService.__index = PlaybackService

export type PlaybackState = "stopped" | "playing" | "paused"

function PlaybackService.new(rig: RigService.RigInfo)
	local self = setmetatable({
		_rig = rig,
		_animation = nil :: Animation?,
		_track = nil :: AnimationTrack?,
		_curveAnimation = nil :: Instance?,
		_state = "stopped" :: PlaybackState,
		_duration = 0,
		_lastStep = 0,
		_renderConn = nil :: RBXScriptConnection?,
		StateChanged = Signal.new(),
		TimeChanged = Signal.new(),
	}, PlaybackService)
	return self
end

function PlaybackService:loadFromCurveAnimation(curveAnim: CurveAnimation): boolean
	self:stop()

	-- Clean up previous
	if self._curveAnimation then
		self._curveAnimation:Destroy()
		self._curveAnimation = nil
	end

	curveAnim.Parent = self._rig.model
	self._curveAnimation = curveAnim

	-- Register with AnimationClipProvider to get a local content ID,
	-- then load via Animator to get an AnimationTrack
	local ok, result = pcall(function()
		local clipId = AnimationClipProvider:RegisterAnimationClip(curveAnim)
		local anim = Instance.new("Animation")
		anim.AnimationId = clipId
		return self._rig.animator:LoadAnimation(anim)
	end)

	if not ok or not result then
		warn("[RoMotion] Failed to load CurveAnimation:", result)
		return false
	end

	self._track = result
	self._duration = result.Length
	return true
end

function PlaybackService:loadFromInstance(curveAnimation: Instance)
	self:stop()

	if self._curveAnimation then
		self._curveAnimation:Destroy()
		self._curveAnimation = nil
	end

	self._curveAnimation = curveAnimation
	curveAnimation.Parent = self._rig.model

	local track = self._rig.animator:LoadAnimation(curveAnimation :: any)
	self._track = track
	self._duration = track.Length
end

function PlaybackService:play()
	if not self._track then
		return
	end
	if self._state == "paused" then
		self._track:AdjustSpeed(1)
	else
		self._track:Play(0, 1, 1)
	end
	self._state = "playing"
	self.StateChanged:Fire("playing")

	if self._renderConn then
		self._renderConn:Disconnect()
	end
	self._lastStep = os.clock()
	self._renderConn = RunService.Heartbeat:Connect(function()
		if self._track and self._state == "playing" then
			local now = os.clock()
			local dt = now - (self._lastStep or now)
			self._lastStep = now
			-- Manually step the animator in edit mode
			self._rig.animator:StepAnimations(dt)
			self.TimeChanged:Fire(self._track.TimePosition)
		end
	end)
end

function PlaybackService:pause()
	if not self._track or self._state ~= "playing" then
		return
	end
	self._track:AdjustSpeed(0)
	self._state = "paused"
	self.StateChanged:Fire("paused")

	if self._renderConn then
		self._renderConn:Disconnect()
		self._renderConn = nil
	end
end

function PlaybackService:stop()
	if self._track then
		self._track:Stop(0)
	end
	self._state = "stopped"
	self.StateChanged:Fire("stopped")

	if self._renderConn then
		self._renderConn:Disconnect()
		self._renderConn = nil
	end

	-- Reset all Motor6D transforms
	for _, motor in self._rig.motors do
		motor.Transform = CFrame.identity
	end
	self.TimeChanged:Fire(0)
end

function PlaybackService:ensureStepped()
	if not self._track then return end
	-- Make sure the track is active so AnimationConstraint.Transform is populated
	if self._state == "stopped" then
		self._track:Play(0, 1, 0)
		self._state = "paused"
	end
	self._rig.animator:StepAnimations(0)
end

function PlaybackService:seekTo(time: number)
	if not self._track then
		return
	end
	-- Ensure the track is loaded and can be scrubbed even when "stopped"
	if self._state == "stopped" then
		self._track:Play(0, 1, 0) -- play at speed 0 (paused but active)
		self._state = "paused"
	elseif self._state == "playing" then
		self._track:AdjustSpeed(0)
		self._state = "paused"
	end
	self._track.TimePosition = math.clamp(time, 0, self._duration)
	self._rig.animator:StepAnimations(0)
	self.TimeChanged:Fire(self._track.TimePosition)
end

function PlaybackService:getDuration(): number
	return self._duration
end

function PlaybackService:getState(): PlaybackState
	return self._state
end

function PlaybackService:getCurrentTime(): number
	if self._track then
		return self._track.TimePosition
	end
	return 0
end

function PlaybackService:destroy()
	self:stop()
	if self._curveAnimation then
		self._curveAnimation:Destroy()
	end
	self.StateChanged:Destroy()
	self.TimeChanged:Destroy()
end

return PlaybackService
