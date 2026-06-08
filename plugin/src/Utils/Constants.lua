--!strict
-- RoMotion constants

local Constants = {}

Constants.PLUGIN_NAME = "RoMotion"
Constants.PLUGIN_ID = "RoMotion_AnimGen"
Constants.WIDGET_ID = "RoMotion_MainWidget"

Constants.SERVER_URL = "http://localhost:8787"
Constants.POLL_INTERVAL = 2.0 -- seconds between status polls

Constants.FPS = 30
Constants.DEFAULT_DURATION = 3.0

Constants.EFFECTORS = {
	"LeftHand",
	"RightHand",
	"LeftFoot",
	"RightFoot",
	"Hips",
}

Constants.EFFECTOR_COLORS = {
	LeftHand = Color3.fromRGB(66, 165, 245),   -- blue
	RightHand = Color3.fromRGB(239, 83, 80),   -- red
	LeftFoot = Color3.fromRGB(102, 187, 106),  -- green
	RightFoot = Color3.fromRGB(255, 167, 38),  -- orange
	Hips = Color3.fromRGB(171, 71, 188),       -- purple
}

Constants.PROMPT_COLORS = {
	Color3.fromRGB(76, 175, 80),    -- green
	Color3.fromRGB(33, 150, 243),   -- blue
	Color3.fromRGB(255, 152, 0),    -- orange
	Color3.fromRGB(156, 39, 176),   -- purple
	Color3.fromRGB(0, 188, 212),    -- cyan
	Color3.fromRGB(244, 67, 54),    -- red
}

Constants.TIMELINE_HEIGHT = 200
Constants.RULER_HEIGHT = 24
Constants.GUTTER = 86 -- left margin for effector labels/+buttons; time=0 starts here
Constants.PROMPT_TRACK_HEIGHT = 36
Constants.CONSTRAINT_TRACK_HEIGHT = 24
Constants.PLAYHEAD_COLOR = Color3.fromRGB(255, 82, 82)

Constants.SAVE_FOLDER_NAME = "RBX_ANIMSAVES"

return Constants
