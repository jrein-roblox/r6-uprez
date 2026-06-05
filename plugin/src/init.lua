--!strict
-- RoMotion plugin library entry point

return {
	Signal = require(script.Signal),
	State = require(script.State),
	Constants = require(script.Utils.Constants),
	TimelineLayout = require(script.Utils.TimelineLayout),
	BackendService = require(script.Services.BackendService),
	RigService = require(script.Services.RigService),
	PlaybackService = require(script.Services.PlaybackService),
	DataModelService = require(script.Services.DataModelService),
}
