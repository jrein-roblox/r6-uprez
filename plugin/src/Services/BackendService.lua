--!strict
-- HTTP communication with the RoMotion backend server.

local HttpService = game:GetService("HttpService")

local Constants = require(script.Parent.Parent.Utils.Constants)

local BackendService = {}

local function url(path: string): string
	return Constants.SERVER_URL .. path
end

local function postJson(path: string, body: { [string]: any }): { [string]: any }
	local jsonBody = HttpService:JSONEncode(body)
	local response = HttpService:PostAsync(url(path), jsonBody, Enum.HttpContentType.ApplicationJson)
	return HttpService:JSONDecode(response)
end

local function getJson(path: string): { [string]: any }
	local response = HttpService:GetAsync(url(path))
	return HttpService:JSONDecode(response)
end

function BackendService.generate(request: {
	prompts: { { text: string, start_time: number, end_time: number } },
	constraints: { { effector: string, time: number, position: { number }, rotation: { number }? } }?,
	duration: number,
	looped: boolean?,
	seed: number?,
	cfg_weight: number?,
	diffusion_steps: number?,
}): { job_id: string, status: string }
	return postJson("/generate", request) :: any
end

function BackendService.getStatus(jobId: string): {
	job_id: string,
	status: string,
	progress: number,
	message: string,
	error: string?,
	result: { [string]: any }?,
}
	return getJson("/status/" .. jobId) :: any
end

function BackendService.getResult(jobId: string): { [string]: any }
	return getJson("/result/" .. jobId) :: any
end

function BackendService.autoConstraints(request: {
	job_id: string,
	effectors: { string }?,
	min_separation_frames: number?,
}): { constraints: { { effector: string, frame: number, time: number, position: { number } } } }
	return postJson("/auto-constraints", request) :: any
end

function BackendService.importClip(request: {
	asset_id: number,
	sample_fps: number?,
}): { job_id: string, status: string }
	return postJson("/import-clip", request) :: any
end

function BackendService.healthCheck(): boolean
	local ok, _ = pcall(function()
		getJson("/health")
	end)
	return ok
end

return BackendService
