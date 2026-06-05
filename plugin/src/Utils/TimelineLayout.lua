--!strict
-- Time ↔ pixel conversion utilities for the timeline.

local TimelineLayout = {}

function TimelineLayout.timeToPixel(
	time: number,
	scrollOffset: number,
	pixelsPerSecond: number
): number
	return (time - scrollOffset) * pixelsPerSecond
end

function TimelineLayout.pixelToTime(
	pixel: number,
	scrollOffset: number,
	pixelsPerSecond: number
): number
	return pixel / pixelsPerSecond + scrollOffset
end

function TimelineLayout.snapToFrame(time: number, fps: number): number
	return math.floor(time * fps + 0.5) / fps
end

function TimelineLayout.getTickInterval(pixelsPerSecond: number): number
	-- Choose tick spacing so labels don't overlap
	local minPixelsBetweenTicks = 60
	local candidates = { 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0 }
	for _, interval in candidates do
		if interval * pixelsPerSecond >= minPixelsBetweenTicks then
			return interval
		end
	end
	return 10.0
end

return TimelineLayout
