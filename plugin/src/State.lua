--!strict
-- Observable state store. Subscribe to changes, get/set values.

local Signal = require(script.Parent.Signal)

local State = {}
State.__index = State

export type State<T> = typeof(setmetatable({} :: {
	_value: T,
	Changed: typeof(Signal.new()),
}, State))

function State.new<T>(initialValue: T): State<T>
	local self = setmetatable({
		_value = initialValue,
		Changed = Signal.new(),
	}, State)
	return self :: any
end

function State:get()
	return self._value
end

function State:set(newValue: any)
	if self._value == newValue then
		return
	end
	local old = self._value
	self._value = newValue
	self.Changed:Fire(newValue, old)
end

function State:subscribe(fn: (newValue: any, oldValue: any) -> ()): { Disconnect: () -> () }
	return self.Changed:Connect(fn)
end

function State:Destroy()
	self.Changed:Destroy()
end

return State
