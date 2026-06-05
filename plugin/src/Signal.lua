--!strict
-- Lightweight signal/event class for reactive UI updates.

local Signal = {}
Signal.__index = Signal

export type Connection = {
	Disconnect: (self: Connection) -> (),
}

export type Signal<T...> = typeof(setmetatable({} :: {
	_listeners: { (T...) -> () },
}, Signal))

function Signal.new<T...>(): Signal<T...>
	local self = setmetatable({
		_listeners = {},
	}, Signal)
	return self :: any
end

function Signal:Connect(fn: (...any) -> ()): Connection
	table.insert(self._listeners, fn)
	local listeners = self._listeners
	local index = #listeners
	return {
		Disconnect = function(_self)
			for i, listener in listeners do
				if listener == fn then
					table.remove(listeners, i)
					break
				end
			end
		end,
	}
end

function Signal:Fire(...: any)
	for _, listener in self._listeners do
		task.spawn(listener, ...)
	end
end

function Signal:Once(fn: (...any) -> ()): Connection
	local conn: Connection
	conn = self:Connect(function(...)
		conn:Disconnect()
		fn(...)
	end)
	return conn
end

function Signal:Destroy()
	table.clear(self._listeners)
end

return Signal
