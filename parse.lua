midi = require 'MIDI'

function parse(filename)
	local file = io.open(filename, "r")
	local m = midi:midi2score(file:read(*a))
	
	r = {}
	--TODO Add something for multiple notes at same time	
	for k, event in pairs(m) do
		local frame = {}
		--Fill frame with zeros
		for i = 1, 88 1 do frame[i] = 0 end
		frame[event['note']-20] = 1
		r[#r+1] = 1
	end
	
	return r

end
