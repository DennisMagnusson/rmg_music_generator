local midi = require 'MIDI'

function parse(filename)
	local file = assert(io.open(filename, 'r'))
	local m = midi.midi2score(file:read("*all"))
	file:close()
	file = nil
	r = {}
	--TODO Add something for multiple notes at same time	
	for k, event in pairs(m[2]) do
		if event[1] ~= 'note' then goto EOL end
		local frame = {}
		--If the times are the same FIXME NOT WORKING
		if k > 1 and (event[2] == m[2][k-1][2]) then
			frame = r[#r]--TODO test
			frame[event[5]-20] = 1--FIXME not working
			r[#r] = frame
			goto EOL
		end
		--Fill frame with zeros
		for i = 1, 88, 1 do frame[i] = 0 end
		frame[event[5]-20] = 1
		r[#r+1] = frame
		::EOL::
	end
	
	return r
end
