local midi = require 'MIDI'

function parse(filename)
	local file = assert(io.open(filename, 'r'))
	local m = midi.midi2score(file:read("*all"))
	file:close()
	file = nil
	local r = {}
	--Remove all non-notes
	local notes = {}
	for k, event in pairs(m[2]) do
		if event[1] == 'note' then notes[#notes+1] = event end
	end

	local i = 0
	for k, event in pairs(notes) do
		print(i, #r)
		--If the times are the same
		if (i > 1 and event[2] == notes[i-1][2]) then
			r[#r-1][event[5]-20] = 1
			i = i+1
			goto EOL
		end

		--Fill frame with zeros
		local frame = {}
		for i = 1, 88, 1 do frame[i] = 0 end
		frame[event[5]-20] = 1
		r[#r+1] = frame
		i = i+1
		::EOL::
	end
	
	return r
end

function print_index(r)
	for key, frame in pairs(r) do
		for n, i in pairs(frame) do
			if i ~= 0 then io.write(n.." ") end
		end
		print("")
	end
end
