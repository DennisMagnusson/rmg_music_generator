local midi = require 'MIDI'

function parse(filename)
	local file = assert(io.open(filename, 'r'))
	local m = midi.midi2ms_score(file:read("*all"))
	file:close()
	file = nil
	local r = {}
	--Remove all non-notes
	local notes = {}
	for k, event in pairs(m[2]) do
		if event[1] == 'note' then notes[#notes+1] = event end
	end

	notes = sort_by_time(notes)

	local i = 0
	for k, event in pairs(notes) do
		if #r > 1 and event[2]-notes[k-1][2] == 0 then
			--tone
			r[#r][event[5]-20] = 1
			--velocity
			if event[6]/127 > r[#r][89] then r[#r][89] = event[6]/127 end
			--duration
			if event[3] > r[#r][91] then r[#r][91] = event[3] end
			i = i+1
			goto EOL
		end

		--Fill frame with zeros
		local frame = {}
		for i = 1, 88, 1 do frame[i] = 0 end
		--tone
		frame[event[5]-20] = 1
		--velocity
		frame[89] = event[6]/127
		--delta start_time
		if #r <= 1 then 
			frame[90] = 0
		else
			frame[90] = event[2] - notes[k-1][2]
		end
		--duration
		frame[91] = event[3]
		--Insert into r
		r[#r+1] = frame
		i = i+1
		::EOL::
	end
	return r
end

function sort_by_time(r)
	local s = {}
	s[1] = r[1]
	for i=2, #r do
		local l = #s
		for u=1, #s do
			if r[i][2] < s[u][2] then 
				table.insert(s, u, r[i])
				goto cont
			end
		end
		if s[l+1] == nil then s[l+1] = r[i] end
		::cont::
	end
	return s
end
