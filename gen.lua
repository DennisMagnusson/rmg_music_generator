local midi = require 'MIDI'

function generate(pattern, ...)
	local arg = table.pack(...)
	local score = {1, {}}
	local time = 0
	for k, event in pairs(pattern) do
		local tones = {}
		for n, i in pairs(event) do
			if i ~= 0 then tones[#tones+1] = n end
		end

		for i, tone in pairs(tones) do
			score[2][#score[2]+1] = {'note', time, 1, 1, tone+20, 127}
		end
		time = time+1

	end
	print(arg[1])
	if arg[1] then
		local file = assert(io.open("generated/" .. arg[1], 'w'))
		file:write(midi.score2midi(score))
		file:close()
	end

end
