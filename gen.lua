local midi = require 'MIDI'

function generate(pattern, [filename])
	local score = {1000}
	local time = 0
	for k, event in pattern do
		local tones = {}
		--TODO Make something with tones
		for i, tone in tones do
			score[2][#score[2]+1] = {'note', time, 1, tones+20, 127}
		end
		time = time+1
	end

	if filename then
		local file = assert(io.open(filename, 'w'))
		file:write(midi.score2midi(score))
		file:close()
	end
	

end
