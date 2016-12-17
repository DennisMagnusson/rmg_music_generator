require 'midiparse'
require 'lfs'
require 'rnn'
require 'optim'

require 'cltorch'
require 'clnn'

local major = {0, 2, 4, 5, 7, 9, 11}
local minor = {0, 2, 3, 5, 7, 8, 10}

local valid_data = {}

function validate(model, rho, batchsize,dir, criterion)
	if not valid_data[1] then valid_data = create_data(dir) end

	local toterr = 0
	local c = 0
	local bs = 1

	local x = torch.Tensor(batchsize, rho, 93)
	local y = torch.Tensor(batchsize, 93)

	for _, song in pairs(valid_data) do
		for i=1, #song-rho-1 do

			for o=rho, 1, -1 do
				x[bs][o] = song[i+o]
			end
			y[bs] = torch.Tensor(song[rho+i+1])
			bs = bs+1

			if bs == batchsize then
				local pred = model:forward(x:cl())
				local err = criterion:forward(pred, y:cl())
				toterr = toterr + err
				x = torch.Tensor(batchsize, rho, 93)
				y = torch.Tensor(batchsize, 93)
				c = c+1
				bs = 0
			end
			
		end
	end
	
	return toterr / c
end

function normalize(r, col)
	for i=1, #r do
		for u=1, #r[i] do
			if r[i][u][col] > 4000 then r[i][u][col] = 4000 end
			r[i][u][col] = math.log(r[i][u][col]+1) / 8.294
		end
	end
	return r
end

function create_data(dir)
	local songs = {}
	for filename in lfs.dir(dir.."/.") do
		if filename[1] == '.' then goto cont end
		local song = parse(dir.."/"..filename) 
		songs[#songs+1] = song
		::cont::
	end
	songs = normalize(songs, 92)
	songs = normalize(songs, 93)
	return songs
end
