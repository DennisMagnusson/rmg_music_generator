require 'midiparse'
require 'lfs'

local major = {0, 2, 4, 5, 7, 9, 11}
local minor = {0, 2, 3, 5, 7, 8, 10}

function validate(model, rho, dir, criterion)
	local data = create_data(rho, dir)
	local toterr = 0
	for i=1, #data[1] do
		local x = data[1][i]
		local y = data[2][i]

		local pred = model:forward(x)
		local err = criterion:forward(pred, y)
		toterr = torerr + err
	end

	return toterr / #data[1]
end

function create_frame(index)
	local frame = {}
	for i=1, 93 do
		frame[i] = 0
	end

	frame[index] = 1
	frame[91] = 0.7
	frame[92] = math.log(201)/8.29
	frame[93] = math.log(150)/8.29

	return frame
end

function create_data(rho, dir)
	local x = {}
	local y = {}
	--Scales
	for i=40, 52 do
		for k=1, 8 do
			--Major scale
			x[#x+1] = torch.zeros(rho, 93)
			for f=1, #major-1 do
				local frame = create_frame(i+major[f])
				for q=1, 93 do
					x[#x][rho-f][q] = frame[q]
				end
			end
			y[#y+1] = torch.Tensor(create_frame(i+major[#major]))

			--Minor scale
			x[#x+1] = torch.zeros(rho, 93)
			for f=1, #minor-1 do
				local frame = create_frame(i+minor[f])
				for q=1, 93 do
					x[#x][rho-f][q] = frame[q]
				end
			end
			y[#y] = torch.Tensor(create_frame(i+minor[#minor]))
		end
	end

	--Actual songs
	local songs = {}
	for filename in lfs.dir(dir.."/.") do
		if filename[1] == '.' then goto cont end
		local song = parse(dir.."/"..filename) 
		::cont::
	end
	--Should work
	for _, songs in pairs(songs) do
		for i=1, #song-rho-1 do
			x[#x+1] = torch.zeros(rho, 93)
			for k=i, i+rho do
				x[#x][k-i+1] = torch.Tensor(song[k-i+1])
			end
			y[#y+1] = song[i+1]
		end
	end

	return {x, y}
end
