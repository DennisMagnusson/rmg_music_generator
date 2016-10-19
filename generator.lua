require 'midigen'
require 'torch'
require 'nn'
require 'rnn'
json = require 'json'

cmd = torch.CmdLine()
cmd:option('-o', '', 'Output file name')
cmd:option('-model', '', 'Model file name')
cmd:option('-temperature', 1.0, 'Temperature')
cmd:option('-firstnote', 41, 'First note index 1-88')
cmd:option('-len', 100, 'Length of the notes')
cmd:option('-k', 1.5, 'k-value')
opt = cmd:parse(arg or {})

function denormalize_col(r, col)
	local min = meta[col..'min']
	local max = meta[col..'max']
	for i=1, #r do
		r[i][col] = r[i][col]*(max-min)-min
	end
	return r
end

function create_song()
	local song = torch.Tensor(opt.len, data_width)
	local x = torch.zeros(rho, data_width)
	x[rho][opt.firstnote] = 1
	local frame = torch.zeros(data_width)
	for i=1, opt.len do
		for u=2, rho do
			x[u-1] = x[u]
		end
		x[rho] = frame

		local pd = model:forward(x)--Probability distribution... thing
		pd = pd:reshape(data_width)
		frame = sample(pd)

		song[i] = frame
	end
	local t = torch.totable(song)
	t = denormalizecol(r, 92, mint, maxt)
	t = denormalizecol(r, 93, mind, maxd)

	if opt.o ~= '' then 
		generate(torch.totable(song), opt.o)
	else 
		print(get_notes(song)) 
	end
end

function sample(r)
	r = torch.exp(torch.log(r) / opt.temperature)
	r = r / torch.sum(r)
	r = r*(opt.k / torch.sum(r)) --Make the sum of r = k

	local frame = torch.zeros(data_width)
	for i = 1, data_width do
		local rand = math.random()
		if r[i] > rand then frame[i] = 1 end
	end
	return frame
end

function get_notes(r)
	local notes = {}
	for i=1, opt.len  do
		notes[i] = {}
		for u=1, data_width do
			if r[i][u] ~= 0 then
				notes[i][#notes[i]+1] = u
			end
		end
	end
	return notes
end

model = torch.load(opt.model)
data_width = model:get(1).inputSize
rho = model:get(1).rho
file = assert(io.open(opt.model..".meta", 'r'))
str = file:read('*all')
meta = json.decode(str)
create_song()
