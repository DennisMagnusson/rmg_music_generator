require 'midigen'
require 'torch'
require 'nn'
require 'rnn'
json = require 'json'
require 'cltorch'

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
		r[i][col] = math.floor(math.exp(r[i][col]*(max-min)+min)-1)
	end
	return r
end

function create_song()
	local song = torch.Tensor(opt.len, data_width)
	local x = torch.zeros(meta['rho'], data_width)
	x[meta['rho']][opt.firstnote] = 1
	local frame = torch.zeros(data_width)
	for i=1, opt.len do
		for u=2, meta['rho'] do
			x[u-1] = x[u]
		end
		x[meta['rho']] = frame

		local pd = model:forward(x)--Probability distribution... thing
		pd = pd:reshape(data_width)
		frame = sample(pd)

		song[i] = torch.Tensor(frame)
	end
	local r = torch.totable(song)
	r = denormalize_col(r, 92)
	r = denormalize_col(r, 93)

	if opt.o ~= '' then
		generate(r, opt.o)
	else
		print(get_notes(song))
	end
end

function sample(frame)
	local r = torch.zeros(88)
	for i=1, 88 do
		r[i] = frame[i]
	end
	r = torch.exp(torch.log(r) / opt.temperature)
	r = r / torch.sum(r)
	r = r*(opt.k / torch.sum(r)) --Make the sum of r = k

	local empty = true
	for i = 1, 88 do
		local rand = math.random()
		if r[i] > rand then
			r[i] = 1
			empty = false
		else
			r[i] = 0
		end
	end
	if empty then return sample(frame) end

	for i=1, 88 do
		frame[i] = r[i]
	end
	--Pedal
	if math.random() > frame[89] then
		frame[89] = 1
		frame[90] = 0
	elseif math.random() > frame[90] then
		frame[89] = 0
		frame[90] = 1
	else
		frame[89] = 0
		frame[90] = 0
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

model = torch.load(opt.model):double()
data_width = 93
file = assert(io.open(opt.model..".meta", 'r'))
str = file:read('*all')
meta = json.decode(str)
create_song()
