require 'gen'
require 'rnn'--Is this needed?
require 'torch'

cmd = torch.CmdLine()
cmd:option('-o', nil, 'Output file name')
cmd:option('-model', nil, 'Model file name')
cmd:option('-temperature', 1.0, 'Temperature')
cmd:option('-firstnote', 41, 'First note index 1-88')
cmd:option('-len', 100, 'Length of the notes')
cmd:option('-opencl', true, 'OpenCL')

data_width = 88
rho = 50
lr = 0.01

--Needed? TODO Check
if opt.opencl then
	require 'cltorch'
	require 'clnn'
else
	require 'torch'
	require 'nn'
end

model = torch.load("models/"..opt.model)

--TODO test
function create_song()
	local song = torch.Tensor(opt.len, data_width)
	local x = torch.zeros(rho, data_width)
	x[opt.firstnote] = 1
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
	print("Done")

	if opt.o then gen.generate(torch.totable(song), opt.o) end

	return torch.totable(song)
end


--Kind of... empty arrays
--Gotta fix the model or this function FIXME
function sample(r)
	r = torch.exp(torch.log(r) / opt.temp)
	r = r / torch.sum(r)
	local k = 1.5
	r = r*(k / torch.sum(r)) --Make the sum of r = k

	local frame = torch.zeros(data_width)
	for i = 1, data_width do
		local rand = math.random()
		if r[i] > rand then frame[i] = 1 end
	end
	return frame
end

model = torch.load(opt.model)
create_song()
