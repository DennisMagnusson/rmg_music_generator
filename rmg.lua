require 'parse'
require 'gen'--TODO Do something with this
require 'lfs'
require 'rnn'
require 'optim'

data_width = 88
rho = 50
lr = 0.01
hiddensize = 512
batch_size = 128

opencl = true

if opencl then
	require 'cltorch'
	require 'clnn'
else
	require 'torch'
	require 'nn'
end


function create_dataset(dir)
	local d = {}
	for filename in lfs.dir(dir.."/.") do
		if filename[1] == '.' then goto cont end
		local song = parse(dir.."/"..filename)
		if #song > 2 then
			d[#d+1] = song
		end
		::cont::
	end
	return d
end

--TODO test
function create_song(model, firstnote, len, temp)
	firstnote = firstnote or 43
	len = len or 100
	temp = temp or 0.8
	local song = torch.Tensor(len, data_width)
	local x = torch.zeros(rho, data_width)
	x[firstnote] = 1
	local frame = torch.zeros(data_width)
	for i=1, len do
		for u=2, rho do
			x[u-1] = x[u]
		end
		x[rho] = frame

		local pd = model:forward(x)--Probability distribution... thing
		pd = pd:reshape(data_width)
		frame = sample(pd, temp)

		song[i] = frame
	end
	print("Done")

	return torch.totable(song)
end


--Kind of... empty arrays
--Gotta fix the model or this function FIXME
function sample(r, temp)
	r = torch.exp(torch.log(r) / temp)
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

function fit(model, criterion, batch)
	local totloss = 0

	local x = batch[1]
	local y = batch[2]
	local pred = model:forward(x)
	local err = criterion:forward(pred, y)
	totloss = totloss + err
	local gradcrit = criterion:backward(pred, y) 
	model:zeroGradParameters()
	model:backward(x, gradcrit)
	model:updateParameters(lr)

	return totloss
end

function train(model, data, ep)
	math.randomseed(os.time())
	model:training()--Training mode
	local criterion = nn.MSECriterion()
	if opencl then criterion = criterion:cl() end
	
	local totlen = get_total_len(data)
	for e = 1, ep do
		print("Epoch: "..e)
		local totloss = 0
		for i = 1, totlen-batch_size, batch_size do
			io.write("\r"..i.."/"..(totlen-batch_size))
			local batch = create_batch(data, i, rho)
			totloss = totloss + fit(model, criterion, batch)
		end
		print("\rAvg loss", totloss / (totlen-rho-batch_size))
	end

	model:evaluate() --Exit training mode
end

function get_total_len(data)
	local i = 0
	for k, s in pairs(data) do
		i = i + #s
	end
	return i
end

--TODO Lots of improvements to be made
function create_batch(data, start_index)
	local i = start_index
	local song = {}
	--Select the correct song
	for k, s in pairs(data) do
		if #s > i+1+batch_size+rho then
			song = s
			break
		else 
			i = i - #s
		end
		if i < 1 then i = 1 end
	end
	--Create batch
	local x = torch.zeros(batch_size, rho, data_width)
	local y = torch.zeros(batch_size, data_width)

	for u = 1, batch_size do
		for o = rho, 1, -1 do
			x[u][o] = torch.Tensor(song[i+o+u])
		end
		if not song[i+u+rho+1] then print(#song, i, u) end
		y[u] = torch.Tensor(song[i+u+rho+1])
	end

	if opencl then
		x = x:cl()
		y = y:cl()
	end

	return {x, y}
end

function create_model()
	local dropoutprob = 0.2
	local model = nn.Sequential()
	local rnn = nn.Sequential()
	rnn:add(nn.FastLSTM(data_width, hiddensize, rho))
	rnn:add(nn.FastLSTM(hiddensize, hiddensize, rho))
	model:add(nn.SplitTable(1,2))
	model:add(nn.Sequencer(rnn))
	model:add(nn.SelectTable(-1))
	model:add(nn.Linear(hiddensize, hiddensize))
	model:add(nn.ReLU())
	model:add(nn.Linear(hiddensize, data_width))
	model:add(nn.ReLU())
	if opencl then model = model:cl() end
	return model
end


model = create_model()
data = create_dataset("data")
data = {data[1]}
--for i = 1000, #data[1] do
	--table.remove(data[1], i)
--end
train(model, data, 32)
