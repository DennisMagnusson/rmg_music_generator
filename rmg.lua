require 'parse'
require 'lfs'
require 'rnn'
require 'optim'

data_width = 88
rho = 50
lr = 0.01
hiddensize = 256 
batch_size = 350
ep = 32
curr_ep = 1
start_index = 1

totloss = 0
batches = 0

opencl = true
logger = optim.Logger('log.log')
logger:setNames{'epoch', 'loss'}

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

function fit(model, criterion, batch)
	local x = batch[1]
	local y = batch[2]

	local yhat = model:forward(x)
	local loss = criterion:forward(yhat, y)
	totloss = totloss + loss

	local gradcrit = criterion:backward(yhat, y)

	model:zeroGradParameters()
	model:backward(x, gradcrit)
	model:updateParameters(lr)

	return totloss
end


function next_batch()
	start_index = start_index + batch_size
	if start_index >= totlen-batch_size-1 then
		start_index = 1
		print("Epoch "..curr_ep.." done")
		curr_ep=curr_ep+1
		print("loss", totloss/batches)
		logger:add{curr_ep, totloss/batches}
		totloss = 0
		batches = 0
	end

	return create_batch(data, start_index)
end

function feval(p)
	if params ~= p then
		params:copy(p)
	end

	batch = next_batch()
	local x = batch[1]
	local y = batch[2]

	gradparams:zero()
	local yhat = model:forward(x)
	local loss = criterion:forward(yhat, y)
	totloss = totloss + loss
	model:backward(x, criterion:backward(yhat, y))

	return loss, gradparams
end

function train()
	math.randomseed(os.time())
	model:training()--Training mode

	local optim_cfg = {learningRate=lr}

	for e = 1, math.floor(ep*totlen/batch_size) do
		if e % 250 == 0 then print(e) end
		batches = batches + 1
		optim.adagrad(feval, params, optim_cfg)
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
		y[u] = torch.Tensor(song[i+u+rho+1])
	end

	if opencl then
		x = x:cl()
		y = y:cl()
	end

	return {x, y}
end

function create_model()
	local dropoutprob = 0.5
	local model = nn.Sequential()

	local rnn = nn.Sequential()
	rnn:add(nn.FastLSTM(data_width, hiddensize, rho))
	rnn:add(nn.Dropout(0.5))
	rnn:add(nn.FastLSTM(hiddensize, hiddensize, rho))

	model:add(nn.SplitTable(1,2))
	model:add(nn.Sequencer(rnn))
	model:add(nn.SelectTable(-1))
	model:add(nn.Linear(hiddensize, hiddensize))
	model:add(nn.Dropout(dropoutprob))
	model:add(nn.Linear(hiddensize, data_width))
	model:add(nn.ReLU())
	model:add(nn.Dropout(dropoutprob))

	if opencl then 
		return model:cl()
	else 
		return model 
	end
end


model = create_model()
params, gradparams = model:getParameters()
criterion = nn.MSECriterion(true)
if opencl then criterion:cl() end
data = create_dataset("data")
totlen = get_total_len(data)
--for i = 200, #data[1] do
	--table.remove(data[1], i)
--end
train()
