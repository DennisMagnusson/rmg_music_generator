require 'midiparse'
require 'lfs'
require 'rnn'
require 'optim'
require 'xlua'

cmd = torch.CmdLine()
cmd:option('-d', 'data', 'Dataset directory')
cmd:option('-o', '', 'Model filename')
cmd:option('-ep', 1, 'Number of epochs')
cmd:option('-batchsize', 256, 'Batch Size')
cmd:option('-rho', 50, 'Rho value')
cmd:option('-recurrentlayers', 1, 'Number of recurrent layers')
cmd:option('-denselayers', 1, 'Number of dense layers')
cmd:option('-hiddensize', 256, 'Size of hidden layers')
cmd:option('-dropout', 0.5, 'Dropout probability')
cmd:option('-lr', 0.01, 'Learning rate')
cmd:option('-opencl', true, 'Use OpenCL')
cmd:option('-log', '', 'Log file')
opt = cmd:parse(arg or {})


data_width = 88
curr_ep = 1
start_index = 1

totloss = 0
batches = 0

if opt.log ~= '' then
	logger = optim.Logger(opt.log)
	logger:setNames{'epoch', 'loss'}
end

if opt.opencl then
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

function next_batch()
	start_index = start_index + opt.batchsize
	if start_index >= totlen-opt.batchsize-1 then
		start_index = 1
		print("Epoch "..curr_ep.." done")
		curr_ep=curr_ep+1
		print("loss", totloss/batches)
		if logger then
			logger:add{curr_ep, totloss/batches}
		end
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

	local optim_cfg = {learningRate=opt.lr}

	for e = 1, math.floor(opt.ep*totlen/opt.batchsize) do
		xlua.progress(e, math.floor(opt.ep*totlen/opt.batchsize))
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
		if #s > i+1+opt.batchsize+opt.rho then
			song = s
			break
		else 
			i = i - #s
		end
		if i < 1 then i = 1 end
	end
	--Create batch
	local x = torch.zeros(opt.batchsize, opt.rho, data_width)
	local y = torch.zeros(opt.batchsize, data_width)

	for u = 1, opt.batchsize do
		for o = opt.rho, 1, -1 do
			x[u][o] = torch.Tensor(song[i+o+u])
		end
		y[u] = torch.Tensor(song[i+u+opt.rho+1])
	end

	if opt.opencl then
		x = x:cl()
		y = y:cl()
	end

	return {x, y}
end

function create_model()
	local model = nn.Sequential()

	local rnn = nn.Sequential()
	rnn:add(nn.FastLSTM(data_width, opt.hiddensize, opt.rho))
	for i=1, opt.recurrentlayers-1 do
		rnn:add(nn.Dropout(opt.dropout))
		rnn:add(nn.FastLSTM(opt.hiddensize, opt.hiddensize, opt.rho))
	end
	model:add(nn.SplitTable(1,2))
	model:add(nn.Sequencer(rnn))
	model:add(nn.SelectTable(-1))
	for i=1, opt.denselayers-1 do
		model:add(nn.Dropout(opt.dropout))
		model:add(nn.Linear(opt.hiddensize, opt.hiddensize))
		model:add(nn.ReLU())
	end
	model:add(nn.Linear(opt.hiddensize, data_width))
	model:add(nn.ReLU())

	if opt.opencl then 
		return model:cl()
	else 
		return model 
	end
end


model = create_model()
params, gradparams = model:getParameters()
criterion = nn.MSECriterion(true)
if opt.opencl then criterion:cl() end
data = create_dataset(opt.d)
totlen = get_total_len(data)
train()
if opt.o ~= '' then
	torch.save(opt.o, model)
end
