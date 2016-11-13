require 'midiparse'
require 'lfs'
require 'rnn'
require 'optim'
require 'xlua'
json = require 'json'

cmd = torch.CmdLine()
cmd:option('-d', 'data', 'Dataset directory')
cmd:option('-datasize', 0, 'Size of dataset (for benchmarking)')
cmd:option('-o', '', 'Model filename')
cmd:option('-ep', 1, 'Number of epochs')
cmd:option('-batchsize', 256, 'Batch Size')
cmd:option('-rho', 50, 'Rho value')
cmd:option('-recurrenttype', 'FastLSTM', 'Type of recurrent layer (FastLSTM, LSTM, GRU)')
cmd:option('-recurrentlayers', 1, 'Number of recurrent layers')
cmd:option('-denselayers', 1, 'Number of dense layers')
cmd:option('-hiddensizes', '100,100', 'Sizes of hidden layers, seperated by commas')
cmd:option('-dropout', 0.5, 'Dropout probability')
cmd:option('-lr', 0.01, 'Learning rate')
cmd:option('-lrdecay', 1e-5, 'Learning rate decay')
cmd:option('-cpu', false, 'Use CPU')
opt = cmd:parse(arg or {})

opt.opencl = not opt.cpu

local h = opt.hiddensizes
opt.hiddensizes = {}
while true do
	if h:len() == 0 then break end
	local c = h:find(',') or h:len()+1
	local str = h:sub(1, c-1)
	h = h:sub(c+1, h:len())
	opt.hiddensizes[#opt.hiddensizes+1] = tonumber(str)
end
if #opt.hiddensizes ~= opt.recurrentlayers+opt.denselayers then
	assert(false, "Number of hiddensizes is not equal to number of layers")
end

data_width = 93
curr_ep = 1
start_index = 1

totloss = 0
batches = 0

meta = {batchsize=opt.batchsize, 
        rho=opt.rho, 
		recurrenttype=opt.recurrenttype,
		recurrentlayers=opt.recurrentlayers, 
		denselayers=opt.denselayers, 
		hiddensizes=opt.hiddensizes,
		dropout=opt.dropout,
		lr=opt.lr,
		lrdecay=opt.lrdecay,
		dataset=opt.d}

if opt.opencl then
	require 'cltorch'
	require 'clnn'
else
	require 'torch'
	require 'nn'
end

-- Min-Maxed logarithms for data with long tail
-- x_n = (ln(x+1)-ln(x_min)) / (ln(x_max)-ln(m_min))
function normalize_col(r, col)
	local min = 99990
	local max = 0
	for i=1, #r do
		for u=1, #r[i] do
			--Clamp max dt to 4s
			if r[i][u][col] > 4000 then r[i][u][col] = 4000 end
			r[i][u][col] = math.log(r[i][u][col]+1)-- +1 to prevent ln(0)
			local val = r[i][u][col]
			if min > val then min = val end
			if max < val then max = val end
		end
	end

	for i=1, #r do
		for u=1, #r[i] do
			r[i][u][col] = (r[i][u][col] - min)/(max - min)
		end
	end

	meta[col..'min'] = min
	meta[col..'max'] = max

	return r
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
		print("Epoch "..curr_ep.." loss=", totloss/batches)
		curr_ep=curr_ep+1
		if logger then
			logger:add{curr_ep, totloss/batches}
		end
		if(curr_ep % 10 == 0 and opt.o ~= '') then torch.save(opt.o, model) end --Autosave
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

	local optim_cfg = {learningRate=opt.lr, learningRateDecay=opt.lrdecay}

	for e = 1, math.floor(opt.ep*totlen/opt.batchsize)-opt.batchsize do
		xlua.progress(e, math.floor(opt.ep*totlen/opt.batchsize)-opt.batchsize)
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
	local l = 1
	
	if opt.recurrenttype == 'FastLSTM' then recurrent = nn.FastLSTM
	elseif opt.recurrenttype == 'LSTM' then recurrent = nn.LSTM
	elseif opt.recurrenttype == 'GRU'  then recurrent = nn.GRU
	else assert(false, "Error: Invalid recurrent type") end

	--Recurrent input layer
	rnn:add(recurrent(data_width, opt.hiddensizes[l], opt.rho))
	rnn:add(nn.SoftSign())
	for i=1, opt.recurrentlayers-1 do
		l = l + 1
		rnn:add(nn.Dropout(opt.dropout))
		rnn:add(recurrent(opt.hiddensizes[l-1], opt.hiddensizes[l], opt.rho))
		rnn:add(nn.SoftSign())
	end
	model:add(nn.SplitTable(1,2))
	model:add(nn.Sequencer(rnn))
	model:add(nn.SelectTable(-1))
	--Dense layers
	for i=1, opt.denselayers do
		l = l + 1
		model:add(nn.Dropout(opt.dropout))
		model:add(nn.Linear(opt.hiddensizes[l-1], opt.hiddensizes[l]))
		model:add(nn.SoftSign())
	end
	--Output layer
	model:add(nn.Dropout(opt.dropout))
	model:add(nn.Linear(opt.hiddensizes[l], data_width))
	model:add(nn.Sigmoid())

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

data = normalize_col(data, 92)
data = normalize_col(data, 93)

if opt.datasize ~= 0 then
	local l = #data
	for i=opt.datasize, l do
		data[i] = nil
	end
end

totlen = get_total_len(data)

if opt.log ~= '' then
	logger = optim.Logger(opt.o..".log")
	logger:setNames{'epoch', 'loss'}
else print("WARNING: No output file!") end --To prevent future fuckups

train()

if opt.o ~= '' then
	torch.save(opt.o, model)
	local file = assert(io.open(opt.o..".meta", 'w'))
	file:write(json.encode(meta))
	file:close()
end
