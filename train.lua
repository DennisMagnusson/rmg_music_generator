require 'midiparse'
require 'validate'
require 'lfs'
require 'rnn'
require 'optim'
require 'xlua'
json = require 'json'

cmd = torch.CmdLine()
cmd:option('-d', 'data', 'Dataset directory')
cmd:option('-vd', 'slow2test', 'Validation data directory')
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
cmd:option('-weightdecay', 0, 'Weight decay')
opt = cmd:parse(arg or {})

opt.opencl = not opt.cpu

if opt.opencl then
	require 'cltorch'
	require 'clnn'
else
	require 'torch'
	require 'nn'
end

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
start_ep = 0
start_index = 1

totloss = 0
loss = 0
batches = 0

resume = false

prev_valid = 0


meta = {batchsize=opt.batchsize,
		rho=opt.rho,
		ep=opt.ep,
		recurrenttype=opt.recurrenttype,
		recurrentlayers=opt.recurrentlayers,
		denselayers=opt.denselayers,
		hiddensizes=opt.hiddensizes,
		dropout=opt.dropout,
		lr=opt.lr,
		lrdecay=opt.lrdecay,
		weightdecay=opt.weightdecay,
		dataset=opt.d,
		v_data=opt.vd}

-- Min-Maxed logarithms for data with long tail
-- x_n = (ln(x+1)-ln(x_min)) / (ln(x_max)-ln(m_min))
function normalize_col(r, col)
	local min = 99990
	local max = 0
	for i=1, #r do
		for u=1, r[i]:size()[1] do
			--Clamp max dt to 4s
			if r[i][u][col] > 4000 then r[i][u][col] = 4000 end
			r[i][u][col] = math.log(r[i][u][col]+1)-- +1 to prevent ln(0)
			local val = r[i][u][col]
			if min > val then min = val end
			if max < val then max = val end
		end
	end

	for i=1, #r do
		for u=1, r[i]:size()[1] do
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
		if opt.datasize ~= 0 and #d >= opt.datasize then return d end
		local song = parse(dir.."/"..filename)
		if #song > 2 then
			d[#d+1] = torch.Tensor(song)
		end
		::cont::
	end
	return d
end

function next_batch()
	start_index = start_index + opt.batchsize
	if start_index >= totlen-opt.batchsize-opt.rho-1 then
		start_index = 1
		local prev_loss = loss
		loss = totloss/batches
		local delta = loss-prev_loss
		model:evaluate()
		validation_err = validate(model, opt.rho, opt.batchsize, opt.vd, criterion)
		model:training()
		local v_delta = validation_err - prev_valid
		prev_valid = validation_err

		print(string.format("Ep %d loss=%.8f  dl=%.6e  valid=%.8f  dv=%.6e", curr_ep, loss, delta, validation_err, v_delta))
		if logger then
			logger:add{curr_ep, loss, delta, validation_err, v_delta}
		end

		curr_ep=curr_ep+1

		if(curr_ep % 10 == 0 and opt.o ~= '') then torch.save(opt.o, model) end --Autosave
		totloss = 0
		batches = 0
	end

	return create_batch(start_index)
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
	model:training()--Training mode
	math.randomseed(os.time())

	local optim_cfg = {learningRate=opt.lr, learningRateDecay=opt.lrdecay, weightdecay=opt.weightdecay}
	local progress = -1

	for e = 1, math.floor(opt.ep*totlen/opt.batchsize)-opt.batchsize do
		if progress ~= math.floor(100*(start_index/totlen)) then
			progress = math.floor(100*(start_index/totlen))
			xlua.progress(100*(curr_ep-start_ep-1)+progress, 100*opt.ep)
		end

		batches = batches + 1

		optim.adagrad(feval, params, optim_cfg)
	end
	
	--Get loss from last epoch
	local prev_loss = loss
	loss = totloss/batches
	local delta = loss-prev_loss
	
	model:evaluate()
	validation_err = validate(model, opt.rho, opt.batchsize, opt.vd, criterion)
	model:training()
	local v_delta = validation_err - prev_valid
	prev_valid = validation_err

	print(string.format("Ep %d loss=%.8f  dl=%.6e  valid=%.8f  dv=%.6e", curr_ep, loss, delta, validation_err, v_delta))

	if logger then
		logger:add{curr_ep, loss, delta, validation_err, v_delta}
	end

	curr_ep=curr_ep+1

	model:evaluate() --Exit training mode
end

function get_total_len(data)
	local i = 0
	for k, s in pairs(data) do
		i = i + s:size()[1]
	end
	return i
end

function create_batch(start_index)
	local i = start_index
	local song = torch.Tensor()
	local songindex = 0
	--Select the correct song
	for k, s in pairs(data) do
		if s:size()[1] > i then
			song = s
			songindex = k
			break
		else
			i = i - s:size()[1]
		end
		if i < 1 then i = 1 end
	end
	--Create batch
	local x = torch.Tensor(opt.batchsize, opt.rho, data_width)
	local y = torch.Tensor(opt.batchsize, data_width)

	for u = 1, opt.batchsize do
		::s::
		if song:size()[1] < i+u+opt.rho+1 then
			song = data[songindex+1]
			songindex = songindex+1
			i=1
			goto s
		end

		for o = opt.rho, 1, -1 do
			x[u][o] = song[i+o+u]
		end
		y[u] = song[i+u+opt.rho+1]
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

if lfs.attributes(opt.o) then--Resume training WIP
	model = torch.load(opt.o)
	resume = true
	--Read JSON
	local file = assert(io.open(opt.o..".meta", 'r'))
	meta = json.decode(file:read('*all'))
	file:close()
	print(meta)
	curr_ep = meta['ep']+1
	start_ep = meta['ep']
	opt.lr = meta['lr']/(1+meta['lrdecay']*meta['ep'])--Restore decayed lr
	meta['ep'] = meta['ep'] + opt.ep
	logger = optim.Logger(opt.o..".log2")
else
	model = create_model()
	
	if opt.o ~= '' then
		logger = optim.Logger(opt.o..".log")
		logger:setNames{'epoch', 'loss', 'delta', 'v_loss', 'v_delta'}
	else print("\n\n\nWARNING: No output file!\n\n\n") end --To prevent future fuckups
end

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

print(curr_ep)
print(start_ep)

train()

if opt.o ~= '' then
	torch.save(opt.o, model)
	local file = assert(io.open(opt.o..".meta", 'w'))
	file:write(json.encode(meta))
	file:close()
	--Merge the logs
	if resume then os.execute("cat "..opt.o..".log2 >> "..opt.o..".log") end
end
