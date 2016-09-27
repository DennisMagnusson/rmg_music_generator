require 'torch'
require 'parse'
require 'nn'
require 'lfs'
require 'rnn'
require 'optim'

function create_dataset(dir)
	local d = {}
	local maxlen = 0
	for filename in lfs.dir(dir.."/.") do
		if filename[1] == '.' then goto cont end
		local song = parse(dir.."/"..filename)
		if #song > 2 then
			--if maxlen > #song then maxlen = #song end
			d[#d+1] = song
		end
		::cont::
	end
	return d
end

function fit(model, criterion, lr, batch)
	--for i = 1, #batch[1] do
	for i = 1, 330 do
		print(i)
		local y = batch[2][i]--And here
		local x = batch[1][i]--Breaks here
		local pred = model:forward(x)
		--local err  = criterion:forward(pred, y)
		local gradcrit = criterion:backward(pred, y) 
		model:zeroGradParameters()
		model:backward(x, gradcrit)
		model:updateParameters(lr)
	end
end
--[[ TODO Let's get shit sorted out here, x in a batch is a tensor of dim=(#data, 50, 88)
y in a batch is a tensor of dim=(#data, 88) or dim=(#data, 1, 88) or, in worst case(#data, 50, 88) where the first 49 timesteps are zeros.
]]
function train(model, data, ep)
	model:training()
	--local criterion = nn.ClassNLLCriterion()
	local criterion = nn.MSECriterion()
	local trainer = nn.StochasticGradient(model, criterion)
	trainer.learningRate = 0.01
	trainer.maxIteration = ep or 1--TODO Do custom epochs?
		
	local maxlen = get_maxlen(data)
	--[[
	for i = 2, maxlen do
		--local x = {}
		--local y = {}
		local x = torch.Tensor(#data, maxlen, 88)
		local y = torch.Tensor(#data, 88)
		for key, d in pairs(data) do
			local p = {}
			for k = 1, i-1 do
				for u = 1, 88 do
					x[key][i][u] = d[k][u]
				end
			end
			--x[#x+1] = p
			--y[#y+1] = d[i]
			for u = 1, 88 do
				y[key] = d[i][u]
			end
		end
		--x = torch.expand(torch.Tensor(x), maxlen, 88)
		--x = fill(x, maxlen, 88)
		--x = torch.zeros(maxlen, 88)--TODO Convert to tensor filled with zeros of same size to solve problem
		x = torch.Tensor(x)
		y = torch.Tensor(y)
		print(x:size(), y:size())
		--local batch = {torch.Tensor(x), torch.Tensor(y)}
		local batch = {x, y}
		--trainer.train(batch)
		fit(model, criterion, 0.01, batch)
	end
	]]	
	--data = fill(data, maxlen, 88)
	local lr = 0.01
	--for key, s in pairs(data) do
	--for i = 1, #data do
	for i = 1, maxlen do
		local batch = create_batch(s, 50, 1)
		print(key, "/", #data)
		--trainer:train(batch)
		fit(model, criterion, lr, batch)
	end

	model.evaluate() --Exit training mode
end

function fill(r, rows, cols)
	for i = 1, rows do
		for u = 1, cols do
			if not r[i][u] then r[i][u] = 0 end
		end
	end
	local x = torch.Tensor(r)
	return x
end

function get_maxlen(data)
	local max = 0
	for key, d in pairs(data) do
		if #d > max then max =  #d end
	end
	return max
end

--Input = 1 song. len=length of x
--TODO Test
--Check for infinite loops?
function create_batch(data, len, i)
	len = len or 50
	x = torch.Tensor(#data, len, 88)
	y = torch.Tensor(#data, 88)
	--local n = 0
	--if i < len then n = i else n = len end
	local start = i - len 
	if i < 1 then i = 1 end
	for key, s in pairs(data) do
		for u = 1, len, do
			for k = 1, 88 do
				x[key][u][k] = s[u+start][k]
			end
			x = fill(x, len, 88)--TODO Test
		end
		for k = 1, 88 do
			y[key][k] = s[len+1][k]
		end
	end
	return {x, y}
	--[[
	local x = {}
	local y = {}
	for n = 2, #data do
		local l = 0
		local p = {}	
		for k, frame in pairs(data) do
			p[#p+1] = frame
		end
		x[#x+1] = p
		p = nil
		--y[#y+1] = {data[n]}
		y[#y+1] = data[n]
	end
	print(#x, #y)
	collectgarbage()
	batch = {torch.Tensor(x), torch.Tensor(y)}--This is where mem fucks up
	print("batch done")
	collectgarbage()
	return batch
	]]
end

function create_model()
	local rho = 100
	local hiddensize = 256
	local dropoutprob = 0.2
	local model = nn.Sequential()
	local layer1 = nn.SeqLSTM(88,88, 100)
	layer1:maskZero()
	model:add(layer1)
	--rho(Steps to backpropagate) = 100
	--layer1=nn.FastLSTM(88, hiddensize, rho)
	--layer1:maskZero(1)
	--model:add(layer1)
	--model:add(nn.FastLSTM(88, hiddensize, rho)):MaskZero()--Hope this works
	model:add(nn.Tanh())
	--model:add(nn.FastLSTM(hiddensize, hiddensize, rho))
	--model:add(nn.Tanh())
	model:add(nn.Linear(hiddensize, hiddensize))
	model:add(nn.Linear(hiddensize, 88))
	model:add(nn.ReLU())
	return model
end

--torch.setdefaulttensortype('torch.FloatTensor')
