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
		local x = batch[1][i]--Breaks here
		local y = batch[2][i]--And here
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
	model:training()--Training mode
	local criterion = nn.CrossEntropyCriterion()
	local trainer = nn.StochasticGradient(model, criterion)
	trainer.learningRate = 0.01
	trainer.maxIteration = ep or 1--TODO Do custom epochs?
		
	local maxlen = get_maxlen(data)
	local lr = 0.01
	for i = 1, maxlen do
		--[[Okay, okay I think I got this shit sorted out. A batch is a table of tensors.
		#table = rho. In the example the target is a tensor of the same dims, which is a shame. I'm gonna have to figure out a way to do that. TODO Check out the sequence-to-one.lua example for guidelines.
		]]

		local batch = create_batch(data, 50, i)
		--trainer:train(batch)
		fit(model, criterion, lr, batch)
		print(key, "/", #data)
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
	local len = len or 50
	local x = torch.zeros(#data, len, 88)
	local y = torch.zeros(#data, 88)
	--local n = 0
	--if i < len then n = i else n = len end
	local start = i - len 
	if start < 1 then start = 1 end
	for key, s in pairs(data) do
		for u = 1, len do
			for k = 1, 88 do
				x[key][u][k] = s[u+start][k]
			end
			--x = fill(x, len, 88)
		end
		for k = 1, 88 do
			y[key][k] = s[len+1][k]
		end
	end
	collectgarbage()
	return {x, y}
end

function create_model()
	local rho = 100
	local hiddensize = 256
	local dropoutprob = 0.2
	local model = nn.Sequential()
	local layer1 = nn.SeqLSTM(88,88, 100)
	layer1:maskZero()
	model:add(layer1)
	model:add(nn.Tanh())
	model:add(nn.Linear(hiddensize, hiddensize))
	model:add(nn.Linear(hiddensize, 88))
	model:add(nn.ReLU())
	return model
end

--torch.setdefaulttensortype('torch.FloatTensor')
