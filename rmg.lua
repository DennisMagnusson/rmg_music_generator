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
	for i = 1, 88 do
		print(i)
		local x = batch[1][i]
		local y = batch[2][i]
		local pred = model:forward(x)
		local err  = criterion:forward(pred, y)
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
	--local criterion = nn.CrossEntropyCriterion()
	--CrossEntropy is acting weird, use MSE instead
	local criterion = nn.MSECriterion()
	local trainer = nn.StochasticGradient(model, criterion)
	trainer.learningRate = 0.01
	trainer.maxIteration = ep or 1--TODO Do custom epochs?
		
	local maxlen = get_maxlen(data)
	local totlen = get_total_len(data)
	local lr = 0.01
	--local rho = 50
	local rho = 88
	for i = 1, totlen-rho, rho do
		--local batch = create_batch(data, 50, i, rho)
		local batch = create_batch(data, 88, i, rho)
		print(batch)
		fit(model, criterion, lr, batch)
		--trainer:train(batch)
	end


	--[[
	for i = 1, maxlen do
		--[[Okay, okay I think I got this shit sorted out. A batch is a table of tensors.
		#table = rho. In the example the target is a tensor of the same dims, which is a shame. I'm gonna have to figure out a way to do that. TODO Check out the sequence-to-one.lua example for guidelines.
		]

		local batch = create_batch(data, 50, i)
		--trainer:train(batch)
		fit(model, criterion, lr, batch)
		print(key, "/", #data)
	end
	]]

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
		if #d > max then max = #d end
	end
	return max
end

function get_total_len(data)
	local i = 0
	for k, s in pairs(data) do
		i = i + #s
	end
	return i
end

--Input = 1 song. len=length of x
--TODO Test
--[[
Should return table of doubletensor where: 
length = rho
size= batch_size x 88
This makes no sense
####
Attempt no. 2 at reasonable explanation:
A Batch is a table of examples. Could be 1d or 2d I think.
length of the batch is the batch size.
function create_batch(data, batch_size, i)
	local len = batch_size or 50
	local x = torch.zeros(#data, len, 88)
	local y = torch.zeros(#data, 88)
	--local n = 0
	--if i < len then n = i else n = len end
	local start = i - len 
	if start < 1 then start = 1 end
	for key, s in pairs(data) do
		for u = 1, len do
			x[key][u] = torch.Tensor(s[u+start])
			--[[for k = 1, 88 do
				x[key][u][k] = s[u+start][k]
			end
			--End of loop
			--x = fill(x, len, 88)
		end
		for k = 1, 88 do
			y[key][k] = s[len+1][k]
		end
	end
	collectgarbage()
	print(x)--Wtf? Hangs up here
	return {x, y}
end
--]]
--TODO Lots of improvements to be made
function create_batch(data, batch_size, start_index, rho)
	local i = start_index
	local x = torch.zeros(batch_size, rho, 88)--Correct sizes
	local y = torch.zeros(batch_size, 88)--Correct sizes
	local song = {}
	--I need a way to index the data conveniently.
	--Select the correct song
	for k, s in pairs(data) do
		if #s > i+1+batch_size then
			song = s
			break
		else 
			i = i - #s
		end
	end
	--print(i)
	if i < 1 then i = 1 end
	
	for u = 1, batch_size do
		--for o = i+rho+u, i+u, -1 do
		for o = rho, 1, -1 do
			x[u][o] = torch.Tensor(song[i+o+u])
		end
		y[u] = torch.Tensor(song[i+u+rho+1])
	end
	return {x, y}	
end

function create_model()
	--local rho = 50
	local rho = 88
	local hiddensize = 88 
	local dropoutprob = 0.2
	local model = nn.Sequential()
	--local layer1 = nn.SeqLSTM(88,88, 100)
	--layer1:maskZero()
	--model:add(layer1)
	--model:add(nn.LookupTable(88, hiddensize))
	--model:add(nn.SplitTable(1, 2))
	model:add(nn.FastLSTM(88, hiddensize, rho))
	model:add(nn.Tanh())
	--model:add(nn.View(88*1))--This guy: https://groups.google.com/forum/#!topic/torch7/ohfYBmXbaXI
	model:add(nn.Narrow(1, 88))--FUCK YES
	model:add(nn.Linear(hiddensize, 88))
	model:add(nn.ReLU())
	return model
end

--torch.setdefaulttensortype('torch.FloatTensor')
