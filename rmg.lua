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
	--Can I just use a table, then convert to tensor when training?
	--[[
	--Fill with zeros
	local empty_frame = {}
	for i = 1, 88 do empty_frame[i] = 0 end
	for key, songs in pairs(d) do
		for k = 1, #d do
			if not d[key][k] then d[key][k] = empty_frame end
		end
	end
	
	--local tensor = torch.zeros(#d, maxlen, 88)
	local tensor = torch.Tensor(d)
	[[--
	for n, s in pairs(d) do
		for key, frame in pairs(s) do
			tensor[n][key] = frame
		end
	end

	--return tensor 
	]]
end

function fit(model, criterion, lr, batch)
	--for i = 1, #batch[1] do
	for i = 1, 330 do
		--print(i)
		local x = batch[1][i]
		local y = batch[2][i]
		local pred = model:forward(x)
		--local err  = criterion:forward(pred, y)
		local gradcrit = criterion:backward(pred, y) 
		model:zeroGradParameters()
		model:backward(x, gradcrit)
		model:updateParameters(lr)
	end
end

function train(model, data, ep)
	model:training()
	--local criterion = nn.ClassNLLCriterion()
	local criterion = nn.MSECriterion()
	local trainer = nn.StochasticGradient(model, criterion)
	trainer.learningRate = 0.01
	trainer.maxIteration = ep or 1--TODO Do custom epochs?
	
	local maxlen = get_maxlen(data)
	for i = 2, maxlen do
		local x = {}
		local y = {}
		for key, d in pairs(data) do
			local p = {}
			for k = 1, i-1 do
				p[#p+1] = d[k]
			end
			x[#x+1] = p
			y[#y+1] = d[i]
		end
		print(#x[1], #y[1])
		--x = torch.expand(torch.Tensor(x), maxlen, 88)
		x = torch.zeros(maxlen, 88)--Something with this --TODO Convert to tensor filled with zeros of same size to solve problem
		y = torch.Tensor(y)
		--local batch = {torch.Tensor(x), torch.Tensor(y)}
		local batch = {x, y}
		--trainer.train(batch)
		fit(model, criterion, 0.01, batch)
	end
	
	--[[
	local lr = 0.01
	for key, s in pairs(data) do
		local batch = create_batch(s, 10)
		print(key, "/", #data)
		--trainer:train(batch)
		fit(model, criterion, lr, batch)
	end
	]]
	model.evaluate() --Exit training mode
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
function create_batch(data, len)
	len = 100 or len
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
end

function create_model()
	local rho = 100
	local hiddensize = 256
	local dropoutprob = 0.2
	local model = nn.Sequential()
	--rho(Steps to backpropagate) = 100
	--layer1=nn.FastLSTM(88, hiddensize, rho)
	--layer1:maskZero(1)
	--model:add(layer1)
	model:add(nn.FastLSTM(88, hiddensize, rho))--Hope this works
	model:add(nn.Tanh())
	--model:add(nn.FastLSTM(hiddensize, hiddensize, rho))
	--model:add(nn.Tanh())
	model:add(nn.Linear(hiddensize, hiddensize))
	model:add(nn.Linear(hiddensize, 88))
	model:add(nn.ReLU())
	return model
end

--torch.setdefaulttensortype('torch.FloatTensor')
