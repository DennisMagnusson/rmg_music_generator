require 'torch'
require 'parse'
require 'nn'
require 'lfs'
require 'rnn'

function create_dataset(dir)
	local d = {}
	local maxlen = 0
	for filename in lfs.dir(".") do
		local song = parse(filename)
		if maxlen > #song then maxlen = #song end
		d[#d+1] = song
	end
	local tensor = torch.zeros(#d, maxlen, 88)
	--TODO append the things

	return tensor 
end

function create_model()
	local model = nn.Sequenial()
	--rho(Steps to backpropagate) = 100
	layer1=nn.FastLSTM(88, 256, 100)
	layer1:maskZero(1)
	model:add(layer1)
	model:add(nn.FastLSTM((88, #len), 256, 50))
	model:add(nn.FastLSTM((256, 50), (512), 50))
	model.add(nn.Linear(512, 512))
	model:add(nn.ReLU())

	return model
end


torch.setdefaulttensortype('torch.FloatTensor')
print(create_dataset(dir))




