require 'torch'
require 'parse'
require 'nn'
require 'lfs'

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
	local model = --IDK
	

	return model
end


torch.setdefaulttensortype('torch.FloatTensor')
print(create_dataset(dir))




