local midi = require 'MIDI'

function parse(filename)
	local file = assert(io.open(filename, 'r'))
	local m = midi.midi2score(file:read("*all"))
	file:close()
	file = nil
	r = {}
	--TODO Add something for multiple notes at same time	
	for k, event in pairs(m[2]) do
		if event[1] ~= 'note' then goto EOL end
		local frame = {}
		--Fill frame with zeros
		for i = 1, 88, 1 do frame[i] = 0 end
		frame[event[5]-20] = 1
		r[#r+1] = frame
		::EOL::
	end
	
	return r
end

function parse_ind(filename)
	local file = assert(io.open(filename, 'r'))
	local m = midi.midi2score(file:read("*all"))
	file:close()
	file = nil
	r = {}
	--TODO Add something for multiple notes at same time	
	for k, event in pairs(m[2]) do
		if event[1] ~= 'note' then goto EOL end
		local frame = {}
		--Fill frame with zeros
		for i = 1, 88, 1 do frame[i] = 0 end
		frame[event[5]-20] = 1
		r[#r+1] = frame
		::EOL::
	end
	
	return r
end



function print_r ( t )
local print_r_cache={}
local function sub_print_r(t,indent)
if (print_r_cache[tostring(t)]) then
	print(indent.."*"..tostring(t))
	else
		print_r_cache[tostring(t)]=true
		if (type(t)=="table") then
			for pos,val in pairs(t) do
				if (type(val)=="table") then
					print(indent.."["..pos.."] => "..tostring(t).." {")
					sub_print_r(val,indent..string.rep(" ",string.len(pos)+8))
					print(indent..string.rep(" ",string.len(pos)+6).."}")
				else
					print(indent.."["..pos.."] => "..tostring(val))
				end
			end
		else
			print(indent..tostring(t))
		end
	end
end
sub_print_r(t," ")
end

print_r(parse("data/elise.mid"))


