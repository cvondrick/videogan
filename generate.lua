require 'torch'
require 'nn'
require 'image'
require 'cunn'
require 'cudnn'

opt = {
  model = 'models/golf/iter65000_net.t7',
  batchSize = 128,
  gpu = 1,
  cudnn = 1,
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- if using GPU, select indicated one
cutorch.setDevice(opt.gpu)

net  = torch.load(opt.model) 
net:evaluate()
net:cuda()
net = cudnn.convert(net, cudnn)

print('Generator:')
print(net)

local noise = torch.Tensor(opt.batchSize, 100):normal():cuda()

local gen = net:forward(noise)
local video = net.modules[2].output[1]:float()
local mask = net.modules[2].output[2]:float()
local static = net.modules[2].output[3]:float()
local mask = mask:repeatTensor(1,3,1,1,1)

function WriteGif(filename, movie)
  for fr=1,movie:size(3) do
    image.save(filename .. '.' .. string.format('%08d', fr) .. '.png', image.toDisplayTensor(movie:select(3,fr)))
  end
  cmd = "ffmpeg -f image2 -i " .. filename .. ".%08d.png -y " .. filename
  print('==> ' .. cmd)
  sys.execute(cmd)
  for fr=1,movie:size(3) do
    os.remove(filename .. '.' .. string.format('%08d', fr) .. '.png')
  end
end

paths.mkdir('vis/')
WriteGif('vis/gen.gif', gen) 
WriteGif('vis/video.gif', video) 
WriteGif('vis/videomask.gif', torch.cmul(video, mask))
WriteGif('vis/mask.gif', mask)
image.save('vis/static.jpg', image.toDisplayTensor(static))
