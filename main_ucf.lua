require 'torch'
require 'nn'
require 'optim'

-- to specify these at runtime, you can do, e.g.:
--    $ lr=0.001 th main.lua
opt = {
  dataset = 'video3',   -- indicates what dataset load to use (in data.lua)
  nThreads = 16,        -- how many threads to pre-fetch data
  batchSize = 256,      -- self-explanatory
  loadSize = 256,       -- when loading images, resize first to this size
  fineSize = 64,       -- crop this size from the loaded image 
  frameSize = 32,
  lr = 0.0002,          -- learning rate
  lr_decay = 1000,         -- how often to decay learning rate (in epoch's)
  lambda = 0.1,
  beta1 = 0.5,          -- momentum term for adam
  meanIter = 0,         -- how many iterations to retrieve for mean estimation
  saveIter = 100,    -- write check point on this interval
  niter = 100,          -- number of iterations through dataset
  max_iter = 1000,
  ntrain = math.huge,   -- how big one epoch should be
  gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  cudnn = 1,            -- whether to use cudnn or not
  name = 'ucf101',        -- the name of the experiment
  randomize = 1,        -- whether to shuffle the data file or not
  cropping = 'random',  -- options for data augmentation
  display_port = 8001,  -- port to push graphs
  display_id = 1,       -- window ID when pushing graphs
  mean = {0,0,0},
  data_root = '/data/vision/torralba/hallucination/UCF101/frames-stable-nofail/videos',
  data_list = '/data/vision/torralba/hallucination/UCF101/gan/train.txt'
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

torch.manualSeed(0)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- if using GPU, select indicated one
if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  cutorch.setDevice(opt.gpu)
end

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- define the model
local net = torch.load("checkpoints/all100/iter95000_netD.t7")
net:remove(#net.modules)
net:remove(#net.modules)
--for i=1,#net.modules do net.modules[i].accGradParameters = function() end end -- freeze all but last
net:add(nn.VolumetricDropout(0.5))
net:add(nn.VolumetricConvolution(512,101, 2,4,4, 1,1,1, 0,0,0))
net:add(nn.View(-1, 101))

-- net:reset() -- random weights?

print('Net:')
print(net)

-- define the loss
local criterion = nn.CrossEntropyCriterion()

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.frameSize, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize)
local err
local accuracy

-- timers to roughly profile performance
local tm = torch.Timer()
local data_tm = torch.Timer()

-- ship everything to GPU if needed
if opt.gpu > 0 then
  input = input:cuda()
  label = label:cuda()
  net:cuda()
  criterion:cuda()
end

-- conver to cudnn if needed
-- if this errors on you, you can disable, but will be slightly slower
if opt.gpu > 0 and opt.cudnn > 0 then
  net = cudnn.convert(net, cudnn)
end

-- get a vector of parameters
local parameters, gradParameters = net:getParameters()

-- show graphics
disp = require 'display'
disp.url = 'http://localhost:' .. opt.display_port .. '/events'

-- optimization closure
-- the optimizer will call this function to get the gradients
local data_im,data_label
local fx = function(x)
  gradParameters:zero()

  -- fetch data
  data_tm:reset(); data_tm:resume()
  data_im,data_label = data:getBatch()
  data_tm:stop()

  input:copy(data_im)
  label:copy(data_label)

  local output = net:forward(input)
  err = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  net:backward(input, df_do)

  _,pred_cat = torch.max(output:float(), 2)
  accuracy = pred_cat:float():eq(data_label):float():mean()

  return err, gradParameters
end

local counter = 0
local history = {}

-- parameters for the optimization
-- very important: you must only create this table once! 
-- the optimizer will add fields to this table (such as momentum)
local optimState = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

-- train main loop
for epoch = 1,opt.niter do -- for each epoch
  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do -- for each mini-batch
    collectgarbage() -- necessary sometimes
    
    tm:reset()

    -- do one iteration
    optim.adam(fx, parameters, optimState)

    if counter % 10 == 0 then
      table.insert(history, {counter, err, errD})
      disp.plot(history, {win=opt.display_id+1, title=opt.name, labels = {"iteration", "err"}})
    end

    counter = counter + 1
    
    print(('%s: Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
              .. '  Err: %.4f  Accuracy: %.4f'):format(
            opt.name, epoch, ((i-1) / opt.batchSize),
            math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
            tm:time().real, data_tm:time().real,
            err and err or -1, accuracy))

    -- save checkpoint
    -- :clearState() compacts the model so it takes less space on disk
    if counter % opt.saveIter == 0 then
      print('Saving ' .. opt.name .. '/iter' .. counter .. '_net.t7')
      paths.mkdir('checkpoints')
      paths.mkdir('checkpoints/' .. opt.name)
      torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_net.t7', net:clearState())
      torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_history.t7', history)
    end
  end
  
  -- decay the learning rate, if requested
  if opt.lr_decay > 0 and epoch % opt.lr_decay == 0 then
    opt.lr = opt.lr / 10
    print('Decreasing learning rate to ' .. opt.lr)

    -- create new optimState to reset momentum
    optimState = {
      learningRate = opt.lr,
      beta1 = opt.beta1,
    }
  end

  if counter > opt.max_iter then
    break
  end
end
