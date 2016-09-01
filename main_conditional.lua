require 'torch'
require 'nn'
require 'optim'

-- to specify these at runtime, you can do, e.g.:
--    $ lr=0.001 th main.lua
opt = {
  dataset = 'video2',   -- indicates what dataset load to use (in data.lua)
  nThreads = 32,        -- how many threads to pre-fetch data
  batchSize = 32,      -- self-explanatory
  loadSize = 128,       -- when loading images, resize first to this size
  fineSize = 64,       -- crop this size from the loaded image 
  frameSize = 32,
  lr = 0.0002,          -- learning rate
  lr_decay = 1000,         -- how often to decay learning rate (in epoch's)
  lambda = 10,
  beta1 = 0.5,          -- momentum term for adam
  meanIter = 0,         -- how many iterations to retrieve for mean estimation
  saveIter = 1000,    -- write check point on this interval
  niter = 100,          -- number of iterations through dataset
  ntrain = math.huge,   -- how big one epoch should be
  gpu = 1,              -- which GPU to use; consider using CUDA_VISIBLE_DEVICES instead
  cudnn = 1,            -- whether to use cudnn or not
  finetune = '',        -- if set, will load this network instead of starting from scratch
  name = 'condbeach7',        -- the name of the experiment
  randomize = 1,        -- whether to shuffle the data file or not
  cropping = 'random',  -- options for data augmentation
  display_port = 8000,  -- port to push graphs
  display_id = 1,       -- window ID when pushing graphs
  mean = {0,0,0},
  data_root = '/data/vision/torralba/crossmodal/flickr_videos/',
  data_list = '/data/vision/torralba/crossmodal/flickr_videos/scene_extract/lists-full/_b_beach.txt.train',
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
  cutorch.setDevice(opt.gpu)
end

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())

-- define the model
local net
local netD
local mask_net
local motion_net 
local static_net
if opt.finetune == '' then -- build network from scratch
  net = nn.Sequential()

  local encode_net = nn.Sequential()
  encode_net:add(nn.SpatialConvolution(3,128, 4,4, 2,2, 1,1))
  encode_net:add(nn.ReLU(true))
  encode_net:add(nn.SpatialConvolution(128,256, 4,4, 2,2, 1,1))
  encode_net:add(nn.SpatialBatchNormalization(256,1e-3)):add(nn.ReLU(true))
  encode_net:add(nn.SpatialConvolution(256,512, 4,4, 2,2, 1,1))
  encode_net:add(nn.SpatialBatchNormalization(512,1e-3)):add(nn.ReLU(true))
  encode_net:add(nn.SpatialConvolution(512,1024, 4,4, 2,2, 1,1))
  encode_net:add(nn.SpatialBatchNormalization(1024,1e-3)):add(nn.ReLU(true))
  net:add(encode_net)

  static_net = nn.Sequential()
  static_net:add(nn.SpatialFullConvolution(1024, 512, 4,4, 2,2, 1,1))
  static_net:add(nn.SpatialBatchNormalization(512)):add(nn.ReLU(true))
  static_net:add(nn.SpatialFullConvolution(512, 256, 4,4, 2,2, 1,1))
  static_net:add(nn.SpatialBatchNormalization(256)):add(nn.ReLU(true))
  static_net:add(nn.SpatialFullConvolution(256, 128, 4,4, 2,2, 1,1))
  static_net:add(nn.SpatialBatchNormalization(128)):add(nn.ReLU(true))
  static_net:add(nn.SpatialFullConvolution(128, 3, 4,4, 2,2, 1,1))
  static_net:add(nn.Tanh())

  local net_video = nn.Sequential()
  net_video:add(nn.View(-1, 1024, 1, 4, 4))
  net_video:add(nn.VolumetricFullConvolution(1024, 1024, 2,1,1))
  net_video:add(nn.VolumetricBatchNormalization(1024)):add(nn.ReLU(true))
  net_video:add(nn.VolumetricFullConvolution(1024, 512, 4,4,4, 2,2,2, 1,1,1))
  net_video:add(nn.VolumetricBatchNormalization(512)):add(nn.ReLU(true))
  net_video:add(nn.VolumetricFullConvolution(512, 256, 4,4,4, 2,2,2, 1,1,1))
  net_video:add(nn.VolumetricBatchNormalization(256)):add(nn.ReLU(true))
  net_video:add(nn.VolumetricFullConvolution(256, 128, 4,4,4, 2,2,2, 1,1,1))
  net_video:add(nn.VolumetricBatchNormalization(128)):add(nn.ReLU(true))

  local mask_out = nn.VolumetricFullConvolution(128,1, 4,4,4, 2,2,2, 1,1,1)
  mask_net = nn.Sequential():add(mask_out):add(nn.Sigmoid())
  gen_net = nn.Sequential():add(nn.VolumetricFullConvolution(128,3, 4,4,4, 2,2,2, 1,1,1)):add(nn.Tanh())
  net_video:add(nn.ConcatTable():add(gen_net):add(mask_net))

  -- [1] is generated video, [2] is mask, and [3] is static
  net:add(nn.ConcatTable():add(net_video):add(static_net)):add(nn.FlattenTable())

  -- video .* mask (with repmat on mask)
  motion_net = nn.Sequential():add(nn.ConcatTable():add(nn.SelectTable(1))
                                                   :add(nn.Sequential():add(nn.SelectTable(2))
                                                                       :add(nn.Squeeze())
                                                                       :add(nn.Replicate(3, 2)))) -- for color chan 
                              :add(nn.CMulTable())

  -- static .* (1-mask) (then repmatted)
  local sta_part = nn.Sequential():add(nn.ConcatTable():add(nn.Sequential():add(nn.SelectTable(3))
                                                                           :add(nn.Replicate(opt.frameSize, 3))) -- for time
                                                       :add(nn.Sequential():add(nn.SelectTable(2))
                                                                           :add(nn.Squeeze())
                                                                           :add(nn.MulConstant(-1))
                                                                           :add(nn.AddConstant(1))
                                                                           :add(nn.Replicate(3, 2)))) -- for color chan
                                  :add(nn.CMulTable())

  net:add(nn.ConcatTable():add(motion_net):add(sta_part)):add(nn.CAddTable())

  netD = nn.Sequential()

  netD:add(nn.VolumetricConvolution(3,128, 4,4,4, 2,2,2, 1,1,1))
  netD:add(nn.LeakyReLU(0.2, true))
  netD:add(nn.VolumetricConvolution(128,256, 4,4,4, 2,2,2, 1,1,1))
  netD:add(nn.VolumetricBatchNormalization(256,1e-3)):add(nn.LeakyReLU(0.2, true))
  netD:add(nn.VolumetricConvolution(256,512, 4,4,4, 2,2,2, 1,1,1))
  netD:add(nn.VolumetricBatchNormalization(512,1e-3)):add(nn.LeakyReLU(0.2, true))
  netD:add(nn.VolumetricConvolution(512,1024, 4,4,4, 2,2,2, 1,1,1))
  netD:add(nn.VolumetricBatchNormalization(1024,1e-3)):add(nn.LeakyReLU(0.2, true))
  netD:add(nn.VolumetricConvolution(1024,2, 2,4,4, 1,1,1, 0,0,0))
  netD:add(nn.View(2):setNumInputDims(4)) 

  -- initialize the model
  local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
    end
  end
  net:apply(weights_init) -- loop over all layers, applying weights_init
  netD:apply(weights_init)

  mask_out.weight:normal(0, 0.01)
  mask_out.bias:fill(0)

else -- load in existing network
  print('loading ' .. opt.finetune)
  net = torch.load(opt.finetune)
end

print('Generator:')
print(net)
print('Discriminator:')
print(netD)

-- define the loss
local criterion = nn.CrossEntropyCriterion()
local criterionReg = nn.AbsCriterion()
local real_label = 1
local fake_label = 2

-- create the data placeholders
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local target = torch.Tensor(opt.batchSize, 3, opt.frameSize, opt.fineSize, opt.fineSize)
local video = torch.Tensor(opt.batchSize, 3, opt.frameSize, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize)
local err, errD, errReg

-- timers to roughly profile performance
local tm = torch.Timer()
local data_tm = torch.Timer()

-- ship everything to GPU if needed
if opt.gpu > 0 then
  input = input:cuda()
  target = target:cuda()
  video = video:cuda()
  label = label:cuda()
  net:cuda()
  netD:cuda()
  criterion:cuda()
  criterionReg:cuda()
end

-- conver to cudnn if needed
-- if this errors on you, you can disable, but will be slightly slower
if opt.gpu > 0 and opt.cudnn > 0 then
  require 'cudnn'
  net = cudnn.convert(net, cudnn)
  netD = cudnn.convert(netD, cudnn)
end

-- get a vector of parameters
local parameters, gradParameters = net:getParameters()
local parametersD, gradParametersD = netD:getParameters()

-- show graphics
disp = require 'display'
disp.url = 'http://localhost:' .. opt.display_port .. '/events'

-- optimization closure
-- the optimizer will call this function to get the gradients
local data_im,data_label
local fDx = function(x)
  gradParametersD:zero()

  -- fetch data
  data_tm:reset(); data_tm:resume()
  data_im = data:getBatch()
  data_tm:stop()

  -- ship to GPU
  input:copy(data_im:select(3,1))
  target:copy(data_im)
  video:copy(data_im)
  label:fill(real_label)

  -- forward/backwards real examples
  local output = netD:forward(video)
  errD = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward(video, df_do)

  -- generate fake examples
  local fake = net:forward(input)
  video:copy(fake)
  label:fill(fake_label)

  -- forward/backwards fake examples
  local output = netD:forward(video)
  errD = errD + criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward(video, df_do)

  errD = errD / 2

  return errD, gradParametersD
end

local fx = function(x)
  gradParameters:zero()

  label:fill(real_label)
  local output = netD.output
  err = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local df_dg = netD:updateGradInput(video, df_do)

  errReg = criterionReg:forward(video:select(3,1), target:select(3,1)) * opt.lambda
  local df_reg = criterionReg:backward(video:select(3,1), target:select(3,1)) * opt.lambda

  df_dg[{ {}, {}, 1, {}, {} }]:add(df_reg)

  net:backward(input, df_dg)

  return err + errReg, gradParameters
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
local optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

-- train main loop
for epoch = 1,opt.niter do -- for each epoch
  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do -- for each mini-batch
    collectgarbage() -- necessary sometimes
    
    tm:reset()

    -- do one iteration
    optim.adam(fDx, parametersD, optimStateD)
    optim.adam(fx, parameters, optimState)

    if counter % 10 == 0 then
      table.insert(history, {counter, err, errD, errReg})
      disp.plot(history, {win=opt.display_id+1, title=opt.name, labels = {"iteration", "err", "errD", "errR"}})
    end
    
    if counter % 100 == 0 then
      local vis = net.output:float()
      local vis_tab = {}
      for i=1,opt.frameSize do table.insert(vis_tab, vis[{ {}, {}, i, {}, {} }]) end
      disp.image(torch.cat(vis_tab, 3), {win=opt.display_id, title=(opt.name .. ' gen')})

      local vis = motion_net.output:float()
      local vis_tab = {}
      for i=1,opt.frameSize do table.insert(vis_tab, vis[{ {}, {}, i, {}, {} }]) end
      disp.image(torch.cat(vis_tab, 3), {win=opt.display_id+3, title=(opt.name .. ' motion')})

      local vis = static_net.output:float()
      disp.image(vis, {win=opt.display_id+4, title=(opt.name .. ' static')})

      local vis = mask_net.output:float():squeeze() 
      local vis_lo = vis:min()
      local vis_hi = vis:max()
      local vis_tab = {}
      for i=1,opt.frameSize do table.insert(vis_tab, vis[{ {}, i, {}, {} }]) end
      disp.image(torch.cat(vis_tab, 2), {win=opt.display_id+2, title=(opt.name .. ' mask ' .. string.format('%.2f %.2f', vis_lo, vis_hi))})
    end
    counter = counter + 1
    
    print(('%s: Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
              .. '  Err: %.4f  ErrD: %.4f  ErrR: %.4f'):format(
            opt.name, epoch, ((i-1) / opt.batchSize),
            math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
            tm:time().real, data_tm:time().real,
            err and err or -1, errD and errD or -1, errReg and errReg or -1))

    -- save checkpoint
    -- :clearState() compacts the model so it takes less space on disk
    if counter % opt.saveIter == 0 then
      print('Saving ' .. opt.name .. '/iter' .. counter .. '_net.t7')
      paths.mkdir('checkpoints')
      paths.mkdir('checkpoints/' .. opt.name)
      torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_net.t7', net:clearState())
      torch.save('checkpoints/' .. opt.name .. '/iter' .. counter .. '_netD.t7', netD:clearState())
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
end
