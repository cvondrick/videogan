local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}

local result = {}
local unpack = unpack and unpack or table.unpack

function data.new(n, dataset_name, opt_)
   opt_ = opt_ or {}
   local self = {}
   for k,v in pairs(data) do
      self[k] = v
   end

   self.randomize = opt_.randomize

   local donkey_file
   if dataset_name == 'simple' then
       donkey_file = 'donkey_simple.lua'
   elseif dataset_name == 'video2' then
       donkey_file = 'donkey_video2.lua'
   else
      error('Unknown dataset: ' .. dataset_name)
   end

   if n > 0 then
      local options = opt_
      self.threads = Threads(n,
                             function() require 'torch' end,
                             function(idx)
                                opt = options
                                tid = idx
                                local seed = (opt.manualSeed and opt.manualSeed or 0) + idx
                                torch.manualSeed(seed)
                                torch.setnumthreads(1)
                                print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
                                assert(options, 'options not found')
                                assert(opt, 'opt not given')
                                paths.dofile(donkey_file)
                             end
      )
   else
      if donkey_file then paths.dofile(donkey_file) end
      self.threads = {}
      function self.threads:addjob(f1, f2) f2(f1()) end
      function self.threads:dojob() end
      function self.threads:synchronize() end
   end

   local nSamples = 0
   self.threads:addjob(function() return trainLoader:size() end,
         function(c) nSamples = c end)
   self.threads:synchronize()
   self._size = nSamples

   self.jobCount = 0
   for i = 1, n do
      self:queueJob()
   end

   return self
end

function data:queueJob()
  self.jobCount = self.jobCount + 1

  if self.randomize > 0 then
    self.threads:addjob(function()
                          return trainLoader:sample(opt.batchSize)
                        end,
                        self._pushResult)
  else
    local indexStart = (self.jobCount-1) * opt.batchSize + 1
    local indexEnd = (indexStart + opt.batchSize - 1)
    if indexEnd <= self:size() then
      self.threads:addjob(function()
                            return trainLoader:get(indexStart, indexEnd)
                          end,
                          self._pushResult)
    end
  end
end

function data._pushResult(...)
   local res = {...}
   if res == nil then
      self.threads:synchronize()
   end
   result[1] = res
end

function data:getBatch()
   -- queue another job
   local res
   repeat 
      self:queueJob()
      self.threads:dojob()
      res = result[1]
      result[1] = nil
   until torch.type(res) == 'table' 
   return unpack(res)
end

function data:size()
   return self._size
end

return data
