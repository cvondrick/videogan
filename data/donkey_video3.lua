--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

-- Heavily moidifed by Carl to make it simpler

require 'torch'
require 'image'
tds = require 'tds'
torch.setdefaulttensortype('torch.FloatTensor')
local class = require('pl.class')

local dataset = torch.class('dataLoader')

-- this function reads in the data files
function dataset:__init(args)
  for k,v in pairs(args) do self[k] = v end

  assert(self.frameSize > 0)

  if self.filenamePad == nil then
    self.filenamePad = 8
  end

  -- read text file consisting of frame directories and counts of frames
  self.data = tds.Vec()
  self.category = tds.Vec()
  print('reading ' .. args.data_list)
  for line in io.lines(args.data_list) do 
    local split = {}
    for k in string.gmatch(line, "%S+") do table.insert(split, k) end
    self.data:insert(split[1])
    self.category:insert(split[2])
  end

  print('found ' .. #self.data .. ' videos')

end

function dataset:size()
  return #self.data
end

-- converts a table of samples (and corresponding labels) to a clean tensor
function dataset:tableToOutput(dataTable, scalarTable, extraTable)
   local data, scalarLabels, labels
   local quantity = #dataTable
   assert(dataTable[1]:dim() == 4)
   data = torch.Tensor(quantity, 3, self.frameSize, self.fineSize, self.fineSize)
   label = torch.Tensor(quantity)
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
      label[i] = scalarTable[i]
   end
   return data, label, extraTable
end

-- sampler, samples with replacement from the training set.
function dataset:sample(quantity)
   assert(quantity)
   local dataTable = {}
   local scalarTable = {}
   local extraTable = {}
   for i=1,quantity do
      local idx = torch.random(1, #self.data)
      local data_path = self.data_root .. '/' .. self.data[idx]

      local out = self:trainHook(data_path)
      table.insert(dataTable, out)
      table.insert(extraTable, self.data[idx])
      table.insert(scalarTable, self.category[idx])
   end
   return self:tableToOutput(dataTable, scalarTable, extraTable)
end

-- gets data in a certain range
function dataset:get(start_idx,stop_idx)
  assert(false)
end

-- function to load the image, jitter it appropriately (random crops etc.)
function dataset:trainHook(path)
  collectgarbage()

  local oW = self.fineSize
  local oH = self.fineSize 
  local h1
  local w1

  local out = torch.zeros(3, self.frameSize, oW, oH)

  local ok,input = pcall(image.load, path, 3, 'float') 
  if not ok then
     print('warning: failed loading: ' .. path)
     return out
  end

  local count = input:size(2) / opt.loadSize
  local t1 = 1
  
  for fr=1,self.frameSize do
    local off 
    if fr <= count then 
      off = (fr+t1-2) * opt.loadSize+1
    else
      off = (count+t1-2)*opt.loadSize+1 -- repeat the last frame
    end
    
    local crop
    if off+opt.loadSize-1 <= input:size(2) and off > 0 then
      crop = input[{ {}, {off, off+opt.loadSize-1}, {} }]
    else
      print('*** WARNING ***')
      print('  bad size')
      print('  path: ' .. path)
      crop = torch.zeros(3, opt.fineSize, opt.fineSize)
    end
    out[{ {}, fr, {}, {} }]:copy(image.scale(crop, opt.fineSize, opt.fineSize))
  end

  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

  -- subtract mean
  for c=1,3 do
    out[{ c, {}, {} }]:add(-self.mean[c])
  end

  return out
end

-- data.lua expects a variable called trainLoader
trainLoader = dataLoader(opt)
