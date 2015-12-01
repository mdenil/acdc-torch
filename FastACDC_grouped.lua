local FastACDC_grouped, parent = torch.class('acdc.FastACDC_grouped', 'nn.Module')

function FastACDC_grouped:__init(dim, groupSize, opt)
  parent.__init(self)
  local opt = opt or {}

  if opt then
    if opt.use_bias ~= nil then
      self.use_bias = opt.use_bias
    else
      self.use_bias = true
    end
    self.sign_init = opt.sign_init or false
    self.rand_init = opt.rand_init or false
  else
    self.use_bias = true
    self.sign_init = false
    self.rand_init = false
  end

  assert(dim % 2 == 0, "Input size must be even.  Odd sizes are not supported.")

  self.dim = dim
  self.groupSize = groupSize
  self.A = torch.Tensor(dim*groupSize)
  self.Ab = torch.Tensor(dim*groupSize)
  self.D = torch.Tensor(dim*groupSize)
  self.Db = torch.Tensor(dim*groupSize)

  self.gradA = torch.Tensor(dim*groupSize)
  self.gradAb = torch.Tensor(dim*groupSize)
  self.gradD = torch.Tensor(dim*groupSize)
  self.gradDb = torch.Tensor(dim*groupSize)

  self.tmp1 = torch.Tensor()
  self.tmp2 = torch.Tensor()
  self.delta_mid = torch.Tensor()
  self.activationsD = torch.Tensor()

  self:reset()
end

function FastACDC_grouped:reset()
  if self.sign_init then
    self.A:uniform(-1, 1)
    self.A:copy(self.A:gt(0):float():add(-1, self.A:le(0):float()))

    self.D:uniform(-1, 1)
    self.D:copy(self.D:gt(0):float():add(-1, self.D:le(0):float()))
  else
    self.A:fill(2)
    self.D:fill(0.7)
  end

  if self.rand_init then
    self.A:add(self.A:clone():uniform(-0.01, 0.01))
    self.D:add(self.D:clone():uniform(-0.01, 0.01))
  end

  self.Ab:fill(0.1)
  self.Db:fill(0.2)

  self.gradA:fill(0)
  self.gradAb:fill(0)
  self.gradD:fill(0)
  self.gradDb:fill(0)
end

function FastACDC_grouped:updateOutput(input)
  assert(input:size(input:nDimension()) == self.dim,
    "Incorrect input size.  Expected " .. self.dim .. ", got " ..
    input:size(input:nDimension()))

  input = input:contiguous()
  self.output = self.output:resize(input:size(1), self.groupSize, input:size(3)):zero():contiguous()

  self.tmp1 = self.tmp1:resize(2 * self.output:nElement()):zero()
  self.tmp2 = self.tmp2:resize(2 * self.output:nElement()):zero()
  self.delta_mid = self.delta_mid:resize(self.output:nElement()):zero()

  input.nn.Fast_ACDC_updateOutput(self, input)
  
  return self.output
end

function FastACDC_grouped:updateGradInput(input, gradOutput)
  input = input:contiguous()
  gradOutput = gradOutput:contiguous()
  self.gradInput = self.gradInput:resizeAs(gradOutput):contiguous():zero()

  input.nn.Fast_ACDC_updateGradInput(self, input, gradOutput)

  return self.gradInput
end

function FastACDC_grouped:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  assert(scale == 1, "scale != 1 is not implemented")

  local input = input:contiguous()
  local gradOutput = gradOutput:contiguous()

  self.activationsD = self.activationsD:resizeAs(gradOutput):zero()

  input.nn.Fast_ACDC_accGradParams(self, input, gradOutput)

  if not self.use_bias then
    self.gradAb:zero()
    self.gradDb:zero()
  end
end

function FastACDC_grouped:parameters()
  local params = {
    self.A,
    self.Ab,
    self.D,
    self.Db,
  }
  local grads = {
    self.gradA,
    self.gradAb,
    self.gradD,
    self.gradDb,
  }
  return params, grads
end

function FastACDC_grouped:print_parameters()
   print(self.A)
   print(self.Ab)
   print(self.D)
   print(self.Db)
end

function FastACDC_grouped:print_grads()
   print(self.gradA)
   print(self.gradAb)
   print(self.gradD)
   print(self.gradDb)
end

FastACDC_grouped.sharedAccUpdateGradParameters = FastACDC_grouped.accUpdateGradParameters
