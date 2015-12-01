require 'torch'
require 'acdc'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local tester = torch.Tester()

local precision = 1e-6

local fast_acdc_test = {}

function fast_acdc_test.CheckShapesAndTypes()

  local batch_size = 5
  local input_size = 8

  local module_s = acdc.ACDC(input_size):cuda()
  local module_f = acdc.FastACDC(input_size):cuda()

  local params_f, grads_f = module_f:getParameters()
  local params_s, grads_s = module_s:getParameters()

  tester:asserteq(params_f:nElement(), params_s:nElement())
  tester:asserteq(grads_f:nElement(), grads_s:nElement())

  tester:asserteq(torch.typename(params_f), torch.typename(params_s))
  tester:asserteq(torch.typename(grads_f), torch.typename(grads_s))
end

function fast_acdc_test.CheckForward()
  for batch_size = 1, 10 do
    for input_size = 2, 20, 2 do

      local module_s = acdc.ACDC(input_size)
      local module_f = acdc.FastACDC(input_size):cuda()

      local params_f, grads_f = module_f:getParameters()
      local params_s, grads_s = module_s:getParameters()

      params_f:uniform()
      params_s:copy(params_f)

      local input = torch.Tensor(batch_size, input_size):uniform(-1, 1)

      local output_f = module_f:forward(input:cuda()):float()
      local output_s = module_s:forward(input:clone())

      tester:assertTensorEq(output_f, output_s, precision, "output mismatch")
    end
  end
end

function fast_acdc_test.CheckOddSizeInputIsError()
  local function fail()
    acdc.FastACDC(3)
  end
  tester:assertError(fail, "Odd input sizes must fail.")
end

function fast_acdc_test.CheckUpdateGradInput()
  local batch_size = 6
  local input_size = 8

  local input = torch.Tensor(batch_size, input_size):uniform(-1, 1)

  local module_s = acdc.ACDC(input_size)
  local module_f = acdc.FastACDC(input_size):cuda()

  local params_f, grads_f = module_f:getParameters()
  local params_s, grads_s = module_s:getParameters()

  params_f:uniform()
  params_s:copy(params_f)

  grads_f:zero()
  grads_s:zero()

  local output_f = module_f:forward(input:cuda()):float()
  local output_s = module_s:forward(input:clone())

  local delta = output_s:clone():uniform(-1, 1)

  local gradInput_f = module_f:updateGradInput(input:cuda(), delta:cuda())
  local gradInput_s = module_s:updateGradInput(input:clone(), delta:clone())

  tester:assertTensorEq(params_f:float(), params_s, precision, "params mismatch")
  tester:assertTensorEq(grads_f:float(), grads_s, precision, "grads mismatch")
  tester:assertTensorEq(gradInput_f:float(), gradInput_s, precision, "gradInput mismatch")
end


function fast_acdc_test.CheckAccGradParameters()
  local batch_size = 6
  local input_size = 8

  local input = torch.Tensor(batch_size, input_size):uniform(-1, 1)

  local module_s = acdc.ACDC(input_size)
  local module_f = acdc.FastACDC(input_size):cuda()

  module_f.A:uniform(-1, 1)
  module_s.modules[1].weight:copy(module_f.A)
  module_f.Ab:uniform(-1, 1)
  module_s.modules[1].bias:copy(module_f.Ab)
  module_f.D:uniform(-1, 1)
  module_s.modules[3].weight:copy(module_f.D)
  module_f.Db:uniform(-1, 1)
  module_s.modules[3].bias:copy(module_f.Db)


  module_f:zeroGradParameters()
  module_s:zeroGradParameters()

  local output_f = module_f:forward(input:cuda()):float()
  local output_s = module_s:forward(input:clone())

  local delta = output_s:clone():uniform(-1, 1)

  module_f:updateGradInput(input:cuda(), delta:cuda())
  module_s:updateGradInput(input:clone(), delta:clone())

  module_f:accGradParameters(input:cuda(), delta:cuda())
  module_s:accGradParameters(input:clone(), delta:clone())

  tester:assertTensorEq(module_f.gradA:float(), module_s.modules[1].gradWeight, precision, "gradA mismatch")
  tester:assertTensorEq(module_f.gradAb:float(), module_s.modules[1].gradBias, precision, "gradAb mismatch")
  tester:assertTensorEq(module_f.gradD:float(), module_s.modules[3].gradWeight, precision, "gradD mismatch")
  tester:assertTensorEq(module_f.gradDb:float(), module_s.modules[3].gradBias, precision, "gradDb mismatch")
end

function fast_acdc_test.CheckSignInit()
  local module = acdc.FastACDC(10, {
    sign_init = true,
  })

  tester:assert(torch.all(torch.eq(module.A:eq(1) + module.A:eq(-1), 1)))
  tester:assert(torch.all(torch.eq(module.D:eq(1) + module.D:eq(-1), 1)))
end

function fast_acdc_test.CheckRandInit()
  local module = acdc.FastACDC(10, {
    rand_init = true,
  })

  tester:assert(torch.all(torch.eq(torch.le(torch.abs(module.A - 1), 0.01), 1)))
  tester:assert(torch.all(torch.eq(torch.le(torch.abs(module.D - 1), 0.01), 1)))
end

function fast_acdc_test.CheckCombinedRandInitAndSignInit()
  local module = acdc.FastACDC(10, {
    sign_init = true,
    rand_init = true,
  })

  local function check(x)
    return torch.all(torch.eq(torch.le(torch.abs(x - 1), 0.01) + torch.le(torch.abs(x + 1), 0.01), 1))
  end

  tester:assert(check(module.A))
  tester:assert(check(module.D))
end

tester:add(fast_acdc_test)

tester:run()
